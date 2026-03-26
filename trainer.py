import logging
import os
import random
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import DiceLoss
from datasets.dataset_tn3k import TN3KDataset, RandomGenerator
from torch.cuda.amp import autocast, GradScaler


def get_largest_connected_component(binary_mask):
    import skimage.measure as measure
    labeled_mask, num_components = measure.label(binary_mask, connectivity=2, return_num=True)
    if num_components == 0:
        return binary_mask

    max_label = 0
    max_area = 0
    for i in range(1, num_components + 1):
        area = (labeled_mask == i).sum()
        if area > max_area:
            max_area = area
            max_label = i

    largest_component = (labeled_mask == max_label).astype(np.uint8)
    return largest_component


def build_boundary_target(label_batch, boundary_width=2):
    mask = label_batch.float().unsqueeze(1)
    k = boundary_width * 2 + 1
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=boundary_width)
    ero = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=boundary_width)
    boundary = (dil - ero > 0).float()
    return boundary


def binary_dice_loss_with_logits(logits, target, smooth=1e-5):
    prob = torch.sigmoid(logits)
    dims = (2, 3)
    intersection = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def boundary_loss(edge_logits, edge_target):
    pos = edge_target.sum()
    neg = edge_target.numel() - pos
    pos_weight = ((neg + 1.0) / (pos + 1.0)).clamp(1.0, 10.0)
    pos_weight = pos_weight.to(edge_logits.device, dtype=edge_logits.dtype)

    loss_bce = F.binary_cross_entropy_with_logits(edge_logits, edge_target, pos_weight=pos_weight)
    loss_dice = binary_dice_loss_with_logits(edge_logits, edge_target)
    return 0.5 * loss_bce + 0.5 * loss_dice, loss_bce, loss_dice


def trainer_synapse(args, model, snapshot_path):
    raise NotImplementedError('This modified trainer only implements TN3K training.')


def trainer_tn3k(args, model, snapshot_path):
    accumulation_steps = args.accumulation_steps if args.accumulation_steps else 1
    use_amp = torch.cuda.is_available() and args.amp_opt_level != 'O0'
    scaler = GradScaler(enabled=use_amp)

    rank = args.local_rank if args.local_rank != -1 else 0
    is_main_process = rank == 0

    if is_main_process:
        logging.basicConfig(filename=os.path.join(snapshot_path, 'log.txt'), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(f'Gradient accumulation steps: {accumulation_steps}')
        logging.info(f'AMP enabled: {use_amp}')

    num_classes = args.num_classes
    batch_size = args.batch_size
    base_lr = args.base_lr

    db_train = TN3KDataset(base_dir=args.root_path,
                           list_dir=args.list_dir,
                           split='train',
                           img_size=args.img_size,
                           transform=RandomGenerator(output_size=[args.img_size, args.img_size]))

    if is_main_process:
        logging.info('The length of train set is: %d', len(db_train))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)

    if args.local_rank != -1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(db_train, shuffle=True)
    else:
        train_sampler = None

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        sampler=train_sampler,
        drop_last=False,
    )

    model.train()
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer_name = getattr(args, 'optimizer', 'adamw').lower()
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    iter_num = 0
    max_epoch = args.max_epochs
    updates_per_epoch = (len(trainloader) + accumulation_steps - 1) // accumulation_steps
    max_iterations = max_epoch * updates_per_epoch
    edge_warmup_epochs = max(1, int(max_epoch * 0.1))

    if is_main_process:
        logging.info('%d iterations per epoch. %d max optimizer updates.', len(trainloader), max_iterations)

    iterator = tqdm(range(max_epoch), ncols=100) if is_main_process else range(max_epoch)
    optimizer.zero_grad(set_to_none=True)

    for epoch_num in iterator:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_num)

        edge_lambda = args.edge_lambda * min(1.0, float(epoch_num + 1) / float(edge_warmup_epochs))

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch['image'].cuda(rank, non_blocking=True) if torch.cuda.is_available() else sampled_batch['image']
            label_batch = sampled_batch['label'].cuda(rank, non_blocking=True) if torch.cuda.is_available() else sampled_batch['label']
            edge_target = build_boundary_target(label_batch, boundary_width=args.boundary_width)

            sync_now = ((i_batch + 1) % accumulation_steps == 0) or ((i_batch + 1) == len(trainloader))
            sync_context = nullcontext()
            if hasattr(model, 'no_sync') and not sync_now:
                sync_context = model.no_sync()

            with sync_context:
                with autocast(enabled=use_amp):
                    outputs, edge_logits = model(image_batch, return_edge=True)
                    loss_ce = ce_loss(outputs, label_batch.long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    loss_seg = 0.5 * loss_ce + 0.5 * loss_dice

                    loss_edge, loss_edge_bce, loss_edge_dice = boundary_loss(edge_logits, edge_target)
                    loss_total = loss_seg + edge_lambda * loss_edge
                    loss_for_backward = loss_total / accumulation_steps

                scaler.scale(loss_for_backward).backward()

            if sync_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num += 1

                if is_main_process and iter_num % 20 == 0:
                    logging.info(
                        'iteration %d : total=%f seg=%f ce=%f dice=%f edge=%f edge_bce=%f edge_dice=%f edge_lambda=%f lr=%f',
                        iter_num,
                        loss_total.item(),
                        loss_seg.item(),
                        loss_ce.item(),
                        loss_dice.item(),
                        loss_edge.item(),
                        loss_edge_bce.item(),
                        loss_edge_dice.item(),
                        edge_lambda,
                        lr_,
                    )

        if is_main_process:
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            last_save_path = os.path.join(snapshot_path, 'last_model.pth')
            torch.save(state_dict, last_save_path)

            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(state_dict, save_mode_path)

            obsolete_epoch = epoch_num - 100
            if obsolete_epoch >= 0:
                obsolete_model_path = os.path.join(snapshot_path, 'epoch_' + str(obsolete_epoch) + '.pth')
                if os.path.exists(obsolete_model_path):
                    os.remove(obsolete_model_path)

    return 'Training Finished!'
