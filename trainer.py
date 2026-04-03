import logging
import os
import random
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_tn3k import TN3KDataset, RandomGenerator


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


def extract_fg_logits(logits):
    if logits.shape[1] == 1:
        return logits
    if logits.shape[1] == 2:
        return logits[:, 1:2]
    raise ValueError(f'Expected 1 or 2 output channels for binary segmentation, got {logits.shape[1]}')


def binary_seg_loss(logits, target):
    fg_logits = extract_fg_logits(logits)
    target = target.float().unsqueeze(1)
    loss_bce = F.binary_cross_entropy_with_logits(fg_logits, target)
    loss_dice = binary_dice_loss_with_logits(fg_logits, target)
    loss = 0.5 * loss_bce + 0.5 * loss_dice
    return loss, loss_bce, loss_dice


def compute_binary_metrics(logits, target, smooth=1e-5):
    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits) > 0.5).long()
    else:
        pred = torch.argmax(logits, dim=1)

    target = target.long()
    pred_fg = (pred > 0).float()
    target_fg = (target > 0).float()

    dims = (1, 2)
    intersection = (pred_fg * target_fg).sum(dim=dims)
    pred_sum = pred_fg.sum(dim=dims)
    target_sum = target_fg.sum(dim=dims)
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (target_sum + smooth)

    return {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
    }


def reset_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def seed_worker_builder(seed):
    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
    return worker_init_fn


def create_optimizer(args, model):
    optimizer_name = getattr(args, 'optimizer', 'adamw').lower()
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    return optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.01)


def create_dataloaders(args):
    db_train = TN3KDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split='train',
        img_size=args.img_size,
        transform=RandomGenerator(output_size=[args.img_size, args.img_size]),
    )
    db_test = TN3KDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split='test',
        img_size=args.img_size,
        transform=None,
    )

    if args.local_rank != -1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(db_train, shuffle=True)
        test_sampler = DistributedSampler(db_test, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    worker_init_fn = seed_worker_builder(args.seed)
    train_loader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        db_test,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    return db_train, db_test, train_loader, test_loader, train_sampler, test_sampler


@torch.no_grad()
def evaluate_tn3k_official(args, model, test_loader, rank, use_amp):
    was_training = model.training
    model.eval()

    loss_meter = []
    dice_meter = []
    iou_meter = []
    precision_meter = []
    recall_meter = []

    for sampled_batch in test_loader:
        image_batch = sampled_batch['image'].cuda(rank, non_blocking=True) if torch.cuda.is_available() else sampled_batch['image']
        label_batch = sampled_batch['label'].cuda(rank, non_blocking=True) if torch.cuda.is_available() else sampled_batch['label']

        with autocast(enabled=use_amp):
            outputs = model(image_batch, return_edge=False, return_aux=False)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss, _, _ = binary_seg_loss(outputs, label_batch)

        metrics = compute_binary_metrics(outputs, label_batch)
        loss_meter.append(float(loss.item()))
        dice_meter.append(metrics['dice'])
        iou_meter.append(metrics['iou'])
        precision_meter.append(metrics['precision'])
        recall_meter.append(metrics['recall'])

    if was_training:
        model.train()

    return {
        'loss': float(np.mean(loss_meter)) if loss_meter else 0.0,
        'dice': float(np.mean(dice_meter)) if dice_meter else 0.0,
        'iou': float(np.mean(iou_meter)) if iou_meter else 0.0,
        'precision': float(np.mean(precision_meter)) if precision_meter else 0.0,
        'recall': float(np.mean(recall_meter)) if recall_meter else 0.0,
    }


def trainer_synapse(args, model, snapshot_path):
    raise NotImplementedError('This modified trainer only implements TN3K training.')


def trainer_tn3k(args, model, snapshot_path):
    os.makedirs(snapshot_path, exist_ok=True)

    accumulation_steps = args.accumulation_steps if args.accumulation_steps else 1
    use_amp = torch.cuda.is_available() and args.amp_opt_level != 'O0'
    scaler = GradScaler(enabled=use_amp)

    rank = args.local_rank if args.local_rank != -1 else 0
    is_main_process = rank == 0

    if is_main_process:
        logger = reset_logger(os.path.join(snapshot_path, 'log.txt'))
        logger.info(str(args))
        logger.info('Official TN3K train/test split training')
        logger.info(f'Gradient accumulation steps: {accumulation_steps}')
        logger.info(f'AMP enabled: {use_amp}')

    db_train, db_test, train_loader, test_loader, train_sampler, _ = create_dataloaders(args)

    if is_main_process:
        logging.info(f'Train size: {len(db_train)} | Test size: {len(db_test)}')

    optimizer = create_optimizer(args, model)
    iter_num = 0
    max_epoch = args.max_epochs
    updates_per_epoch = max(1, (len(train_loader) + accumulation_steps - 1) // accumulation_steps)
    max_iterations = max_epoch * updates_per_epoch
    edge_warmup_epochs = max(1, int(max_epoch * 0.1))

    best_dice = -1.0
    best_epoch = -1
    optimizer.zero_grad(set_to_none=True)
    epoch_iterator = tqdm(range(max_epoch), ncols=120) if is_main_process else range(max_epoch)

    for epoch_num in epoch_iterator:
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_num)

        base_edge_lambda = getattr(args, 'edge_lambda', 0.2)
        edge_lambda = base_edge_lambda * min(1.0, float(epoch_num + 1) / float(edge_warmup_epochs))
        aux1_weight = getattr(args, 'aux_weight_half', 0.4)
        aux2_weight = getattr(args, 'aux_weight_full', 0.2)

        running_total, running_seg, running_edge = [], [], []
        running_bce, running_dice, running_aux = [], [], []

        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch = sampled_batch['image'].cuda(rank, non_blocking=True) if torch.cuda.is_available() else sampled_batch['image']
            label_batch = sampled_batch['label'].cuda(rank, non_blocking=True) if torch.cuda.is_available() else sampled_batch['label']
            edge_target = build_boundary_target(label_batch, boundary_width=args.boundary_width)

            sync_now = ((i_batch + 1) % accumulation_steps == 0) or ((i_batch + 1) == len(train_loader))
            sync_context = nullcontext()
            if hasattr(model, 'no_sync') and not sync_now:
                sync_context = model.no_sync()

            with sync_context:
                with autocast(enabled=use_amp):
                    outputs, edge_logits, aux_outputs = model(image_batch, return_edge=True, return_aux=True)
                    loss_main, loss_bce, loss_dice = binary_seg_loss(outputs, label_batch)
                    loss_aux1, _, _ = binary_seg_loss(aux_outputs[0], label_batch)
                    loss_aux2, _, _ = binary_seg_loss(aux_outputs[1], label_batch)
                    loss_aux = aux1_weight * loss_aux1 + aux2_weight * loss_aux2
                    loss_seg = loss_main + loss_aux

                    loss_edge, _, _ = boundary_loss(edge_logits, edge_target)
                    loss_total = loss_seg + edge_lambda * loss_edge
                    loss_for_backward = loss_total / accumulation_steps

                scaler.scale(loss_for_backward).backward()

            running_total.append(float(loss_total.item()))
            running_seg.append(float(loss_seg.item()))
            running_edge.append(float(loss_edge.item()))
            running_bce.append(float(loss_bce.item()))
            running_dice.append(float(loss_dice.item()))
            running_aux.append(float(loss_aux.item()))

            if sync_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                iter_num += 1

        test_metrics = evaluate_tn3k_official(args, model, test_loader, rank, use_amp)

        if is_main_process:
            logging.info(
                'epoch=%d train_total=%.6f train_seg=%.6f train_edge=%.6f train_bce=%.6f train_dice=%.6f train_aux=%.6f test_loss=%.6f test_dice=%.6f test_iou=%.6f test_precision=%.6f test_recall=%.6f',
                epoch_num,
                float(np.mean(running_total)) if running_total else 0.0,
                float(np.mean(running_seg)) if running_seg else 0.0,
                float(np.mean(running_edge)) if running_edge else 0.0,
                float(np.mean(running_bce)) if running_bce else 0.0,
                float(np.mean(running_dice)) if running_dice else 0.0,
                float(np.mean(running_aux)) if running_aux else 0.0,
                test_metrics['loss'],
                test_metrics['dice'],
                test_metrics['iou'],
                test_metrics['precision'],
                test_metrics['recall'],
            )

            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, os.path.join(snapshot_path, 'last_model.pth'))

            if test_metrics['dice'] > best_dice:
                best_dice = test_metrics['dice']
                best_epoch = epoch_num
                torch.save(
                    {
                        'epoch': epoch_num,
                        'best_dice': best_dice,
                        'model_state_dict': state_dict,
                    },
                    os.path.join(snapshot_path, 'best_model.pth'),
                )
                logging.info('epoch=%d New best model saved with test_dice=%.6f', epoch_num, best_dice)

        if args.local_rank != -1:
            torch.distributed.barrier()

    return 'Training Finished!', float(best_dice), int(best_epoch)
