import logging
import os
import random
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
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


class DistanceWeightedBCELoss(nn.Module):
    def __init__(self, penalty_factor=0.05):
        super(DistanceWeightedBCELoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.penalty_factor = penalty_factor

    def forward(self, pred, gt, dist_map):
        pred_clamp = torch.clamp(pred, 1e-6, 1.0 - 1e-6)

        # ---------------------------------------------------------
        # I/O 消除：直接使用从 Dataloader 并行加载的 GPU 距离图张量
        # ---------------------------------------------------------
        weight_tensor = 1.0 + self.penalty_factor * dist_map

        base_bce = self.bce(pred_clamp, gt.float())
        weighted_loss = (base_bce * weight_tensor).mean()

        return weighted_loss


def trainer_synapse(args, model, snapshot_path):
    pass


def trainer_tn3k(args, model, snapshot_path):
    accumulation_steps = args.accumulation_steps if args.accumulation_steps else 6
    scaler = GradScaler()

    if args.local_rank == 0:
        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(f"Using Gradient Accumulation: Real Batch Size={args.batch_size}, "
                     f"Accumulation Steps={accumulation_steps}, "
                     f"Virtual Batch Size={args.batch_size * accumulation_steps}")

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size

    db_train = TN3KDataset(base_dir=args.root_path,
                           list_dir=args.list_dir,
                           split="train",
                           img_size=args.img_size,
                           transform=RandomGenerator(output_size=[args.img_size, args.img_size]))

    if args.local_rank == 0:
        print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(db_train, shuffle=True)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn,
                             sampler=train_sampler)

    model.train()

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    dist_loss_fn = DistanceWeightedBCELoss(penalty_factor=0.05).cuda(args.local_rank)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    # ---------------------------------------------------------
    # 动力学控制：课程学习 (Curriculum Learning) 退火参数设置
    # ---------------------------------------------------------
    warmup_epochs = int(max_epoch * 0.2)
    max_dist_alpha = 0.2

    if args.local_rank == 0:
        logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70) if args.local_rank == 0 else range(max_epoch)

    optimizer.zero_grad()

    for epoch_num in iterator:
        train_sampler.set_epoch(epoch_num)

        # 动态计算当前 Epoch 的距离惩罚损失权重系数
        if epoch_num < warmup_epochs:
            current_alpha = 0.0
        else:
            current_alpha = max_dist_alpha * ((epoch_num - warmup_epochs) / (max_epoch - warmup_epochs))

        for i_batch, sampled_batch in enumerate(trainloader):
            # 获取额外的 dist_map 张量
            image_batch, label_batch, dist_map = sampled_batch['image'], sampled_batch['label'], sampled_batch[
                'dist_map']

            image_batch = image_batch.cuda(args.local_rank, non_blocking=True)
            label_batch = label_batch.cuda(args.local_rank, non_blocking=True)
            dist_map = dist_map.cuda(args.local_rank, non_blocking=True)

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            pred_prob = torch.softmax(outputs, dim=1)[:, 1:2, :, :]
            gt_mask = label_batch.unsqueeze(1).float()

            loss_dist = dist_loss_fn(pred_prob, gt_mask, dist_map)

            # 引入动态权重 current_alpha 控制距离惩罚强度
            loss = 0.3 * loss_ce + 0.5 * loss_dice + current_alpha * loss_dist

            loss = loss / accumulation_steps
            loss.backward()

            if (i_batch + 1) % accumulation_steps == 0 or (i_batch + 1) == len(trainloader):
                optimizer.step()
                optimizer.zero_grad()

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1

                if args.local_rank == 0:
                    if iter_num % 20 == 0:
                        logging.info(
                            'iteration %d : loss: %f, ce: %f, dice: %f, dist: %f, alpha: %f' %
                            (iter_num, loss.item() * accumulation_steps, loss_ce.item(), loss_dice.item(),
                             loss_dist.item(), current_alpha))

        if args.local_rank == 0:
            last_save_path = os.path.join(snapshot_path, 'last_model.pth')
            torch.save(model.state_dict(), last_save_path)

            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)

            obsolete_epoch = epoch_num - 100
            if obsolete_epoch >= 0:
                obsolete_model_path = os.path.join(snapshot_path, 'epoch_' + str(obsolete_epoch) + '.pth')
                if os.path.exists(obsolete_model_path):
                    os.remove(obsolete_model_path)

    return "Training Finished!"