import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils import DiceLoss
from datasets.dataset_tn3k import TN3KDataset, RandomGenerator


# ================= 边界提取与边界损失 =================
def get_boundary(mask, kernel_size=3):
    dilation = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    erosion = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    boundary = dilation - erosion
    return boundary


class BoundaryLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BoundaryLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, gt):
        pred_boundary = get_boundary(pred)
        gt_boundary = get_boundary(gt)

        pred_b_flat = pred_boundary.reshape(-1)
        gt_b_flat = gt_boundary.reshape(-1)

        intersection = (pred_b_flat * gt_b_flat).sum()
        union = pred_b_flat.sum() + gt_b_flat.sum()
        boundary_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        pred_boundary_clamp = torch.clamp(pred_boundary, 1e-6, 1.0 - 1e-6)

        # 关键：关闭 autocast，避免 BCE 报错
        with torch.cuda.amp.autocast(enabled=False):
            bce_loss = F.binary_cross_entropy(
                pred_boundary_clamp.float(),
                gt_boundary.float()
            )

        loss = bce_loss + (1.0 - boundary_dice)
        return loss


# ================= 评估指标 =================
def compute_dice(pred, gt, eps=1e-5):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter + eps) / (pred.sum() + gt.sum() + eps)


def compute_iou(pred, gt, eps=1e-5):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + eps) / (union + eps)


def get_surface(mask):
    if mask.sum() == 0:
        return mask
    eroded = binary_erosion(mask)
    surface = mask ^ eroded
    return surface


def compute_hd95(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 100.0

    pred_surface = get_surface(pred)
    gt_surface = get_surface(gt)

    dt_pred = distance_transform_edt(~pred_surface)
    dt_gt = distance_transform_edt(~gt_surface)

    dist_pred_to_gt = dt_gt[pred_surface]
    dist_gt_to_pred = dt_pred[gt_surface]

    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return np.percentile(all_dists, 95)


@torch.no_grad()
def evaluate_tn3k(model, dataloader, device):
    model.eval()

    dice_list = []
    iou_list = []
    hd95_list = []

    for sampled_batch in dataloader:
        image_batch = sampled_batch["image"].to(device, non_blocking=True)
        label_batch = sampled_batch["label"].to(device, non_blocking=True)

        with autocast():
            outputs = model(image_batch)

        pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        gt_np = label_batch.squeeze(0).cpu().numpy().astype(np.uint8)

        dice_list.append(compute_dice(pred_np, gt_np))
        iou_list.append(compute_iou(pred_np, gt_np))
        hd95_list.append(compute_hd95(pred_np, gt_np))

    metrics = {
        "dice": float(np.mean(dice_list)),
        "iou": float(np.mean(iou_list)),
        "hd95": float(np.mean(hd95_list)),
    }

    model.train()
    return metrics


def trainer_synapse(args, model, snapshot_path):
    pass


def trainer_tn3k(args, model, snapshot_path):
    accumulation_steps = args.accumulation_steps if args.accumulation_steps else 6
    scaler = GradScaler()
    device = torch.device("cuda", args.local_rank if args.local_rank != -1 else 0)

    os.makedirs(snapshot_path, exist_ok=True)

    if args.local_rank == 0:
        logging.basicConfig(
            filename=os.path.join(snapshot_path, "log.txt"),
            level=logging.INFO,
            format='[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%H:%M:%S'
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(
            f"Using Gradient Accumulation: Real Batch Size={args.batch_size}, "
            f"Accumulation Steps={accumulation_steps}, "
            f"Virtual Batch Size={args.batch_size * accumulation_steps}"
        )

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size

    # ================= 数据集 =================
    db_train = TN3KDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        img_size=args.img_size,
        transform=RandomGenerator(output_size=[args.img_size, args.img_size])
    )

    db_val = TN3KDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="val",
        img_size=args.img_size,
        transform=None
    )

    db_test = TN3KDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="test",
        img_size=args.img_size,
        transform=None
    )

    if args.local_rank == 0:
        print("The length of train set is: {}".format(len(db_train)))
        print("The length of val set is: {}".format(len(db_val)))
        print("The length of test set is: {}".format(len(db_test)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # ================= train 仍然走 DDP sampler =================
    train_sampler = DistributedSampler(db_train, shuffle=True)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        sampler=train_sampler,
    )

    # ================= 方案1：val/test 只在 rank0 上单卡评估 =================
    if args.local_rank == 0:
        valloader = DataLoader(
            db_val,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
        )

        testloader = DataLoader(
            db_test,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
        )
    else:
        valloader = None
        testloader = None

    model.train()

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    boundary_loss_fn = BoundaryLoss().to(device)

    # 你后面可再调，这里先保留
    max_alpha = 0.2

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    warmup_epochs = int(max_epoch * 0.2)

    best_val_dice = -1.0
    best_epoch = -1
    best_metrics = None

    if args.local_rank == 0:
        logging.info("{} iterations per epoch. {} max iterations".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70) if args.local_rank == 0 else range(max_epoch)

    optimizer.zero_grad()

    for epoch_num in iterator:
        train_sampler.set_epoch(epoch_num)

        if epoch_num < warmup_epochs:
            current_alpha = 0.0
        else:
            current_alpha = max_alpha * ((epoch_num - warmup_epochs) / max(1, (max_epoch - warmup_epochs)))

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch["image"].to(device, non_blocking=True)
            label_batch = sampled_batch["label"].to(device, non_blocking=True)

            with autocast():
                outputs = model(image_batch)

                loss_ce = ce_loss(outputs, label_batch.long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)

                pred_prob = torch.softmax(outputs, dim=1)[:, 1:2, :, :]
                gt_mask = label_batch.unsqueeze(1).float()

                loss_boundary = boundary_loss_fn(pred_prob, gt_mask)

                loss = 0.4 * loss_ce + 0.6 * loss_dice + current_alpha * loss_boundary
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i_batch + 1) % accumulation_steps == 0 or (i_batch + 1) == len(trainloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                iter_num += 1
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_

                if args.local_rank == 0 and iter_num % 20 == 0:
                    logging.info(
                        "iteration %d : loss: %f, loss_ce: %f, loss_dice: %f, loss_boundary: %f, alpha: %f"
                        % (
                            iter_num,
                            loss.item() * accumulation_steps,
                            loss_ce.item(),
                            loss_dice.item(),
                            loss_boundary.item(),
                            current_alpha,
                        )
                    )

        # ================= 方案1：只有 rank0 做验证 =================
        if args.local_rank == 0 and (epoch_num + 1) % args.eval_interval == 0:
            model_to_eval = model.module if hasattr(model, "module") else model
            val_metrics = evaluate_tn3k(model_to_eval, valloader, device)

            logging.info(
                f"Epoch {epoch_num}: "
                f"val_dice={val_metrics['dice']:.4f}, "
                f"val_iou={val_metrics['iou']:.4f}, "
                f"val_hd95={val_metrics['hd95']:.4f}"
            )

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                best_epoch = epoch_num
                best_metrics = val_metrics

                best_save_path = os.path.join(snapshot_path, "best_model.pth")
                torch.save(model_to_eval.state_dict(), best_save_path)
                logging.info(
                    f"save best model to {best_save_path}, "
                    f"best_epoch={best_epoch}, best_val_dice={best_val_dice:.4f}"
                )

        # 每个 epoch 后同步一下，避免 rank0 在评估时其他卡继续乱跑
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # ================= 常规保存仍只在 rank0 =================
        if args.local_rank == 0:
            model_to_save = model.module if hasattr(model, "module") else model

            last_save_path = os.path.join(snapshot_path, 'last_model.pth')
            torch.save(model_to_save.state_dict(), last_save_path)
            logging.info(f"save last model to {last_save_path}")

    # ================= 方案1：只有 rank0 做最终 test =================
    if args.local_rank == 0:
        logging.info("Training finished. Loading best model for final test evaluation...")

        model_to_eval = model.module if hasattr(model, "module") else model

        best_path = os.path.join(snapshot_path, "best_model.pth")
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"best_model.pth not found at {best_path}")

        state_dict = torch.load(best_path, map_location=device)
        model_to_eval.load_state_dict(state_dict)

        test_metrics = evaluate_tn3k(model_to_eval, testloader, device)

        logging.info(
            f"Final TEST metrics using best model: "
            f"dice={test_metrics['dice']:.4f}, "
            f"iou={test_metrics['iou']:.4f}, "
            f"hd95={test_metrics['hd95']:.4f}"
        )

        result_dict = {
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_val_metrics": best_metrics,
            "test_metrics": test_metrics,
        }

        with open(os.path.join(snapshot_path, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return "Training Finished!"