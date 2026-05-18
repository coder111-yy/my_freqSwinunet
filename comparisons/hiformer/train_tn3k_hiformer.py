import argparse
import os
import random
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from scipy import ndimage


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from models.HiFormer import HiFormer
from configs.HiFormer_configs import (
    get_hiformer_s_configs,
    get_hiformer_b_configs,
    get_hiformer_l_configs,
)




try:
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
except AttributeError:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


# =========================================================
# Data augmentation, aligned with your TN3K training setting
# =========================================================
def random_rot_flip(image, label):
    """
    image: H x W x C
    label: H x W
    """
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    return image, label


def random_rotate(image, label):
    """
    image: H x W x C
    label: H x W
    """
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(
        image,
        angle,
        order=1,
        reshape=False,
        mode="reflect"
    )
    label = ndimage.rotate(
        label,
        angle,
        order=0,
        reshape=False,
        mode="nearest"
    )
    return image, label


def apply_elastic_transform(image, label, alpha_scale=0.15, sigma_scale=0.04):
    """
    image: H x W x C
    label: H x W
    """
    shape = image.shape[:2]
    alpha = shape[0] * alpha_scale
    sigma = shape[0] * sigma_scale

    random_state = np.random.RandomState(None)

    dx = ndimage.gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma,
        mode="constant",
        cval=0
    ) * alpha

    dy = ndimage.gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma,
        mode="constant",
        cval=0
    ) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    image_t = np.empty_like(image)
    for i in range(image.shape[2]):
        image_t[:, :, i] = ndimage.map_coordinates(
            image[:, :, i],
            indices,
            order=1,
            mode="reflect"
        ).reshape(shape)

    label_t = ndimage.map_coordinates(
        label,
        indices,
        order=0,
        mode="reflect"
    ).reshape(shape)

    return image_t, label_t


def add_speckle_noise(image, variance=0.04):
    """
    image range: [0, 1]
    """
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch) * variance
    noisy_image = image + image * gauss
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    return noisy_image


# =========================================================
# TN3K Dataset
# =========================================================
class TN3KSimpleDataset(Dataset):
    def __init__(self, root_path, list_dir, split="train", img_size=384, use_aug=True):
        self.root_path = root_path
        self.list_dir = list_dir
        self.split = split
        self.img_size = img_size
        self.use_aug = use_aug and split == "train"

        list_file = os.path.join(list_dir, split + ".txt")
        with open(list_file, "r") as f:
            self.sample_list = [line.strip() for line in f.readlines() if line.strip()]

        self.mean = np.array([0.3915, 0.2582, 0.2558], dtype=np.float32)
        self.std = np.array([0.1656, 0.1430, 0.1631], dtype=np.float32)

    def _find_file(self, name, is_mask=False):
        """
        TN3K directory example:
        root_path/
          train/
            images/
            masks/
          test/
            images/
            masks/
        """
        name = name.strip()
        name_no_ext = os.path.splitext(name)[0]
        folder = "masks" if is_mask else "images"

        candidates = []
        exts = ["", ".png", ".jpg", ".jpeg", ".bmp"]

        candidates.append(os.path.join(self.root_path, self.split, folder, name))

        for ext in exts:
            candidates.append(os.path.join(self.root_path, self.split, folder, name_no_ext + ext))

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Cannot find {'mask' if is_mask else 'image'} for sample: {name}. "
            f"Checked folder: {os.path.join(self.root_path, self.split, folder)}"
        )

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]

        image_path = self._find_file(name, is_mask=False)
        mask_path = self._find_file(name, is_mask=True)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.img_size, self.img_size), BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), NEAREST)

        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32)
        mask = (mask > 127).astype(np.float32)

        if self.use_aug:
            if random.random() > 0.5:
                image, mask = random_rot_flip(image, mask)

            if random.random() > 0.5:
                image, mask = random_rotate(image, mask)

            if random.random() > 0.7:
                image, mask = apply_elastic_transform(image, mask)

            if random.random() > 0.7:
                image = add_speckle_noise(image, variance=0.04)

        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask, name


# =========================================================
# Loss and metrics
# =========================================================
def extract_fg_logits(logits):
    """
    logits: [B, 1, H, W] or [B, 2, H, W]
    """
    if isinstance(logits, dict):
        if "out" in logits:
            logits = logits["out"]
        elif "logits" in logits:
            logits = logits["logits"]
        else:
            raise ValueError(f"Unsupported dict output keys: {logits.keys()}")

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if logits.shape[1] == 1:
        return logits
    elif logits.shape[1] == 2:
        return logits[:, 1:2]
    else:
        raise ValueError(f"Expected 1 or 2 output channels, got {logits.shape[1]}")


def dice_loss_with_logits(logits, target, smooth=1e-5):
    prob = torch.sigmoid(logits)

    intersection = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def seg_loss(logits, target):
    """
    logits: [B, 2, H, W] or [B, 1, H, W]
    target: [B, 1, H, W]
    """
    fg_logits = extract_fg_logits(logits)

    if fg_logits.shape[-2:] != target.shape[-2:]:
        fg_logits = F.interpolate(
            fg_logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    target = target.float()

    bce = F.binary_cross_entropy_with_logits(fg_logits, target)
    dice = dice_loss_with_logits(fg_logits, target)

    loss = 0.5 * bce + 0.5 * dice
    return loss


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()

    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []

    smooth = 1e-5

    for images, masks, _ in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            fg_logits = extract_fg_logits(logits)

            if fg_logits.shape[-2:] != masks.shape[-2:]:
                fg_logits = F.interpolate(
                    fg_logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            preds = (torch.sigmoid(fg_logits) > 0.5).float()

        intersection = (preds * masks).sum(dim=(1, 2, 3))
        pred_sum = preds.sum(dim=(1, 2, 3))
        target_sum = masks.sum(dim=(1, 2, 3))
        union = pred_sum + target_sum - intersection

        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        iou = (intersection + smooth) / (union + smooth)
        precision = (intersection + smooth) / (pred_sum + smooth)
        recall = (intersection + smooth) / (target_sum + smooth)

        dice_list.extend(dice.detach().cpu().numpy().tolist())
        iou_list.extend(iou.detach().cpu().numpy().tolist())
        precision_list.extend(precision.detach().cpu().numpy().tolist())
        recall_list.extend(recall.detach().cpu().numpy().tolist())

    return {
        "dice": float(np.mean(dice_list)),
        "iou": float(np.mean(iou_list)),
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
    }


# =========================================================
# Model
# =========================================================
def build_model(img_size=384, model_name="hiformer-b"):
    if model_name == "hiformer-s":
        config = get_hiformer_s_configs()
    elif model_name == "hiformer-b":
        config = get_hiformer_b_configs()
    elif model_name == "hiformer-l":
        config = get_hiformer_l_configs()
    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Choose from hiformer-s, hiformer-b, hiformer-l."
        )

    config.image_size = img_size
    config.num_classes = 2

    model = HiFormer(
        config=config,
        img_size=img_size,
        in_chans=3,
        n_classes=2
    )

    return model


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker_builder(seed):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--list_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=1)

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--no_amp", action="store_true", help="Disable AMP mixed precision.")
    parser.add_argument("--no_aug", action="store_true", help="Disable training augmentation.")

    parser.add_argument("--deterministic", type=int, default=0)

    parser.add_argument(
        "--model_name",
        type=str,
        default="hiformer-b",
        choices=["hiformer-s", "hiformer-b", "hiformer-l"],
        help="HiFormer variant."
    )

    args = parser.parse_args()

    use_amp = not args.no_amp

    set_seed(args.seed)

    if args.deterministic:
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = False

    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "train_log.txt")
    with open(log_path, "w") as f:
        f.write(str(args) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==================================================")
    print(f"Device: {device}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Accumulation steps: {args.accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"AMP: {use_amp}")
    print(f"Use augmentation: {not args.no_aug}")
    print("==================================================")

    train_set = TN3KSimpleDataset(
        root_path=args.root_path,
        list_dir=args.list_dir,
        split="train",
        img_size=args.img_size,
        use_aug=not args.no_aug
    )

    test_set = TN3KSimpleDataset(
        root_path=args.root_path,
        list_dir=args.list_dir,
        split="test",
        img_size=args.img_size,
        use_aug=False
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker_builder(args.seed),
        generator=generator,
        drop_last=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker_builder(args.seed),
        drop_last=False
    )

    print(f"Train samples: {len(train_set)}")
    print(f"Test samples: {len(test_set)}")
    print(f"Train iterations per epoch: {len(train_loader)}")

    model = build_model(
        img_size=args.img_size,
        model_name=args.model_name
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    scaler = GradScaler(enabled=use_amp)

    best_dice = 0.0
    best_metrics = None

    best_path = os.path.join(args.output_dir, "best_model.pth")
    last_path = os.path.join(args.output_dir, "last_model.pth")

    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch + 1}/{args.max_epochs}",
            ncols=120
        )

        for step, (images, masks, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits = model(images)
                loss = seg_loss(logits, masks)

                loss_for_backward = loss / args.accumulation_steps

            scaler.scale(loss_for_backward).backward()

            should_update = (
                (step + 1) % args.accumulation_steps == 0
                or (step + 1) == len(train_loader)
            )

            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=use_amp
        )

        log_msg = (
            f"Epoch {epoch + 1}/{args.max_epochs}: "
            f"loss={avg_loss:.4f}, "
            f"dice={metrics['dice']:.4f}, "
            f"iou={metrics['iou']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}"
        )

        print(log_msg)

        with open(log_path, "a") as f:
            f.write(log_msg + "\n")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "metrics": metrics,
                "args": vars(args),
            },
            last_path
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_metrics = metrics

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "metrics": metrics,
                    "args": vars(args),
                },
                best_path
            )

            print(f"Saved best model to {best_path}, best dice={best_dice:.4f}")

            with open(log_path, "a") as f:
                f.write(f"Saved best model: epoch={epoch + 1}, best_dice={best_dice:.4f}\n")

    print("==================================================")
    print("Training finished.")
    print(f"Best Dice = {best_dice:.4f}")

    if best_metrics is not None:
        print(
            f"Best metrics: "
            f"dice={best_metrics['dice']:.4f}, "
            f"iou={best_metrics['iou']:.4f}, "
            f"precision={best_metrics['precision']:.4f}, "
            f"recall={best_metrics['recall']:.4f}"
        )

    print(f"Best model path: {best_path}")
    print("==================================================")


if __name__ == "__main__":
    main()