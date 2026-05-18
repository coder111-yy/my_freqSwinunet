import argparse
import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "/home/ub/pql/Swin-Unet-main"

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg



class TN3KDeepLabDataset(Dataset):
    def __init__(self, root_path, list_dir, split="train", img_size=384):
        self.root_path = root_path
        self.list_dir = list_dir
        self.split = split
        self.img_size = img_size

        list_file = os.path.join(list_dir, split + ".txt")
        with open(list_file, "r") as f:
            self.sample_list = [line.strip() for line in f.readlines() if line.strip()]

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3915, 0.2582, 0.2558],
                                 std=[0.1656, 0.1430, 0.1631])
        ])

        self.mask_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=Image.NEAREST
        )

    def _find_file(self, name, is_mask=False):
        name = name.strip()
        name_no_ext = os.path.splitext(name)[0]

        folder = "masks" if is_mask else "images"
        base_dir = os.path.join(self.root_path, self.split, folder)

        candidates = []
        exts = ["", ".png", ".jpg", ".jpeg", ".bmp"]

        candidates.append(os.path.join(base_dir, name))

        for ext in exts:
            candidates.append(os.path.join(base_dir, name_no_ext + ext))

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Cannot find {'mask' if is_mask else 'image'} for sample: {name}. "
            f"Checked folder: {base_dir}"
        )

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]

        image_path = self._find_file(name, is_mask=False)
        mask_path = self._find_file(name, is_mask=True)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)

        mask = self.mask_resize(mask)
        mask = np.array(mask)
        mask = (mask > 127).astype(np.int64)

        mask = torch.from_numpy(mask)

        return image, mask, name


def extract_fg_logits(logits):
    """
    DeepLabV3+ 输出为 [B, 2, H, W]。
    为了和你的 SPFD-Net 评估逻辑一致，这里取前景通道 logits[:, 1:2]。
    """
    if logits.shape[1] == 1:
        return logits

    if logits.shape[1] == 2:
        return logits[:, 1:2]

    raise ValueError(f"Expected 1 or 2 output channels, got {logits.shape[1]}")


def binary_dice_loss_with_logits(logits, target, smooth=1e-5):
    """
    logits: [B, 1, H, W]
    target: [B, 1, H, W]
    """
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def binary_seg_loss(logits, target):
    """
    与你 SPFD-Net 的主分割损失保持一致：
    BCE + Dice。
    """
    fg_logits = extract_fg_logits(logits)
    target = target.float().unsqueeze(1)

    loss_bce = F.binary_cross_entropy_with_logits(fg_logits, target)
    loss_dice = binary_dice_loss_with_logits(fg_logits, target)

    loss = 0.5 * loss_bce + 0.5 * loss_dice
    return loss, loss_bce, loss_dice


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    dice_meter = []
    iou_meter = []
    precision_meter = []
    recall_meter = []

    smooth = 1e-5

    for images, masks, _ in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device)

        logits = model(images)

        if logits.shape[1] == 1:
            pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
        elif logits.shape[1] == 2:
            pred = (torch.sigmoid(logits[:, 1:2]) > 0.5).long().squeeze(1)
        else:
            raise ValueError(f"Expected 1 or 2 output channels, got {logits.shape[1]}")

        target = masks.long()

        pred_fg = (pred > 0).float()
        target_fg = (target > 0).float()

        intersection = (pred_fg * target_fg).sum(dim=(1, 2))
        pred_sum = pred_fg.sum(dim=(1, 2))
        target_sum = target_fg.sum(dim=(1, 2))
        union = pred_sum + target_sum - intersection

        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        iou = (intersection + smooth) / (union + smooth)
        precision = (intersection + smooth) / (pred_sum + smooth)
        recall = (intersection + smooth) / (target_sum + smooth)

        # 与你的 SPFD-Net 评估风格保持一致：batch mean 再求平均
        dice_meter.append(dice.mean().item())
        iou_meter.append(iou.mean().item())
        precision_meter.append(precision.mean().item())
        recall_meter.append(recall.mean().item())

    return {
        "dice": float(np.mean(dice_meter)) if dice_meter else 0.0,
        "iou": float(np.mean(iou_meter)) if iou_meter else 0.0,
        "precision": float(np.mean(precision_meter)) if precision_meter else 0.0,
        "recall": float(np.mean(recall_meter)) if recall_meter else 0.0,
    }


def build_model(args):
    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = args.img_size
    config.freeze()

    model = ViT_seg(
        config,
        img_size=args.img_size,
        num_classes=2
    )

    # 如果 config 里设置了预训练权重，就加载；没有就跳过
    try:
        model.load_from(config)
    except Exception as e:
        print("Warning: failed to load pretrained weights:", e)
        print("Continue training from scratch.")

    return model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
    parser.add_argument("--resume", default="")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default="O1")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")

    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--list_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--pretrained_backbone", action="store_true",
                        help="use ImageNet pretrained backbone if available")

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    if torch.cuda.is_available():
        print("CUDA current device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    train_set = TN3KDeepLabDataset(
        root_path=args.root_path,
        list_dir=args.list_dir,
        split="train",
        img_size=args.img_size
    )

    test_set = TN3KDeepLabDataset(
        root_path=args.root_path,
        list_dir=args.list_dir,
        split="test",
        img_size=args.img_size
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = build_model(args).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    best_dice = 0.0
    best_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{args.max_epochs}")

        for images, masks, _ in pbar:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device)

            logits = model(images)

            loss, loss_bce, loss_dice = binary_seg_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                bce=f"{loss_bce.item():.4f}",
                dice=f"{loss_dice.item():.4f}"
            )

        metrics = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch + 1}: "
            f"loss={total_loss / len(train_loader):.4f}, "
            f"dice={metrics['dice']:.4f}, "
            f"iou={metrics['iou']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}"
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}, best dice={best_dice:.4f}")

    print(f"Training finished. Best Dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()