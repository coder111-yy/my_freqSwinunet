import argparse
import os
import random
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from Models import U_Net, AttU_Net, NestedUNet


class TN3KSimpleDataset(Dataset):
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

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        ])

    def _find_file(self, name, is_mask=False):
        """
        适配你的 TN3K 数据结构：
        root_path/
          train/
            images/
            masks/
          test/
            images/
            masks/
        """

        # 如果 train.txt / test.txt 里已经带扩展名，例如 xxx.jpg
        # 这里先保留原名，同时也准备一个去掉扩展名的名字
        name = name.strip()
        name_no_ext = os.path.splitext(name)[0]

        folder = "masks" if is_mask else "images"

        candidates = []

        # 常见扩展名
        exts = ["", ".png", ".jpg", ".jpeg", ".bmp"]

        # 情况 1：txt 里写的是完整文件名，例如 0001.jpg
        candidates.append(os.path.join(self.root_path, self.split, folder, name))

        # 情况 2：txt 里写的是不带扩展名的名字，例如 0001
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

        image = self.image_transform(image)

        mask = self.mask_transform(mask)
        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask, name


def dice_loss_with_logits(logits, target, smooth=1e-5):
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def seg_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_with_logits(logits, target)
    return 0.5 * bce + 0.5 * dice


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []

    smooth = 1e-5

    for images, masks, _ in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        intersection = (preds * masks).sum(dim=(1, 2, 3))
        pred_sum = preds.sum(dim=(1, 2, 3))
        target_sum = masks.sum(dim=(1, 2, 3))
        union = pred_sum + target_sum - intersection

        dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
        iou = (intersection + smooth) / (union + smooth)
        precision = (intersection + smooth) / (pred_sum + smooth)
        recall = (intersection + smooth) / (target_sum + smooth)

        dice_list.extend(dice.cpu().numpy().tolist())
        iou_list.extend(iou.cpu().numpy().tolist())
        precision_list.extend(precision.cpu().numpy().tolist())
        recall_list.extend(recall.cpu().numpy().tolist())

    return {
        "dice": float(np.mean(dice_list)),
        "iou": float(np.mean(iou_list)),
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
    }


def build_model(model_name):
    model_name = model_name.lower()

    if model_name == "unet":
        return U_Net(in_ch=3, out_ch=1)

    if model_name == "attention_unet":
        return AttU_Net(img_ch=3, output_ch=1)

    if model_name == "unetpp":
        return NestedUNet(in_ch=3, out_ch=1)

    raise ValueError(f"Unknown model: {model_name}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "attention_unet", "unetpp"])
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--list_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = TN3KSimpleDataset(
        root_path=args.root_path,
        list_dir=args.list_dir,
        split="train",
        img_size=args.img_size
    )

    test_set = TN3KSimpleDataset(
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
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = build_model(args.model).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.01)

    best_dice = 0.0
    best_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{args.max_epochs}")
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = seg_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

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