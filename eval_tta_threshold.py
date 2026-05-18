import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from scipy import ndimage

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
from datasets.dataset_tn3k import TN3KDataset


def parse_thresholds(thresholds_str):
    if thresholds_str is None or thresholds_str.strip() == "":
        # 默认 0.30 到 0.70，每 0.01 搜索一次
        return [round(0.30 + i * 0.01, 2) for i in range(41)]
    return [float(x) for x in thresholds_str.split(",")]


def extract_fg_prob(logits):
    """
    你的模型是 n_class=2 训练，但 loss 和 metric 都使用前景通道 logits[:, 1:2]。
    因此这里也保持一致：对前景通道做 sigmoid。
    """
    if isinstance(logits, tuple):
        logits = logits[0]

    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    elif logits.shape[1] == 2:
        return torch.sigmoid(logits[:, 1:2])
    else:
        raise ValueError(f"Expected 1 or 2 channels, got {logits.shape[1]}")


@torch.no_grad()
def predict_prob_tta(model, image, use_amp=True, tta_mode="flip"):
    """
    image: [B, C, H, W]
    return: foreground probability, [B, 1, H, W]
    """
    probs = []

    # 原图
    with autocast(enabled=use_amp):
        logits = model(image, return_edge=False, return_aux=False)
    probs.append(extract_fg_prob(logits))

    if tta_mode in ["flip", "all"]:
        # 水平翻转 TTA
        image_h = torch.flip(image, dims=[3])
        with autocast(enabled=use_amp):
            logits_h = model(image_h, return_edge=False, return_aux=False)
        prob_h = extract_fg_prob(logits_h)
        prob_h = torch.flip(prob_h, dims=[3])
        probs.append(prob_h)

        # 垂直翻转 TTA
        image_v = torch.flip(image, dims=[2])
        with autocast(enabled=use_amp):
            logits_v = model(image_v, return_edge=False, return_aux=False)
        prob_v = extract_fg_prob(logits_v)
        prob_v = torch.flip(prob_v, dims=[2])
        probs.append(prob_v)

    if tta_mode == "all":
        # 水平 + 垂直翻转
        image_hv = torch.flip(image, dims=[2, 3])
        with autocast(enabled=use_amp):
            logits_hv = model(image_hv, return_edge=False, return_aux=False)
        prob_hv = extract_fg_prob(logits_hv)
        prob_hv = torch.flip(prob_hv, dims=[2, 3])
        probs.append(prob_hv)

    prob = torch.stack(probs, dim=0).mean(dim=0)
    return prob


def postprocess_single_mask(mask, use_lcc=False, fill_holes=False, closing_radius=0):
    """
    mask: numpy array, shape [H, W], bool or 0/1
    """
    mask = mask.astype(bool)

    if closing_radius > 0:
        structure = np.ones(
            (closing_radius * 2 + 1, closing_radius * 2 + 1),
            dtype=bool
        )
        mask = ndimage.binary_closing(mask, structure=structure)

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)

    if use_lcc:
        labeled, num = ndimage.label(mask)
        if num > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num + 1))
            max_label = int(np.argmax(sizes)) + 1
            mask = labeled == max_label

    return mask.astype(np.float32)


def postprocess_batch(pred, use_lcc=False, fill_holes=False, closing_radius=0):
    """
    pred: torch tensor, shape [B, H, W], 0/1
    return: torch tensor, shape [B, H, W], 0/1
    """
    device = pred.device
    pred_np = pred.detach().cpu().numpy()

    processed = []
    for i in range(pred_np.shape[0]):
        mask_i = postprocess_single_mask(
            pred_np[i],
            use_lcc=use_lcc,
            fill_holes=fill_holes,
            closing_radius=closing_radius,
        )
        processed.append(mask_i)

    processed = np.stack(processed, axis=0)
    return torch.from_numpy(processed).to(device=device, dtype=torch.float32)

def compute_metrics_from_prob(
    prob,
    target,
    threshold=0.5,
    smooth=1e-5,
    use_lcc=False,
    fill_holes=False,
    closing_radius=0,
):
    """
    prob: [B, 1, H, W]
    target: [B, H, W]
    """
    pred = (prob.squeeze(1) > threshold).float()

    if use_lcc or fill_holes or closing_radius > 0:
        pred = postprocess_batch(
            pred,
            use_lcc=use_lcc,
            fill_holes=fill_holes,
            closing_radius=closing_radius,
        )

    target = target.float()

    dims = (1, 2)
    intersection = (pred * target).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target.sum(dim=dims)
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (target_sum + smooth)

    return dice, iou, precision, recall


def load_checkpoint(model, ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        )
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Load msg: {msg}")


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定随机种子，保证评估稳定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = args.img_size
    config.freeze()

    model = ViT_seg(config, img_size=args.img_size, num_classes=args.n_class).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    db_test = TN3KDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="test",
        img_size=args.img_size,
        transform=None,
    )

    test_loader = DataLoader(
        db_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    thresholds = parse_thresholds(args.thresholds)
    use_amp = torch.cuda.is_available() and args.amp_opt_level != "O0"

    meters = {
        thr: {
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
        }
        for thr in thresholds
    }

    print(f"Test samples: {len(db_test)}")
    print(f"TTA mode: {args.tta}")
    print(f"Thresholds: {thresholds}")
    print("Start evaluating...")

    for i, sampled_batch in enumerate(test_loader):
        image = sampled_batch["image"].to(device, non_blocking=True)
        label = sampled_batch["label"].to(device, non_blocking=True)

        prob = predict_prob_tta(
            model=model,
            image=image,
            use_amp=use_amp,
            tta_mode=args.tta,
        )

        for thr in thresholds:
            dice, iou, precision, recall = compute_metrics_from_prob(
                prob=prob,
                target=label,
                threshold=thr,
                use_lcc=args.post_use_lcc,
                fill_holes=args.post_fill_holes,
                closing_radius=args.post_closing_radius,
            )

            meters[thr]["dice"].append(dice.cpu())
            meters[thr]["iou"].append(iou.cpu())
            meters[thr]["precision"].append(precision.cpu())
            meters[thr]["recall"].append(recall.cpu())

    results = []

    for thr in thresholds:
        dice = torch.cat(meters[thr]["dice"]).mean().item()
        iou = torch.cat(meters[thr]["iou"]).mean().item()
        precision = torch.cat(meters[thr]["precision"]).mean().item()
        recall = torch.cat(meters[thr]["recall"]).mean().item()

        results.append({
            "threshold": thr,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
        })

    results = sorted(results, key=lambda x: x["dice"], reverse=True)

    print("\n========== Top 10 Results ==========")
    for r in results[:10]:
        print(
            f"thr={r['threshold']:.2f} | "
            f"dice={r['dice']:.6f} | "
            f"iou={r['iou']:.6f} | "
            f"precision={r['precision']:.6f} | "
            f"recall={r['recall']:.6f}"
        )

    best = results[0]
    print("\n========== Best Result ==========")
    print(
        f"best_thr={best['threshold']:.2f} | "
        f"dice={best['dice']:.6f} | "
        f"iou={best['iou']:.6f} | "
        f"precision={best['precision']:.6f} | "
        f"recall={best['recall']:.6f}"
    )

    if args.save_result:
        os.makedirs(os.path.dirname(args.save_result), exist_ok=True)
        with open(args.save_result, "w") as f:
            f.write("threshold,dice,iou,precision,recall\n")
            for r in sorted(results, key=lambda x: x["threshold"]):
                f.write(
                    f"{r['threshold']:.4f},"
                    f"{r['dice']:.6f},"
                    f"{r['iou']:.6f},"
                    f"{r['precision']:.6f},"
                    f"{r['recall']:.6f}\n"
                )
        print(f"\nSaved threshold search results to: {args.save_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--list_dir", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--dataset", type=str, default="TN3K")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--amp-opt-level", type=str, default="O1", choices=["O0", "O1", "O2"])

    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
    parser.add_argument("--resume", default="")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--post_use_lcc", action="store_true")
    parser.add_argument("--post_fill_holes", action="store_true")
    parser.add_argument("--post_closing_radius", type=int, default=0)
    # tta: none / flip / all
    # none: 只用原图
    # flip: 原图 + 水平翻转 + 垂直翻转
    # all: 原图 + 水平翻转 + 垂直翻转 + 水平垂直翻转
    parser.add_argument("--tta", type=str, default="flip", choices=["none", "flip", "all"])

    # 例如 "0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60"
    parser.add_argument("--thresholds", type=str, default="")

    parser.add_argument("--save_result", type=str, default="./output_tn3k_edge01_aux02/tta_threshold_results.csv")

    args = parser.parse_args()
    evaluate(args)