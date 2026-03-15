import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_tn3k import TN3KDataset
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
import glob
import re
from medpy import metric

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        # 1. 计算 Dice 和 IoU
        dice = (2.0 * (pred * gt).sum()) / (pred.sum() + gt.sum())
        iou = (pred * gt).sum() / (pred.sum() + gt.sum() - (pred * gt).sum())

        # 2. 计算 HD95 和 ASD (注意：这里的默认体素间距 spacing 是 1)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)

        return dice, iou, hd95, asd

    elif pred.sum() == 0 and gt.sum() == 0:
        # 完美预测出背景 (没有目标)
        # Dice 和 IoU 是完美的 1.0，距离误差是完美的 0.0
        return 1.0, 1.0, 0.0, 0.0
    else:
        # 完全预测错误 (有目标预测无，或无目标预测有)
        # 此时无法计算表面距离，因此给予一个极大的惩罚值 (例如 50 或 100，视图像大小而定)
        # Dice 和 IoU 为 0.0
        penalty_distance = 100.0
        return 0.0, 0.0, penalty_distance, penalty_distance


def test_single_model(args, model_path, db_test):
    """测试单个权重，返回 avg_dice, avg_iou, avg_hd95, avg_asd"""
    # 1. 配置与模型初始化
    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = args.img_size
    config.MODEL.SWIN.PATCH_SIZE = 4
    config.MODEL.SWIN.WINDOW_SIZE = 12
    config.freeze()

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.n_class).cuda()

    # 2. 加载训练好的权重
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict, strict=True)
    net.eval()

    # 3. 准备测试数据
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)

    metric_list = []

    print(f"Testing {os.path.basename(model_path)}... ", end="")

    # 4. 开始推理
    with torch.no_grad():
        for sampled_batch in tqdm(testloader, leave=False):
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            output = net(image_batch)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
            label = label_batch.squeeze(0).squeeze(0).cpu().detach().numpy()

            # 接收新增的指标
            dice, iou, hd95, asd = calculate_metric_percase(pred, label)
            metric_list.append([dice, iou, hd95, asd])

    metric_list = np.array(metric_list)
    avg_dice = np.mean(metric_list[:, 0])
    avg_iou = np.mean(metric_list[:, 1])
    avg_hd95 = np.mean(metric_list[:, 2])
    avg_asd = np.mean(metric_list[:, 3])

    print(f"Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, HD95: {avg_hd95:.4f}, ASD: {avg_asd:.4f}")
    return avg_dice, avg_iou, avg_hd95, avg_asd


def main(args):
    # 1. 寻找所有需要测试的 .pth 文件
    model_dir = args.output_dir
    pth_files = glob.glob(os.path.join(model_dir, "epoch_*.pth"))

    if not pth_files:
        print(f"在 {model_dir} 中没有找到任何 epoch_*.pth 文件！")
        return

    pth_files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', os.path.basename(x)).group(1)))
    print(f"共找到 {len(pth_files)} 个检查点，准备开始批量验证...")

    # 2. 提前加载 Dataset
    db_test = TN3KDataset(base_dir=args.root_path,
                          list_dir=args.list_dir,
                          split="test",
                          transform=None,
                          img_size=args.img_size)
    print(f"测试集图片数量: {len(db_test)}\n")

    results = []

    # 3. 循环测试所有模型
    for pth in pth_files:
        avg_dice, avg_iou, avg_hd95, avg_asd = test_single_model(args, pth, db_test)
        epoch_name = os.path.basename(pth)
        results.append({
            "epoch": epoch_name,
            "dice": avg_dice,
            "iou": avg_iou,
            "hd95": avg_hd95,
            "asd": avg_asd
        })

        torch.cuda.empty_cache()

    # 4. 打印最终排行榜
    print("\n" + "=" * 70)
    print(" 🏆 批量测试完成！最终排行榜 (按 Dice 降序) 🏆")
    print("=" * 70)

    results.sort(key=lambda x: x["dice"], reverse=True)

    # 优化表头，由于 HD95 和 ASD 是越小越好，加上 ↓ 标识
    print(
        f"{'排名':<4} | {'权重名称':<13} | {'Avg Dice ↑':<10} | {'Avg IoU ↑':<10} | {'Avg HD95 ↓':<10} | {'Avg ASD ↓':<10}")
    print("-" * 70)
    for i, res in enumerate(results):
        star = " 🌟" if i == 0 else ""
        print(
            f"{i + 1:<4} | {res['epoch']:<15} | {res['dice']:.4f}{star:<6} | {res['iou']:.4f}     | {res['hd95']:.4f}     | {res['asd']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/ub/pql/Swin-Unet-main/data/TN3K')
    parser.add_argument('--list_dir', type=str, default='/home/ub/pql/Swin-Unet-main/lists/TN3K')
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--cfg', type=str, default='configs/swin_base_patch4_window12_384_22k.yaml')
    parser.add_argument('--output_dir', type=str, default='./output_tn3k')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--opts", default=None, nargs='+')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O1')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--model_path', type=str, default='')

    args = parser.parse_args()

    main(args)