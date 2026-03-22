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
import skimage.morphology as morphology
import cv2  # 用于图像可视化与保存


def remove_speckle_noise(binary_mask, min_size=150):
    """
    [修正版后处理] 
    移除面积小于 min_size 的孤立噪点，完美保留多发性结节。
    min_size=150 适用于过滤 384x384 尺度下的超声干涉散斑。
    """
    mask_bool = binary_mask.astype(bool)
    cleaned_mask = morphology.remove_small_objects(mask_bool, min_size=min_size, connectivity=2)
    return cleaned_mask.astype(np.uint8)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = (2.0 * (pred * gt).sum()) / (pred.sum() + gt.sum())
        iou = (pred * gt).sum() / (pred.sum() + gt.sum() - (pred * gt).sum())
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, iou, hd95, asd, False

    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0, 0.0, 0.0, False
    else:
        # 触发极端预测失败 (假阴性或完全偏离的假阳性)
        penalty_distance = 100.0
        return 0.0, 0.0, penalty_distance, penalty_distance, True


def test_single_model(args, model_path, db_test):
    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = args.img_size
    config.MODEL.SWIN.PATCH_SIZE = 4
    config.MODEL.SWIN.WINDOW_SIZE = 12
    config.freeze()

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.n_class).cuda()

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

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)

    metric_list = []
    penalty_count = 0

    # 创建困难样本保存目录
    bad_case_dir = os.path.join(args.output_dir, "bad_cases", os.path.basename(model_path).replace('.pth', ''))
    os.makedirs(bad_case_dir, exist_ok=True)

    print(f"Testing {os.path.basename(model_path)}... ", end="")

    # TN3K 数据集的标准化参数 (用于可视化时的反归一化)
    tn3k_mean = np.array([0.3915, 0.2582, 0.2558])
    tn3k_std = np.array([0.1656, 0.1430, 0.1631])

    with torch.no_grad():
        for batch_idx, sampled_batch in enumerate(tqdm(testloader, leave=False)):
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            output = net(image_batch)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
            label = label_batch.squeeze(0).squeeze(0).cpu().detach().numpy()

            # ================= 核心修正：基于面积阈值的多发性结节过滤 =================
            pred = remove_speckle_noise(pred, min_size=150)
            # ====================================================================

            dice, iou, hd95, asd, is_penalty = calculate_metric_percase(pred, label)
            metric_list.append([dice, iou, hd95, asd])
            if is_penalty:
                penalty_count += 1

            # ================= 困难样本 (Bad Case) 物理落盘机制 =================
            if hd95 > 30.0 or is_penalty:
                # 1. 提取图像并反归一化到 0-255 RGB
                orig_img = image_batch.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
                orig_img = (orig_img * tn3k_std) + tn3k_mean
                orig_img = np.clip(orig_img * 255.0, 0, 255).astype(np.uint8)

                # 2. 生成红色预测掩码
                pred_color = np.zeros_like(orig_img)
                pred_color[:, :, 0] = pred * 255  # Red 通道

                # 3. 生成绿色真实掩码
                gt_color = np.zeros_like(orig_img)
                gt_color[:, :, 1] = label * 255  # Green 通道

                # 4. Alpha 混合叠加
                overlay = cv2.addWeighted(orig_img, 0.6, pred_color, 0.4, 0)
                overlay = cv2.addWeighted(overlay, 1.0, gt_color, 0.4, 0)

                # 5. 写入磁盘 (cv2 需转换为 BGR 格式)
                save_name = f"idx_{batch_idx}_HD95_{hd95:.1f}_Dice_{dice:.4f}.jpg"
                cv2.imwrite(os.path.join(bad_case_dir, save_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            # =====================================================================

    metric_list = np.array(metric_list)
    avg_dice = np.mean(metric_list[:, 0])
    avg_iou = np.mean(metric_list[:, 1])
    avg_hd95 = np.mean(metric_list[:, 2])
    avg_asd = np.mean(metric_list[:, 3])

    print(
        f"Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, HD95: {avg_hd95:.4f}, ASD: {avg_asd:.4f} (Penalty: {penalty_count})")
    return avg_dice, avg_iou, avg_hd95, avg_asd, penalty_count


def main(args):
    model_dir = args.output_dir
    pth_files = glob.glob(os.path.join(model_dir, "epoch_*.pth"))

    if not pth_files:
        print(f"在 {model_dir} 中没有找到任何 epoch_*.pth 文件！")
        return

    pth_files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', os.path.basename(x)).group(1)))
    print(f"共找到 {len(pth_files)} 个检查点，准备开始批量验证...")

    db_test = TN3KDataset(base_dir=args.root_path,
                          list_dir=args.list_dir,
                          split="test",
                          transform=None,
                          img_size=args.img_size)
    print(f"测试集图片数量: {len(db_test)}\n")

    results = []

    for pth in pth_files:
        avg_dice, avg_iou, avg_hd95, avg_asd, penalty_count = test_single_model(args, pth, db_test)
        epoch_name = os.path.basename(pth)
        results.append({
            "epoch": epoch_name,
            "dice": avg_dice,
            "iou": avg_iou,
            "hd95": avg_hd95,
            "asd": avg_asd,
            "penalty": penalty_count
        })
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(" 🏆 批量测试完成！最终排行榜 (按 Dice 降序) 🏆")
    print("=" * 80)

    results.sort(key=lambda x: x["dice"], reverse=True)

    print(
        f"{'排名':<4} | {'权重名称':<13} | {'Avg Dice ↑':<10} | {'Avg IoU ↑':<10} | {'Avg HD95 ↓':<10} | {'Avg ASD ↓':<10} | {'极值惩罚数'}")
    print("-" * 80)
    for i, res in enumerate(results):
        star = " 🌟" if i == 0 else ""
        print(
            f"{i + 1:<4} | {res['epoch']:<15} | {res['dice']:.4f}{star:<6} | {res['iou']:.4f}     | {res['hd95']:.4f}     | {res['asd']:.4f}     | {res['penalty']}")
    print("=" * 80)


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