import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from datasets.dataset_tn3k import TN3KDataset
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
import sys


# 简单的 Dice 和 IoU 计算函数
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = (2.0 * (pred * gt).sum()) / (pred.sum() + gt.sum())
        iou = (pred * gt).sum() / (pred.sum() + gt.sum() - (pred * gt).sum())
        return dice, iou
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0
    else:
        return 0.0, 0.0


def test(args):
    # 1. 配置与模型初始化
    config = get_config(args)

    # [关键修改 1] 强制覆盖配置，确保与训练时一致
    config.defrost()
    config.DATA.IMG_SIZE = args.img_size
    # 强制指定 Patch Size 为 4，防止默认值 7 导致分辨率不匹配报错
    config.MODEL.SWIN.PATCH_SIZE = 4
    config.MODEL.SWIN.WINDOW_SIZE = 12  # 对应 Swin Base 384
    config.freeze()

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.n_class).cuda()

    # 2. 加载训练好的权重
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型文件: {args.model_path}")

    print(f"Loading checkpoint from: {args.model_path}")

    # 使用 cpu map_location 防止显卡不匹配
    checkpoint = torch.load(args.model_path, map_location='cpu')

    # 兼容性处理
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 去除 'module.' 前缀 (针对 DDP 训练的权重)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # 加载清洗后的权重
    msg = net.load_state_dict(new_state_dict, strict=True)
    print(f"Weights loaded successfully! Msg: {msg}")
    net.eval()

    # 3. 准备测试数据
    # [关键修改 2] 必须传入 img_size=args.img_size，否则 dataset 会默认 resize 成 224
    db_test = TN3KDataset(base_dir=args.root_path,
                          list_dir=args.list_dir,
                          split="test",
                          transform=None,
                          img_size=args.img_size)  # <--- 这里非常重要

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)

    print(f"测试集图片数量: {len(db_test)}")

    # 4. 准备保存预测结果的文件夹
    if args.save_prediction:
        save_dir = os.path.join(args.output_dir, "predictions")
        os.makedirs(save_dir, exist_ok=True)

    metric_list = []

    # 5. 开始推理
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # 前向传播
            output = net(image_batch)

            # 获取预测结果
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            pred = pred.cpu().detach().numpy()

            label = label_batch.squeeze(0).squeeze(0).cpu().detach().numpy()

            # 计算指标
            dice, iou = calculate_metric_percase(pred, label)
            metric_list.append([dice, iou])

            # 保存可视化图片
            if args.save_prediction:
                # 原始 Mask 是 0/1，转为 0/255
                pred_img = (pred * 255).astype(np.uint8)
                img_name = db_test.sample_list[i_batch]
                # 保存 PNG
                Image.fromarray(pred_img).save(os.path.join(save_dir, f"{img_name}.png"))

    # 6. 输出平均结果
    metric_list = np.array(metric_list)
    avg_dice = np.mean(metric_list[:, 0])
    avg_iou = np.mean(metric_list[:, 1])

    print("\n" + "=" * 20)
    print(f"Test Finished!")
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Average IoU : {avg_iou:.4f}")
    print("=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/ub/pql/Swin-Unet-main/data/TN3K',
                        help='root dir for data')
    parser.add_argument('--list_dir', type=str, default='/home/ub/pql/Swin-Unet-main/lists/TN3K', help='list dir')
    parser.add_argument('--n_class', type=int, default=2, help='number of classes')

    # [关键修改 3] 将默认 img_size 改为 384
    parser.add_argument('--img_size', type=int, default=384, help='input patch size of network input')

    # [关键修改 4] 将默认 cfg 改为 Swin Base 的 yaml
    parser.add_argument('--cfg', type=str, default='configs/swin_base_patch4_window12_384_22k.yaml',
                        help='path to config file')

    parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')

    # 默认加载的模型路径，如果有 best_model.pth 可以改这里，或者运行时指定
    parser.add_argument('--model_path', type=str, default='./output_tn3k/best_model.pth', help='trained model path')
    parser.add_argument('--output_dir', type=str, default='./output_tn3k', help='output dir')
    parser.add_argument('--save_prediction', type=bool, default=True, help='whether to save predicted masks')

    # 其他兼容参数
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+', )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'])
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args = parser.parse_args()

    # 简单的日志重定向，将输出同时也保存到 txt
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # sys.stdout = open(os.path.join(args.output_dir, 'test_result.txt'), 'w')  # 重定向输出到文件
    # 为了让控制台也能看到，可以使用 tee 的逻辑，但这里为了简单直接覆盖
    # 如果希望控制台也显示，注释掉上面一行 sys.stdout = ... 即可

    test(args)

  #   python test_tn3k.py --cfg configs/swin_base_patch4_window12_384_22k.yaml --img_size 384 --model_path ./output_tn3k/epoch_39.pth