import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import create_dataloaders, evaluate_tn3k_official


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', type=str,
                        default='/home/ub/pql/Swin-Unet-main/data/TN3K',
                        help='TN3K root dir')
    parser.add_argument('--list_dir', type=str,
                        default='/home/ub/pql/Swin-Unet-main/lists/TN3K',
                        help='directory containing train.txt and test.txt')
    parser.add_argument('--dataset', type=str, default='TN3K')
    parser.add_argument('--cfg', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to best_model.pth or checkpoint')
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=2)

    # 关键：这里默认设成 4，因为你训练 log 里 batch_size=4
    # create_dataloaders 里 test batch size 会变成 batch_size // 2，也就是 2
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)

    # 为了和训练时 evaluate_tn3k_official 的 use_amp 保持一致
    parser.add_argument('--amp-opt-level', type=str, default='O1',
                        choices=['O0', 'O1', 'O2'])

    # 下面这些参数 create_dataloaders / config 可能不直接用，但保留可兼容训练 args
    parser.add_argument('--output_dir', type=str, default='./test_output')
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--max_iterations', type=int, default=30000)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--opts', default=None, nargs='+')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part',
                        choices=['no', 'full', 'part'])
    parser.add_argument('--resume', default='')
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--edge_lambda', type=float, default=0.2)
    parser.add_argument('--aux_weight_half', type=float, default=0.4)
    parser.add_argument('--aux_weight_full', type=float, default=0.2)
    parser.add_argument('--boundary_width', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd'])

    return parser.parse_args()


def load_checkpoint(model, model_path):
    print(f'Loading checkpoint from: {model_path}')
    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Checkpoint best_dice: {checkpoint.get('best_dice', 'N/A')}")
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    msg = model.load_state_dict(new_state_dict, strict=True)
    print(f'Weights loaded successfully! Msg: {msg}')


def main():
    args = parse_args()

    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = args.img_size
    config.freeze()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rank = 0

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.n_class).to(device)

    load_checkpoint(net, args.model_path)

    # 直接复用训练时的数据加载方式
    db_train, db_test, train_loader, test_loader, train_sampler, test_sampler = create_dataloaders(args)

    print(f'Train size: {len(db_train)} | Test size: {len(db_test)}')
    print(f'Test batch size: {test_loader.batch_size}')

    # 直接复用训练时的评估方式
    use_amp = torch.cuda.is_available() and args.amp_opt_level != 'O0'
    print(f'AMP enabled: {use_amp}')

    metrics = evaluate_tn3k_official(
        args=args,
        model=net,
        test_loader=test_loader,
        rank=rank,
        use_amp=use_amp
    )

    print('========== Official TN3K Evaluation ==========')
    print(f"Test Loss : {metrics['loss']:.6f}")
    print(f"Dice      : {metrics['dice']:.6f}")
    print(f"IoU       : {metrics['iou']:.6f}")
    print(f"Precision : {metrics['precision']:.6f}")
    print(f"Recall    : {metrics['recall']:.6f}")


if __name__ == '__main__':
    main()