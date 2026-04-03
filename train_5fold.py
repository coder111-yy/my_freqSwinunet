import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer_5fold import prepare_tn3k_kfold_splits, trainer_tn3k_one_fold


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True, help='TN3K root dir, e.g. ./data/TN3K')
parser.add_argument('--dataset', type=str, default='TN3K', help='experiment_name')
parser.add_argument('--list_dir', type=str, required=True, help='directory containing official train.txt and test.txt')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--output_dir', type=str, required=True, help='output dir')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-4, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=384, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar='FILE', help='path to config file')
parser.add_argument('--opts', default=None, nargs='+', help="Modify config options by adding 'KEY VALUE' pairs.")
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'])
parser.add_argument('--resume', default='', help='resume from checkpoint for a single fold')
parser.add_argument('--accumulation-steps', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--use-checkpoint', action='store_true', help='whether to use gradient checkpointing to save memory')
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'])
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--n_class', default=2, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--eval_interval', default=1, type=int)
parser.add_argument('--edge_lambda', type=float, default=0.1, help='weight for boundary loss')
parser.add_argument('--boundary_width', type=int, default=2, help='boundary band radius in pixels')
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)), help='local rank for distributed training')

# k-fold options
parser.add_argument('--num_folds', type=int, default=5, help='number of folds for cross-validation')
parser.add_argument('--fold', type=int, default=-1, help='-1 means run all folds; 0~num_folds-1 means run one specific fold only')
parser.add_argument('--save_epoch_ckpt', action='store_true', help='save every epoch checkpoint in each fold')
args = parser.parse_args()

config = get_config(args)
config.defrost()
config.DATA.IMG_SIZE = args.img_size
config.freeze()


def set_random_seed(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)



def build_model(device, rank):
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.n_class).to(device)

    if args.resume:
        if rank == 0:
            print(f'Loading custom weights from: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint.get('state_dict', checkpoint)))
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        msg = net.load_state_dict(new_state_dict, strict=False)
        if rank == 0:
            print(f'Weights loaded successfully! Msg: {msg}')
    else:
        net.load_from(config)

    if args.local_rank != -1:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
    return net


if __name__ == '__main__':
    if args.dataset != 'TN3K':
        raise ValueError('This script is only for TN3K 5-fold cross-validation.')

    distributed = args.local_rank != -1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        rank = torch.distributed.get_rank()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    set_random_seed(args.seed, rank)
    os.makedirs(args.output_dir, exist_ok=True)

    args.num_classes = args.n_class
    all_records, folds = prepare_tn3k_kfold_splits(args)

    if rank == 0:
        with open(os.path.join(args.output_dir, 'all_records.txt'), 'w', encoding='utf-8') as f:
            for item in all_records:
                f.write(f'{item}\n')

    fold_ids = list(range(args.num_folds)) if args.fold < 0 else [args.fold]
    fold_results = []

    for fold_idx in fold_ids:
        train_records, val_records = folds[fold_idx]
        fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_idx}')
        os.makedirs(fold_output_dir, exist_ok=True)

        # make runs reproducible but still different across folds
        set_random_seed(args.seed + fold_idx * 1000, rank)
        model = build_model(device, rank)

        result = trainer_tn3k_one_fold(
            args=args,
            model=model,
            snapshot_path=fold_output_dir,
            train_records=train_records,
            val_records=val_records,
            fold_idx=fold_idx,
        )
        fold_results.append({'fold': fold_idx, 'best_epoch': result['best_epoch'], 'best_dice': result['best_dice']})

        if distributed:
            torch.distributed.barrier()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rank == 0:
        best_dices = [x['best_dice'] for x in fold_results]
        summary = {
            'num_folds': len(fold_results),
            'fold_results': fold_results,
            'mean_best_dice': float(np.mean(best_dices)) if best_dices else 0.0,
            'std_best_dice': float(np.std(best_dices)) if best_dices else 0.0,
        }
        with open(os.path.join(args.output_dir, 'cv_results.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print('\n========== 5-Fold Cross Validation Summary ==========' )
        for item in fold_results:
            print(f"Fold {item['fold']}: best_epoch={item['best_epoch']} best_dice={item['best_dice']:.6f}")
        print(f"Mean best Dice: {summary['mean_best_dice']:.6f}")
        print(f"Std  best Dice: {summary['std_best_dice']:.6f}")
