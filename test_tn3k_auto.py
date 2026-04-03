import argparse
import json
import os
from copy import deepcopy

import numpy as np
import torch

from config import get_config
from datasets.dataset_tn3k import load_tn3k_records, make_kfold_splits
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import evaluate_tn3k


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True)
parser.add_argument('--list_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True, help='directory containing fold_* subfolders')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--img_size', type=int, default=384)
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_folds', type=int, default=5)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cv_source_splits', nargs='+', default=['train', 'test'])
parser.add_argument('--use_tta', type=str2bool, default=True)
parser.add_argument('--post_use_lcc', type=str2bool, default=True)
parser.add_argument('--post_min_size', type=int, default=0)
parser.add_argument('--post_fill_holes', type=str2bool, default=True)
parser.add_argument('--post_closing_radius', type=int, default=3)
parser.add_argument('--opts', help='Modify config options by adding "KEY VALUE" pairs.', default=None, nargs='+')
args = parser.parse_args()


def build_model(local_args):
    config = get_config(local_args)
    config.defrost()
    config.DATA.IMG_SIZE = local_args.img_size
    config.MODEL.SWIN.PATCH_SIZE = 4
    config.MODEL.SWIN.WINDOW_SIZE = 12
    config.freeze()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ViT_seg(config, img_size=local_args.img_size, num_classes=local_args.n_class).to(device)
    return net


def load_checkpoint(net, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=True)
    net.eval()


def main():
    records = load_tn3k_records(args.root_path, args.list_dir, source_splits=tuple(args.cv_source_splits))
    fold_pairs = make_kfold_splits(records, num_folds=args.num_folds, seed=args.seed)

    all_results = []
    for fold_id, (_, val_records) in enumerate(fold_pairs):
        fold_dir = os.path.join(args.output_dir, f'fold_{fold_id}')
        ckpt_path = os.path.join(fold_dir, 'best_model.pth')
        if not os.path.exists(ckpt_path):
            print(f'[Skip] Missing checkpoint: {ckpt_path}')
            continue

        net = build_model(args)
        load_checkpoint(net, ckpt_path)
        fold_args = deepcopy(args)
        metrics = evaluate_tn3k(
            fold_args,
            net,
            val_records,
            batch_size=1,
            use_tta=args.use_tta,
            desc=f'Fold {fold_id} Test',
        )
        metrics['fold_id'] = fold_id
        all_results.append(metrics)
        print(f"Fold {fold_id}: Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f} HD95={metrics['hd95']:.4f} ASD={metrics['asd']:.4f}")
        del net
        torch.cuda.empty_cache()

    if not all_results:
        raise RuntimeError('No fold checkpoints were found for evaluation.')

    summary = {'num_folds': len(all_results)}
    for key in ['dice', 'iou', 'hd95', 'asd']:
        values = np.asarray([item[key] for item in all_results], dtype=np.float32)
        summary[f'{key}_mean'] = float(values.mean())
        summary[f'{key}_std'] = float(values.std(ddof=0))

    print('=' * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(os.path.join(args.output_dir, 'cv_summary_recomputed.json'), 'w', encoding='utf-8') as f:
        json.dump({'folds': all_results, 'summary': summary}, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
