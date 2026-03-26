import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from trainer import trainer_synapse, trainer_tn3k


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, required=True, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iteration number')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-4, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar='FILE', help='path to config file')
parser.add_argument('--opts', default=None, nargs='+', help="Modify config options by adding 'KEY VALUE' pairs.")
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'])
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--use-checkpoint', action='store_true', help='whether to use gradient checkpointing to save memory')
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'])
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--n_class', default=9, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--eval_interval', default=1, type=int)
parser.add_argument('--edge_lambda', type=float, default=0.1, help='weight for boundary loss')
parser.add_argument('--boundary_width', type=int, default=2, help='boundary band radius in pixels')
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)), help='local rank for distributed training')

args = parser.parse_args()
if args.dataset == 'Synapse':
    args.root_path = os.path.join(args.root_path, 'train_npz')

config = get_config(args)
config.defrost()
config.DATA.IMG_SIZE = args.img_size
config.freeze()


if __name__ == '__main__':
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

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    dataset_name = args.dataset
    dataset_config = {
        args.dataset: {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.n_class,
        },
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    os.makedirs(args.output_dir, exist_ok=True)

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)

    if args.resume:
        if rank == 0:
            print(f'Loading custom weights from: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        msg = net.load_state_dict(new_state_dict, strict=False)
        if rank == 0:
            print(f'Weights loaded successfully! Msg: {msg}')
    else:
        net.load_from(config)

    if distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    if args.dataset == 'Synapse':
        trainer_synapse(args, net, args.output_dir)
    elif args.dataset == 'TN3K':
        trainer_tn3k(args, net, args.output_dir)
    else:
        raise ValueError(f'Dataset not supported: {args.dataset}')
