import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config
from trainer import trainer_synapse, trainer_tn3k

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
# parser.add_argument("--dataset_name", default="datasets")
parser.add_argument("--n_class", default=9, type=int)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--eval_interval", default=1, type=int)
parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")

config = get_config(args)
config.defrost()
config.DATA.IMG_SIZE = args.img_size
config.freeze()

if __name__ == "__main__":
    # ================= 1. DDP 初始化 (必须放在最前面) =================
    # 这一步非常关键，它告诉当前进程：“你是第几号进程，你应该用第几号显卡”
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # ===================================================================
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        args.dataset: {
            'root_path': args.root_path,
            'list_dir': f'./lists/{args.dataset}',
            'num_classes': args.n_class,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda(args.local_rank)
    if args.resume:
        # 如果指定了 --resume，加载你训练过的权重
        if args.local_rank == 0:
            print(f"Loading custom weights from: {args.resume}")

        checkpoint = torch.load(args.resume, map_location='cpu')

        # 处理 DDP 带来的 'module.' 前缀
        state_dict = checkpoint
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        msg = net.load_state_dict(new_state_dict, strict=False)
        if args.local_rank == 0:
            print(f"Weights loaded successfully! Msg: {msg}")
    else:
        # 否则加载官方预训练权重
        net.load_from(config)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
    )
    # trainer = {'Synapse': trainer_synapse}


    # trainer_synapse(args, net, args.output_dir)//原代码
    if args.dataset == "Synapse":
        trainer_synapse(args, net, args.output_dir)
    elif args.dataset == "TN3K":
        trainer_tn3k(args, net, args.output_dir)
    else:
        raise ValueError("Dataset not supported: {}".format(args.dataset))

