import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse, trainer_tn3k
from config import get_config


parser = argparse.ArgumentParser()

# ================= 基础参数 =================
parser.add_argument('--root_path', type=str, default='./data/TN3K', help='root dir for dataset')
parser.add_argument('--dataset', type=str, default='TN3K', help='dataset name: Synapse or TN3K')
parser.add_argument('--list_dir', type=str, default='./lists/TN3K', help='list dir')
parser.add_argument('--output_dir', type=str, default='./results', help='output root dir')
parser.add_argument('--exp_name', type=str, default='proposed', help='experiment name, e.g. baseline/proposed')

# ================= 训练参数 =================
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--n_class', type=int, default=2, help='alias of num_classes')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch size per gpu')
parser.add_argument('--n_gpu', type=int, default=3, help='total gpu number (for logging/reference)')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='base learning rate before scaling')
parser.add_argument('--img_size', type=int, default=384, help='input patch size')
parser.add_argument('--seed', type=int, default=3407, help='random seed')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
parser.add_argument('--eval_interval', type=int, default=1, help='validate every N epochs')
parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')

# ================= 模型/配置参数 =================
parser.add_argument(
    '--cfg',
    type=str,
    default='./configs/swin_base_patch4_window12_384_22k.yaml',
    metavar="FILE",
    help='path to config file'
)
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs.",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument(
    '--cache-mode',
    type=str,
    default='part',
    choices=['no', 'full', 'part'],
    help='no: no cache, full: cache all data, part: cache shard'
)
parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
parser.add_argument('--use-checkpoint', action='store_true', help='use gradient checkpointing')
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='amp opt level')
parser.add_argument('--tag', type=str, default='', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='test throughput only')

# ================= DDP 参数 =================
parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')

args = parser.parse_args()


def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def build_dataset_config(args):
    dataset_config = {
        'Synapse': {
            'root_path': os.path.join(args.root_path, "train_npz") if not args.root_path.endswith("train_npz") else args.root_path,
            'list_dir': './lists/Synapse',
            'num_classes': 9,
        },
        'TN3K': {
            'root_path': args.root_path,
            'list_dir': './lists/TN3K',
            'num_classes': 2,
        }
    }

    if args.dataset not in dataset_config:
        raise ValueError("Dataset not supported: {}".format(args.dataset))

    return dataset_config[args.dataset]


def load_custom_weights(net, resume_path, local_rank=0):
    if local_rank == 0:
        print(f"Loading custom weights from: {resume_path}")

    checkpoint = torch.load(resume_path, map_location='cpu')

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    msg = net.load_state_dict(new_state_dict, strict=False)

    if local_rank == 0:
        print(f"Weights loaded successfully! Msg: {msg}")


if __name__ == "__main__":
    # ================= DDP 初始化：优先读取 torchrun 环境变量 =================
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        distributed = True
    else:
        rank = 0
        world_size = 1
        distributed = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= 随机种子与 cudnn =================
    set_random_seed(args.seed, deterministic=bool(args.deterministic))

    # ================= 数据集配置 =================
    dataset_cfg = build_dataset_config(args)
    args.root_path = dataset_cfg['root_path']
    args.list_dir = dataset_cfg['list_dir']
    args.num_classes = dataset_cfg['num_classes']
    args.n_class = dataset_cfg['num_classes']

    # ================= 配置文件 =================
    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = args.img_size
    config.freeze()

    # ================= 输出目录 =================
    args.output_dir = os.path.join(args.output_dir, args.exp_name, f"seed_{args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ================= 学习率按有效 batch size 缩放 =================
    # effective_batch_size = per_gpu_batch * world_size * accumulation_steps
    reference_batch_size = 24
    effective_batch_size = args.batch_size * world_size * args.accumulation_steps
    args.base_lr = args.base_lr * effective_batch_size / reference_batch_size

    if rank == 0:
        print("========== Training Configuration ==========")
        print(f"dataset               : {args.dataset}")
        print(f"root_path             : {args.root_path}")
        print(f"list_dir              : {args.list_dir}")
        print(f"output_dir            : {args.output_dir}")
        print(f"cfg                   : {args.cfg}")
        print(f"img_size              : {args.img_size}")
        print(f"num_classes           : {args.num_classes}")
        print(f"local_rank            : {args.local_rank}")
        print(f"rank                  : {rank}")
        print(f"world_size            : {world_size}")
        print(f"per_gpu_batch_size    : {args.batch_size}")
        print(f"accumulation_steps    : {args.accumulation_steps}")
        print(f"effective_batch_size  : {effective_batch_size}")
        print(f"scaled_base_lr        : {args.base_lr}")
        print(f"max_epochs            : {args.max_epochs}")
        print(f"seed                  : {args.seed}")
        print(f"eval_interval         : {args.eval_interval}")
        print("===========================================")

    # ================= 构建模型 =================
    if distributed:
        net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda(args.local_rank)
    else:
        net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)

    # ================= 加载权重 =================
    if args.resume:
        load_custom_weights(net, args.resume, local_rank=rank)
    else:
        net.load_from(config)

    # ================= DDP 包装 =================
    if distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    # ================= 训练入口 =================
    if args.dataset == "Synapse":
        trainer_synapse(args, net, args.output_dir)
    elif args.dataset == "TN3K":
        trainer_tn3k(args, net, args.output_dir)
    else:
        raise ValueError("Dataset not supported: {}".format(args.dataset))