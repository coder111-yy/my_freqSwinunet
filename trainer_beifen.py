import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss
from datasets.dataset_tn3k import TN3KDataset, RandomGenerator
from torch.cuda.amp import autocast, GradScaler


# ================= 新增：边界提取与边界损失计算模块 =================
def get_boundary(mask, kernel_size=3):
    """
    使用形态学梯度 (膨胀 - 腐蚀) 提取张量的边界
    """
    # 膨胀操作 (Dilation)
    dilation = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    # 腐蚀操作 (Erosion)
    erosion = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    # 形态学梯度即为边界
    boundary = dilation - erosion
    return boundary


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, gt):
        # 提取预测概率图和真实标签的边界
        pred_boundary = get_boundary(pred)
        gt_boundary = get_boundary(gt)
        # 约束两条边界尽可能重合 (使用均方误差)
        loss = F.mse_loss(pred_boundary, gt_boundary)
        return loss


# ====================================================================

def trainer_synapse(args, model, snapshot_path):

    pass


def trainer_tn3k(args, model, snapshot_path):
    import logging
    import os
    import sys
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torchvision import transforms
    from tqdm import tqdm

    accumulation_steps = args.accumulation_steps if args.accumulation_steps else 6
    scaler = GradScaler()

    if args.local_rank == 0:
        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(f"Using Gradient Accumulation: Real Batch Size={args.batch_size}, "
                     f"Accumulation Steps={accumulation_steps}, "
                     f"Virtual Batch Size={args.batch_size * accumulation_steps}")

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size

    db_train = TN3KDataset(base_dir=args.root_path,
                           list_dir=args.list_dir,
                           split="train",
                           img_size=args.img_size,
                           transform=RandomGenerator(output_size=[args.img_size, args.img_size]))

    if args.local_rank == 0:
        print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_sampler = DistributedSampler(db_train, shuffle=True)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn,
                             sampler=train_sampler)

    model.train()

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # [修改点 1] 初始化边界损失函数
    boundary_loss_fn = BoundaryLoss().cuda()
    alpha = 0.5  # 边界损失的权重系数

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    if args.local_rank == 0:
        logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70) if args.local_rank == 0 else range(max_epoch)

    optimizer.zero_grad()

    for epoch_num in iterator:
        train_sampler.set_epoch(epoch_num)

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(non_blocking=True), label_batch.cuda(non_blocking=True)

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            # ================= [修改点 2] 计算并融合边界损失 =================
            # 提取模型预测为前景 (结节) 的概率图: shape (B, 1, H, W)
            pred_prob = torch.softmax(outputs, dim=1)[:, 1:2, :, :]
            # 将真实标签转换维度以匹配计算: shape (B, 1, H, W)
            gt_mask = label_batch.unsqueeze(1).float()

            loss_boundary = boundary_loss_fn(pred_prob, gt_mask)

            # 将新的损失项加入总 Loss
            loss = 0.4 * loss_ce + 0.6 * loss_dice + alpha * loss_boundary
            # =================================================================

            # Loss 标准化 (梯度累加)
            loss = loss / accumulation_steps

            loss.backward()

            if (i_batch + 1) % accumulation_steps == 0 or (i_batch + 1) == len(trainloader):
                optimizer.step()
                optimizer.zero_grad()

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1

                # [修改点 3] 日志打印中加入 loss_boundary 的监控
                if args.local_rank == 0:
                    if iter_num % 20 == 0:
                        logging.info('iteration %d : loss: %f, loss_ce: %f, loss_dice: %f, loss_boundary: %f' %
                                     (iter_num, loss.item() * accumulation_steps, loss_ce.item(), loss_dice.item(),
                                      loss_boundary.item()))

        if args.local_rank == 0:
            last_save_path = os.path.join(snapshot_path, 'last_model.pth')
            torch.save(model.state_dict(), last_save_path)
            logging.info(f"save last model to {last_save_path}")

            # ================= 修改后：滑动窗口保存最新的 100 个 epoch =================
            # 1. 保存当前 epoch 的权重
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            # 2. 计算需要删除的旧 epoch 编号（当前 epoch 向前推 100 个）
            obsolete_epoch = epoch_num - 100

            # 3. 如果旧 epoch 存在，则将其从磁盘删除，释放空间
            if obsolete_epoch >= 0:
                obsolete_model_path = os.path.join(snapshot_path, 'epoch_' + str(obsolete_epoch) + '.pth')
                if os.path.exists(obsolete_model_path):
                    os.remove(obsolete_model_path)
                    # 可选：如果你不想日志里全是删除信息，可以把下面这行注释掉
                    logging.info("remove obsolete model {}".format(obsolete_model_path))
            # =========================================================================

    return "Training Finished!"