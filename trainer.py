import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss
from datasets.dataset_tn3k import TN3KDataset, RandomGenerator

from torch.cuda.amp import autocast, GradScaler

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            image_batch, label_batch = image_batch.cuda(non_blocking=True), label_batch.cuda(non_blocking=True)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
            #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()

                batch_ce_loss /= len(val_loader)
                batch_dice_loss /= len(val_loader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                    epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
                if batch_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = batch_loss
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"


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

    # [修改点 1] 设置累加步数
    # 你的真实 Batch Size 是 2，如果你想要模拟 Batch Size = 12 的效果
    # 那么 accumulation_steps = 12 / 2 = 6
    # 你也可以在命令行通过 --accumulation-steps 6 传入，如果没有传入，这里默认设为 6
    accumulation_steps = args.accumulation_steps if args.accumulation_steps else 6

    scaler = GradScaler()

    if args.local_rank == 0:
        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        # [日志] 打印累加配置
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

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    if args.local_rank == 0:
        logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70) if args.local_rank == 0 else range(max_epoch)

    #首次循环前，必须先清空梯度
    optimizer.zero_grad()

    for epoch_num in iterator:
        train_sampler.set_epoch(epoch_num)

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(non_blocking=True), label_batch.cuda(non_blocking=True)

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            #  Loss 标准化
            # 因为梯度是累加的（求和），所以 loss 必须除以累加步数，保持梯度幅值正常
            loss = loss / accumulation_steps

            # 反向传播 (只计算梯度，不更新权重)
            # 这里的梯度会累积到 param.grad 中
            loss.backward()

            # 判断是否达到累加步数，或者是否是 Epoch 的最后一个 Batch
            if (i_batch + 1) % accumulation_steps == 0 or (i_batch + 1) == len(trainloader):
                # 执行更新
                optimizer.step()
                # 清空梯度
                optimizer.zero_grad()

                # 更新学习率和计数器 (建议放在 step 之后)
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1

                # 打印日志 (注意：还原 loss 数值以便观察)
                if args.local_rank == 0:
                    if iter_num % 20 == 0:
                        # 打印时把 loss 乘回去，方便你和之前的日志对比
                        logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                                     (iter_num, loss.item() * accumulation_steps, loss_ce.item(), loss_dice.item()))

        if args.local_rank == 0:
            # 1. 始终保存最新的模型为 'last_model.pth' (覆盖式保存，占用空间小，用于断点续训)
            last_save_path = os.path.join(snapshot_path, 'last_model.pth')
            torch.save(model.state_dict(), last_save_path)
            logging.info(f"save last model to {last_save_path}")

            # 2. 只保存最后 50 个 epoch 的权重
            # 计算起始保存轮数 (例如 100 - 50 = 50，从第 50 轮开始保存)
            save_start_epoch = args.max_epochs - 50

            # 如果配置的 max_epochs 小于 50，则可能会出现负数，这里做一个 max(0, ...) 保护
            save_start_epoch = max(0, save_start_epoch)

            if epoch_num >= save_start_epoch:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

    return "Training Finished!"

