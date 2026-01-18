import torch
import torch.nn as nn
from types import SimpleNamespace

from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from config import get_config
import os


def build_model():
    # 仿造 train.py 里 argparse 的参数，把会被用到的字段都补上
    args = SimpleNamespace(
        cfg="configs/swin_tiny_patch4_window7_224_lite.yaml",
        opts=None,

        # 下面这些都是 update_config 里会访问的字段
        batch_size=None,
        zip=False,
        cache_mode="part",        # 和你训练时日志里的 cache_mode 一致
        pretrained=None,
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level="O0",       # 这里就当成不用 AMP
        tag=None,
        eval=False,
        throughput=False,
    )

    config = get_config(args)

    model = SwinTransformerSys(
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=config.MODEL.SWIN.DEPTHS,
        depths_decoder=config.MODEL.SWIN.DECODER_DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        window_size=config.MODEL.SWIN.WINDOW_SIZE,
        mlp_ratio=config.MODEL.MLP_RATIO,
        qkv_bias=config.MODEL.QKV_BIAS,
        qk_scale=config.MODEL.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.SWIN.APE,
        patch_norm=config.MODEL.SWIN.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )

    # 加载预训练权重，跟正式训练保持一致
    pretrained_path = "./pretrained_ckpt/swin_tiny_patch4_window7_224.pth"
    if os.path.exists(pretrained_path):
        print("==> loading pretrained:", pretrained_path)
        model.load_from(config)
    else:
        print("==> WARNING: pretrained not found, using random init")

    return model


if __name__ == "__main__":
    # 先把 cudnn 关掉，避免 cudnn 某些 kernel 出问题
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = build_model().to(device)
    model.train()

    # 一个极简 batch：1 张 224x224 单通道图
    x = torch.randn(1, 1, 224, 224, device=device)
    y = torch.randint(0, 9, (1, 224, 224), device=device)  # 9 类，和 cfg 里一致

    criterion = nn.CrossEntropyLoss()

    out = model(x)
    print("out.shape:", out.shape)

    loss = criterion(out, y.long())
    print("loss:", loss.item())

    loss.backward()
    print("backward ok")
