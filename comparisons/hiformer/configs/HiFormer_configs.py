import ml_collections
import os

try:
    import wget
except ImportError:
    wget = None


os.makedirs('./weights', exist_ok=True)


# HiFormer-S Configs
def get_hiformer_s_configs():
    
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 384
    cfg.patch_size = 4
    cfg.num_classes = 2

    # 384 输入时，patch 后特征图为 96×96，需要 window_size=12
    cfg.swin_window_size = 12

    # 先关闭 Swin-Tiny 224 预训练权重，避免 window7/224 权重与 window12/384 不匹配
    cfg.swin_pretrained_path = ""

    # CNN Configs
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.resnet_pretrained = False

    # DLF Configs
    cfg.depth = [[1, 1, 0]]
    cfg.num_heads = (3, 3)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# HiFormer-B Configs
def get_hiformer_b_configs():

    cfg = ml_collections.ConfigDict()
    
    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 384
    cfg.patch_size = 4
    cfg.num_classes = 2
    cfg.swin_window_size = 12
    cfg.swin_pretrained_path = ""

    # CNN Configs
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024]
    cfg.resnet_pretrained = False

    # DLF Configs
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# HiFormer-L Configs
def get_hiformer_l_configs():
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 384
    cfg.patch_size = 4
    cfg.num_classes = 2
    cfg.swin_window_size = 12
    cfg.swin_pretrained_path = ""

    # CNN Configs
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 4, 0]]
    cfg.num_heads = (6, 6)
    cfg.mlp_ratio=(4., 4., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = False

    return cfg