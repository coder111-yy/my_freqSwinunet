# TPAMI 2024：Frequency-aware Feature Fusion for Dense Image Prediction (Integrated with CFCA)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np


# ==================== 创新模块：显式结构先验注入 (ESPI) ====================
class StructurePriorInjection(nn.Module):
    def __init__(self, channels):
        super(StructurePriorInjection, self).__init__()
        # 1. 结构先验提取器：使用深度可分离卷积 (Depthwise Conv)
        # 它像可学习的 Sobel/Laplacian 算子，从干净的低频语义中提取局部边界梯度
        self.spatial_gradient = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

        # 2. 动态注入门控：决定在哪些空间位置、注入多少强度的结构先验
        self.injection_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, hf_feat, lf_feat):
        # 1. 从干净的低频特征中提取“结构/边缘先验”
        struct_prior = self.spatial_gradient(lf_feat)

        # 2. 结合高低频信息，计算自适应的注入权重图 (0~1 之间)
        # 只有在确实需要增强边缘的地方，权重才会高
        gate_weight = self.injection_gate(torch.cat([hf_feat, struct_prior], dim=1))

        # 3. 核心创新：残差注入 (Residual Injection)
        # 做加法：保留高频原始所有信息的基础上，显式增强边界
        enhanced_hf = hf_feat + (struct_prior * gate_weight)

        return enhanced_hf.contiguous()


# =========================================================================

try:
    from mmcv.ops.carafe import normal_init, xavier_init, carafe
except ImportError:
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None: nn.init.constant_(module.bias, bias)


    def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None: nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None: nn.init.constant_(module.bias, bias)


    def carafe(x, normed_mask, kernel_size, group=1, up=1):
        b, c, h, w = x.shape
        _, m_c, m_h, m_w = normed_mask.shape
        pad = kernel_size // 2
        pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
        unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
        unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, h, w)
        unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
        expected_mask_c = group * kernel_size * kernel_size
        if m_c != expected_mask_c:
            if m_c > expected_mask_c and m_c % expected_mask_c == 0:
                ratio = m_c // expected_mask_c
                normed_mask = normed_mask.view(b, ratio, expected_mask_c, m_h, m_w).mean(dim=1)
        normed_mask = normed_mask.reshape(b, group, kernel_size * kernel_size, m_h, m_w)
        if group == 1:
            res = unfold_x * normed_mask
        else:
            unfold_x = unfold_x.view(b, group, c // group, kernel_size * kernel_size, m_h, m_w)
            res = unfold_x * normed_mask.unsqueeze(2)
            res = res.view(b, c, kernel_size * kernel_size, m_h, m_w)
        return res.sum(dim=2).reshape(b, c, m_h, m_w)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None: nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None: nn.init.constant_(module.bias, bias)


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def hamming2D(M, N):
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    return np.outer(hamming_x, hamming_y)


class FreqFusion(nn.Module):
    def __init__(self, hr_channels, lr_channels, scale_factor=1, lowpass_kernel=5, highpass_kernel=3,
                 up_group=1, encoder_kernel=3, encoder_dilation=1, compressed_channels=64,
                 align_corners=False, upsample_mode='nearest', feature_resample=False,
                 feature_resample_group=4, comp_feat_upsample=True, use_high_pass=True,
                 use_low_pass=True, hr_residual=True, semi_conv=True, hamming_window=True,
                 feature_resample_norm=True, **kwargs):
        super().__init__()
        self.scale_factor, self.lowpass_kernel, self.highpass_kernel = scale_factor, lowpass_kernel, highpass_kernel
        self.up_group, self.encoder_kernel, self.encoder_dilation = up_group, encoder_kernel, encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoder = nn.Conv2d(self.compressed_channels, lowpass_kernel ** 2 * up_group * scale_factor ** 2,
                                         encoder_kernel, padding=int((encoder_kernel - 1) * encoder_dilation / 2),
                                         dilation=encoder_dilation)

        self.align_corners, self.upsample_mode, self.hr_residual = align_corners, upsample_mode, hr_residual
        self.use_high_pass, self.use_low_pass, self.semi_conv = use_high_pass, use_low_pass, semi_conv
        self.feature_resample, self.comp_feat_upsample = feature_resample, comp_feat_upsample

        if self.feature_resample:
            from .freq_fusion import LocalSimGuidedSampler
            self.dysampler = LocalSimGuidedSampler(in_channels=compressed_channels, scale=2, style='lp',
                                                   groups=feature_resample_group, use_direct_scale=True,
                                                   kernel_size=encoder_kernel, norm=feature_resample_norm)

        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(self.compressed_channels,
                                              highpass_kernel ** 2 * up_group * scale_factor ** 2,
                                              encoder_kernel, padding=int((encoder_kernel - 1) * encoder_dilation / 2),
                                              dilation=encoder_dilation)

            # 注册 ESPI 模块
            self.espi = StructurePriorInjection(channels=hr_channels)

        self.hamming_window = hamming_window
        if self.hamming_window:
            self.register_buffer('hamming_lowpass',
                                 torch.FloatTensor(hamming2D(lowpass_kernel, lowpass_kernel))[None, None,])
            self.register_buffer('hamming_highpass',
                                 torch.FloatTensor(hamming2D(highpass_kernel, highpass_kernel))[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass: normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None: mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))
        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w).permute(0, 1, 4, 5, 2, 3).reshape(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        return mask.view(n, mask_channel, h, w, -1).permute(0, 1, 4, 2, 3).reshape(n, -1, h, w).contiguous()

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)

        mask_hr = None

        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_high_pass:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                    mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel,
                                                          hamming=self.hamming_highpass)
                    compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat,
                                                                                          mask_hr_init,
                                                                                          self.highpass_kernel,
                                                                                          self.up_group, 1)
                    mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                    mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel,
                                                          hamming=self.hamming_lowpass)
                    mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                    mask_lr_lr_feat = F.interpolate(
                        carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2),
                        size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat
                    mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
                    mask_hr_lr_feat = F.interpolate(
                        carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel,
                               self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                else:
                    raise NotImplementedError
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(
                    self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(
                        self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
        else:
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:],
                                         mode='nearest') + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_high_pass: mask_hr = self.content_encoder2(compressed_x)

        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)

        # 统一变量名以防混淆
        if self.semi_conv:
            lr_feat_up = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)
        else:
            lr_feat_up = resize(input=lr_feat, size=hr_feat.shape[2:], mode=self.upsample_mode,
                                align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
            lr_feat_up = carafe(lr_feat_up, mask_lr, self.lowpass_kernel, self.up_group, 1)

        if self.use_high_pass and mask_hr is not None:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)

            # 1. 提取原始高频特征 (包含脆弱边界与散斑噪声)
            hr_feat_hf_raw = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)

            # 2. ESPI 介入：将低频的结构先验显式注入到高频特征中
            # 这里的 lr_feat_up 是已经上采样好的、干净的低频语义特征
            hr_feat_hf = self.espi(hr_feat_hf_raw, lr_feat_up)

            # 3. 融合回主干
            if self.hr_residual:
                hr_feat = (hr_feat_hf + hr_feat).contiguous()
            else:
                hr_feat = hr_feat_hf.contiguous()

        if self.feature_resample:
            # dysampler 同样调用上采样后的特征
            lr_feat_up = self.dysampler(hr_x=compressed_hr_feat, lr_x=compressed_lr_feat, feat2sample=lr_feat_up)

        return mask_lr, hr_feat, lr_feat_up