# TPAMI 2024：Frequency-aware Feature Fusion for Dense Image Prediction (Integrated with CFCA & DAG)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np


# ==================== 创新模块升级：深度感知 & 语义门控结构先验注入 (SGSPI) ====================
class StructurePriorInjection(nn.Module):
    def __init__(self, channels, stage_idx=None, total_stages=4):
        super(StructurePriorInjection, self).__init__()
        self.sobel_conv = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, groups=channels, bias=False)
        self._init_sobel_prior(channels)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 动态注入门控 (注意：此处开启 bias=True，以便接受深度先验初始化)
        self.injection_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # -------------------------------------------------------------
        # 深度先验初始化：浅层 bias 为正 (鼓励高频)，深层 bias 为负 (压制伪影)
        # -------------------------------------------------------------
        if stage_idx is not None:
            # stage_idx 从深到浅通常为 1, 2, 3
            bias_val = (float(stage_idx) / float(total_stages - 1)) * 2.0 - 1.0
            nn.init.constant_(self.injection_gate[0].bias, bias_val)

    def _init_sobel_prior(self, channels, scale_factor=0.1):
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ], dtype=torch.float32) * scale_factor

        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ], dtype=torch.float32) * scale_factor

        weight = torch.zeros((channels * 2, 1, 3, 3), dtype=torch.float32)
        for i in range(channels):
            weight[i * 2, 0, :, :] = sobel_x
            weight[i * 2 + 1, 0, :, :] = sobel_y

        with torch.no_grad():
            self.sobel_conv.weight.copy_(weight)
            self.sobel_conv.weight.requires_grad = False

    def forward(self, hf_feat, lf_feat):
        semantic_gate = self.spatial_attention(lf_feat)

        edges = self.sobel_conv(lf_feat)
        edges_x = edges[:, 0::2, :, :]
        edges_y = edges[:, 1::2, :, :]

        struct_prior_raw = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-6)
        struct_prior_masked = struct_prior_raw * semantic_gate

        gate_weight = self.injection_gate(torch.cat([hf_feat, struct_prior_masked], dim=1))
        enhanced_hf = hf_feat + (struct_prior_masked * gate_weight)

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
        """
        Pure PyTorch fallback for CARAFE.
        Important: only upsample H/W, never the kernel dimension K^2.
        The previous implementation called F.interpolate on a 5D tensor
        shaped [B, C, K^2, H, W], which also doubled K^2 when up=2, causing
        a mismatch like 50 vs 25 at dim=2.
        """
        b, c, h, w = x.shape
        _, m_c, m_h, m_w = normed_mask.shape
        pad = kernel_size // 2

        pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
        unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
        # [B, C*K^2, H, W]
        unfold_x = unfold_x.view(b, c * kernel_size * kernel_size, h, w)

        if up != 1:
            # Upsample only spatial dimensions.
            unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')

        unfold_x = unfold_x.view(b, c, kernel_size * kernel_size, unfold_x.shape[-2], unfold_x.shape[-1])

        expected_mask_c = group * kernel_size * kernel_size
        if m_c != expected_mask_c:
            if m_c > expected_mask_c and m_c % expected_mask_c == 0:
                ratio = m_c // expected_mask_c
                normed_mask = normed_mask.view(b, ratio, expected_mask_c, m_h, m_w).mean(dim=1)
            else:
                raise RuntimeError(
                    f"CARAFE mask channels mismatch: got {m_c}, expected {expected_mask_c} "
                    f"(group={group}, kernel={kernel_size})."
                )

        normed_mask = normed_mask.view(b, group, kernel_size * kernel_size, m_h, m_w)

        if group == 1:
            # normed_mask is already [B, 1, K^2, H, W] when group=1.
            # Do NOT unsqueeze again, otherwise broadcasting creates a 6D tensor
            # and sum(dim=2) will reduce the wrong axis.
            res = unfold_x * normed_mask
        else:
            if c % group != 0:
                raise RuntimeError(f"Input channels {c} must be divisible by group {group}.")
            unfold_x = unfold_x.view(b, group, c // group, kernel_size * kernel_size, m_h, m_w)
            res = unfold_x * normed_mask.unsqueeze(2)
            res = res.view(b, c, kernel_size * kernel_size, m_h, m_w)

        return res.sum(dim=2).reshape(b, c, m_h, m_w)


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def hamming2D(M, N):
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    return np.outer(hamming_x, hamming_y)


class FreqFusion(nn.Module):
    # [新增] stage_idx 与 total_stages 参数
    def __init__(self, hr_channels, lr_channels, scale_factor=1, lowpass_kernel=5, highpass_kernel=3,
                 up_group=1, encoder_kernel=3, encoder_dilation=1, compressed_channels=64,
                 align_corners=False, upsample_mode='nearest', feature_resample=False,
                 feature_resample_group=4, comp_feat_upsample=True, use_high_pass=True,
                 use_low_pass=True, hr_residual=True, semi_conv=True, hamming_window=True,
                 feature_resample_norm=True, stage_idx=None, total_stages=4, **kwargs):
        super().__init__()
        self.stage_idx = stage_idx
        self.total_stages = total_stages

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

        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(self.compressed_channels,
                                              highpass_kernel ** 2 * up_group * scale_factor ** 2,
                                              encoder_kernel, padding=int((encoder_kernel - 1) * encoder_dilation / 2),
                                              dilation=encoder_dilation)
            # 注册具有深度感知的 ESPI 模块
            self.espi = StructurePriorInjection(channels=hr_channels, stage_idx=stage_idx, total_stages=total_stages)

        # ==================== 深度感知门控融合模块 (DAG) ====================
        if stage_idx is not None:
            # 高频强度随深度变浅而增强 (深层0.33 -> 浅层1.0)
            hf_init = float(stage_idx) / float(total_stages - 1)
            hf_init = min(max(hf_init, 1e-3), 0.99)
            self.depth_hf_weight = nn.Parameter(torch.tensor([np.log(hf_init / (1.0 - hf_init))], dtype=torch.float32))

            # 低频语义主干采用以 1.0 为中心的有界可学习门控
            self.depth_lf_weight = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

            # SE Channel Attention 动态调配高低频贡献
            self.dag_se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hr_channels + lr_channels, (hr_channels + lr_channels) // 4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d((hr_channels + lr_channels) // 4, hr_channels + lr_channels, 1, bias=False),
                nn.Sigmoid()
            )
            self.dag_conv = nn.Conv2d(hr_channels + lr_channels, hr_channels, 1, bias=False)
        # ====================================================================

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

        compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:],
                                     mode='nearest') + compressed_hr_feat
        mask_lr = self.content_encoder(compressed_x)
        if self.use_high_pass: mask_hr = self.content_encoder2(compressed_x)

        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)

        # 提取并上采样低频语义特征 (LF)
        lr_feat_up = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, self.scale_factor)

        if self.use_high_pass and mask_hr is not None:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
            hr_feat_hf_raw = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
            # 提取经过门控的高频物理先验 (HF)
            hr_feat_hf = self.espi(hr_feat_hf_raw, lr_feat_up)

            if self.hr_residual:
                hr_feat = (hr_feat_hf + hr_feat).contiguous()
            else:
                hr_feat = hr_feat_hf.contiguous()

        # ==================== 终极合并：修复原先高频特征丢失的 Bug ====================
        if self.stage_idx is not None:
            # 乘上深度可学习标量权重（有界，避免负权重破坏稳定性）
            hf_w = torch.sigmoid(self.depth_hf_weight)
            lf_w = 1.0 + 0.5 * torch.tanh(self.depth_lf_weight)
            hr_feat_w = hr_feat * hf_w
            lr_feat_up_w = lr_feat_up * lf_w

            # 拼接高低频特征，经 SE 注意力调控后融合
            concat_feat = torch.cat([hr_feat_w, lr_feat_up_w], dim=1)
            gate = self.dag_se(concat_feat)
            concat_feat = concat_feat * gate
            fused_out = self.dag_conv(concat_feat) + lr_feat_up

            # 将 fused_out 作为最终完整的高低频融合特征返回给第 3 个参数
            return mask_lr, hr_feat, fused_out
        else:
            # 兼容非 DAG 模式的硬性相加
            fused_out = hr_feat + lr_feat_up
            return mask_lr, hr_feat, fused_out