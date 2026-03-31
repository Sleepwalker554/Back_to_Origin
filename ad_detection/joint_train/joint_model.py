import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
import fairseq
from pathlib import Path

from frcrn import FRCRN
from joint_config import PROJECT_ROOT


class JointSSLModel(nn.Module):
    """
    XLSR-53 封装，支持:
    - 解冻最后 finetune_last_n 层 transformer
    - 对 frozen 层用 gradient checkpointing 省显存
    """

    def __init__(self, device, finetune_last_n=3, use_checkpoint=True):
        super().__init__()

        cp_path = str(PROJECT_ROOT / "models/xlsr2_300m.pt")
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].to(device)
        self.device = device
        self.out_dim = 1024
        self.finetune_last_n = finetune_last_n
        self.use_checkpoint = use_checkpoint

        num_layers = len(self.model.encoder.layers)  # 24
        self.frozen_layers = num_layers - finetune_last_n  # 21

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻最后 finetune_last_n 层
        if finetune_last_n > 0:
            for layer in self.model.encoder.layers[-finetune_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # 解冻 encoder 最终的 layer_norm
            for param in self.model.encoder.layer_norm.parameters():
                param.requires_grad = True

        self.model.eval()

        # 统计参数量
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"XLSR: {total:,} total params, {trainable:,} trainable "
              f"(last {finetune_last_n} layers), checkpoint={use_checkpoint}")

    def extract_feat(self, input_data):
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]

        # ====== Feature extraction (CNN, frozen) ======
        # 不能用 torch.no_grad()! 虽然权重 frozen (requires_grad=False),
        # 但梯度需要穿过计算图回传到 FRCRN
        features = self.model.feature_extractor(input_data)

        features = features.transpose(1, 2)  # (B, T, C)
        features = self.model.layer_norm(features)

        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)

        # ====== Positional encoding (frozen) ======
        x_conv = self.model.encoder.pos_conv(features.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = features + x_conv

        if not self.model.encoder.layer_norm_first:
            x = self.model.encoder.layer_norm(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # ====== Transformer layers ======
        for i, layer in enumerate(self.model.encoder.layers):
            if i < self.frozen_layers and self.use_checkpoint:
                # Frozen 层: gradient checkpointing, 只返回 x
                x = checkpoint(
                    self._run_layer, layer, x, None,
                    use_reentrant=False,
                )
            else:
                # Trainable 层: 正常 forward
                x, _ = layer(x, self_attn_padding_mask=None, need_weights=False)

        # Final layer norm
        x = self.model.encoder.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x

    @staticmethod
    def _run_layer(layer, x, padding_mask):
        """Wrapper for gradient checkpointing, 只返回 x"""
        x, _ = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
        return x


class JointFRCRN(nn.Module):
    """
    FRCRN 封装: 冻结 stft + istft + unet1, 只训练 unet2
    """

    def __init__(self, pretrained_path=None):
        super().__init__()

        self.frcrn = FRCRN(
            complex=True,
            model_complexity=45,
            model_depth=14,
            log_amp=False,
            padding_mode="zeros",
            win_len=640,
            win_inc=320,
            fft_len=640,
            win_type="hann",
        )

        # 加载预训练权重
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.frcrn.load_state_dict(state_dict, strict=False)
            print(f"FRCRN: loaded pretrained weights from {pretrained_path}")

        # 冻结 stft, istft, unet1
        for param in self.frcrn.stft.parameters():
            param.requires_grad = False
        for param in self.frcrn.istft.parameters():
            param.requires_grad = False
        for param in self.frcrn.unet.parameters():
            param.requires_grad = False

        # unet2 保持可训练
        trainable = sum(p.numel() for p in self.frcrn.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.frcrn.parameters())
        print(f"FRCRN: {total:,} total params, {trainable:,} trainable (unet2 only)")

    def forward(self, inputs):
        """
        Args:
            inputs: (B, T) raw waveform
        Returns:
            denoised: (B, T) enhanced waveform (wav_l2)
            out_list: full output list for optional loss computation
        """
        out_list = self.frcrn(inputs)
        denoised = out_list[4]  # wav_l2: 第二级UNet最终输出
        return denoised, out_list


class PoolAttFF(nn.Module):
    """Attention pooling (复制自 ad_detection/train/model.py)"""

    def __init__(self, dim_hidden, dropout):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.linear1 = nn.Linear(self.dim_hidden, 2 * self.dim_hidden)
        self.linear2 = nn.Linear(2 * self.dim_hidden, 1)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2, 1)
        if mask is not None:
            expanded_mask = mask.unsqueeze(1)
            mask_positions = (expanded_mask == 0)
            att = att.masked_fill(mask_positions, float('-inf'))
        att = F.softmax(att, dim=2)
        x_pooled = torch.bmm(att, x).squeeze(1)
        return x_pooled


class AD_XLSR_Model(nn.Module):
    """AD分类器 (复制自 ad_detection/train/model.py)"""

    def __init__(self, dropout=0.4):
        super().__init__()
        self.norm = nn.BatchNorm1d(1024)
        self.conv1 = nn.Conv1d(1024, 32, kernel_size=5, padding=2)
        self.bn_conv = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout)
        self.pool_ad = PoolAttFF(dim_hidden=32, dropout=dropout)
        self.output_layer = nn.Linear(32, 2)

    def forward(self, x, mask=None):
        x = self.norm(x.permute(0, 2, 1))
        x = self.conv1(x)
        x = self.bn_conv(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x_pooled = self.pool_ad(x, mask)
        out = self.output_layer(x_pooled)
        return out


class JointDenoiseADModel(nn.Module):
    """
    Joint Training 模型: FRCRN → XLSR → AD分类器
    """

    def __init__(self, frcrn_model, xlsr_model, ad_model, xlsr_max_time_steps=3000):
        super().__init__()
        self.frcrn_model = frcrn_model
        self.xlsr_model = xlsr_model
        self.ad_model = ad_model
        self.xlsr_max_time_steps = xlsr_max_time_steps

    def forward(self, raw_audio):
        """
        Args:
            raw_audio: (B, T) raw waveform, T = 60s * 16kHz = 960000

        Returns:
            denoised: (B, T) denoised waveform
            logits: (B, 2) AD classification logits
        """
        # 1. FRCRN 降噪
        denoised, _ = self.frcrn_model(raw_audio)  # (B, T)

        # 2. XLSR 特征提取
        xlsr_feat = self.xlsr_model.extract_feat(denoised)  # (B, T/320, 1024)

        # 3. Pad/truncate + mask
        seq_len = xlsr_feat.shape[1]
        if seq_len > self.xlsr_max_time_steps:
            xlsr_feat = xlsr_feat[:, :self.xlsr_max_time_steps, :]
            mask = torch.ones(xlsr_feat.shape[0], self.xlsr_max_time_steps,
                              device=xlsr_feat.device)
        elif seq_len < self.xlsr_max_time_steps:
            pad_len = self.xlsr_max_time_steps - seq_len
            xlsr_feat = F.pad(xlsr_feat, (0, 0, 0, pad_len))
            mask = torch.ones(xlsr_feat.shape[0], self.xlsr_max_time_steps,
                              device=xlsr_feat.device)
            mask[:, seq_len:] = 0
        else:
            mask = torch.ones(xlsr_feat.shape[0], seq_len, device=xlsr_feat.device)

        # 4. AD 分类
        logits = self.ad_model(xlsr_feat, mask)  # (B, 2)

        return denoised, logits
