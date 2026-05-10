import torch
from torch import Tensor, nn
import torch.nn.functional as F

from XLSR_model.model import PoolAttFF
from .config import DROPOUT, PRE_POOL_KERNEL


class AD_SLS_Model(nn.Module):
    """
    AD-detection head for SLS-style cached features.

    Input:
        x:    (B, L, T, 1024)  — pre-extracted XLS-R layer features
        mask: (B, T)            — 1 for real frames, 0 for padding (optional)

    Pipeline:
        1. Mask-aware mean over T per layer       -> (B, L, 1024)
        2. Linear(1024, 1) + sigmoid              -> (B, L, 1) layer weights
        3. Per-frame weighted sum over layers     -> (B, T, 1024)
        4. XLSR-style head with pre-CNN pooling:
              BatchNorm1d(1024)
              -> AvgPool1d(k=4, s=4)              # 20ms->80ms per frame, T 3000->750
              -> Conv1d(1024, 32, k=5)            # 5*80ms = 400ms receptive field
              -> BN1d(32) -> ReLU -> Dropout
              -> PoolAttFF(32) -> Linear(32, 2)
        5. -> (B, 2) raw logits  (use nn.CrossEntropyLoss directly)

    Pre-CNN AvgPool acts as a regularizer at small N: AD-relevant signals
    (pauses, disfluency, prosody) live at 100ms-second scales, so 20ms/
    frame XLS-R resolution is wasted capacity that the Conv1d can otherwise
    overfit to.
    """

    def __init__(self, dropout: float = DROPOUT, pre_pool_kernel: int = PRE_POOL_KERNEL):
        super().__init__()
        # Layer attention (SLS-specific, ~1K params)
        self.fc0 = nn.Linear(1024, 1)

        # XLSR-style head — mirrors AD_XLSR_Model + AvgPool1d before Conv1d
        self.pre_pool_kernel = pre_pool_kernel
        self.norm = nn.BatchNorm1d(1024)
        self.pre_pool = nn.AvgPool1d(kernel_size=pre_pool_kernel, stride=pre_pool_kernel)
        self.conv1 = nn.Conv1d(1024, 32, kernel_size=5, padding=2)
        self.bn_conv = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout)
        self.pool_ad = PoolAttFF(dim_hidden=32, dropout=dropout)
        self.output_layer = nn.Linear(32, 2)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 1. Mask-aware mean over T per layer -> (B, L, 1024)
        if mask is not None:
            mask_t = mask[:, None, :, None]                       # (B, 1, T, 1)
            denom = mask.sum(dim=1).clamp(min=1)[:, None, None]   # (B, 1, 1)
            layer_pooled = (x * mask_t).sum(dim=2) / denom        # (B, L, 1024)
        else:
            layer_pooled = x.mean(dim=2)                          # (B, L, 1024)

        # 2. Per-layer attention weights -> (B, L, 1)
        y0 = torch.sigmoid(self.fc0(layer_pooled))                # (B, L, 1)

        # 3. Per-frame weighted sum over layers -> (B, T, 1024)
        weighted = (x * y0.unsqueeze(-1)).sum(dim=1)              # (B, T, 1024)

        # 4. XLSR-style head with pre-CNN AvgPool
        h = self.norm(weighted.permute(0, 2, 1))                  # (B, 1024, T)
        h = self.pre_pool(h)                                      # (B, 1024, T')  T' = T // k
        h = self.conv1(h)                                         # (B, 32, T')
        h = self.bn_conv(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = h.permute(0, 2, 1)                                    # (B, T', 32)

        # Pool mask along T to match: any real frame in window keeps the slot real
        if mask is not None:
            pooled_mask = F.max_pool1d(
                mask.unsqueeze(1).float(),
                kernel_size=self.pre_pool_kernel,
                stride=self.pre_pool_kernel,
            ).squeeze(1)                                          # (B, T')
        else:
            pooled_mask = None

        h_pooled = self.pool_ad(h, pooled_mask)                   # (B, 32)
        logits = self.output_layer(h_pooled)                      # (B, 2)
        return logits
