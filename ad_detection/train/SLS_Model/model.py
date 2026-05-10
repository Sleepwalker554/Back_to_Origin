import torch
from torch import Tensor, nn
import torch.nn.functional as F

from XLSR_model.model import PoolAttFF


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
        4. Reused XLSR head:
              BatchNorm1d(1024) -> Conv1d(1024,32,k=5) -> BN1d(32)
              -> ReLU -> Dropout -> PoolAttFF(32) -> Linear(32, 2)
        5. -> (B, 2) raw logits  (use nn.CrossEntropyLoss directly)

    The only architectural difference vs AD_XLSR_Model is the
    layer-attention preamble (~1025 params). After step 3 the tensor
    shape (B, T, 1024) matches XLSR's cached feature shape exactly,
    so the same head applies unchanged.
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        # Layer attention (SLS-specific, ~1K params)
        self.fc0 = nn.Linear(1024, 1)

        # XLSR-style head — mirrors AD_XLSR_Model
        self.norm = nn.BatchNorm1d(1024)
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

        # 4. XLSR-style head
        h = self.norm(weighted.permute(0, 2, 1))                  # (B, 1024, T)
        h = self.conv1(h)                                         # (B, 32, T)
        h = self.bn_conv(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = h.permute(0, 2, 1)                                    # (B, T, 32)
        h_pooled = self.pool_ad(h, mask)                          # (B, 32)
        logits = self.output_layer(h_pooled)                      # (B, 2)
        return logits
