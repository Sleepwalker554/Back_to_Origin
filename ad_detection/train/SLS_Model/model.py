import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AD_SLS_Model(nn.Module):
    """
    AD-detection head for SLS-style cached features.

    Input:
        x: (batch_size, L, T, 1024) — pre-extracted XLS-R layer features
            where L = number of transformer layers (24 or 25 depending
            on fairseq version), T = time steps after 320x downsampling.
        mask: (batch_size, T) — 1 for real frames, 0 for padding (optional).

    Pipeline (parallels SLS getAttenF + classifier):
        1. Mask-aware mean over T per layer    -> (B, L, 1024)
        2. Linear(1024, 1) + sigmoid           -> (B, L, 1) layer weights
        3. Weighted sum over layers            -> (B, T, 1024)
        4. BatchNorm2d + SELU + AdaptiveMaxPool2d((67, 341)) -> (B, 1, 67, 341)
        5. Flatten -> Linear(22847, 1024) -> SELU -> Dropout -> Linear(1024, 2)

    Returns raw logits of shape (B, 2). Use nn.CrossEntropyLoss directly
    (no LogSoftmax — the original SLS used LogSoftmax + CrossEntropyLoss
    which double-applies softmax).

    The fixed AdaptiveMaxPool2d output shape (67, 341) keeps the
    Linear(22847, 1024) weight unchanged from the original SLS at
    T=201; at T=3000 the temporal pool window is ~45 frames instead
    of the original 3 (more aggressive smoothing, but fc1 stays small).
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.fc1 = nn.Linear(22847, 1024)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # x: (B, L, T, 1024)

        # 1. Mask-aware mean over time per layer -> (B, L, 1024)
        if mask is not None:
            mask_t = mask[:, None, :, None]                  # (B, 1, T, 1)
            denom = mask.sum(dim=1).clamp(min=1)[:, None, None]  # (B, 1, 1)
            layer_pooled = (x * mask_t).sum(dim=2) / denom   # (B, L, 1024)
        else:
            layer_pooled = x.mean(dim=2)                     # (B, L, 1024)

        # 2. Per-layer attention weights -> (B, L, 1) -> (B, L, 1, 1)
        y0 = torch.sigmoid(self.fc0(layer_pooled))           # (B, L, 1)
        y0 = y0.unsqueeze(-1)                                # (B, L, 1, 1)

        # 3. Weighted sum over layers -> (B, T, 1024)
        weighted = x * y0                                    # (B, L, T, 1024)
        summed = weighted.sum(dim=1)                         # (B, T, 1024)

        # 4. 2-D pooling head
        h = summed.unsqueeze(1)                              # (B, 1, T, 1024)
        h = self.first_bn(h)
        h = self.selu(h)
        h = F.adaptive_max_pool2d(h, (67, 341))              # (B, 1, 67, 341)

        # 5. Classifier head
        h = torch.flatten(h, 1)                              # (B, 22847)
        h = self.fc1(h)
        h = self.selu(h)
        h = self.dropout(h)
        logits = self.fc3(h)                                 # (B, 2)
        return logits
