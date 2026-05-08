import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PoolAttFF(nn.Module):
    """
    Attention pooling module

    Args:
        dim_hidden: Hidden dimension of input features
        dropout: Dropout rate for attention network
    """

    def __init__(self, dim_hidden, dropout):
        super().__init__()
        self.dim_hidden = dim_hidden

        self.linear1 = nn.Linear(self.dim_hidden, 2 * self.dim_hidden)
        self.linear2 = nn.Linear(2 * self.dim_hidden, 1)

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2, 1)

        if mask is not None:
            expanded_mask = mask.unsqueeze(1)
            mask_positions = (expanded_mask == 0)
            att = att.masked_fill(mask_positions, float('-inf'))

        att = F.softmax(att, dim=2)

        x_pooled = torch.bmm(att, x).squeeze(1)

        return x_pooled


class AD_EGE_Model(nn.Module):
    """
    AD detection model specifically for eGeMAPS features (25-dim)

    Simple and Effective Architecture:
        1. Feature expansion layer
        2. Two-layer MLP with dropout
        3. Attention pooling (aggregate time dimension)
        4. Output classification layer

    Input:
        - x: (batch_size, 10, 25) - eGeMAPS features

    Output:
        - logits: (batch_size, 2) - Control and Dementia logits
    """

    def __init__(self, dim_input=25, dim_hidden=14, dropout=0.3):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dropout = nn.Dropout(dropout)

        self.linear_layer1 = nn.Linear(25, 64)
        self.norm1 = nn.BatchNorm1d(64)

        self.linear_layer2 = nn.Linear(64, 32)
        self.norm2 = nn.BatchNorm1d(32)

        # Attention pooling (aggregate time dimension)
        self.pool_ad = PoolAttFF(dim_hidden=32, dropout=dropout)

        # Output mapping layer (64 → 2)
        self.output_layer = nn.Linear(32, 2)  # Binary classification

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, 25) - eGeMAPS features
            mask: Not used for eGeMAPS (no padding needed)

        Returns:
            out: (batch_size, 2) - AD classification logits
        """
        x = self.linear_layer1(x)
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear_layer2(x)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)

        x_pooled = self.pool_ad(x, mask)

        out = self.output_layer(x_pooled)
        return out
