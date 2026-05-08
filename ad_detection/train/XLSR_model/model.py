import torch
from torch import Tensor, nn
import torch.nn.functional as F
from utils.config import PROJECT_ROOT
import fairseq

########################XLSR-53-300m####################################
class SSLModel(nn.Module):
    """
    Args:
        device: Device (cuda/cpu)
        freeze_xlsr: Whether to freeze XLSR parameters
            - True: Freeze all parameters, only extract features (no XLSR update)
            - False: Unfreeze parameters, allow fine-tuning (will update XLSR)
    """
    def __init__(self, device, freeze_xlsr=True):
        super(SSLModel, self).__init__()

        print("XLSR: Loading pretrained model")
        cp_path = str(PROJECT_ROOT / "models/xlsr2_300m.pt")

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].to(device)
        self.device = device
        self.out_dim = 1024
        self.freeze_xlsr = freeze_xlsr

        # Set to eval mode and freeze parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


    def extract_feat(self, input_data):
        """
        Extract XLSR features
        
        Args:
            input_data: Audio input
        
        Returns:
            embedding: Output features from the last layer
            layerresult: Outputs from all layers
        """
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(device=input_data.device)
        if next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(dtype=input_data.dtype)
        
        # The XLSR-53 model expects input in the format: (Batch, Time)
        # Stereo audio is (Batch, Time, Channels), e.g., (32, 48000, 2)
        #   32 = batch size, 48000 = time steps, 2 = stereo channels
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]  # Keep only the first channel; it will automatically
                                            # be reduced to 2D (Batch, Time)
        # Mono audio, e.g., (32, 48000)
        else:
            input_tmp = input_data

        # Extract features
        model_output = self.model(input_tmp, mask=False, features_only=True)
        embedding = model_output['x']  # Features from the last layer (Batch, Time_downsampled, 1024),
                                    # Time_downsampled = original time steps / 320 (downsampling rate)
        layerresult = model_output['layer_results']  # Features from all layers

        return embedding, layerresult

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

        # Attention network: hidden -> 2*hidden -> 1 (attention weights)
        self.linear1 = nn.Linear(self.dim_hidden, 2 * self.dim_hidden)
        self.linear2 = nn.Linear(2 * self.dim_hidden, 1)

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) - 1 for real data, 0 for padding (optional)

        Returns:
            x_pooled: (batch_size, hidden_dim) - pooled features
        """
        # Compute attention scores
        # x: (B, L, H) -> (B, L, 2H) -> (B, L, 1)
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))

        # Transpose for masking and softmax
        # (B, L, 1) -> (B, 1, L)
        att = att.transpose(2, 1)

        # Apply mask - set padding positions to -inf so they become 0 after softmax
        if mask is not None:
            # (batch, seq_len) -> (batch, 1, seq_len)
            expanded_mask = mask.unsqueeze(1)
            
            #  1 -> False -> keep, 0 -> True ->mask out
            mask_positions = (expanded_mask == 0)

            # Fill masked positions with -inf
            att = att.masked_fill(mask_positions, float('-inf'))

        # softmax(-inf) = 0
        att = F.softmax(att, dim=2)  # (B, 1, L), sum over L = 1.0

        # att: (B, 1, L), x: (B, L, H) -> bmm -> (B, 1, H) -> squeeze -> (B, H)
        x_pooled = torch.bmm(att, x).squeeze(1)

        return x_pooled


############################################################
# Model classes for XLSR features
############################################################

class AD_XLSR_Model(nn.Module):
    """
    AD detection model specifically for XLSR features (1024-dim)
    CNN + Linear classification head to reduce parameters and capture temporal patterns.

    Input:
        - x: (batch_size, seq_len, 1024) - XLSR features
        - mask: (batch_size, seq_len) - attention mask (optional)

    Output:
        - logits: (batch_size, 2) - Control and Dementia logits
    """

    def __init__(self, dropout=0.4):
        super().__init__()

        # BatchNorm normalization
        self.norm = nn.BatchNorm1d(1024)

        # Conv1d: 1024 → 32 (kernel_size=3, padding=1 preserves seq_len)
        self.conv1 = nn.Conv1d(1024, 32, kernel_size=5, padding=2)
        self.bn_conv = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(dropout)

        # Attention pooling
        self.pool_ad = PoolAttFF(
            dim_hidden=32,
            dropout=dropout)

        # Output mapping layer (32 → 2)
        self.output_layer = nn.Linear(32, 2)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, 1024) - XLSR features
            mask: (batch_size, seq_len) - attention mask (1=real, 0=padding), optional

        Returns:
            out: (batch_size, 2) - AD classification logits
        """
        # BatchNorm: (B, L, 1024) -> (B, 1024, L) -> normalize -> (B, 1024, L)
        x = self.norm(x.permute(0, 2, 1))

        # Conv1d: (B, 1024, L) -> (B, 32, L)
        x = self.conv1(x)
        x = self.bn_conv(x)
        x = F.relu(x)
        x = self.dropout(x)

        # (B, 32, L) -> (B, L, 32) for attention pooling
        x = x.permute(0, 2, 1)

        # Attention pooling
        x_pooled = self.pool_ad(x, mask)

        # Output mapping layer
        out = self.output_layer(x_pooled)

        return out