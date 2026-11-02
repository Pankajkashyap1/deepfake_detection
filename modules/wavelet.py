import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.residual import ResidualConvBlock

class WaveletTransform(nn.Module):
    """Fast fixed Haar DWT."""
    def __init__(self):
        super().__init__()

        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2
        lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2
        hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2

        filters = torch.stack([ll, lh, hl, hh])  # (4, 2, 2)
        self.register_buffer("filters", filters.unsqueeze(1))  # (4,1,2,2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        out = F.conv2d(x, self.filters, stride=2)
        out = out.view(B, C * 4, H // 2, W // 2)  # (B,12,112,112)
        return out


class FrequencyEncoder(nn.Module):
    """Wavelet-frequency branch with residual blocks + token normalization."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()

        self.dwt = WaveletTransform()

        self.net = nn.Sequential(
            ResidualConvBlock(12, 64, stride=2),   # 112 -> 56
            ResidualConvBlock(64, 128, stride=2),  # 56 -> 28
            ResidualConvBlock(128, 256, stride=2), # 28 -> 14
            nn.Conv2d(256, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=embed_dim),
        )

        self.token_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_maps = self.dwt(x)                        # (B,12,112,112)
        features = self.net(freq_maps)                # (B,768,14,14)
        tokens = features.flatten(2).transpose(1, 2) # (B,196,768)
        tokens = self.token_norm(tokens)
        return tokens