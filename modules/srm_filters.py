import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.residual import ResidualConvBlock

class SRMFilters(nn.Module):
    """Fixed SRM filter bank: the 3 normalized filters of Zhou et al. (2018)."""

    def __init__(self):
        super().__init__()

        base_filters = torch.tensor([
            [[0, 0, 0, 0, 0],
             [0, -1, 2, -1, 0],
             [0, 2, -4, 2, 0],
             [0, -1, 2, -1, 0],
             [0, 0, 0, 0, 0]],

            [[-1, 2, -2, 2, -1],
             [2, -6, 8, -6, 2],
             [-2, 8, -12, 8, -2],
             [2, -6, 8, -6, 2],
             [-1, 2, -2, 2, -1]],

            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        ], dtype=torch.float32)
        base_filters = base_filters / torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1)

        self.register_buffer("filters", base_filters.unsqueeze(1))  # (3,1,5,5)
        self.tlu = nn.Hardtanh(min_val=-3.0, max_val=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        filters = self.filters.repeat(C, 1, 1, 1)  # (3*C,1,5,5)
        residuals = F.conv2d(x, filters, padding=2, groups=C)  # (B,3*C,H,W)
        residuals = self.tlu(residuals)
        return residuals  # for RGB: (B,9,H,W)


class NoiseEncoder(nn.Module):
    """SRM residual branch with learnable expansion."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()

        self.srm = SRMFilters()

        self.stem = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
        )

        self.net = nn.Sequential(
            ResidualConvBlock(32, 64, stride=2),    # 224 -> 112
            ResidualConvBlock(64, 128, stride=2),   # 112 -> 56
            ResidualConvBlock(128, 256, stride=2),  # 56 -> 28
            ResidualConvBlock(256, 256, stride=2),  # 28 -> 14
            nn.Conv2d(256, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=embed_dim),
        )

        self.token_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise_maps = self.srm(x)                     # (B,9,224,224)
        features = self.stem(noise_maps)
        features = self.net(features)               # (B,768,14,14)
        tokens = features.flatten(2).transpose(1, 2) # (B,196,768)
        tokens = self.token_norm(tokens)
        return tokens
