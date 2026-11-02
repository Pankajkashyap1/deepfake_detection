import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """Global + local gated multimodal fusion with normalization and residual skip."""

    def __init__(self, input_dim: int = 768, gate_dim: int = 256):
        super().__init__()

        self.rgb_norm = nn.LayerNorm(input_dim)
        self.freq_norm = nn.LayerNorm(input_dim)
        self.noise_norm = nn.LayerNorm(input_dim)

        self.global_gate = nn.Sequential(
            nn.Linear(input_dim * 3, gate_dim),
            nn.GELU(),
            nn.Linear(gate_dim, 3)
        )

        self.local_gate = nn.Sequential(
            nn.Linear(input_dim * 3, gate_dim),
            nn.GELU(),
            nn.Linear(gate_dim, 3)
        )

        self.out_norm = nn.LayerNorm(input_dim)
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: dict) -> torch.Tensor:
        rgb = self.rgb_norm(features["rgb"])      # (B,N,D)
        freq = self.freq_norm(features["freq"])
        noise = self.noise_norm(features["noise"])

        # 1. Global gate
        g_rgb = rgb.mean(dim=1)
        g_freq = freq.mean(dim=1)
        g_noise = noise.mean(dim=1)

        global_input = torch.cat([g_rgb, g_freq, g_noise], dim=-1)
        global_weights = torch.softmax(self.global_gate(global_input), dim=-1)  # (B,3)

        # 2. Local gate
        local_input = torch.cat([rgb, freq, noise], dim=-1)   # (B,N,3D)
        local_weights = torch.softmax(self.local_gate(local_input), dim=-1)     # (B,N,3)

        # 3. Combine global and local
        g_w = global_weights.unsqueeze(1)                     # (B,1,3)
        weights = local_weights * g_w                         # (B,N,3)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        w1 = weights[..., 0:1]
        w2 = weights[..., 1:2]
        w3 = weights[..., 2:3]

        fused = w1 * rgb + w2 * freq + w3 * noise

        # 4. Residual preserve semantic structure
        fused = fused + self.res_scale * rgb

        # 5. Output normalization
        fused = self.out_norm(fused)

        return fused