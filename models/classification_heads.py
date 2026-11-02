import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, global_token):
        global_token = self.norm(global_token)
        return self.mlp(global_token)


class LocalizationHead(nn.Module):
    def __init__(self, embed_dim=768, init_temperature=0.1):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim ** -0.5
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

    def forward(self, global_token, patch_tokens, image_size=(224, 224)):
        B, N, D = patch_tokens.shape

        q = self.q_proj(global_token)     # (B,D)
        k = self.k_proj(patch_tokens)     # (B,N,D)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(k, q.unsqueeze(-1)).squeeze(-1)  # (B,N)
        attn = attn * self.scale

        temp = torch.clamp(self.temperature, min=0.05, max=1.0)
        attn = torch.softmax(attn / temp, dim=-1)

        grid = int(N ** 0.5)
        assert grid * grid == N, f"Patch count {N} is not a square number"

        heatmap = attn.view(B, 1, grid, grid)

        heatmap = F.interpolate(
            heatmap,
            size=image_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        heatmap = heatmap - heatmap.amin(dim=(1, 2), keepdim=True)
        heatmap = heatmap / (heatmap.amax(dim=(1, 2), keepdim=True) + 1e-6)

        return heatmap