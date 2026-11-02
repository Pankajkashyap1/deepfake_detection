import torch
import torch.nn as nn


class ResidualRefineBlock(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(16, embed_dim)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(16, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        return x


class MultiScaleTokens(nn.Module):
    """Multi-scale token extractor with residual local refinement and type encoding."""

    def __init__(self, embed_dim: int = 768, num_patches: int = 196, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.grid_size = int(num_patches ** 0.5)

        self.input_norm = nn.LayerNorm(embed_dim)
        self.global_norm = nn.LayerNorm(embed_dim)

        self.pixel_conv = ResidualRefineBlock(embed_dim)

        self.global_proj = nn.Linear(embed_dim, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 2 * num_patches, embed_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 3, embed_dim))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = fused_tokens.shape

        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        assert self.grid_size * self.grid_size == N, "num_patches must form a square grid"

        fused_tokens = self.input_norm(fused_tokens)

        # 1. Global token with residual projection
        pooled = fused_tokens.mean(dim=1, keepdim=True)                 # (B,1,D)
        global_token = self.global_proj(self.global_norm(pooled)) + pooled

        # 2. Patch tokens
        patch_tokens = fused_tokens

        # 3. Local refinement tokens
        grid = fused_tokens.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)
        pixel_feat = self.pixel_conv(grid)
        pixel_tokens = pixel_feat.flatten(2).transpose(1, 2)

        # residual interaction with original patches
        pixel_tokens = pixel_tokens + patch_tokens

        # 4. Type embeddings
        global_token = global_token + self.type_embed[:, 0:1]
        patch_tokens = patch_tokens + self.type_embed[:, 1:2]
        pixel_tokens = pixel_tokens + self.type_embed[:, 2:3]

        # 5. Concatenate + positional encoding
        tokens = torch.cat([global_token, patch_tokens, pixel_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]

        tokens = self.dropout(tokens)
        tokens = self.norm(tokens)
        return tokens