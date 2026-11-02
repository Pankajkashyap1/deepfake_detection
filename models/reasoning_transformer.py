import torch
import torch.nn as nn


class ForgeryReasoningTransformer(nn.Module):
    """Transformer reasoning block over tokens that already include a global token."""

    def __init__(self, embed_dim: int = 768, num_layers: int = 6, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()

        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor):
        # tokens already contain:
        # [global_token, patch_tokens, pixel_tokens]

        x = self.input_dropout(tokens)
        x = self.transformer(x)
        x = self.norm(x)

        global_token = x[:, 0]   # use existing global token
        return x, global_token