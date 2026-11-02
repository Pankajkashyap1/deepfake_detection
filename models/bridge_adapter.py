import torch
import torch.nn as nn


class PromptGuidedBridge(nn.Module):
    """Bidirectional prompt-visual bridge with gated fusion and transformer-style refinement."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()

        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_t = nn.LayerNorm(embed_dim)
        self.norm_t2 = nn.LayerNorm(embed_dim)
        self.norm_z = nn.LayerNorm(embed_dim)

        self.attn_pv = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_vp = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.prompt_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.visual_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, visual_tokens: torch.Tensor, prompt_tokens: torch.Tensor) -> torch.Tensor:
        V = visual_tokens
        T = prompt_tokens

        # 1. Prompt attends to visual
        V_norm = self.norm_v(V)
        T_norm = self.norm_t(T)
        A_pv, _ = self.attn_pv(T_norm, V_norm, V_norm)
        T_prime = T + A_pv
        T_prime = T_prime + self.prompt_ffn(self.norm_t2(T_prime))

        # 2. Visual attends to updated prompt
        A_vp, _ = self.attn_vp(self.norm_v(V), self.norm_t2(T_prime), self.norm_t2(T_prime))

        # 3. Gated visual update
        gate_input = torch.cat([V, A_vp], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        Z = V + gate * A_vp

        # 4. Visual FFN refinement
        Z = Z + self.visual_ffn(self.norm_z(Z))

        return Z