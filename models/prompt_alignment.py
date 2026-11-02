import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DualPromptAlignment(nn.Module):
    """
    Two-class prompt module:
    - 768-d prompt tokens for reasoning/bridge
    - CLIP-space class prototypes for contrastive alignment
    """

    def __init__(
        self,
        clip_backbone,
        embed_dim: int = 768,
        clip_proj_dim: int = 512,
        num_prompt_tokens: int = 4,
        proto_scale: float = 0.2,
    ):
        super().__init__()

        self.clip = clip_backbone
        self.embed_dim = embed_dim
        self.clip_proj_dim = clip_proj_dim
        self.num_prompt_tokens = num_prompt_tokens
        self.proto_scale = proto_scale

        # For reasoning tokens only
        self.text_to_token = nn.Linear(clip_proj_dim, embed_dim)

        self.real_templates = [
            "a real human face with natural textures and consistent lighting",
            "an authentic face photo with realistic facial details",
            "a genuine portrait with consistent skin texture"
        ]

        self.fake_templates = [
            "a fake face with artifacts, blending inconsistencies, and unnatural textures",
            "a manipulated synthetic face with visual forgery traces",
            "an altered portrait with unrealistic facial texture"
        ]

        with torch.no_grad():
            real_anchor = self._encode_templates(self.real_templates)   # (512,)
            fake_anchor = self._encode_templates(self.fake_templates)   # (512,)

        self.register_buffer("real_anchor", real_anchor)
        self.register_buffer("fake_anchor", fake_anchor)

        # Learnable prompt tokens for reasoning path (768-d)
        real_token_base = self.text_to_token(real_anchor.unsqueeze(0)).unsqueeze(1).repeat(1, num_prompt_tokens, 1)
        fake_token_base = self.text_to_token(fake_anchor.unsqueeze(0)).unsqueeze(1).repeat(1, num_prompt_tokens, 1)

        self.real_tokens = nn.Parameter(real_token_base + 0.02 * torch.randn_like(real_token_base))
        self.fake_tokens = nn.Parameter(fake_token_base + 0.02 * torch.randn_like(fake_token_base))

        # Learnable CLIP-space prototype residuals for contrastive path
        self.real_proto_delta = nn.Parameter(torch.zeros(clip_proj_dim))
        self.fake_proto_delta = nn.Parameter(torch.zeros(clip_proj_dim))

        self.token_norm = nn.LayerNorm(embed_dim)

    def _encode_templates(self, texts):
        embeds = self.clip.encode_text(texts)         # (T,512), normalized by CLIPBackbone
        embeds = embeds.mean(dim=0)                   # (512,)
        embeds = embeds / (embeds.norm(dim=-1, keepdim=False) + 1e-6)
        return embeds

    def get_class_prototypes(self):
        real_proto = self.real_anchor + self.proto_scale * self.real_proto_delta
        fake_proto = self.fake_anchor + self.proto_scale * self.fake_proto_delta

        real_proto = real_proto / (real_proto.norm(dim=-1, keepdim=True) + 1e-6)
        fake_proto = fake_proto / (fake_proto.norm(dim=-1, keepdim=True) + 1e-6)

        prototypes = torch.stack([real_proto, fake_proto], dim=0)  # (2,512)
        return prototypes

    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # reasoning prompt tokens (768)
        real_tokens = self.real_tokens.expand(batch_size, -1, -1)
        fake_tokens = self.fake_tokens.expand(batch_size, -1, -1)
        prompt_tokens = torch.cat([real_tokens, fake_tokens], dim=1)
        prompt_tokens = self.token_norm(prompt_tokens)

        # contrastive prototypes (2,512)
        class_prototypes = self.get_class_prototypes()

        return prompt_tokens, class_prototypes

    def anchor_regularization(self):
        return (
            F.mse_loss(self.real_anchor + self.proto_scale * self.real_proto_delta, self.real_anchor) +
            F.mse_loss(self.fake_anchor + self.proto_scale * self.fake_proto_delta, self.fake_anchor)
        )