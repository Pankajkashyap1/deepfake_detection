import torch
import torch.nn as nn

from .clip_backbone import CLIPBackbone
from modules.wavelet import FrequencyEncoder
from modules.srm_filters import NoiseEncoder


class MultiModalForgeryEncoder(nn.Module):
    """
    Multi-modal forgery encoder.

    Outputs:
        {
            "rgb": (B,196,embed_dim),
            "freq": (B,196,embed_dim),
            "noise": (B,196,embed_dim),
            "image_embeds": (B,proj_dim)
        }
    """

    def __init__(
        self,
        embed_dim: int = 768,
        use_wavelet: bool = True,
        use_srm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.use_wavelet = use_wavelet
        self.use_srm = use_srm
        self.embed_dim = embed_dim

        self.clip = CLIPBackbone()

        if self.use_wavelet:
            self.freq_encoder = FrequencyEncoder(embed_dim)
            self.freq_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.freq_encoder = None
            self.freq_proj = None

        if self.use_srm:
            self.noise_encoder = NoiseEncoder(embed_dim)
            self.noise_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.noise_encoder = None
            self.noise_proj = None

        self.rgb_proj = nn.Linear(768, embed_dim)

        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.freq_norm = nn.LayerNorm(embed_dim)
        self.noise_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.modality_embed = nn.Parameter(torch.zeros(1, 3, embed_dim))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> dict:
        # CLIP patch tokens: (B,196,768)
        rgb_tokens, image_embeds = self.clip.get_visual_features(images)

        # RGB residual projection
        rgb_proj = self.rgb_proj(rgb_tokens)
        if rgb_proj.shape[-1] == rgb_tokens.shape[-1]:
            rgb_tokens = self.rgb_norm(rgb_proj + rgb_tokens)
        else:
            rgb_tokens = self.rgb_norm(rgb_proj)
        rgb_tokens = rgb_tokens + self.modality_embed[:, 0:1]
        rgb_tokens = self.dropout(rgb_tokens)

        if self.use_wavelet:
            freq_tokens = self.freq_encoder(images)
            freq_tokens = self.freq_norm(self.freq_proj(freq_tokens) + freq_tokens)
            freq_tokens = freq_tokens + self.modality_embed[:, 1:2]
            freq_tokens = self.dropout(freq_tokens)
        else:
            freq_tokens = torch.zeros_like(rgb_tokens)

        if self.use_srm:
            noise_tokens = self.noise_encoder(images)
            noise_tokens = self.noise_norm(self.noise_proj(noise_tokens) + noise_tokens)
            noise_tokens = noise_tokens + self.modality_embed[:, 2:3]
            noise_tokens = self.dropout(noise_tokens)
        else:
            noise_tokens = torch.zeros_like(rgb_tokens)

        assert rgb_tokens.shape == freq_tokens.shape == noise_tokens.shape, \
            "Mismatch in encoder outputs"

        return {
            "rgb": rgb_tokens,
            "freq": freq_tokens,
            "noise": noise_tokens,
            "image_embeds": image_embeds,
        }