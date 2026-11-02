import os
os.environ["TRANSFORMERS_NO_TORCH_LOAD"] = "1"

import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from typing import List


class CLIPBackbone(nn.Module):
    """Frozen CLIP vision/text backbone with consistent projected embedding space."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        super().__init__()

        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(
            model_name, use_safetensors=False
        )
        self.text_model = CLIPTextModelWithProjection.from_pretrained(
            model_name, use_safetensors=False
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.text_model.parameters():
            p.requires_grad = False

        self.hidden_dim = self.vision_model.config.hidden_size       # 768
        self.proj_dim = self.vision_model.config.projection_dim      # usually 512

    def get_visual_features(self, images: torch.Tensor):
        """Patch tokens (B, 196, 768) and image embedding (B, proj_dim) from one forward pass."""
        outputs = self.vision_model(pixel_values=images)
        tokens = outputs.last_hidden_state[:, 1:]  # exclude CLS
        tokens = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-6)
        image_embeds = outputs.image_embeds
        image_embeds = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-6)
        return tokens, image_embeds

    def get_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Patch tokens in hidden space: (B, 196, 768)."""
        outputs = self.vision_model(pixel_values=images)
        tokens = outputs.last_hidden_state[:, 1:]  # exclude CLS
        tokens = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-6)
        return tokens

    def get_visual_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Global projected image embedding: (B, proj_dim)."""
        outputs = self.vision_model(pixel_values=images)
        image_embeds = outputs.image_embeds
        image_embeds = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-6)
        return image_embeds

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Projected text embedding: (B, proj_dim)."""
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.text_model(**inputs)
        text_embeds = outputs.text_embeds
        text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-6)
        return text_embeds