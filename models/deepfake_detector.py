import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .multimodal_encoder import MultiModalForgeryEncoder
from .fusion_module import GatedFusion
from .multiscale_tokens import MultiScaleTokens
from .prompt_alignment import DualPromptAlignment
from .bridge_adapter import PromptGuidedBridge
from .reasoning_transformer import ForgeryReasoningTransformer
from .classification_heads import ClassificationHead, LocalizationHead
from .explanation_generator import ExplanationGenerator


class DeepfakeVLMDetector(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        use_wavelet = config.get("use_wavelet", True)
        use_srm = config.get("use_srm", True)
        embed_dim = config.get("embed_dim", 768)
        gate_dim = config.get("gate_dim", 256)
        self.use_bridge = config.get("use_bridge", True)
        self.use_reasoning = config.get("use_reasoning", True)
        self.heatmap_mode = config.get("heatmap_mode", "fused")
        self.gradcam_weight = config.get("gradcam_weight", 0.7)

        self.encoder = MultiModalForgeryEncoder(
            embed_dim=embed_dim,
            use_wavelet=use_wavelet,
            use_srm=use_srm,
        )

        self.fusion = GatedFusion(input_dim=embed_dim, gate_dim=gate_dim)
        self.multiscale = MultiScaleTokens(embed_dim=embed_dim, num_patches=196)

        # returns: prompt_tokens (for reasoning), class_prototypes (for contrastive)
        self.prompt_align = DualPromptAlignment(
            self.encoder.clip,
            embed_dim=embed_dim,
            clip_proj_dim=self.encoder.clip.proj_dim,
            num_prompt_tokens=4,
        )

        self.bridge = PromptGuidedBridge(embed_dim=embed_dim, num_heads=12)
        self.reasoning = ForgeryReasoningTransformer(
            embed_dim=embed_dim,
            num_layers=config.get("num_reasoning_layers", 3),
            num_heads=12,
        )

        self.classifier = ClassificationHead(embed_dim=embed_dim)
        self.localizer = LocalizationHead(embed_dim=embed_dim)

        if config.get("use_llm", False):
            self.explainer = ExplanationGenerator(
                llm_name=config.get("llm_model"),
                embed_dim=embed_dim
            )
        else:
            self.explainer = None

        self.stage = 1
    
    def _normalize_map(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.amin(dim=(1, 2), keepdim=True)
        x = x / (x.amax(dim=(1, 2), keepdim=True) + 1e-6)
        return x


    def _compute_token_gradcam_heatmap(
        self,
        score: torch.Tensor,
        pre_reason_tokens: torch.Tensor,
        image_size,
    ) -> torch.Tensor:
        grads = torch.autograd.grad(
            outputs=score,
            inputs=pre_reason_tokens,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        acts = pre_reason_tokens[:, 1:197, :]   # patch tokens only
        grads = grads[:, 1:197, :]

        weights = grads.mean(dim=1, keepdim=True)   # (B,1,D)
        cam = (acts * weights).sum(dim=-1)          # (B,196)
        cam = F.relu(cam)

        B = cam.size(0)
        cam = cam.view(B, 1, 14, 14)
        cam = F.interpolate(
            cam,
            size=image_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        cam = self._normalize_map(cam)
        return cam


    def set_stage(self, stage: int):
        self.stage = stage

        if stage == 1:
            # keep CLIP frozen in detector training
            for param in self.encoder.clip.parameters():
                param.requires_grad = False

            if self.explainer is not None:
                for param in self.explainer.parameters():
                    param.requires_grad = False

        elif stage == 2:
            # only explainer trains
            for name, param in self.named_parameters():
                param.requires_grad = False

            if self.explainer is not None:
                for name, param in self.explainer.named_parameters():
                    if "lora_" in name or "visual_proj" in name:
                        param.requires_grad = True

        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        explanations: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:

        # 1. Encoder
        features = self.encoder(images)
        # features:
        #   rgb, freq, noise -> (B,196,D)
        #   image_embeds -> (B,proj_dim)

        # 2. Fusion
        fused = self.fusion(features)   # (B,196,D)

        # 3. Multi-scale tokens
        tokens = self.multiscale(fused) # (B,393,D)
        # [global(1), patch(196), pixel(196)]

        # 4. Prompt alignment
        prompt_tokens, class_prototypes = self.prompt_align(images.size(0))

        # 5. Prompt-guided bridge
        if self.use_bridge:
            bridged_tokens = self.bridge(tokens, prompt_tokens)
        else:
            bridged_tokens = tokens
        
        # for Stage 2 Grad-CAM heatmap generation with frozen detector
        if self.stage == 2 and self.explainer is not None and self.heatmap_mode in {"gradcam", "fused"}:
            bridged_tokens = bridged_tokens.detach().requires_grad_(True)
        
        # 6. Reasoning
        if self.use_reasoning:
            reasoned_tokens, global_token = self.reasoning(bridged_tokens)
        else:
            reasoned_tokens = bridged_tokens
            global_token = reasoned_tokens[:, 0]

        # 7. Classification
        logits = self.classifier(global_token)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        # 8. Localization
        patch_tokens = reasoned_tokens[:, 1:197]   # assumes no extra CLS added in reasoning block
        localizer_heatmap = self.localizer(global_token, patch_tokens, image_size=images.shape[-2:])
        localizer_heatmap = self._normalize_map(localizer_heatmap)

        gradcam_heatmap = None
        final_heatmap = localizer_heatmap

        if self.stage == 2 and self.explainer is not None and self.heatmap_mode in {"gradcam", "fused"}:
            target_score = logits.gather(1, preds.unsqueeze(1)).sum()
            gradcam_heatmap = self._compute_token_gradcam_heatmap(
                score=target_score,
                pre_reason_tokens=bridged_tokens,
                image_size=images.shape[-2:]
            )

            if self.heatmap_mode == "gradcam":
                final_heatmap = gradcam_heatmap
            else:
                final_heatmap = (
                    self.gradcam_weight * gradcam_heatmap +
                    (1.0 - self.gradcam_weight) * localizer_heatmap
                )
                final_heatmap = self._normalize_map(final_heatmap)

        outputs = {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "global_token": global_token,
            "tokens": reasoned_tokens,
            "patch_tokens": patch_tokens,
            "pre_reason_tokens": bridged_tokens,
            "fused_tokens": fused,
            "freq_tokens": features["freq"],
            "prompt_tokens": prompt_tokens,
            "class_prototypes": class_prototypes,
            "image_embeds": features["image_embeds"],
            "heatmap": final_heatmap,
            "localizer_heatmap": localizer_heatmap
        }
        if gradcam_heatmap is not None:
            outputs["gradcam_heatmap"] = gradcam_heatmap

        # 9. Stage 2 explanation
        if self.stage == 2 and self.explainer is not None:
            pred_labels = ["real" if p.item() == 0 else "fake" for p in preds]

            if explanations is not None:
                questions = explanations["questions"]
                answers = explanations["answers"]
            else:
                questions = ["Does the image look real or fake?" for _ in pred_labels]
                answers = [
                    "The image appears fake due to visible facial inconsistencies."
                    if label == "fake"
                    else "The image appears real with consistent facial appearance."
                    for label in pred_labels
                ]

            exp_loss, generated_explanations = self.explainer(
                tokens=reasoned_tokens,
                heatmap=final_heatmap,
                labels_text=pred_labels,
                questions_text=questions,
                answers_text=answers,
            )

            outputs["exp_loss"] = exp_loss
            outputs["explanations"] = generated_explanations

        return outputs