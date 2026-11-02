# models/__init__.py

from .deepfake_detector import DeepfakeVLMDetector

# Optional (for research/debugging)
from .clip_backbone import CLIPBackbone
from .multimodal_encoder import MultiModalForgeryEncoder
from .fusion_module import GatedFusion
from .multiscale_tokens import MultiScaleTokens
from .prompt_alignment import DualPromptAlignment
from .bridge_adapter import PromptGuidedBridge
from .reasoning_transformer import ForgeryReasoningTransformer
from .classification_heads import ClassificationHead, LocalizationHead
from .explanation_generator import ExplanationGenerator

__all__ = [
    # Main model
    "DeepfakeVLMDetector",

    # Optional components
    "CLIPBackbone",
    "MultiModalForgeryEncoder",
    "GatedFusion",
    "MultiScaleTokens",
    "DualPromptAlignment",
    "PromptGuidedBridge",
    "ForgeryReasoningTransformer",
    "ClassificationHead",
    "LocalizationHead",
    "ExplanationGenerator",
]