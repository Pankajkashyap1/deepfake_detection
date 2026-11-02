import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, class_weights):
        return F.cross_entropy(
            logits, targets,
            weight=class_weights,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1.0 - pt) ** self.gamma * ce).mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        mask_pos  = (labels.view(-1, 1) == labels.view(1, -1)) & ~mask_self
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        sim = sim.masked_fill(mask_self, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        # diagonal is -inf; zero it so -inf * 0 does not turn the sum into NaN
        log_prob = log_prob.masked_fill(mask_self, 0.0)
        num_pos = mask_pos.float().sum(dim=1)
        loss = -(log_prob * mask_pos.float()).sum(dim=1) / num_pos.clamp(min=1.0)
        # average only over anchors that have at least one positive
        return loss[num_pos > 0].mean()


class ProtoAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, labels, class_prototypes=None):
        # NOTE: class_prototypes from DualPromptAlignment is (2, 512) — CLIP proj space.
        # global_token is (B, 768) — transformer hidden space.
        # These are incompatible dims, so we always use batch-mean in 768-d space.
        device = features.device
        features = F.normalize(features, dim=1)
        loss = torch.tensor(0.0, device=device)
        n = 0
        for cls_idx in labels.unique():
            mask = labels == cls_idx
            cls_feats = features[mask]
            proto = F.normalize(cls_feats.mean(0, keepdim=True), dim=1)
            cos_sim = F.cosine_similarity(
                cls_feats, proto.expand_as(cls_feats), dim=1
            )
            loss = loss + (1.0 - cos_sim).mean()
            n += 1
        return loss / max(n, 1)


class DeepfakeLoss(nn.Module):
    """
    Combined loss for DeepfakeVLMDetector.

    STAGE 1  (Stage1Trainer)
    ─────────────────────────
    Constructor call in stage1_trainer.py:
        loss_cfg = {
            "lambda_cls": 1.0, "lambda_contrast": 0.1, "lambda_proto": 0.01,
            "temperature": 0.07, "label_smoothing": 0.0,
        }
        class_weights = torch.tensor([w_real, w_fake])   # (2,) from dataset balance
        self.loss_fn = DeepfakeLoss(loss_cfg, class_weights=class_weights)
        self.loss_fn = self.loss_fn.to(self.accelerator.device)  # buffer must move

    Forward call in stage1_trainer.py:
        losses = self.loss_fn(outputs, labels, stage=1)
        total_batch_loss = losses["total"]               # only key trainer reads

    outputs from deepfake_detector.py:
        "logits"        (B, 2)    ClassificationHead: Linear(768->768->256->2)
        "global_token"  (B, 768)  ForgeryReasoningTransformer x[:,0]

    STAGE 2  (Stage2Trainer)
    ─────────────────────────
    Stage2Trainer NEVER calls this loss.
    It reads outputs["exp_loss"] from ExplanationGenerator directly.
    forward(stage=2) is kept as a safe no-op for eval scripts only.
    """

    def __init__(self, cfg: dict, class_weights: torch.Tensor = None):
        super().__init__()
        self.lambda_cls      = float(cfg.get("lambda_cls",      1.0))
        self.lambda_contrast = float(cfg.get("lambda_contrast", 0.1))
        self.lambda_proto    = float(cfg.get("lambda_proto",    0.01))
        label_smoothing      = float(cfg.get("label_smoothing", 0.0))
        temperature          = float(cfg.get("temperature",     0.07))

        # register_buffer so .to(device) in stage1_trainer moves it correctly
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", torch.ones(2, dtype=torch.float32))

        self.ce     = WeightedCrossEntropyLoss(label_smoothing=label_smoothing)
        self.focal  = FocalLoss(alpha=0.25, gamma=2.0)
        self.supcon = SupConLoss(temperature=temperature)
        self.proto  = ProtoAlignmentLoss()

    def forward(
        self,
        outputs: dict,
        labels: torch.Tensor,
        stage: int = 1,
    ) -> dict:
        """
        outputs : dict from DeepfakeVLMDetector.forward()
            "logits"       (B, 2)    always present
            "global_token" (B, 768)  always present, used in stage=1
        labels  : (B,) long   0=real  1=fake
        stage   : 1 or 2
        returns : {"total", "cls", "contrast", "proto"}
        """
        logits = outputs["logits"]          # (B, 2)
        labels = labels.long()
        device = logits.device

        # classification loss — always active in stage 1
        loss_cls = (
            self.ce(logits, labels, self.class_weights)
            + self.focal(logits, labels)
        )

        loss_contrast = torch.tensor(0.0, device=device)
        loss_proto    = torch.tensor(0.0, device=device)

        # feature losses — stage 1 only (detector is frozen in stage 2)
        if stage == 1 and "global_token" in outputs:
            features = outputs["global_token"]          # (B, 768)
            if self.lambda_contrast > 0:
                loss_contrast = self.supcon(features, labels)
            if self.lambda_proto > 0:
                loss_proto = self.proto(features, labels)

        total = (
            self.lambda_cls      * loss_cls
            + self.lambda_contrast * loss_contrast
            + self.lambda_proto    * loss_proto
        )

        return {
            "total":    total,
            "cls":      loss_cls,
            "contrast": loss_contrast,
            "proto":    loss_proto,
        }
