import os
import sys
import argparse
import torch
from PIL import Image
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.deepfake_detector import DeepfakeVLMDetector
from datasets.transforms import get_transforms
from utils.visualization import visualize_result


def build_model_config(cfg):
    return OmegaConf.create({
        **OmegaConf.to_container(cfg.model, resolve=True),
        **OmegaConf.to_container(cfg.fusion, resolve=True),
    })


def load_model(ckpt_path: str, config_path: str, device: str):
    cfg = OmegaConf.load(config_path)
    model_cfg = build_model_config(cfg)

    model = DeepfakeVLMDetector(model_cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys count   :", len(missing))
    print("Unexpected keys count:", len(unexpected))

    model.set_stage(1)
    model.to(device)
    model.eval()
    return model


def predict(model, image_tensor, threshold: float):
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)

        fake_prob = probs[0, 1].item()
        real_prob = probs[0, 0].item()

        pred_idx = 1 if fake_prob >= threshold else 0
        pred_label = "FAKE" if pred_idx == 1 else "REAL"
        confidence = fake_prob if pred_idx == 1 else real_prob

        heatmap = outputs["localizer_heatmap"][0].detach().cpu().numpy()

    return pred_label, confidence, fake_prob, heatmap


def main():
    parser = argparse.ArgumentParser(description="Final detector inference with localization head")

    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--config", type=str, default="configs/train_stage1.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/stage1_final_detector_no_reasoning.pth")
    parser.add_argument("--threshold", type=float, default=0.29)
    parser.add_argument("--output", type=str, default="figures/localization/result.png")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.ckpt, args.config, device)

    image = Image.open(args.image_path).convert("RGB")
    transform = get_transforms(train=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    pred_label, confidence, fake_prob, heatmap = predict(model, image_tensor, args.threshold)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    visualize_result(
        image=image,
        heatmap=heatmap,
        pred=f"{pred_label} | conf={confidence:.2%} | fake_prob={fake_prob:.4f}",
        save_path=args.output,
    )

    print("\n===== FINAL RESULT =====")
    print(f"Image      : {args.image_path}")
    print(f"Prediction : {pred_label}")
    print(f"Confidence : {confidence:.2%}")
    print(f"Fake prob  : {fake_prob:.4f}")
    print(f"Threshold  : {args.threshold:.2f}")
    print(f"Saved to   : {args.output}")


if __name__ == "__main__":
    main()