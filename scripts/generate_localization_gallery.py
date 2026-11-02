import os
import sys
import json
import argparse
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.ffpp_dataset import FFPPDataset
from datasets.transforms import get_transforms
from models.deepfake_detector import DeepfakeVLMDetector
from utils.gradcam_tokens import token_gradcam


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
    print("Missing keys   :", missing)
    print("Unexpected keys:", unexpected)

    model.set_stage(1)
    model.to(device)
    model.eval()
    return model, cfg


def build_dataset(cfg, split: str):
    return FFPPDataset(
        root=cfg.data.root,
        split=split,
        transform=get_transforms(train=False),
        train_ratio=cfg.data.get("train_ratio", 0.7),
        val_ratio=cfg.data.get("val_ratio", 0.15),
        test_ratio=cfg.data.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
        max_frames_per_video=cfg.data.get("max_frames_per_video", None),
        balance=False,
    )


def normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = hm.astype(np.float32)
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-6)
    return hm


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import cv2

    heatmap = normalize_heatmap(heatmap)
    h, w = image_rgb.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        else:
            image_rgb = image_rgb.astype(np.uint8)

    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def save_panel(
    image_path: str,
    localizer_heatmap: np.ndarray,
    gradcam_heatmap: np.ndarray,
    save_path: str,
    title_text: str,
):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    localizer_heatmap = normalize_heatmap(localizer_heatmap)
    gradcam_heatmap = normalize_heatmap(gradcam_heatmap)

    localizer_overlay = overlay_heatmap(image_np, localizer_heatmap)
    gradcam_overlay = overlay_heatmap(image_np, gradcam_heatmap)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(localizer_heatmap, cmap="jet")
    plt.title("LocalizationHead Heatmap")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(localizer_overlay)
    plt.title("Localization Overlay")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(gradcam_heatmap, cmap="jet")
    plt.title("Token Grad-CAM")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(gradcam_overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")

    plt.suptitle(title_text, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def get_category(true_label: int, pred_label: int) -> str:
    if true_label == 1 and pred_label == 1:
        return "TP_fake"
    if true_label == 0 and pred_label == 0:
        return "TN_real"
    if true_label == 0 and pred_label == 1:
        return "FP_real_as_fake"
    return "FN_fake_as_real"


def main():
    parser = argparse.ArgumentParser(description="Generate localization gallery from Stage 1 detector")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_per_category", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="results/localization_gallery")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in ["TP_fake", "TN_real", "FP_real_as_fake", "FN_fake_as_real"]:
        os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, cfg = load_model(args.ckpt, args.config, device)
    dataset = build_dataset(cfg, args.split)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    counts = {
        "TP_fake": 0,
        "TN_real": 0,
        "FP_real_as_fake": 0,
        "FN_fake_as_real": 0,
    }
    metadata: List[Dict] = []

    for batch_idx, batch in enumerate(loader):
        if all(v >= args.max_per_category for v in counts.values()):
            break

        images = batch["image"].to(device)
        labels = batch["label"]
        paths = batch["path"]
        videos = batch["video"]

        for i in range(images.size(0)):
            image_tensor = images[i:i+1]
            true_label = int(labels[i].item())
            image_path = paths[i]
            video_name = videos[i]

            # detector forward
            model.set_stage(1)
            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)

                fake_prob = float(probs[0, 1].item())
                pred_label = 1 if fake_prob >= args.threshold else 0
                pred_text = "FAKE" if pred_label == 1 else "REAL"
                true_text = "FAKE" if true_label == 1 else "REAL"

                localizer_heatmap = outputs["heatmap"][0].detach().cpu().numpy()

            # grad-cam
            target_class = pred_label
            gradcam, _ = token_gradcam(model, image_tensor, target_class=target_class)
            gradcam_heatmap = gradcam[0].detach().cpu().numpy()

            category = get_category(true_label, pred_label)
            if counts[category] >= args.max_per_category:
                continue

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_name = f"{counts[category]:03d}_{video_name}_{base_name}.png"
            save_path = os.path.join(args.out_dir, category, save_name)

            title_text = (
                f"Category: {category} | True: {true_text} | Pred: {pred_text} | "
                f"fake_prob={fake_prob:.4f} | threshold={args.threshold:.2f}"
            )

            save_panel(
                image_path=image_path,
                localizer_heatmap=localizer_heatmap,
                gradcam_heatmap=gradcam_heatmap,
                save_path=save_path,
                title_text=title_text,
            )

            metadata.append({
                "category": category,
                "image_path": image_path,
                "video": video_name,
                "true_label": true_text,
                "pred_label": pred_text,
                "fake_prob": fake_prob,
                "threshold": args.threshold,
                "saved_file": save_path,
            })

            counts[category] += 1

        if batch_idx % 50 == 0:
            print(f"Processed batch {batch_idx} | counts = {counts}")

    meta_path = os.path.join(args.out_dir, f"{args.split}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "split": args.split,
            "threshold": args.threshold,
            "counts": counts,
            "items": metadata,
        }, f, indent=2)

    print("\nDone.")
    print("Saved counts:", counts)
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()