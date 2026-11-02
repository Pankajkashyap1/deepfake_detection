import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.celebdf_dataset import CelebDFDataset
from datasets.transforms import get_transforms
from models.deepfake_detector import DeepfakeVLMDetector
from utils.metrics import compute_metrics


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate final detector on Celeb-DF")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_stage1.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/stage1_final_detector_no_reasoning.pth")
    parser.add_argument("--threshold", type=float, default=0.29)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_frames_per_video", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="results/celebdf_eval")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.ckpt, args.config, device)

    dataset = CelebDFDataset(
        root=args.data_root,
        split="test",
        transform=get_transforms(train=False),
        max_frames_per_video=args.max_frames_per_video,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_logits = []
    all_labels = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]

            outputs = model(images)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_paths.extend(list(batch["path"]))
    
    if len(all_logits) == 0:
        raise RuntimeError(
            "No Celeb-DF samples were loaded. "
            "Check data_root, List_of_testing_videos.txt, and resolved frame-folder paths."
        )

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.cat(all_probs, dim=0)

    metrics = compute_metrics(logits, labels, threshold=args.threshold)

    print("\n===== Celeb-DF Results =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"AUC      : {metrics['auc']:.4f}")
    print(f"EER      : {metrics['eer']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")

    out = {
        "dataset": "Celeb-DF-v1",
        "threshold": args.threshold,
        "metrics": metrics,
        "num_samples": int(len(labels)),
        "paths": all_paths,
        "probs": probs.numpy().tolist(),
        "labels": labels.numpy().tolist(),
    }

    out_path = os.path.join(args.out_dir, "celebdf_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()