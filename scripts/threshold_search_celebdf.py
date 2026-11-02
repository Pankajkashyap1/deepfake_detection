import os
import sys
import json
import argparse
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.celebdf_dataset import CelebDFDataset
from datasets.transforms import get_transforms
from models.deepfake_detector import DeepfakeVLMDetector


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
    return model, cfg


def build_dataset(cfg, split: str):
    return CelebDFDataset(
        root=cfg.data.root,
        split=split,
        transform=get_transforms(train=False),
        train_ratio=cfg.data.get("train_ratio", 0.85),
        seed=cfg.get("seed", 42),
        max_frames_per_video=cfg.data.get("max_frames_per_video", None),
    )


def compute_eer(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def evaluate_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    preds = (probs >= threshold).astype(np.int64)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def collect_probs(model, loader, device: str):
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_paths: List[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]

            outputs = model(images)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_paths.extend(list(batch["path"]))

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return probs, labels, all_paths


def save_threshold_table(results: List[Dict], out_csv: str):
    import csv
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Threshold search for Celeb-DF-trained detector")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/threshold_search_celebdf")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--start", type=float, default=0.01)
    parser.add_argument("--end", type=float, default=0.99)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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

    probs, labels, paths = collect_probs(model, loader, device)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    eer = compute_eer(labels, probs)

    print(f"\nSplit: {args.split}")
    print(f"Samples: {len(labels)}")
    print(f"AUC (threshold-free): {auc:.4f}")
    print(f"EER (threshold-free): {eer:.4f}")

    raw_path = os.path.join(args.out_dir, f"{args.split}_probs.json")
    with open(raw_path, "w") as f:
        json.dump(
            {
                "split": args.split,
                "auc": float(auc),
                "eer": float(eer),
                "labels": labels.tolist(),
                "probs": probs.tolist(),
                "paths": paths,
            },
            f,
            indent=2,
        )

    if args.threshold is not None:
        result = evaluate_at_threshold(labels, probs, args.threshold)
        result["auc"] = float(auc)
        result["eer"] = float(eer)

        out_json = os.path.join(args.out_dir, f"{args.split}_threshold_{args.threshold:.2f}.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

        print("\nSingle-threshold evaluation")
        print(json.dumps(result, indent=2))
        print(f"Saved summary to: {out_json}")
        return

    thresholds = np.arange(args.start, args.end + 1e-9, args.step)
    results = [evaluate_at_threshold(labels, probs, float(t)) for t in thresholds]

    best_acc = max(results, key=lambda x: x["accuracy"])
    best_f1 = max(results, key=lambda x: x["f1"])

    print("\nBest threshold by accuracy")
    print(json.dumps(best_acc, indent=2))

    print("\nBest threshold by F1")
    print(json.dumps(best_f1, indent=2))

    out_csv = os.path.join(args.out_dir, f"{args.split}_threshold_table.csv")
    save_threshold_table(results, out_csv)

    summary = {
        "split": args.split,
        "auc": float(auc),
        "eer": float(eer),
        "best_by_accuracy": best_acc,
        "best_by_f1": best_f1,
    }
    out_json = os.path.join(args.out_dir, f"{args.split}_threshold_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved threshold table to: {out_csv}")
    print(f"Saved summary to: {out_json}")


if __name__ == "__main__":
    main()