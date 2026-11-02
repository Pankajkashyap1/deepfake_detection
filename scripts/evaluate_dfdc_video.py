import os
import sys
import json
import argparse
from collections import defaultdict

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
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.dfdc_dataset import DFDCDataset
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
    return model


def compute_eer_from_probs(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_metrics_from_probs(labels: np.ndarray, probs: np.ndarray, threshold: float):
    preds = (probs >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        "eer": float(compute_eer_from_probs(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def aggregate_video_scores(video_prob_dict, mode="mean"):
    out = {}
    for video, probs in video_prob_dict.items():
        arr = np.array(probs, dtype=np.float32)
        if mode == "mean":
            out[video] = float(arr.mean())
        elif mode == "max":
            out[video] = float(arr.max())
        elif mode == "median":
            out[video] = float(np.median(arr))
        else:
            raise ValueError(f"Unsupported aggregation mode: {mode}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Video-level DFDC evaluation")
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--labels_path", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/train_stage1.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/stage1_final_detector_no_reasoning.pth")
    parser.add_argument("--threshold", type=float, default=0.29)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_frames_per_video", type=int, default=None)
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "max", "median"])
    parser.add_argument("--out_dir", type=str, default="results/dfdc_video_eval")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.ckpt, args.config, device)

    dataset = DFDCDataset(
        frames_root=args.frames_root,
        labels_path=args.labels_path,
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

    video_prob_dict = defaultdict(list)
    video_label_dict = {}
    frame_count = 0
    has_labels = args.labels_path is not None

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            videos = batch["video"]

            outputs = model(images)
            probs = torch.softmax(outputs["logits"], dim=1)[:, 1].detach().cpu().numpy()

            if has_labels:
                labels = batch["label"].cpu().numpy()

            for i, vid in enumerate(videos):
                video_prob_dict[vid].append(float(probs[i]))
                if has_labels:
                    lab = int(labels[i])
                    if vid in video_label_dict and video_label_dict[vid] != lab:
                        raise ValueError(f"Inconsistent labels for video {vid}")
                    video_label_dict[vid] = lab

            frame_count += len(videos)

    if len(video_prob_dict) == 0:
        raise RuntimeError("No DFDC samples were loaded. Check frames_root and labels.")

    video_probs_map = aggregate_video_scores(video_prob_dict, mode=args.agg)

    records = []
    for vid in sorted(video_probs_map.keys()):
        prob = video_probs_map[vid]
        rec = {
            "video": vid,
            "fake_prob": float(prob),
            "pred": int(prob >= args.threshold),
            "num_frames_used": len(video_prob_dict[vid]),
        }
        if has_labels:
            rec["label"] = int(video_label_dict[vid])
        records.append(rec)

    out = {
        "dataset": "DFDC",
        "level": "video",
        "aggregation": args.agg,
        "threshold": args.threshold,
        "num_videos": len(video_probs_map),
        "num_frames": frame_count,
        "records": records,
    }

    if has_labels:
        video_names = sorted(video_probs_map.keys())
        video_probs = np.array([video_probs_map[v] for v in video_names], dtype=np.float32)
        video_labels = np.array([video_label_dict[v] for v in video_names], dtype=np.int64)

        metrics = compute_metrics_from_probs(video_labels, video_probs, threshold=args.threshold)
        out["metrics"] = metrics

        print("\n===== DFDC Video-Level Results =====")
        print(f"Aggregation: {args.agg}")
        print(f"Videos     : {len(video_names)}")
        print(f"Frames     : {frame_count}")
        print(f"Accuracy   : {metrics['accuracy']:.4f}")
        print(f"AUC        : {metrics['auc']:.4f}")
        print(f"EER        : {metrics['eer']:.4f}")
        print(f"F1         : {metrics['f1']:.4f}")
        print(f"Precision  : {metrics['precision']:.4f}")
        print(f"Recall     : {metrics['recall']:.4f}")
    else:
        print("\n===== DFDC Prediction-Only Results =====")
        print(f"Aggregation: {args.agg}")
        print(f"Videos     : {len(video_probs_map)}")
        print(f"Frames     : {frame_count}")
        print("No labels provided, so metrics were not computed.")

    out_path = os.path.join(args.out_dir, f"dfdc_video_results_{args.agg}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()