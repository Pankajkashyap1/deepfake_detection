"""Evaluate every trained stage-1 detector on every test set.

    python scripts/evaluate_all.py

For each run in RUNS it loads runs/<run>/checkpoints/stage1_best.pth with the run's own
config, then reports frame-level and video-level (mean of frame probs) AUC / EER on the
FF++, Celeb-DF and DFDC test sets, and frame-level on DeepFakeFace (single images).
It also picks the threshold with the best accuracy on the run's own validation set
(saved in the checkpoint) and reports in-domain test accuracy at that threshold.
Results go to runs/<run>/eval_all.json and results/evaluate_all.json.
"""
import os
import sys
import json
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve

from datasets.ffpp_dataset import FFPPDataset
from datasets.celebdf_dataset import CelebDFDataset
from datasets.dfdc_dataset import DFDCDataset
from datasets.deepfakeface_dataset import DeepFakeFaceDataset
from datasets.transforms import get_transforms
from models.deepfake_detector import DeepfakeVLMDetector

RUNS = ["ff_stage1_full", "celebdf_stage1", "multisource_ffpp_celebdf", "multisource_ffpp_celebdf_dff"]
DATA = "/data/datasets/deepfake_datasets"
FRAMES_PER_VIDEO = 32


def build_test_sets():
    tf = get_transforms(train=False)
    return {
        "FF++": FFPPDataset(root=f"{DATA}/FF_fixed", split="test", transform=tf,
                            max_frames_per_video=FRAMES_PER_VIDEO),
        "Celeb-DF": CelebDFDataset(root=f"{DATA}/Celeb-DF-v2", split="test", transform=tf,
                                   max_frames_per_video=FRAMES_PER_VIDEO),
        "DFDC": DFDCDataset(frames_root=f"{DATA}/DFDC/frames", labels_path=f"{DATA}/DFDC/labels.json",
                            transform=tf, max_frames_per_video=FRAMES_PER_VIDEO),
        "DeepFakeFace": DeepFakeFaceDataset(root=f"{DATA}/DeepFakeFace", split="test", transform=tf),
    }


def load_model(run, device):
    cfg = OmegaConf.load(f"{PROJECT_ROOT}/runs/{run}/.hydra/config.yaml")
    model_cfg = OmegaConf.create({
        **OmegaConf.to_container(cfg.model, resolve=True),
        **OmegaConf.to_container(cfg.fusion, resolve=True),
    })
    model = DeepfakeVLMDetector(model_cfg)
    ckpt = torch.load(f"{PROJECT_ROOT}/runs/{run}/checkpoints/stage1_best.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.set_stage(1)
    return model.to(device).eval(), ckpt


def auc_eer(labels, probs):
    labels, probs = np.asarray(labels), np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return None, None
    fpr, tpr, _ = roc_curve(labels, probs)
    idx = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float(roc_auc_score(labels, probs)), float((fpr[idx] + 1 - tpr[idx]) / 2)


def best_threshold(labels, probs):
    labels, probs = np.asarray(labels), np.asarray(probs)
    thresholds = np.arange(0.01, 1.0, 0.01)
    accs = [((probs >= t).astype(int) == labels).mean() for t in thresholds]
    return float(thresholds[int(np.argmax(accs))])


@torch.no_grad()
def predict(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    probs, labels, videos = [], [], []
    for batch in loader:
        p = torch.softmax(model(batch["image"].to(device))["logits"], dim=1)[:, 1]
        probs.extend(p.cpu().tolist())
        labels.extend(batch["label"].tolist())
        videos.extend(batch["video"])
    return np.array(probs), np.array(labels), videos


def video_level(probs, labels, videos):
    vp, vl = defaultdict(list), {}
    for p, l, v in zip(probs, labels, videos):
        vp[v].append(p)
        vl[v] = l
    names = sorted(vp)
    return [vl[v] for v in names], [float(np.mean(vp[v])) for v in names]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_sets = build_test_sets()
    all_results = {}

    for run in RUNS:
        if not os.path.exists(f"{PROJECT_ROOT}/runs/{run}/checkpoints/stage1_best.pth"):
            print(f"skip {run}: no checkpoint")
            continue
        print(f"\n===== {run} =====")
        model, ckpt = load_model(run, device)
        thr = best_threshold(ckpt["val_labels"], ckpt["val_probs"])
        res = {"best_epoch": ckpt["epoch"], "val_auc": ckpt["auc"], "threshold": thr}

        test_file = f"{PROJECT_ROOT}/runs/{run}/checkpoints/stage1_test_results.pth"
        if os.path.exists(test_file):
            t = torch.load(test_file, map_location="cpu")["test_metrics"]
            tl, tp = np.array(t["labels_list"]), np.array(t["probs_list"])
            res["in_domain_test_acc_at_0.5"] = float(((tp >= 0.5).astype(int) == tl).mean())
            res["in_domain_test_acc_at_val_threshold"] = float(((tp >= thr).astype(int) == tl).mean())

        for name, ds in test_sets.items():
            probs, labels, videos = predict(model, ds, device)
            f_auc, f_eer = auc_eer(labels, probs)
            entry = {"frames": len(probs), "frame_auc": f_auc, "frame_eer": f_eer}
            if name != "DeepFakeFace":
                vl, vp = video_level(probs, labels, videos)
                v_auc, v_eer = auc_eer(vl, vp)
                entry.update({"videos": len(vl), "video_auc": v_auc, "video_eer": v_eer})
            res[name] = entry
            print(f"  {name:13s} frame AUC {f_auc:.4f} EER {f_eer:.4f}"
                  + (f" | video AUC {entry['video_auc']:.4f} EER {entry['video_eer']:.4f}" if "video_auc" in entry else ""))

        with open(f"{PROJECT_ROOT}/runs/{run}/eval_all.json", "w") as f:
            json.dump(res, f, indent=2)
        all_results[run] = res
        del model
        torch.cuda.empty_cache()

    os.makedirs(f"{PROJECT_ROOT}/results", exist_ok=True)
    with open(f"{PROJECT_ROOT}/results/evaluate_all.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n===== Summary: video-level AUC (frame-level for DeepFakeFace) =====")
    names = list(test_sets)
    print(f"{'run':32s}" + "".join(f"{n:>14s}" for n in names))
    for run, res in all_results.items():
        row = [res[n].get("video_auc", res[n]["frame_auc"]) for n in names]
        print(f"{run:32s}" + "".join(f"{v:14.4f}" for v in row))


if __name__ == "__main__":
    main()
