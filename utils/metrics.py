import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)
import numpy as np


def get_fake_probs(logits):
    return F.softmax(logits, dim=1)[:, 1]


def compute_accuracy(logits, labels, threshold=0.5):
    probs = get_fake_probs(logits)
    preds = (probs >= threshold).long()
    correct = (preds == labels).float().mean()
    return correct.item()


def compute_auc(logits, labels):
    probs = get_fake_probs(logits).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    if len(np.unique(labels)) < 2:
        return 0.0

    try:
        return roc_auc_score(labels, probs)
    except Exception:
        return 0.0


def compute_eer(logits, labels):
    probs = get_fake_probs(logits).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    if len(np.unique(labels)) < 2:
        return 0.0

    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[idx] + fnr[idx]) / 2.0
        return float(eer)
    except Exception:
        return 0.0


def compute_f1(logits, labels, threshold=0.5):
    probs = get_fake_probs(logits).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = (probs >= threshold).astype(int)

    try:
        return f1_score(labels, preds, zero_division=0)
    except Exception:
        return 0.0


def compute_precision(logits, labels, threshold=0.5):
    probs = get_fake_probs(logits).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = (probs >= threshold).astype(int)

    try:
        return precision_score(labels, preds, zero_division=0)
    except Exception:
        return 0.0


def compute_recall(logits, labels, threshold=0.5):
    probs = get_fake_probs(logits).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = (probs >= threshold).astype(int)

    try:
        return recall_score(labels, preds, zero_division=0)
    except Exception:
        return 0.0


def compute_metrics(logits, labels, threshold=0.5):
    return {
        "accuracy": compute_accuracy(logits, labels, threshold),
        "auc": compute_auc(logits, labels),
        "eer": compute_eer(logits, labels),
        "f1": compute_f1(logits, labels, threshold),
        "precision": compute_precision(logits, labels, threshold),
        "recall": compute_recall(logits, labels, threshold),
    }