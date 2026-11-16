"""Accuracy, precision, recall, F1 (weighted + macro + per class) and confusion matrix."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, labels: list[str], target_names: list[str] | None = None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("weighted", "macro"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out[avg] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out["per_class"] = {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                        for i, lab in enumerate(labels)}
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    out["labels"] = list(labels)
    out["report"] = classification_report(y_true, y_pred, labels=labels, target_names=target_names,
                                          digits=2, zero_division=0)
    return out


# ---------------------------------------------------------------------------
# Probabilistic metrics for SkinGAN (roadmap issue 05)
# ---------------------------------------------------------------------------
def auc_score(y_true, proba) -> float:
    """Binary AUC (proba: (N,2) or (N,)) or macro one-vs-rest AUC for K classes (proba: (N,K))."""
    from sklearn.metrics import roc_auc_score

    y_true, proba = np.asarray(y_true), np.asarray(proba)
    if proba.ndim == 1 or proba.shape[1] == 2:
        return float(roc_auc_score(y_true, proba if proba.ndim == 1 else proba[:, 1]))
    present = np.unique(y_true)
    return float(roc_auc_score(y_true, proba[:, present] / proba[:, present].sum(1, keepdims=True),
                               multi_class="ovr", average="macro", labels=present))


def probabilistic_metrics(y_true, proba, class_names: list[str]) -> dict:
    """AUC (binary or macro-OvR + per class), balanced accuracy, and the hard-label report."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = proba.argmax(1)
    out = {"auc": auc_score(y_true, proba), "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
           "accuracy": float((pred == y_true).mean()), "n": int(len(y_true))}
    if proba.shape[1] > 2:
        out["auc_per_class"] = {c: float(roc_auc_score(y_true == k, proba[:, k]))
                                for k, c in enumerate(class_names) if 0 < (y_true == k).sum() < len(y_true)}
    names = [class_names[i] for i in y_true], [class_names[i] for i in pred]
    out["hard"] = classification_metrics(*names, labels=list(class_names))
    return out


def bootstrap_ci(y_true, proba, metric=auc_score, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap 95 % CI of ``metric(y_true, proba)`` over test examples."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            vals.append(metric(y_true[idx], proba[idx]))
        except ValueError:
            continue
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(metric(y_true, proba)), "low": float(lo), "high": float(hi), "n_boot": len(vals)}
