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
