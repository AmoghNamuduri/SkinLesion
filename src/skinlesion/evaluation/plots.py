"""Figures that mirror the paper: Fig. 5 (CNN history), Fig. 6 (model comparison),
Table 4 as a bar chart, and confusion matrices."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..utils.io import resolve  # noqa: E402

PAPER_COLOR = "#9aa5b1"
OURS_COLOR = "#2a6fdb"


def _save(fig, path) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_history(history: dict, path) -> Path:
    """Fig. 5 — training/validation accuracy and loss per epoch."""
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history["train_acc"], label="train")
    ax[0].plot(ep, history["val_acc"], label="test")
    ax[0].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax[1].plot(ep, history["train_loss"], label="train")
    ax[1].plot(ep, history["val_loss"], label="test")
    ax[1].set(title="Loss", xlabel="epoch", ylabel="categorical cross-entropy")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
    return _save(fig, path)


def plot_confusion(cm, labels, path, title="Confusion matrix") -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    return _save(fig, path)


def plot_paper_vs_ours(names, paper_vals, our_vals, path, title, ylabel="accuracy") -> Path:
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - .2, paper_vals, .4, label="paper", color=PAPER_COLOR)
    ax.bar(x + .2, our_vals, .4, label="ours", color=OURS_COLOR)
    ax.set_xticks(x, names)
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="y", alpha=.3)
    ax.legend()
    return _save(fig, path)
