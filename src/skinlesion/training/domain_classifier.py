"""Train the ImageNet DenseNet-121 classifier (roadmap issue 09).

``fit_classifier`` is shared: the domain classifier (real train split) and every
synthetic-learning model (issue 12) are trained by the same function, so the only
thing that differs between experiment arms is the training data.

Recipe (after SpineGAN's SpineClassifier): class-weighted loss, Adam, light augmentation
(flips, small rotation, random resized crop), best epoch chosen by validation AUC.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data.dataset import load_manifest
from ..data.splits import class_names, encode_labels, ensure_splits
from ..evaluation.metrics import bootstrap_ci, probabilistic_metrics
from ..evaluation.plots import plot_roc
from ..gan.dataset_export import load_rgb
from ..models.densenet import build_densenet, normalize_imagenet, probabilities, save_classifier
from ..utils.io import ensure_dir, resolve, save_json


class PreloadedImages(Dataset):
    """Images centre-cropped and resized once (same crop as the GAN export), kept in RAM as uint8."""

    def __init__(self, paths, labels, size: int, train: bool):
        self.images = np.stack([load_rgb(p, size) for p in paths])
        self.labels = np.asarray(labels, dtype=np.int64)
        self.train = train
        if train:
            from torchvision import transforms as T
            self.aug = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15),
                                  T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(1.0, 1.0), antialias=True)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1).float().div(255)
        if self.train:
            x = self.aug(x)
        return normalize_imagenet(x.unsqueeze(0))[0], self.labels[i]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def cap_class(df: pd.DataFrame, dx: str, cap: int | None, seed: int) -> pd.DataFrame:
    """Keep at most ``cap`` rows of one diagnosis (nv is ~67 % of HAM10000)."""
    if not cap or (df["dx"] == dx).sum() <= cap:
        return df
    keep = df[df["dx"] == dx].sample(n=cap, random_state=seed)
    return pd.concat([df[df["dx"] != dx], keep]).sort_index()


def loss_fn(num_classes: int, train_labels: np.ndarray, weighted: bool, device):
    counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    if num_classes == 2:
        pos_weight = torch.tensor([counts[0] / max(counts[1], 1)], device=device) if weighted else None
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return lambda logits, y: bce(logits[:, 0], y.float())
    w = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=w if weighted else None)


@torch.no_grad()
def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        ps.append(probabilities(logits.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def fit_classifier(cfg, train_df, val_df, test_df, seed: int = 0, epochs: int | None = None,
                   out_dir=None, tag: str = "model", n_boot: int = 1000, save_model: bool = True,
                   pretrained: bool | None = None) -> dict:
    p = cfg.domain_classifier
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device(p.device)
    names = class_names(cfg)
    k = len(names)
    sets = {}
    for name, df, train in (("train", train_df, True), ("val", val_df, False), ("test", test_df, False)):
        if df is not None and len(df):
            sets[name] = PreloadedImages(list(df["path"]), encode_labels(df["dx"], cfg), p.image_size, train)
    g = torch.Generator().manual_seed(seed)
    loaders = {n: DataLoader(s, batch_size=p.batch_size, shuffle=(n == "train"), generator=g if n == "train" else None,
                             num_workers=p.num_workers, pin_memory=device.type == "cuda") for n, s in sets.items()}

    model = build_densenet(k, pretrained=p.pretrained if pretrained is None else pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p.lr)
    crit = loss_fn(k, sets["train"].labels, p.class_weighted, device)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best = {"auc": -1, "epoch": -1, "state": None}
    history = []
    t0 = time.time()
    for epoch in range(1, (epochs or p.epochs) + 1):
        model.train()
        running = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(y)
        row = {"epoch": epoch, "train_loss": running / len(sets["train"])}
        if "val" in loaders:
            yv, pv = predict_proba(model, loaders["val"], device)
            row["val_auc"] = probabilistic_metrics(yv, pv, names)["auc"]
            if row["val_auc"] > best["auc"]:
                best = {"auc": row["val_auc"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        history.append(row)
        print(f"[{tag}] epoch {epoch:2d} loss {row['train_loss']:.4f}"
              + (f" val_auc {row['val_auc']:.4f}" if "val_auc" in row else "") + f"  [{time.time() - t0:.0f}s]", flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    result = {"tag": tag, "seed": seed, "classes": names, "n_train": len(sets["train"]),
              "best_epoch": best["epoch"], "best_val_auc": best["auc"], "history": history,
              "train_seconds": time.time() - t0}
    if "test" in loaders:
        yt, pt = predict_proba(model, loaders["test"], device)
        result["test"] = probabilistic_metrics(yt, pt, names)
        result["test"]["auc_ci"] = bootstrap_ci(yt, pt, n=n_boot, seed=seed)
        if out_dir:
            plot_roc(yt, pt, names, Path(out_dir) / f"roc_{tag}.png", title=f"ROC — {tag}")
    if out_dir and save_model:
        meta = {"num_classes": k, "classes": names, "image_size": p.image_size, "label_set": cfg.labels.set}
        result["checkpoint"] = str(save_classifier(model, Path(out_dir) / f"{tag}.pt", meta))
    result["model"] = model
    return result


def train_domain_classifier(cfg) -> dict:
    """Real train split → domain_classifier.pt, chosen on val AUC, scored on the real test split."""
    splits = ensure_splits(cfg)
    train = cap_class(splits["train"], "nv", cfg.domain_classifier.max_nv, cfg.domain_classifier.seed)
    ckpt = resolve(cfg.paths.domain_classifier)
    out_dir = ensure_dir(ckpt.parent)
    res = fit_classifier(cfg, train, splits["val"], splits["test"], seed=cfg.domain_classifier.seed,
                         out_dir=out_dir, tag=ckpt.stem)
    res.pop("model")
    save_json(res, out_dir / "results.json")
    t = res["test"]
    print(f"\ntest AUC {t['auc']:.4f} (95% CI {t['auc_ci']['low']:.3f}–{t['auc_ci']['high']:.3f}), "
          f"balanced acc {t['balanced_accuracy']:.4f}  -> {res['checkpoint']}")
    return res
