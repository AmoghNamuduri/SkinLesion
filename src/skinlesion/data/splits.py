"""Train/test splitting.

``augment_before_split=True`` reproduces the paper: the 1,400 augmented images
are split at random, so an image and its mirrored twin can land on opposite
sides. ``False`` splits by original ``image_id`` so twins always stay together
(a leakage-free variant for later experiments).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.io import ensure_dir, resolve


def split_indices(labels: np.ndarray, groups: np.ndarray, test_size: float, seed: int,
                  augment_before_split: bool = True, stratify: bool = False):
    n = len(labels)
    if augment_before_split:
        idx = np.arange(n)
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed,
                                  stratify=labels if stratify else None)
        return np.sort(tr), np.sort(te)

    uniq, first = np.unique(groups, return_index=True)
    g_tr, g_te = train_test_split(uniq, test_size=test_size, random_state=seed,
                                  stratify=labels[first] if stratify else None)
    in_test = np.isin(groups, g_te)
    return np.where(~in_test)[0], np.where(in_test)[0]


# ---------------------------------------------------------------------------
# SkinGAN: lesion-level split of the full dataset + label sets (roadmap issue 03)
# ---------------------------------------------------------------------------
SPLIT_NAMES = ("train", "val", "test")


def lesion_level_split(meta: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.2,
                       seed: int = 340) -> pd.DataFrame:
    """Add a ``split`` column (train/val/test) so that every ``lesion_id`` sits in exactly one split.

    HAM10000 has several images of the same lesion; splitting by image would leak
    near-duplicates into the test set. Lesions are stratified by diagnosis, and rows are
    sorted first so the split is identical whether the metadata came from Kaggle or Dataverse.
    """
    df = meta.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)
    lesions = df.groupby("lesion_id", sort=True)["dx"].first()
    ids, dx = lesions.index.to_numpy(), lesions.to_numpy()
    rest_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=seed, stratify=dx)
    rest_dx = lesions.loc[rest_ids].to_numpy()
    train_ids, val_ids = train_test_split(rest_ids, test_size=val_size / (1 - test_size),
                                          random_state=seed, stratify=rest_dx)
    split = pd.Series("train", index=ids)
    split[val_ids], split[test_ids] = "val", "test"
    df["split"] = df["lesion_id"].map(split)
    return df


def save_splits(df: pd.DataFrame, splits_dir) -> Path:
    """Write one CSV per split. Paths are machine-specific, so only ids and labels are stored;
    ``ensure_splits`` attaches local image paths from the metadata when loading."""
    out = ensure_dir(splits_dir)
    cols = [c for c in ("image_id", "lesion_id", "dx") if c in df.columns]
    for name in SPLIT_NAMES:
        df.loc[df["split"] == name, cols].to_csv(out / f"ham10000_{name}.csv", index=False)
    return out


def load_split(splits_dir, name: str) -> pd.DataFrame:
    p = resolve(splits_dir) / f"ham10000_{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run scripts/export_gan_dataset.py (it creates the splits)")
    return pd.read_csv(p)


def ensure_splits(cfg, meta: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Load the split CSVs, creating them from the metadata if they don't exist yet."""
    from .metadata import load_metadata

    d = resolve(cfg.paths.splits_dir)
    if meta is None:
        try:
            meta = load_metadata(cfg.paths.raw_dir)
        except FileNotFoundError:
            meta = None  # no local dataset: splits can still be read, but without image paths
    if not all((d / f"ham10000_{n}.csv").exists() for n in SPLIT_NAMES):
        if meta is None:
            raise FileNotFoundError(f"no split CSVs in {d} and no HAM10000 metadata in {cfg.paths.raw_dir}")
        s = cfg.split
        save_splits(lesion_level_split(meta, s.val_size, s.test_size, s.seed), d)
    splits = {n: load_split(d, n) for n in SPLIT_NAMES}
    if meta is not None:  # attach image paths for this machine
        paths = dict(zip(meta["image_id"], meta["path"]))
        for df in splits.values():
            df["path"] = df["image_id"].map(paths)
    return splits


def class_names(cfg) -> list[str]:
    """Class order used by the GAN labels and every classifier."""
    return ["benign", "malignant"] if cfg.labels.set == "binary" else list(cfg.data.classes)


def encode_labels(dx, cfg) -> np.ndarray:
    """Diagnosis codes → integer labels for the configured label set (``7class`` or ``binary``)."""
    dx = np.asarray(dx, dtype=str)
    names = class_names(cfg)
    malignant = set(cfg.labels.malignant)
    binary = cfg.labels.set == "binary"
    # synthetic manifests already use class names ("malignant", "mel", …); real rows use dx codes
    return np.array([names.index(d) if d in names else (int(d in malignant) if binary else -1) for d in dx],
                    dtype=np.int64)
