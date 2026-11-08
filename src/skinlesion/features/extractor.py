"""Concatenate the global descriptors per image and persist them in HDF5 (as the paper does)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..utils.io import resolve
from .descriptors import DESCRIPTORS


def extract_global_features(images: np.ndarray, bins: int = 8, descriptors: list[str] | None = None) -> np.ndarray:
    names = descriptors or list(DESCRIPTORS)
    feats = []
    for im in images:
        feats.append(np.hstack([DESCRIPTORS[n](im, bins=bins) for n in names]))
    return np.asarray(feats, dtype=np.float64)


def save_features_h5(path: str | Path, key: str, features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "a") as f:
        if key in f:
            del f[key]
        g = f.create_group(key)
        g.create_dataset("features", data=features)
        g.create_dataset("labels", data=labels.astype("S"))
        g.create_dataset("groups", data=groups.astype("S"))
    return p


def load_features_h5(path: str | Path, key: str):
    with h5py.File(resolve(path), "r") as f:
        g = f[key]
        return (g["features"][()], g["labels"][()].astype(str), g["groups"][()].astype(str))
