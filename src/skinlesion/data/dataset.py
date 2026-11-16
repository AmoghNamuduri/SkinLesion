"""Load the sampled subset into memory as arrays (Step 2 resize + Step 3 flip)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .augmentation import hflip_offline
from .sampling import load_subset_manifest


@dataclass
class ImageDataset:
    images: np.ndarray        # (N, H, W, 3) uint8, BGR (OpenCV order, as in the paper's cv2 code)
    labels: np.ndarray        # (N,) str  — diagnosis code, e.g. "mel"
    groups: np.ndarray        # (N,) str  — original image_id (shared by an image and its flip)
    is_augmented: np.ndarray  # (N,) bool

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, idx) -> "ImageDataset":
        return ImageDataset(self.images[idx], self.labels[idx], self.groups[idx], self.is_augmented[idx])


def load_image(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"could not read {path}")
    return cv2.resize(img, (size, size))


def load_manifest(path_or_df) -> "pd.DataFrame":
    """Read a manifest (columns: image_id, dx, path, …) from a CSV path, a folder containing
    ``manifest.csv``, or a DataFrame. Real split CSVs and synthetic sets share this format,
    so downstream code doesn't care whether images are real or generated (roadmap issue 08)."""
    import pandas as pd
    from pathlib import Path

    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df.copy()
    p = Path(path_or_df)
    if p.is_dir():
        p = p / "manifest.csv"
    df = pd.read_csv(p)
    # synthetic manifests store paths relative to their folder
    df["path"] = [str(x) if Path(x).is_absolute() else str(p.parent / x) for x in df["path"]]
    return df


def dataset_from_manifest(manifest, size: int) -> ImageDataset:
    df = load_manifest(manifest)
    images = np.stack([load_image(p, size) for p in df["path"]])
    labels = df["dx"].to_numpy(dtype=str)
    groups = df.get("lesion_id", df["image_id"]).to_numpy(dtype=str)
    return ImageDataset(images, labels, groups, np.zeros(len(df), bool))


def build_image_dataset(cfg, size: int, hflip: bool | None = None) -> ImageDataset:
    """Images resized to ``size``×``size``; if ``hflip`` (default: config) each image gets a mirrored twin."""
    manifest = load_subset_manifest(cfg.paths.subset_dir)
    manifest = manifest[manifest["dx"].isin(cfg.data.classes)]
    images = np.stack([load_image(p, size) for p in manifest["path"]])
    labels = manifest["dx"].to_numpy(dtype=str)
    groups = manifest["image_id"].to_numpy(dtype=str)
    is_aug = np.zeros(len(labels), bool)
    if cfg.data.hflip_augmentation if hflip is None else hflip:
        images, labels, groups, is_aug = hflip_offline(images, labels, groups)
    return ImageDataset(images, labels, groups, is_aug)
