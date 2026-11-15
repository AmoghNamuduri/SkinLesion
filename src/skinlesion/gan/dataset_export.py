"""Export HAM10000 as a conditional StyleGAN2-ADA training archive (roadmap issue 04).

Each 600×450 image is centre-cropped to 450×450 (the lesion is centred in HAM10000)
and resized to ``resolution``². The zip holds the images plus ``dataset.json``
(``{"labels": [[file, class_index], ...]}``), which StyleGAN's ImageFolderDataset reads
directly, so ``dataset_tool.py`` is not needed. ``class_names.json`` records the label order.
Only the configured splits (default: train) are exported; the GAN never sees val/test images.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..data.splits import class_names, encode_labels, ensure_splits
from ..utils.io import resolve


def center_crop_resize(image_bgr: np.ndarray, resolution: int) -> np.ndarray:
    """Centre square crop + area resize. Returns RGB uint8."""
    h, w = image_bgr.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    sq = image_bgr[y:y + s, x:x + s]
    sq = cv2.resize(sq, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)


def load_rgb(path: str, resolution: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"could not read {path}")
    return center_crop_resize(img, resolution)


def encode_image(rgb: np.ndarray, fmt: str = "jpg", quality: int = 95) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg" if fmt == "jpg" else ".png", bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else [])
    if not ok:
        raise IOError("image encoding failed")
    return buf.tobytes()


def select_rows(splits: dict[str, pd.DataFrame], names, max_per_class: int | None, seed: int) -> pd.DataFrame:
    df = pd.concat([splits[n] for n in names], ignore_index=True)
    if max_per_class:
        df = (df.groupby("dx", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), max_per_class), random_state=seed)))
    return df.sort_values("image_id").reset_index(drop=True)


def export_stylegan_dataset(cfg, out_zip: str | Path | None = None, meta: pd.DataFrame | None = None) -> dict:
    """Write the training archive; returns a summary (path, image count, per-class counts)."""
    g = cfg.gan_data
    splits = ensure_splits(cfg, meta)
    rows = select_rows(splits, list(g.splits), g.max_per_class, cfg.split.seed)
    labels = encode_labels(rows["dx"], cfg)
    names = class_names(cfg)
    out = resolve(out_zip or cfg.paths.gan_dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if g.format == "jpg" else "png"
    entries = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for i, (r, y) in enumerate(zip(rows.itertuples(), labels)):
            fname = f"{r.dx}/{r.image_id}.{ext}"
            zf.writestr(fname, encode_image(load_rgb(r.path, g.resolution), ext, g.jpg_quality))
            entries.append([fname, int(y)])
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(rows)} images")
        zf.writestr("dataset.json", json.dumps({"labels": entries}))
        zf.writestr("class_names.json", json.dumps(names))
    counts = {names[k]: int((labels == k).sum()) for k in range(len(names))}
    return {"path": str(out), "images": len(rows), "resolution": g.resolution, "classes": names, "counts": counts}


def read_archive_labels(zip_path) -> tuple[list[list], list[str]]:
    with zipfile.ZipFile(resolve(zip_path)) as zf:
        labels = json.loads(zf.read("dataset.json"))["labels"]
        names = json.loads(zf.read("class_names.json")) if "class_names.json" in zf.namelist() else []
    return labels, names


def read_archive_image(zip_path, fname: str) -> np.ndarray:
    with zipfile.ZipFile(resolve(zip_path)) as zf:
        data = np.frombuffer(zf.read(fname), np.uint8)
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
