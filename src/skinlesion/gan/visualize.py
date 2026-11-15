"""Image grids for SkinGAN: real vs generated per class, and training progression strips."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from ..utils.io import resolve


def make_grid(rows: list[list[np.ndarray]], row_labels: list[str] | None = None, tile: int = 128,
              pad: int = 4, label_w: int = 150, bg: int = 255) -> np.ndarray:
    """Stack RGB tiles into a grid. ``rows`` = list of rows, each a list of HxWx3 uint8 images."""
    ncols = max(len(r) for r in rows)
    lw = label_w if row_labels else 0
    H = len(rows) * (tile + pad) + pad
    W = lw + ncols * (tile + pad) + pad
    grid = np.full((H, W, 3), bg, np.uint8)
    for i, row in enumerate(rows):
        y = pad + i * (tile + pad)
        if row_labels:
            cv2.putText(grid, str(row_labels[i]), (8, y + tile // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (40, 40, 40), 1, cv2.LINE_AA)
        for j, img in enumerate(row):
            x = lw + pad + j * (tile + pad)
            grid[y:y + tile, x:x + tile] = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
    return grid


def save_rgb(img: np.ndarray, path) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return p


def real_class_grid(zip_path, n_per_class: int = 8, seed: int = 0, tile: int = 128) -> np.ndarray:
    """One row per class of real images taken from a StyleGAN training archive."""
    from .dataset_export import read_archive_image, read_archive_labels

    labels, names = read_archive_labels(zip_path)
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[str]] = {}
    for fname, y in labels:
        by_class.setdefault(int(y), []).append(fname)
    rows, row_labels = [], []
    for k in sorted(by_class):
        files = by_class[k]
        pick = rng.choice(len(files), size=min(n_per_class, len(files)), replace=False)
        rows.append([read_archive_image(zip_path, files[i]) for i in pick])
        row_labels.append(f"{names[k] if names else k} ({len(files)})")
    return make_grid(rows, row_labels, tile=tile)


def _kimg_of(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def progression_strip(run_dir, n_tiles: int = 6, tile_px: int | None = None, max_snapshots: int = 8,
                      out_tile: int = 112) -> np.ndarray:
    """Crop the same top-left tiles from every ``fakes*.png`` snapshot grid StyleGAN wrote
    (fixed latents, so you watch the same samples improve) and stack them by kimg."""
    run_dir = resolve(run_dir)
    snaps = sorted(run_dir.glob("fakes*.png"), key=_kimg_of)
    if not snaps:
        raise FileNotFoundError(f"no fakes*.png in {run_dir}")
    if len(snaps) > max_snapshots:  # keep first, last and evenly spaced in between
        idx = np.linspace(0, len(snaps) - 1, max_snapshots).round().astype(int)
        snaps = [snaps[i] for i in idx]
    rows, labels = [], []
    for s in snaps:
        img = cv2.cvtColor(cv2.imread(str(s)), cv2.COLOR_BGR2RGB)
        t = tile_px or _guess_tile(img)
        rows.append([img[0:t, j * t:(j + 1) * t] for j in range(n_tiles)])
        labels.append("init" if "init" in s.stem else f"{_kimg_of(s)} kimg")
    return make_grid(rows, labels, tile=out_tile, label_w=110)


def _guess_tile(grid_img: np.ndarray) -> int:
    """StyleGAN fakes grids are ~7680×4320 px of square tiles; infer the tile size."""
    h, w = grid_img.shape[:2]
    for t in (1024, 512, 256, 128, 64, 32):
        if w % t == 0 and h % t == 0 and 7 <= w // t <= 32 and 4 <= h // t <= 32:
            return t
    return min(h, w) // 4
