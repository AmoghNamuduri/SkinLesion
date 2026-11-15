"""Generate synthetic datasets from a StyleGAN snapshot (roadmap issue 08).

Writes ``<out_dir>/<class>/<image_id>.png`` plus ``manifest.csv`` in the same format as the
real split CSVs (image_id, lesion_id, dx, path, …), so real, synthetic and mixed sets all
load through ``skinlesion.data.dataset.load_manifest``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir, resolve


def _pick_device(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_generator(pkl, stylegan_dir, device="auto"):
    """Load the EMA generator from a StyleGAN .pkl (needs the vendored code on sys.path to unpickle)."""
    sg = str(resolve(stylegan_dir))
    if sg not in sys.path:
        sys.path.insert(0, sg)
    import legacy  # noqa: E402  (from third_party/stylegan2-ada-pytorch)
    import dnnlib  # noqa: E402

    dev = _pick_device(device)
    with dnnlib.util.open_url(str(pkl)) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(dev).eval()
    return G


def generate_images(G, class_idx: int, seeds, truncation: float = 1.0, batch: int = 16) -> np.ndarray:
    """Generate one image per seed for ``class_idx``. Returns (N, H, W, 3) uint8 RGB."""
    import torch

    dev = next(G.parameters()).device
    out = []
    seeds = list(seeds)
    for i in range(0, len(seeds), batch):
        chunk = seeds[i:i + batch]
        z = torch.from_numpy(np.stack([np.random.RandomState(s).randn(G.z_dim) for s in chunk])).float().to(dev)
        c = torch.zeros(len(chunk), G.c_dim, device=dev)
        if G.c_dim:
            c[:, class_idx] = 1
        with torch.no_grad():
            img = G(z, c, truncation_psi=truncation, noise_mode="const")
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        out.append(img)
    return np.concatenate(out)


def class_counts(policy: str, classes: list[str], per_class: int, real_train: pd.DataFrame | None = None,
                 label_of=None) -> dict[str, int]:
    """``balanced``: per_class each; ``match_real``: same class distribution and size as the real train split."""
    if policy == "balanced":
        return {c: per_class for c in classes}
    if policy == "match_real":
        if real_train is None or label_of is None:
            raise ValueError("match_real needs the real training split")
        labels = label_of(real_train["dx"])
        return {c: int((labels == k).sum()) for k, c in enumerate(classes)}
    raise ValueError(f"unknown policy {policy!r}")


def generate_dataset(G, classes: list[str], counts: dict[str, int], out_dir, truncation: float = 1.0,
                     seed0: int = 0, batch: int = 16, source: str = "gan") -> pd.DataFrame:
    import cv2

    out = ensure_dir(out_dir)
    rows, seed = [], seed0
    for k, c in enumerate(classes):
        n = counts.get(c, 0)
        if not n:
            continue
        seeds = range(seed, seed + n)
        imgs = generate_images(G, k, seeds, truncation, batch)
        cls_dir = ensure_dir(out / c)
        for s, img in zip(seeds, imgs):
            iid = f"syn_{c}_{s:07d}"
            cv2.imwrite(str(cls_dir / f"{iid}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rows.append({"image_id": iid, "lesion_id": iid, "dx": c, "path": f"{c}/{iid}.png",
                         "seed": s, "psi": truncation, "source": source})
        seed += n
        print(f"  {c}: {n} images")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    return manifest


def class_sample_grid(G, classes: list[str], n_per_class: int = 8, truncation: float = 1.0, seed0: int = 0,
                      tile: int = 128) -> np.ndarray:
    """One row of generated samples per class (same seeds in every row)."""
    from .visualize import make_grid

    rows = [list(generate_images(G, k, range(seed0, seed0 + n_per_class), truncation)) for k in range(len(classes))]
    return make_grid(rows, classes, tile=tile)
