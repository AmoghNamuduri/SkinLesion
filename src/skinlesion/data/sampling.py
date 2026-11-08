"""Choose the class-balanced subset used by the paper (100 images per class = 700)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.io import ensure_dir, resolve
from .organize import _link_or_copy

MANIFEST = "manifest.csv"


def sample_per_class(meta: pd.DataFrame, classes: list[str], n: int, seed: int) -> pd.DataFrame:
    parts = []
    for dx in classes:
        pool = meta[meta["dx"] == dx]
        if len(pool) < n:
            raise ValueError(f"class {dx} has only {len(pool)} images (< {n})")
        parts.append(pool.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def materialize_subset(subset: pd.DataFrame, subset_dir: str | Path) -> Path:
    """Write ``subset_dir/<dx>/<image_id>.jpg`` plus a ``manifest.csv`` describing the subset."""
    out = ensure_dir(subset_dir)
    rows = []
    for _, r in subset.iterrows():
        dst = ensure_dir(out / r["dx"]) / f"{r['image_id']}.jpg"
        _link_or_copy(Path(r["path"]), dst)
        rows.append({"image_id": r["image_id"], "lesion_id": r["lesion_id"], "dx": r["dx"], "path": str(dst)})
    pd.DataFrame(rows).to_csv(out / MANIFEST, index=False)
    return out


def load_subset_manifest(subset_dir: str | Path) -> pd.DataFrame:
    path = resolve(subset_dir) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/prepare_data.py first")
    return pd.read_csv(path)
