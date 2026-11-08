"""Step 1 of the paper — "ordering the dataset": put every image into a folder
named after its diagnosis (``dx``) using ``image_id`` and ``dx`` from the metadata."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from ..utils.io import ensure_dir


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hard link: no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def organize_by_class(meta: pd.DataFrame, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    for dx, group in meta.groupby("dx"):
        cls_dir = ensure_dir(out / dx)
        for p in group["path"]:
            _link_or_copy(Path(p), cls_dir / Path(p).name)
    return out
