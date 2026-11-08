"""Read ``HAM10000_metadata.csv`` (lesion_id, image_id, dx, dx_type, age, sex, localization)
and attach the on-disk path of every image.

Images are located by searching ``raw_dir`` recursively for ``<image_id>.jpg``,
so any layout works: Kaggle's ``HAM10000_images_part_1/`` + ``part_2/`` folders,
the extracted Dataverse zips, or a single flat ``images/`` folder.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.io import resolve

METADATA_NAMES = ("HAM10000_metadata.csv", "HAM10000_metadata.tab", "HAM10000_metadata")


def find_metadata(raw_dir: Path) -> Path:
    for name in METADATA_NAMES:
        hits = list(raw_dir.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No HAM10000_metadata.csv under {raw_dir}. Place the dataset there (see data/README.md) "
        f"or run scripts/download_data.py."
    )


def index_images(raw_dir: Path) -> dict[str, str]:
    """image_id -> path for every .jpg under raw_dir (first hit wins)."""
    index: dict[str, str] = {}
    for p in sorted(raw_dir.rglob("*.jpg")):
        index.setdefault(p.stem, str(p))
    return index


def load_metadata(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = resolve(raw_dir)
    df = pd.read_csv(find_metadata(raw_dir), sep=None, engine="python")  # sniff comma vs tab
    df.columns = [c.strip().strip('"') for c in df.columns]
    for col in ("lesion_id", "image_id", "dx"):
        df[col] = df[col].astype(str).str.strip().str.strip('"')
    index = index_images(raw_dir)
    df["path"] = df["image_id"].map(index)
    missing = df["path"].isna().sum()
    if missing:
        print(f"  [warn] {missing} of {len(df)} metadata rows have no image file; they are ignored")
        df = df.dropna(subset=["path"]).reset_index(drop=True)
    return df


def class_counts(df: pd.DataFrame) -> pd.Series:
    """Images per diagnosis (paper: akiec 327, bcc 514, bkl 1099, df 115, mel 1113, nv 6705, vasc 142)."""
    return df["dx"].value_counts().sort_index()
