"""Download HAM10000 (10,015 dermoscopic images + metadata).

Two interchangeable sources:

* ``dataverse`` — Harvard Dataverse (the official HAM10000 host), no login.
* ``kaggle``    — the Kaggle mirror cited by the paper; needs the ``kaggle``
                  package and ``~/.kaggle/kaggle.json``.

After download the zips are extracted to ``<raw_dir>/images`` and the metadata
is stored as ``<raw_dir>/HAM10000_metadata.csv``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from ..utils.io import ensure_dir

DATAVERSE_API = "https://dataverse.harvard.edu/api"
WANTED = ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip", "HAM10000_metadata.tab")
HEADERS = {"User-Agent": "skinlesion-reproduction/0.1"}  # Dataverse rejects urllib's default UA


def _open(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS))


def _stream(url: str, dest: Path, expected: int | None = None) -> None:
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        print(f"  [skip] {dest.name} already downloaded")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {dest.name}")
    with _open(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or expected or 0)
        done, last = 0, -1
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = int(100 * done / total)
                if pct != last and pct % 5 == 0:
                    print(f"         {pct:3d}%  ({done / 1e6:.0f} MB)", flush=True)
                    last = pct
    tmp.replace(dest)


def download_dataverse(raw_dir: Path, doi: str) -> None:
    meta_url = f"{DATAVERSE_API}/datasets/:persistentId/?persistentId={doi}"
    with _open(meta_url) as resp:
        files = json.load(resp)["data"]["latestVersion"]["files"]
    by_name = {f["dataFile"]["filename"]: f["dataFile"] for f in files}
    for name in WANTED:
        info = by_name[name]
        url = f"{DATAVERSE_API}/access/datafile/{info['id']}"
        if name.endswith(".tab"):  # ingested tabular file -> ask for the original CSV
            _stream(url + "?format=original", raw_dir / "HAM10000_metadata.csv")
        else:
            _stream(url, raw_dir / name, info.get("filesize"))


def download_kaggle(raw_dir: Path, dataset: str) -> None:
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir), "--unzip"]
    print("  [run ]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_images(raw_dir: Path) -> Path:
    """Unzip every image archive into ``raw_dir/images`` (flat)."""
    img_dir = ensure_dir(raw_dir / "images")
    for z in sorted(raw_dir.glob("HAM10000_images_part_*.zip")):
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".jpg")]
            if all((img_dir / Path(m).name).exists() for m in members):
                print(f"  [skip] {z.name} already extracted")
                continue
            print(f"  [unz ] {z.name} ({len(members)} images)")
            for m in members:
                with zf.open(m) as src, open(img_dir / Path(m).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    # Kaggle mirror ships already-unzipped folders
    for folder in raw_dir.glob("HAM10000_images_part_*"):
        if folder.is_dir():
            for jpg in folder.glob("*.jpg"):
                target = img_dir / jpg.name
                if not target.exists():
                    shutil.copy2(jpg, target)
    n = len(list(img_dir.glob("*.jpg")))
    print(f"  [ok  ] {n} images in {img_dir}")
    return img_dir


def download(cfg) -> Path:
    raw_dir = ensure_dir(cfg.paths.raw_dir)
    src = cfg.download.source
    print(f"Downloading HAM10000 from {src} -> {raw_dir}")
    if src == "dataverse":
        download_dataverse(raw_dir, cfg.download.dataverse_doi)
    elif src == "kaggle":
        download_kaggle(raw_dir, cfg.download.kaggle_dataset)
    else:
        raise ValueError(f"Unknown download source {src!r}")
    return extract_images(raw_dir)
