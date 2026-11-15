"""Preview the GAN's input on your laptop (no GPU): real HAM10000 training images per class,
centre-cropped and resized to 256² exactly as the StyleGAN export does.

    python gan_test/make_real_grid.py                 # -> gan_test/output/real_samples.png
    python gan_test/make_real_grid.py --n 10

Also creates the lesion-level split CSVs in data/splits/ if they don't exist yet.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from skinlesion.data.metadata import load_metadata  # noqa: E402
from skinlesion.data.splits import ensure_splits  # noqa: E402
from skinlesion.gan.dataset_export import load_rgb  # noqa: E402
from skinlesion.gan.visualize import make_grid, save_rgb  # noqa: E402
from skinlesion.utils import load_config  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8, help="images per class")
    ap.add_argument("--config", default=str(ROOT / "gan_test" / "test_config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    splits = ensure_splits(cfg, load_metadata(cfg.paths.raw_dir))
    train = splits["train"]
    rows, labels = [], []
    for dx in cfg.data.classes:
        df = train[train["dx"] == dx]
        pick = df.sample(n=min(args.n, len(df)), random_state=0)
        rows.append([load_rgb(p, cfg.gan_data.resolution) for p in pick["path"]])
        labels.append(f"{dx} ({len(df)})")
    out = save_rgb(make_grid(rows, labels, tile=128), ROOT / "gan_test" / "output" / "real_samples.png")
    sizes = {k: len(v) for k, v in splits.items()}
    print(f"split sizes: {sizes}  (lesion-level, seed {cfg.split.seed})")
    print("real training images per class:", dict(train["dx"].value_counts().sort_index()))
    print("grid ->", out)
