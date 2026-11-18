"""Pictures comparing the two GAN arms.

    python skinGan/compare_grids.py --data /content/data/ham10000_256_7class.zip \
        --baseline-runs .../gan/baseline --skingan-runs .../gan/skingan --out .../figures

Writes:
  real_vs_gan_vs_skingan.png  per class: 4 real | 4 standard GAN | 4 SkinGAN (same latent seeds
                              for both GANs, so differences come from the classifier loss)
  progression_baseline.png    same samples across snapshots of the latest baseline run
  progression_skingan.png     same for SkinGAN
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from skinlesion.gan.dataset_export import read_archive_image, read_archive_labels  # noqa: E402
from skinlesion.gan.generate import generate_images, load_generator  # noqa: E402
from skinlesion.gan.train_gan import latest_snapshot  # noqa: E402
from skinlesion.gan.visualize import make_grid, progression_strip, save_rgb  # noqa: E402

TILE, PAD, LABEL_W, N = 112, 4, 120, 4


def header(titles: list[str], width: int) -> np.ndarray:
    """White strip with one title centred over each group of N columns."""
    h = np.full((30, width, 3), 255, np.uint8)
    group = N * (TILE + PAD)
    for i, t in enumerate(titles):
        x = LABEL_W + PAD + i * group + group // 2 - len(t) * 5
        cv2.putText(h, t, (max(0, x), 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return h


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--baseline-runs", required=True)
    ap.add_argument("--skingan-runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stylegan", default=str(ROOT / "third_party" / "stylegan2-ada-pytorch"))
    ap.add_argument("--truncation", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)

    labels, names = read_archive_labels(args.data)
    gens = {}
    for arm, runs in (("baseline", args.baseline_runs), ("skingan", args.skingan_runs)):
        pkl = latest_snapshot(runs)
        if pkl is None:
            raise SystemExit(f"no snapshot in {runs}")
        print(f"{arm}: {pkl}")
        gens[arm] = load_generator(pkl, args.stylegan)
        run_dir = pkl.parent
        try:
            print("  ->", save_rgb(progression_strip(run_dir, out_tile=TILE), out / f"progression_{arm}.png"))
        except FileNotFoundError as e:
            print("  (no progression):", e)

    rng = np.random.default_rng(0)
    rows = []
    for k, name in enumerate(names):
        files = [f for f, y in labels if int(y) == k]
        real = [read_archive_image(args.data, files[i]) for i in rng.choice(len(files), min(N, len(files)), replace=False)]
        seeds = range(1000, 1000 + N)
        rows.append(real + [im for arm in ("baseline", "skingan")
                            for im in generate_images(gens[arm], k, seeds, args.truncation)])
    grid = make_grid(rows, names, tile=TILE, pad=PAD, label_w=LABEL_W)
    grid = np.vstack([header(["Real", "Standard GAN", "SkinGAN"], grid.shape[1]), grid])
    print("->", save_rgb(grid, out / "real_vs_gan_vs_skingan.png"))
