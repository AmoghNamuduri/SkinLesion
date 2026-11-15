"""Turn a StyleGAN run into the pictures for the presentation.

    python gan_test/show_results.py --runs /content/drive/MyDrive/SkinGAN/runs/test \
                                    --data /content/data/ham10000_256_7class.zip

Writes into the run folder:
  progression.png        the same generated samples at each snapshot (how the GAN learns)
  real_vs_generated.png  per class: 4 real images | 4 generated images (latest snapshot)
  class_grid.png         8 generated samples per class
Needs a GPU for the generated grids (CPU works, slowly).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from skinlesion.gan.dataset_export import read_archive_image, read_archive_labels  # noqa: E402
from skinlesion.gan.generate import class_sample_grid, generate_images, load_generator  # noqa: E402
from skinlesion.gan.visualize import make_grid, progression_strip, save_rgb  # noqa: E402


def latest_run(runs: Path) -> Path:
    cands = sorted(d for d in runs.iterdir() if d.is_dir() and list(d.glob("network-snapshot-*.pkl")))
    if not cands:
        raise SystemExit(f"no run with snapshots in {runs}")
    return cands[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="gan.outdir / paths.gan_runs (folder of StyleGAN runs)")
    ap.add_argument("--data", required=True, help="the training zip (for real images and class names)")
    ap.add_argument("--stylegan", default=str(ROOT / "third_party" / "stylegan2-ada-pytorch"))
    ap.add_argument("--truncation", type=float, default=1.0)
    args = ap.parse_args()

    run = latest_run(Path(args.runs))
    pkl = sorted(run.glob("network-snapshot-*.pkl"))[-1]
    print("run:", run, "\nsnapshot:", pkl.name)
    print("progression ->", save_rgb(progression_strip(run), run / "progression.png"))

    labels, names = read_archive_labels(args.data)
    G = load_generator(pkl, args.stylegan)
    print("class grid ->", save_rgb(class_sample_grid(G, names, 8, args.truncation), run / "class_grid.png"))

    rng = np.random.default_rng(0)
    rows, row_labels = [], []
    for k, name in enumerate(names):
        files = [f for f, y in labels if int(y) == k]
        real = [read_archive_image(args.data, files[i]) for i in rng.choice(len(files), min(4, len(files)), replace=False)]
        fake = list(generate_images(G, k, range(100, 104), args.truncation))
        rows.append(real + fake)
        row_labels.append(name)
    out = save_rgb(make_grid(rows, row_labels, tile=128), run / "real_vs_generated.png")
    print("real | generated ->", out, "(columns 1-4 real, 5-8 generated)")
