"""Memorisation audit of a synthetic dataset against the real train/test splits (issue 15).

    python scripts/audit_memorization.py                                  # data/synthetic/baseline
    python scripts/audit_memorization.py --config configs/skingan/skingan.yaml
    python scripts/audit_memorization.py --set audit.max_images=2000 audit.method=pixels
"""
import numpy as np
from _bootstrap import SKINGAN_CONFIG, parse_args

from skinlesion.data.dataset import load_manifest
from skinlesion.data.splits import ensure_splits
from skinlesion.evaluation.memorization import audit, embed
from skinlesion.gan.dataset_export import load_rgb
from skinlesion.gan.visualize import make_grid, save_rgb
from skinlesion.utils.io import resolve, save_json

if __name__ == "__main__":
    _, cfg = parse_args(__doc__, default_config=SKINGAN_CONFIG)
    a = cfg.get("audit") or {}
    n, method, res = a.get("max_images", 3000), a.get("method", "densenet"), 128
    splits = ensure_splits(cfg)
    syn = load_manifest(resolve(cfg.paths.synthetic_dir))
    sample = lambda df: df.sample(n=min(n, len(df)), random_state=0).reset_index(drop=True)  # noqa: E731
    syn, train, test = sample(syn), sample(splits["train"]), sample(splits["test"])
    imgs = {k: np.stack([load_rgb(p, res) for p in df["path"]]) for k, df in (("syn", syn), ("train", train), ("test", test))}
    feats = {k: embed(v, method) for k, v in imgs.items()}
    rep = audit(feats["syn"], feats["train"], feats["test"])
    out = resolve(cfg.paths.synthetic_dir)
    pairs = [[imgs["syn"][s], imgs["train"][t]] for s, t, _ in rep["closest"][:8]]
    save_rgb(make_grid(pairs, [f"d={d:.3f}" for _, _, d in rep["closest"][:8]], tile=128), out / "closest_pairs.png")
    save_json(rep, out / "memorization.json")
    print(f"median NN distance  synthetic→train {rep['syn_median']:.4f}   test→train {rep['test_median']:.4f}")
    print(f"{100 * rep['frac_below']:.2f}% of synthetic images closer than the {100 * rep['quantile']:.0f}% "
          f"closest real test images (≈{100 * rep['quantile']:.0f}% expected if nothing is copied)")
