"""Generate a synthetic dataset (class folders + manifest.csv) from a GAN snapshot (issue 08).

    python scripts/generate_synthetic.py                                   # latest baseline snapshot
    python scripts/generate_synthetic.py --config configs/skingan/skingan.yaml
    python scripts/generate_synthetic.py --set generate.network=path/to/network-snapshot.pkl generate.per_class=500
    python scripts/generate_synthetic.py --grid-only                       # just a per-class sample grid
"""
from _bootstrap import SKINGAN_CONFIG, parse_args

from skinlesion.data.splits import class_names, encode_labels, ensure_splits
from skinlesion.gan.generate import class_counts, class_sample_grid, generate_dataset, load_generator
from skinlesion.gan.train_gan import latest_snapshot
from skinlesion.gan.visualize import save_rgb
from skinlesion.utils.io import resolve


def _extra(p):
    p.add_argument("--grid-only", action="store_true", help="only save a per-class sample grid")


if __name__ == "__main__":
    args, cfg = parse_args(__doc__, _extra, default_config=SKINGAN_CONFIG)
    gen = cfg.generate
    network = resolve(gen.network) if gen.network else latest_snapshot(cfg.paths.gan_runs)
    if network is None:
        raise SystemExit(f"no snapshot found under {cfg.paths.gan_runs}; pass --set generate.network=...")
    print("network:", network)
    G = load_generator(network, cfg.gan.stylegan_dir, gen.device)
    names = class_names(cfg)
    out_dir = resolve(cfg.paths.synthetic_dir)
    grid = save_rgb(class_sample_grid(G, names, 8, gen.truncation, gen.seed0), out_dir / "sample_grid.png")
    print("sample grid ->", grid)
    if not args.grid_only:
        real = ensure_splits(cfg)["train"] if gen.policy == "match_real" else None
        counts = class_counts(gen.policy, names, gen.per_class, real, lambda dx: encode_labels(dx, cfg))
        manifest = generate_dataset(G, names, counts, out_dir, gen.truncation, gen.seed0, gen.batch)
        print(f"{len(manifest)} images -> {out_dir}")
