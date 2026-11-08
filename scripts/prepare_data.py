"""Step 1 — organise HAM10000 into class folders and sample 100 images per class.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --skip-organize          # only build data/subset/
"""
from _bootstrap import parse_args

from skinlesion.data.metadata import class_counts, load_metadata
from skinlesion.data.organize import organize_by_class
from skinlesion.data.sampling import materialize_subset, sample_per_class


def _extra(p):
    p.add_argument("--skip-organize", action="store_true", help="do not build data/organized/ (all 10,015 images)")


if __name__ == "__main__":
    args, cfg = parse_args(__doc__, _extra)
    meta = load_metadata(cfg.paths.raw_dir)
    print(f"metadata: {len(meta)} images\n{class_counts(meta).to_string()}")
    if not args.skip_organize:
        print("organising by class ->", organize_by_class(meta, cfg.paths.organized_dir))
    subset = sample_per_class(meta, list(cfg.data.classes), cfg.data.images_per_class, cfg.data.sampling_seed)
    out = materialize_subset(subset, cfg.paths.subset_dir)
    print(f"subset: {len(subset)} images ({cfg.data.images_per_class}/class) -> {out}")
