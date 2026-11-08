"""End-to-end reproduction: prepare data → ML pipeline → CNN → report.

    python scripts/run_all.py
    python scripts/run_all.py --config configs/variations/no_leakage.yaml
"""
from _bootstrap import parse_args

from skinlesion.data.metadata import load_metadata
from skinlesion.data.sampling import materialize_subset, sample_per_class
from skinlesion.evaluation.report import build_report
from skinlesion.training.cnn_trainer import run_cnn_pipeline
from skinlesion.training.ml_trainer import run_ml_pipeline


def _extra(p):
    p.add_argument("--skip-ml", action="store_true")
    p.add_argument("--skip-cnn", action="store_true")


if __name__ == "__main__":
    args, cfg = parse_args(__doc__, _extra)
    meta = load_metadata(cfg.paths.raw_dir)
    subset = sample_per_class(meta, list(cfg.data.classes), cfg.data.images_per_class, cfg.data.sampling_seed)
    materialize_subset(subset, cfg.paths.subset_dir)
    if not args.skip_ml:
        run_ml_pipeline(cfg, force_features=True)
    if not args.skip_cnn:
        run_cnn_pipeline(cfg)
    print(build_report(cfg.paths.outputs_dir).read_text(encoding="utf-8"))
