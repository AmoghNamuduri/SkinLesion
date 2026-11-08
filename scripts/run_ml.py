"""Classical ML pipeline → outputs/ml/results.json (Table 4 + ML part of Table 5).

    python scripts/run_ml.py
    python scripts/run_ml.py --force-features                       # re-extract features
    python scripts/run_ml.py --set data.augment_before_split=false  # leakage-free variant
"""
from _bootstrap import parse_args

from skinlesion.evaluation.report import build_report
from skinlesion.training.ml_trainer import run_ml_pipeline


def _extra(p):
    p.add_argument("--force-features", action="store_true", help="ignore the cached HDF5 features")


if __name__ == "__main__":
    args, cfg = parse_args(__doc__, _extra)
    run_ml_pipeline(cfg, force_features=args.force_features)
    print("\nreport ->", build_report(cfg.paths.outputs_dir))
