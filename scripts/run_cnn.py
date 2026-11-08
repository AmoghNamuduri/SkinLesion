"""Customised CNN → outputs/cnn/{results.json, model.pt, history.png} (Tables 6, 7, Fig. 5).

    python scripts/run_cnn.py
    python scripts/run_cnn.py --set cnn.epochs=10       # quick check
"""
from _bootstrap import parse_args

from skinlesion.evaluation.report import build_report
from skinlesion.training.cnn_trainer import run_cnn_pipeline

if __name__ == "__main__":
    _, cfg = parse_args(__doc__)
    run_cnn_pipeline(cfg)
    print("\nreport ->", build_report(cfg.paths.outputs_dir))
