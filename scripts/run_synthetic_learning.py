"""Synthetic learning: train on real / GAN / SkinGAN images, test on real (issue 12). GPU recommended.

    python scripts/run_synthetic_learning.py
    python scripts/run_synthetic_learning.py --set synthetic_learning.seeds=[0] synthetic_learning.epochs=5
"""
from _bootstrap import ROOT, parse_args

from skinlesion.evaluation.report import build_skingan_report
from skinlesion.training.synthetic_learning import run_synthetic_learning

if __name__ == "__main__":
    _, cfg = parse_args(__doc__, default_config=str(ROOT / "configs" / "skingan" / "synthetic_learning.yaml"))
    run_synthetic_learning(cfg)
    print("report ->", build_skingan_report(cfg.paths.outputs_dir))
