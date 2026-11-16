"""Synthetic-learning experiment (roadmap issue 12) — the headline SkinGAN result.

For each training source (real, baseline-GAN images, SkinGAN images, …) train the same
DenseNet with the same recipe (``fit_classifier``) and evaluate on the same *real* test
split. SpineGAN's numbers for reference: real 0.856, GAN 0.814, SpineGAN 0.830 AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.dataset import load_manifest
from ..data.splits import ensure_splits
from ..utils.io import ensure_dir, resolve, save_json
from .domain_classifier import cap_class, fit_classifier


def source_frame(spec, real_train: pd.DataFrame) -> pd.DataFrame:
    return real_train.copy() if spec == "real" else load_manifest(resolve(spec))


def run_synthetic_learning(cfg) -> dict:
    s = cfg.synthetic_learning
    splits = ensure_splits(cfg)
    real_train = cap_class(splits["train"], "nv", cfg.domain_classifier.max_nv, cfg.domain_classifier.seed)
    size = s.train_size or len(real_train)
    out_dir = ensure_dir(resolve(cfg.paths.outputs_dir) / "synthetic_learning")
    results = {"train_size": size, "sources": {}}
    for name, spec in s.sources.items():
        try:
            df = source_frame(spec, real_train)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        if len(df) > size:
            df = df.sample(n=size, random_state=0)
        runs = []
        for seed in s.seeds:
            r = fit_classifier(cfg, df, splits["val"] if s.select_on_real_val else None, splits["test"],
                               seed=seed, epochs=s.epochs, out_dir=out_dir, tag=f"{name}_seed{seed}",
                               n_boot=s.bootstrap, save_model=False)
            r.pop("model")
            runs.append(r)
        aucs = np.array([r["test"]["auc"] for r in runs])
        results["sources"][name] = {"n_train": len(df), "auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std()),
                                    "runs": runs}
        print(f"== {name}: AUC {aucs.mean():.4f} ± {aucs.std():.4f} over {len(runs)} seeds")
    save_json(results, out_dir / "results.json")
    return results
