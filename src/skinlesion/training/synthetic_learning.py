"""Synthetic-learning experiment (roadmap issue 12) — the headline SkinGAN result.

For each training source train the same DenseNet with the same recipe (``fit_classifier``)
and evaluate on the same *real* test split, so the training data are the only difference.
A source is one of:

* ``real``                                  the real training split (nv capped)
* ``path/to/synthetic``                     synthetic images only (SpineGAN-style)
* ``{topup: duplicate, target: 1000}``      real + duplicated real images of each class below
                                            ``target`` (traditional augmentation: the
                                            training transforms flip/rotate/crop each copy)
* ``{topup: path/to/synthetic, target: 1000}``  real + synthetic images up to ``target``

SpineGAN's numbers for reference: real 0.856, GAN 0.814, SpineGAN 0.830 AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.dataset import load_manifest
from ..data.splits import ensure_splits
from ..utils.io import ensure_dir, resolve, save_json
from .domain_classifier import cap_class, fit_classifier


def topup(real: pd.DataFrame, target: int, fill, seed: int = 0) -> pd.DataFrame:
    """Real images plus extra rows for every class with fewer than ``target`` images.
    ``fill`` = "duplicate" (resample real rows) or a synthetic manifest (DataFrame)."""
    parts = [real.assign(source="real")]
    for dx, n in real["dx"].value_counts().items():
        missing = int(target) - int(n)
        if missing <= 0:
            continue
        if isinstance(fill, str) and fill == "duplicate":
            pool = real[real["dx"] == dx]
            extra = pool.sample(n=missing, replace=True, random_state=seed).assign(source="duplicate")
        else:
            pool = fill[fill["dx"] == dx]
            if pool.empty:
                raise ValueError(f"no synthetic images for class {dx!r}")
            extra = pool.sample(n=missing, replace=len(pool) < missing, random_state=seed).assign(source="synthetic")
        parts.append(extra)
    return pd.concat(parts, ignore_index=True)


def source_frame(spec, real_train: pd.DataFrame, cfg=None) -> pd.DataFrame:
    if isinstance(spec, dict):
        if cfg is not None and cfg.labels.set != "7class":
            raise ValueError("top-up sources need labels.set=7class")
        fill = spec["topup"] if spec["topup"] == "duplicate" else load_manifest(resolve(spec["topup"]))
        return topup(real_train, spec.get("target", 1000), fill)
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
            df = source_frame(spec, real_train, cfg)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        if isinstance(spec, str) and spec != "real" and len(df) > size:
            df = df.sample(n=size, random_state=0)  # synthetic-only: same size as the real set
        composition = df.groupby(["dx", df.get("source", pd.Series("real", index=df.index))]).size()
        print(f"\n== {name}: {len(df)} training images\n{composition.unstack(fill_value=0).to_string()}")
        runs = []
        for seed in s.seeds:
            r = fit_classifier(cfg, df, splits["val"] if s.select_on_real_val else None, splits["test"],
                               seed=seed, epochs=s.epochs, out_dir=out_dir, tag=f"{name}_seed{seed}",
                               n_boot=s.bootstrap, save_model=False)
            r.pop("model")
            runs.append(r)
        aucs = np.array([r["test"]["auc"] for r in runs])
        counts = df["dx"].value_counts().to_dict()
        n_syn = int((df["source"] == "synthetic").sum()) if "source" in df else (0 if spec == "real" else len(df))
        results["sources"][name] = {"n_train": len(df), "n_synthetic": n_syn, "class_counts": counts,
                                    "auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std()), "runs": runs}
        print(f"== {name}: AUC {aucs.mean():.4f} ± {aucs.std():.4f} over {len(runs)} seeds")
    save_json(results, out_dir / "results.json")
    return results
