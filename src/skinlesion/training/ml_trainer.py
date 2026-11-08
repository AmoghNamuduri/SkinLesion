"""Classical ML pipeline (Table 4 + ML columns of Table 5).

For each setting ("without_augmentation" = 100×7, "with_hflip_augmentation" = 200×7):

1. load images at 220×220 (+ mirrored twins when augmented)
2. extract global features (colour histogram ‖ Haralick ‖ Hu) → cache in HDF5
3. Min-Max scale, 80/20 split
4. for each classifier: 10-fold CV accuracy on the training split (Table 4),
   then fit on the training split and score the test split (P/R/F1, Table 5)
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..data.dataset import build_image_dataset
from ..data.splits import split_indices
from ..evaluation.metrics import classification_metrics
from ..evaluation.plots import plot_confusion
from ..features.extractor import extract_global_features, load_features_h5, save_features_h5
from ..models.ml_models import build_ml_models
from ..utils.io import ensure_dir, resolve, save_json

SETTINGS = {"without_augmentation": False, "with_hflip_augmentation": True}


def get_features(cfg, augmented: bool, key: str, force: bool = False):
    """Return (X, y, groups); features are cached in ``paths.features_file`` under ``key``."""
    h5 = resolve(cfg.paths.features_file)
    if h5.exists() and not force:
        try:
            return load_features_h5(h5, key)
        except KeyError:
            pass
    t = time.time()
    ds = build_image_dataset(cfg, cfg.ml.image_size, hflip=augmented)
    X = extract_global_features(ds.images, bins=cfg.ml.histogram_bins)
    save_features_h5(h5, key, X, ds.labels, ds.groups)
    print(f"  extracted {X.shape} features for {key} in {time.time() - t:.1f}s")
    return X, ds.labels, ds.groups


def make_scaler(kind: str):
    return {"minmax": MinMaxScaler(feature_range=(0, 1)), "standard": StandardScaler(), "none": None}[kind]


def evaluate_setting(cfg, X, y, groups) -> dict:
    tr, te = split_indices(y, groups, cfg.ml.test_size, cfg.ml.split_seed, cfg.data.augment_before_split)
    scaler = make_scaler(cfg.ml.scaler)
    if scaler is not None:
        # reference implementation fits the scaler on all features before splitting
        scaler.fit(X if cfg.ml.get("scale_before_split", True) else X[tr])
        X = scaler.transform(X)
    Xtr, ytr, gtr, Xte, yte = X[tr], y[tr], groups[tr], X[te], y[te]

    if cfg.data.augment_before_split:
        cv, cv_groups = KFold(n_splits=cfg.ml.kfold, shuffle=True, random_state=cfg.ml.kfold_seed), None
    else:
        cv, cv_groups = GroupKFold(n_splits=cfg.ml.kfold), gtr

    results = {"n_train": int(len(tr)), "n_test": int(len(te))}
    for name, model in build_ml_models(cfg.ml.models).items():
        t = time.time()
        scores = cross_val_score(clone(model), Xtr, ytr, cv=cv, groups=cv_groups, scoring="accuracy")
        model.fit(Xtr, ytr)
        test = classification_metrics(yte, model.predict(Xte), labels=list(cfg.data.classes))
        results[name] = {"cv_scores": scores.tolist(), "cv_mean": float(scores.mean()),
                         "cv_std": float(scores.std()), "test": test}
        print(f"    {name:4s} cv={scores.mean():.4f}±{scores.std():.3f}  test_acc={test['accuracy']:.4f}  "
              f"wF1={test['weighted']['f1']:.4f}  ({time.time() - t:.1f}s)")
    return results


def run_ml_pipeline(cfg, settings: list[str] | None = None, force_features: bool = False) -> dict:
    out_dir = ensure_dir(resolve(cfg.paths.outputs_dir) / "ml")
    all_results = {}
    for key in settings or list(SETTINGS):
        print(f"\n== ML setting: {key} ==")
        X, y, g = get_features(cfg, SETTINGS[key], key, force=force_features)
        res = evaluate_setting(cfg, X, y, g)
        all_results[key] = res
        model_names = [m for m in res if isinstance(res[m], dict)]
        best = max(model_names, key=lambda m: res[m]["cv_mean"])
        plot_confusion(res[best]["test"]["confusion_matrix"], list(cfg.data.classes),
                       out_dir / f"confusion_{key}_{best}.png", f"{best} — {key}")
    save_json(all_results, out_dir / "results.json")
    return all_results
