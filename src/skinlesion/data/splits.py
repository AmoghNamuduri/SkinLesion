"""Train/test splitting.

``augment_before_split=True`` reproduces the paper: the 1,400 augmented images
are split at random, so an image and its mirrored twin can land on opposite
sides. ``False`` splits by original ``image_id`` so twins always stay together
(a leakage-free variant for later experiments).
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def split_indices(labels: np.ndarray, groups: np.ndarray, test_size: float, seed: int,
                  augment_before_split: bool = True, stratify: bool = False):
    n = len(labels)
    if augment_before_split:
        idx = np.arange(n)
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed,
                                  stratify=labels if stratify else None)
        return np.sort(tr), np.sort(te)

    uniq, first = np.unique(groups, return_index=True)
    g_tr, g_te = train_test_split(uniq, test_size=test_size, random_state=seed,
                                  stratify=labels[first] if stratify else None)
    in_test = np.isin(groups, g_te)
    return np.where(~in_test)[0], np.where(in_test)[0]
