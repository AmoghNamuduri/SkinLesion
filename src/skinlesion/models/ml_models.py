"""Registry of the seven classical classifiers compared in the paper (Table 1).

Models are built from the ``ml.models`` section of the config:

    RF: {class: RandomForestClassifier, params: {n_estimators: 200, random_state: 0}}

To try a new classifier, register its class here and add a config entry.
"""
from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ML_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "KNeighborsClassifier": KNeighborsClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "GaussianNB": GaussianNB,
    "SVC": SVC,
    # extra, not in the paper — available for variations
    "GradientBoostingClassifier": GradientBoostingClassifier,
}


def build_ml_models(models_cfg) -> dict:
    """``{"LR": LogisticRegression(random_state=9), ...}`` in config order."""
    out = {}
    for name, spec in models_cfg.items():
        cls = ML_REGISTRY[spec["class"]]
        out[name] = cls(**dict(spec.get("params") or {}))
    return out
