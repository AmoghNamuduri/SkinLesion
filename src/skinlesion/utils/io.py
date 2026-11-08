"""Small filesystem helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def project_root() -> Path:
    """Repository root (the folder that contains ``src/``)."""
    return Path(__file__).resolve().parents[3]


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def ensure_dir(path: str | Path) -> Path:
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=_NumpyEncoder)
    return p


def load_json(path: str | Path) -> Any:
    with open(resolve(path), "r", encoding="utf-8") as fh:
        return json.load(fh)
