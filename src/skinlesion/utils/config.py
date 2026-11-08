"""YAML configuration with attribute access and command-line overrides.

Every hyper-parameter of the paper lives in ``configs/paper.yaml``. Experiments
(variations of the parent paper) should be expressed as a *new* YAML file or as
``--set key.sub=value`` overrides, never by editing code.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A dict that also supports ``cfg.section.key`` access (recursively)."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Config) else copy.deepcopy(v)) for k, v in self.items()}

    def set_by_path(self, dotted: str, value: Any) -> None:
        node = self
        *parents, leaf = dotted.split(".")
        for p in parents:
            if p not in node or not isinstance(node[p], Config):
                node[p] = Config()
            node = node[p]
        node[leaf] = Config(value) if isinstance(value, dict) else value


def _parse_value(raw: str) -> Any:
    """Turn ``"0.5"`` → 0.5, ``"true"`` → True, ``"[1,2]"`` → [1, 2]; else keep the string."""
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML config. ``overrides`` are strings of the form ``a.b.c=value``.

    A config may declare ``base: other.yaml`` to inherit from another file
    (path relative to the config itself) and only change a few keys.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    base = data.pop("base", None)
    if base:
        cfg = load_config(path.parent / base)
        _deep_update(cfg, data)
    else:
        cfg = Config(data)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        cfg.set_by_path(key.strip(), _parse_value(raw))
    return cfg


def _deep_update(cfg: Config, data: dict) -> None:
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), Config):
            _deep_update(cfg[k], v)
        else:
            cfg[k] = Config(v) if isinstance(v, dict) else v
