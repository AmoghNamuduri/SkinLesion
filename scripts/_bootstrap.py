"""Shared CLI plumbing for the scripts: puts ``src/`` on the path and parses
``--config`` / ``--set`` so every script accepts the same overrides."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skinlesion.utils import load_config, set_seed  # noqa: E402


def parse_args(description: str, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "configs" / "paper.yaml"), help="YAML config file")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. --set cnn.epochs=20 data.augment_before_split=false")
    if extra:
        extra(p)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    return args, cfg
