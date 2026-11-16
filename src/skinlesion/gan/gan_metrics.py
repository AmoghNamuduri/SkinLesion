"""GAN quality metrics through StyleGAN's own ``calc_metrics.py`` (roadmap issue 07).

FID (lower = closer to real), KID, and precision / recall (precision = realism,
recall = coverage of the real distribution; SpineGAN's finding was worse precision
but better recall for the domain-loss GAN). Results are appended by StyleGAN to
``metric-<name>.jsonl`` next to the snapshot; we parse and return them.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..utils.io import resolve

QUICK = ["fid10k_full"]
FULL = ["fid50k_full", "kid50k_full", "pr50k3_full"]


def calc_metrics(cfg, network, metrics: list[str] = QUICK, data=None, mirror: bool | None = None) -> dict:
    from .train_gan import stylegan_dir

    network = resolve(network)
    data = resolve(data or cfg.gan.data or cfg.paths.gan_dataset)
    mirror = cfg.gan.mirror if mirror is None else mirror
    cmd = [sys.executable, "calc_metrics.py", f"--metrics={','.join(metrics)}", f"--network={network}",
           f"--data={data}", f"--mirror={int(bool(mirror))}"]
    print(" ".join(cmd))
    import os
    env = dict(os.environ) if cfg.gan.get("show_warnings", False) else dict(os.environ, PYTHONWARNINGS="ignore")
    subprocess.run(cmd, cwd=stylegan_dir(cfg), check=True, env=env)
    return read_metrics(network.parent, network.name)


def read_metrics(run_dir, snapshot_name: str | None = None) -> dict:
    """{metric: value} from the metric-*.jsonl files in a run folder (latest entry per metric)."""
    out = {}
    for f in sorted(Path(run_dir).glob("metric-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if snapshot_name and Path(rec.get("snapshot_pkl", "")).name != snapshot_name:
                continue
            out.update(rec["results"])
    return out
