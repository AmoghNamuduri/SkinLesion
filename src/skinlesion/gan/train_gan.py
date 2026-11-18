"""Launch StyleGAN2-ADA training from a YAML config (roadmap issue 02).

Baseline and SkinGAN run through exactly the same code; SkinGAN only sets
``gan.domain_loss.enabled: true``, which makes the launcher write a small JSON spec
and point the patched StyleGAN loss at it through ``$SKINGAN_DOMAIN_LOSS``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ..utils.io import ensure_dir, project_root, resolve, save_json


def stylegan_dir(cfg) -> Path:
    d = resolve(cfg.gan.stylegan_dir)
    if not (d / "train.py").exists():
        raise FileNotFoundError(f"StyleGAN2-ADA not found at {d}. Run: python third_party/setup_stylegan.py")
    return d


def latest_snapshot(run_root) -> Path | None:
    """Newest network-snapshot-*.pkl under a runs folder (searches all run subfolders)."""
    snaps = sorted(resolve(run_root).glob("*/network-snapshot-*.pkl"), key=lambda p: (p.parent.name, p.name))
    return snaps[-1] if snaps else None


# ---------------------------------------------------------------------------
# Multi-session training (Colab): every resume starts a new StyleGAN run folder whose
# kimg counter restarts at 0, so progress is the sum over run folders.
# ---------------------------------------------------------------------------
_SNAP_RE = re.compile(r"^(?:network-snapshot-|fakes)(\d{6})\.(?:pkl|png)$")


def kimg_done(run_root) -> int:
    """Total kimg trained in a runs folder = Σ over run subfolders of their last snapshot.
    Counts ``fakes*.png`` too, so pruned .pkl files don't lose progress."""
    root = resolve(run_root)
    if not root.exists():
        return 0
    total = 0
    for run in sorted(d for d in root.iterdir() if d.is_dir()):
        ks = [int(m.group(1)) for f in run.iterdir() if (m := _SNAP_RE.match(f.name))]
        total += max(ks, default=0)
    return total


def resume_plan(run_root, start_pkl, target_kimg: int) -> dict:
    """What to run next for one arm: resume from its newest snapshot (or ``start_pkl`` if it
    hasn't started) for the kimg still missing to reach ``target_kimg``."""
    done = kimg_done(run_root)
    latest = latest_snapshot(run_root) if resolve(run_root).exists() else None
    return {"done": done, "remaining": max(0, int(target_kimg) - done),
            "resume": str(latest or start_pkl), "finished": done >= int(target_kimg)}


def prune_snapshots(run_root, keep: int = 2) -> list[Path]:
    """Delete all but the newest ``keep`` network snapshots (they're large; Drive is 15 GB)."""
    snaps = sorted(resolve(run_root).glob("*/network-snapshot-*.pkl"), key=lambda p: (p.parent.name, p.name))
    removed = snaps[:-keep] if keep > 0 else snaps
    for p in removed:
        p.unlink()
    return removed


def build_command(cfg) -> list[str]:
    g = cfg.gan
    data = resolve(g.data or cfg.paths.gan_dataset)
    outdir = resolve(g.outdir or cfg.paths.gan_runs)
    args = [sys.executable, "train.py", f"--outdir={outdir}", f"--data={data}", "--cond=1",
            f"--gpus={g.gpus}", f"--cfg={g.cfg}", f"--kimg={g.kimg}", f"--batch={g.batch}",
            f"--mirror={int(bool(g.mirror))}", f"--aug={g.aug}", f"--snap={g.snap}",
            f"--metrics={g.metrics}", f"--seed={g.seed}", f"--workers={g.workers}"]
    if g.aug == "ada":
        args += [f"--target={g.target}", f"--augpipe={g.augpipe}"]
    if g.gamma is not None:
        args.append(f"--gamma={g.gamma}")
    if g.resume and g.resume != "noresume":
        args.append(f"--resume={resolve(g.resume) if str(g.resume).endswith('.pkl') else g.resume}")
    return args


def domain_loss_spec(cfg) -> dict | None:
    dl = cfg.gan.domain_loss
    if not dl.enabled:
        return None
    from ..data.splits import class_names
    clf = resolve(dl.classifier or cfg.paths.domain_classifier)
    if not clf.exists():
        raise FileNotFoundError(f"domain classifier {clf} not found — run scripts/train_domain_classifier.py first")
    import torch
    meta = torch.load(clf, map_location="cpu", weights_only=False)["meta"]
    from .domain_loss import class_map_for
    cmap = class_map_for(class_names(cfg), meta["classes"], list(cfg.labels.malignant))
    return {"classifier": str(clf), "weight": float(dl.weight), "class_map": cmap}


def train(cfg, dry_run: bool = False) -> int:
    sg = None if dry_run else stylegan_dir(cfg)
    cmd = build_command(cfg)
    env = dict(os.environ, SKINGAN_SRC=str(project_root() / "src"), PYTHONUNBUFFERED="1")
    if not cfg.gan.get("show_warnings", False):
        # 2021-era StyleGAN on new PyTorch prints harmless fallback/DataLoader warnings; hide them
        env["PYTHONWARNINGS"] = "ignore"
    outdir = ensure_dir(cfg.gan.outdir or cfg.paths.gan_runs)
    spec = domain_loss_spec(cfg)
    if spec:
        spec_path = outdir / "domain_loss.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        env["SKINGAN_DOMAIN_LOSS"] = str(spec_path)
    else:
        env.pop("SKINGAN_DOMAIN_LOSS", None)
    save_json({"command": cmd, "domain_loss": spec, "config": cfg.to_dict()}, outdir / "launch.json")
    print("StyleGAN command:\n  " + " ".join(cmd) + ("\n  + SkinGAN domain loss" if spec else "\n  (baseline: no domain loss)"))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=sg, env=env)
