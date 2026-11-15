"""Fetch NVIDIA StyleGAN2-ADA (PyTorch) into third_party/ and apply SkinGAN's two patches.

    python third_party/setup_stylegan.py           # clone (if needed) + patch
    python third_party/setup_stylegan.py --check   # only report patch status

SpineGAN keeps three full copies of StyleGAN2-ADA; we keep one, cloned on demand
(NVIDIA Source Code License — non-commercial research use; see its LICENSE.txt), plus:

1. training/loss.py   — SkinGAN hook: if $SKINGAN_DOMAIN_LOSS points to a spec, the
                        generator's main loss gets  + λ·CE(classifier(G(z,c)), c)  (issue 10).
                        Without the variable the code behaves exactly like the original.
2. torch_utils/misc.py — copy_params_and_buffers skips tensors whose shape differs, so a
                        *conditional* model can start from NVIDIA's *unconditional* FFHQ
                        weights (--resume=ffhq256); the label-embedding layers start fresh.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "stylegan2-ada-pytorch"
REPO = "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
MARK = "# [SkinGAN patch]"

LOSS_IMPORT_ANCHOR = "from torch_utils.ops import conv2d_gradfix\n"
LOSS_IMPORT_ADD = f'''
{MARK} domain-relevant generator loss, enabled by $SKINGAN_DOMAIN_LOSS (see src/skinlesion/gan/domain_loss.py)
import os as _os, sys as _sys
_SKINGAN_DOMAIN = None
if _os.environ.get('SKINGAN_DOMAIN_LOSS'):
    _sys.path.insert(0, _os.environ['SKINGAN_SRC'])
    from skinlesion.gan.domain_loss import DomainLoss as _DomainLoss
    _SKINGAN_DOMAIN = _DomainLoss.from_json(_os.environ['SKINGAN_DOMAIN_LOSS'])
'''
LOSS_GMAIN_OLD = """                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
"""
LOSS_GMAIN_NEW = f"""                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                if _SKINGAN_DOMAIN is not None: {MARK}
                    loss_domain = _SKINGAN_DOMAIN(gen_img, gen_c)
                    training_stats.report('Loss/G/domain', loss_domain)
                    loss_Gmain = loss_Gmain + loss_domain  # after mean(): adversarial + weight * domain
                training_stats.report('Loss/G/loss', loss_Gmain)
"""
MISC_OLD = """        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
"""
MISC_NEW = f"""        if name in src_tensors and src_tensors[name].shape == tensor.shape: {MARK} skip shape mismatches
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
"""


def patch_loss(text: str) -> str:
    if MARK in text:
        return text
    if LOSS_IMPORT_ANCHOR not in text or LOSS_GMAIN_OLD not in text:
        raise RuntimeError("training/loss.py does not match the expected StyleGAN2-ADA version")
    text = text.replace(LOSS_IMPORT_ANCHOR, LOSS_IMPORT_ANCHOR + LOSS_IMPORT_ADD, 1)
    return text.replace(LOSS_GMAIN_OLD, LOSS_GMAIN_NEW, 1)


def patch_misc(text: str) -> str:
    if MARK in text:
        return text
    if MISC_OLD not in text:
        raise RuntimeError("torch_utils/misc.py does not match the expected StyleGAN2-ADA version")
    return text.replace(MISC_OLD, MISC_NEW, 1)


PATCHES = {"training/loss.py": patch_loss, "torch_utils/misc.py": patch_misc}


def apply(target: Path = TARGET, check_only: bool = False) -> dict[str, bool]:
    status = {}
    for rel, fn in PATCHES.items():
        p = target / rel
        text = p.read_text(encoding="utf-8").replace("\r\n", "\n")  # Windows clones may use CRLF
        status[rel] = MARK in text
        if not check_only and not status[rel]:
            p.write_text(fn(text), encoding="utf-8")
            status[rel] = True
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not (TARGET / "train.py").exists():
        if args.check:
            sys.exit(f"not installed: {TARGET}")
        print(f"cloning {REPO} -> {TARGET}")
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(TARGET)], check=True)
    for rel, ok in apply(check_only=args.check).items():
        print(f"  {'patched' if ok else 'NOT patched'}: {rel}")
    print("StyleGAN2-ADA ready:", TARGET)


if __name__ == "__main__":
    main()
