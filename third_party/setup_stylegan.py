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
                        InfiniteSampler no longer passes `dataset` to Sampler.__init__, which
                        newer PyTorch rejects (TypeError: object.__init__() takes exactly one argument).
3. torch_utils/custom_ops.py — use the module returned by cpp_extension.load(); newer PyTorch
                        no longer makes it importable by name ("No module named 'upfirdn2d_plugin'").
4. torch_utils/ops/upfirdn2d.py — set _inited so a failed CUDA build is not retried on every
                        call (which floods the log and stalls training).
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
# Newer PyTorch removed Sampler.__init__(data_source); StyleGAN's InfiniteSampler still passes it
# ("TypeError: object.__init__() takes exactly one argument").
SAMPLER_OLD = "        super().__init__(dataset)\n"
SAMPLER_NEW = f"        super().__init__() {MARK} newer PyTorch: Sampler takes no data_source\n"

# Newer PyTorch's cpp_extension.load() builds the CUDA plugin and *returns* it, but no longer
# makes it importable by name, so importlib.import_module fails ("No module named
# 'upfirdn2d_plugin'") even though the build worked. Use the returned module instead.
OPS_LOAD1_OLD = "            torch.utils.cpp_extension.load(name=module_name, build_directory=build_dir,\n"
OPS_LOAD1_NEW = f"            module = torch.utils.cpp_extension.load(name=module_name, build_directory=build_dir, {MARK}\n"
OPS_LOAD2_OLD = "            torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)\n"
OPS_LOAD2_NEW = f"            module = torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs) {MARK}\n"
OPS_IMPORT_OLD = "        module = importlib.import_module(module_name)\n"
OPS_IMPORT_NEW = f"        {MARK} module comes from cpp_extension.load() above (import by name fails on newer PyTorch)\n"
# upfirdn2d._init never sets _inited (bias_act does), so a failed build is retried on every
# call, flooding the log with tracebacks and stalling training. Mark it as tried once.
UPFIRDN_OLD = "    global _inited, _plugin\n    if not _inited:\n"
UPFIRDN_NEW = f"    global _inited, _plugin\n    if not _inited:\n        _inited = True {MARK} try the build only once\n"

# file -> list of (original text, patched text); each patch is applied independently, so new
# patches also reach clones that were patched by an older version of this script
PATCHES = {
    "training/loss.py": [(LOSS_IMPORT_ANCHOR, LOSS_IMPORT_ANCHOR + LOSS_IMPORT_ADD), (LOSS_GMAIN_OLD, LOSS_GMAIN_NEW)],
    "torch_utils/misc.py": [(MISC_OLD, MISC_NEW), (SAMPLER_OLD, SAMPLER_NEW)],
    "torch_utils/custom_ops.py": [(OPS_LOAD1_OLD, OPS_LOAD1_NEW), (OPS_LOAD2_OLD, OPS_LOAD2_NEW),
                                  (OPS_IMPORT_OLD, OPS_IMPORT_NEW)],
    "torch_utils/ops/upfirdn2d.py": [(UPFIRDN_OLD, UPFIRDN_NEW)],
}


def patch_text(text: str, pairs, name: str = "file") -> str:
    for old, new in pairs:
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(f"{name} does not match the expected StyleGAN2-ADA version")
        text = text.replace(old, new, 1)
    return text


def patch_loss(text: str) -> str:
    return patch_text(text, PATCHES["training/loss.py"], "training/loss.py")


def patch_misc(text: str) -> str:
    return patch_text(text, PATCHES["torch_utils/misc.py"], "torch_utils/misc.py")


def apply(target: Path = TARGET, check_only: bool = False) -> dict[str, bool]:
    status = {}
    for rel, pairs in PATCHES.items():
        p = target / rel
        text = p.read_text(encoding="utf-8").replace("\r\n", "\n")  # Windows clones may use CRLF
        done = all(new in text for _, new in pairs)
        if not check_only and not done:
            p.write_text(patch_text(text, pairs, rel), encoding="utf-8")
            done = True
        status[rel] = done
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
