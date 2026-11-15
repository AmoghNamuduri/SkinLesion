"""Train the baseline GAN or SkinGAN with StyleGAN2-ADA (issues 02, 06, 11). Needs a CUDA GPU.

    python scripts/train_gan.py                                           # baseline
    python scripts/train_gan.py --config configs/skingan/skingan.yaml     # SkinGAN (domain loss)
    python scripts/train_gan.py --set gan.kimg=200 gan.snap=5             # short test
    python scripts/train_gan.py --set gan.resume=outputs/gan/baseline/00000-.../network-snapshot-000200.pkl
    python scripts/train_gan.py --dry-run                                 # print the StyleGAN command only
"""
import sys

from _bootstrap import SKINGAN_CONFIG, parse_args

from skinlesion.gan.train_gan import train


def _extra(p):
    p.add_argument("--dry-run", action="store_true", help="print the command without training")


if __name__ == "__main__":
    args, cfg = parse_args(__doc__, _extra, default_config=SKINGAN_CONFIG)
    sys.exit(train(cfg, dry_run=args.dry_run))
