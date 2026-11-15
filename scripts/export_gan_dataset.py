"""Step 1 of SkinGAN — lesion-level split + StyleGAN training archive (issues 03, 04).

    python scripts/export_gan_dataset.py
    python scripts/export_gan_dataset.py --set labels.set=binary paths.gan_dataset=data/gan/ham10000_256_binary.zip
    python scripts/export_gan_dataset.py --set gan_data.max_per_class=300      # small test archive

Also writes a contact sheet of real images per class next to the zip.
"""
from _bootstrap import SKINGAN_CONFIG, parse_args

from skinlesion.data.metadata import load_metadata
from skinlesion.gan.dataset_export import export_stylegan_dataset
from skinlesion.gan.visualize import real_class_grid, save_rgb
from skinlesion.utils.io import resolve

if __name__ == "__main__":
    _, cfg = parse_args(__doc__, default_config=SKINGAN_CONFIG)
    meta = load_metadata(cfg.paths.raw_dir)
    info = export_stylegan_dataset(cfg, meta=meta)
    print(f"\n{info['images']} images at {info['resolution']}² -> {info['path']}")
    for c, n in info["counts"].items():
        print(f"  {c:10s} {n}")
    sheet = save_rgb(real_class_grid(info["path"]), resolve(info["path"]).with_name("real_samples.png"))
    print("contact sheet ->", sheet)
    print("splits ->", resolve(cfg.paths.splits_dir))
