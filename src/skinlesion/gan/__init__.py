"""SkinGAN: our code around the vendored NVIDIA StyleGAN2-ADA (third_party/stylegan2-ada-pytorch).

dataset_export  HAM10000 split → 256² StyleGAN training archive with class labels   (issue 04)
train_gan       YAML config → StyleGAN train.py launcher (baseline or SkinGAN)      (issue 02)
domain_loss     the SkinGAN domain-relevant generator loss                           (issue 10)
generate        .pkl → synthetic images per class + manifest.csv                    (issue 08)
gan_metrics     FID / KID / precision-recall wrapper                                 (issue 07)
visualize       image grids: real vs generated, per class, training progression
dp_training     correct DP-SGD step + ε accounting (stretch)                         (issue 16)

Modules that need StyleGAN or torch import them lazily, so the data-only parts run anywhere.
"""
