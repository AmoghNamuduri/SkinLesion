"""SkinGAN domain-relevant generator loss (roadmap issue 10).

    L_G = softplus(-D(G(z, c)))  +  λ · CE( C(G(z, c)), c )        C frozen

This is SpineGAN's idea (their ``domain-loss-stylegan2-ada-pytorch-main/training/loss.py``),
with its problems fixed:
* generator output in [-1, 1] is mapped to [0, 1] *before* ImageNet normalisation
* the classifier is resized to its input size instead of a hard-coded pixel crop
* the classifier is put in eval mode with frozen weights (gradients still reach G)
* path, λ and device come from config instead of being hard-coded

The vendored StyleGAN ``training/loss.py`` is patched (third_party/setup_stylegan.py) to
build a DomainLoss from the JSON file named in ``$SKINGAN_DOMAIN_LOSS`` and add its value
to the generator's main loss.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch import nn

from ..models.densenet import normalize_imagenet


class DomainLoss(nn.Module):
    def __init__(self, classifier: nn.Module, weight: float = 0.1, image_size: int = 224,
                 class_map: list[int] | None = None):
        """``class_map[k]`` = classifier target for GAN class k (e.g. 7-class GAN → binary
        malignant classifier). ``None`` means GAN classes and classifier classes are identical."""
        super().__init__()
        self.classifier = classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        self.weight = float(weight)
        self.image_size = int(image_size)
        self.register_buffer("class_map", torch.tensor(class_map) if class_map is not None else torch.empty(0, dtype=torch.long))

    def train(self, mode: bool = True):  # the classifier stays in eval mode (BatchNorm / dropout frozen)
        super().train(mode)
        self.classifier.eval()
        return self

    def preprocess(self, gen_img: torch.Tensor) -> torch.Tensor:
        x = (gen_img.float().clamp(-1, 1) + 1) / 2
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return normalize_imagenet(x)

    def forward(self, gen_img: torch.Tensor, gen_c: torch.Tensor) -> torch.Tensor:
        if next(self.classifier.parameters()).device != gen_img.device:
            self.to(gen_img.device)
        target = gen_c.argmax(dim=1)
        if self.class_map.numel():
            target = self.class_map.to(gen_img.device)[target]
        logits = self.classifier(self.preprocess(gen_img))
        if logits.shape[1] == 1:
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], target.float())
        else:
            loss = F.cross_entropy(logits, target)
        return self.weight * loss

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str) -> "DomainLoss":
        """Build from the spec written by the launcher: {classifier, weight, class_map}."""
        from ..models.densenet import load_classifier

        with open(path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
        model, meta = load_classifier(spec["classifier"])
        loss = cls(model, spec.get("weight", 0.1), meta.get("image_size", 224), spec.get("class_map"))
        print(f"[SkinGAN] domain loss on: λ={loss.weight}, classifier={spec['classifier']}, "
              f"outputs={meta['num_classes']}, class_map={spec.get('class_map')}")
        return loss


def class_map_for(gan_classes: list[str], clf_classes: list[str], malignant: list[str]) -> list[int] | None:
    """Map GAN class indices to classifier targets when the label sets differ."""
    if gan_classes == clf_classes:
        return None
    if clf_classes == ["benign", "malignant"]:
        return [int(c in malignant) for c in gan_classes]
    raise ValueError(f"cannot map GAN classes {gan_classes} to classifier classes {clf_classes}")
