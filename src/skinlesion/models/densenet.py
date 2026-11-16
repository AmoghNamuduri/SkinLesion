"""ImageNet DenseNet-121 classifier (roadmap issue 09).

Used twice, as in SpineGAN: trained on real images it is the frozen *domain classifier*
inside the SkinGAN loss; trained on synthetic images it is the *synthetic-learning* test.
Unlike SpineGAN (DenseNet → Linear(1000→1) → Sigmoid) we replace the ImageNet head and
return logits, so the same model serves binary (1 output) and 7-class tasks.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_densenet(num_classes: int, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    """DenseNet-121 with a fresh head: 1 output for binary tasks, ``num_classes`` otherwise."""
    from torchvision.models import densenet121

    model = densenet121(weights="IMAGENET1K_V1" if pretrained else None)
    out = 1 if num_classes == 2 else num_classes
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier.in_features, out))
    return model


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """x: (N,3,H,W) in [0,1] → ImageNet-normalised."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    """(N,1) logits → (N,2) probabilities; (N,K) → softmax."""
    if logits.shape[1] == 1:
        p = torch.sigmoid(logits)
        return torch.cat([1 - p, p], dim=1)
    return torch.softmax(logits, dim=1)


def save_classifier(model: nn.Module, path, meta: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, p)
    return p


def load_classifier(path, map_location="cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = ckpt["meta"]
    model = build_densenet(meta["num_classes"], pretrained=False, dropout=meta.get("dropout", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), meta


def feature_extractor(pretrained: bool = True) -> nn.Module:
    """DenseNet-121 trunk → 1024-d pooled features (used by the memorisation audit)."""
    from torchvision.models import densenet121

    m = densenet121(weights="IMAGENET1K_V1" if pretrained else None)
    return nn.Sequential(m.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
