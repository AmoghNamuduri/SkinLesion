"""The customised CNN of Table 2, re-implemented in PyTorch.

The paper builds it in Keras; to stay numerically close we copy Keras defaults:
Glorot-uniform weights, zero biases, BatchNorm momentum 0.99 (torch 0.01) and
epsilon 1e-3, and the Keras ordering Conv → ReLU → BatchNorm.

    input 96×96×3
    [Conv32]            → MaxPool 3×3 (96→32) → Dropout .25
    [Conv64, Conv64]    → MaxPool 2×2 (32→16) → Dropout .25
    [Conv128, Conv128]  → MaxPool 2×2 (16→8)  → Dropout .25
    Flatten → Dense 1024 → ReLU → BN → Dropout .5 → Dense 7 (softmax in the loss)

The block list comes from ``cnn.architecture`` in the config, so deeper or
wider variants need only a YAML change.
"""
from __future__ import annotations

import torch
from torch import nn


def _bn(n: int, dims: int = 2) -> nn.Module:
    cls = nn.BatchNorm2d if dims == 2 else nn.BatchNorm1d
    return cls(n, eps=1e-3, momentum=0.01)


class ConvBlock(nn.Module):
    """One or more (Conv3×3 same → ReLU → BN) layers, then MaxPool and Dropout."""

    def __init__(self, in_ch: int, filters: list[int], pool: int, dropout: float, batch_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        for f in filters:
            layers += [nn.Conv2d(in_ch, f, 3, padding=1), nn.ReLU(inplace=True)]
            if batch_norm:
                layers.append(_bn(f))
            in_ch = f
        layers += [nn.MaxPool2d(pool), nn.Dropout(dropout)]
        self.body = nn.Sequential(*layers)
        self.out_channels = in_ch
        self.pool = pool

    def forward(self, x):
        return self.body(x)


class SkinLesionCNN(nn.Module):
    def __init__(self, num_classes: int = 7, image_size: int = 96, conv_blocks=None,
                 dense_units: int = 1024, dense_dropout: float = 0.5, batch_norm: bool = True):
        super().__init__()
        conv_blocks = conv_blocks or [
            {"filters": [32], "pool": 3, "dropout": 0.25},
            {"filters": [64, 64], "pool": 2, "dropout": 0.25},
            {"filters": [128, 128], "pool": 2, "dropout": 0.25},
        ]
        blocks, ch, size = [], 3, image_size
        for b in conv_blocks:
            blk = ConvBlock(ch, list(b["filters"]), int(b["pool"]), float(b["dropout"]), batch_norm)
            blocks.append(blk)
            ch, size = blk.out_channels, size // blk.pool
        self.features = nn.Sequential(*blocks)
        head = [nn.Flatten(), nn.Linear(ch * size * size, dense_units), nn.ReLU(inplace=True)]
        if batch_norm:
            head.append(_bn(dense_units, dims=1))
        head += [nn.Dropout(dense_dropout), nn.Linear(dense_units, num_classes)]
        self.classifier = nn.Sequential(*head)
        self.apply(_keras_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # returns logits
        return self.classifier(self.features(x))


def _keras_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def build_cnn(cfg, num_classes: int) -> SkinLesionCNN:
    a = cfg.cnn.architecture
    return SkinLesionCNN(
        num_classes=num_classes,
        image_size=cfg.cnn.image_size,
        conv_blocks=[dict(b) for b in a.conv_blocks],
        dense_units=a.dense_units,
        dense_dropout=a.dense_dropout,
        batch_norm=a.batch_norm,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
