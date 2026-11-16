"""Memorisation audit: are synthetic images copies of training patients? (roadmap issue 15)

For every synthetic image we find its nearest *training* image in feature space and compare
that distance with the same nearest-training distance for *real test* images. Test images
were never trained on, so their distances show what "a new, different lesion" looks like.
If many synthetic images sit much closer to a training image than test images ever do,
the GAN is reproducing training data.
"""
from __future__ import annotations

import numpy as np


def embed(images: np.ndarray, method: str = "densenet", batch: int = 64, device: str = "auto") -> np.ndarray:
    """images: (N,H,W,3) uint8 RGB → L2-normalised features.
    ``densenet``: ImageNet DenseNet-121 pooled features; ``pixels``: 32×32 thumbnails (fast, for tests)."""
    if method == "pixels":
        import cv2
        f = np.stack([cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).reshape(-1) for im in images]).astype(np.float32)
    else:
        import torch
        import torch.nn.functional as F
        from ..models.densenet import feature_extractor, normalize_imagenet

        dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        net = feature_extractor(pretrained=True).to(dev)
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                x = torch.from_numpy(images[i:i + batch]).permute(0, 3, 1, 2).float().div(255).to(dev)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                feats.append(net(normalize_imagenet(x)).cpu().numpy())
        f = np.concatenate(feats).astype(np.float32)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)


def nearest_neighbors(query: np.ndarray, reference: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distance (1 − cos) from each query row to its nearest reference row, and that row's index."""
    dist, idx = np.empty(len(query), np.float32), np.empty(len(query), np.int64)
    for i in range(0, len(query), chunk):
        sim = query[i:i + chunk] @ reference.T
        j = sim.argmax(1)
        idx[i:i + chunk], dist[i:i + chunk] = j, 1 - sim[np.arange(len(j)), j]
    return dist, idx


def audit(syn_feats: np.ndarray, train_feats: np.ndarray, test_feats: np.ndarray, quantile: float = 0.01) -> dict:
    """Compare synthetic→train with test→train nearest-neighbour distances.

    ``threshold`` = the ``quantile`` (1 %) of test→train distances: closer than almost any
    genuinely new lesion. ``frac_below`` is the share of synthetic images under it
    (≈ quantile if the GAN doesn't copy; much higher means memorisation)."""
    d_syn, i_syn = nearest_neighbors(syn_feats, train_feats)
    d_test, _ = nearest_neighbors(test_feats, train_feats)
    thr = float(np.quantile(d_test, quantile))
    return {"syn_median": float(np.median(d_syn)), "test_median": float(np.median(d_test)),
            "threshold": thr, "quantile": quantile, "frac_below": float((d_syn < thr).mean()),
            "closest": [(int(k), int(i_syn[k]), float(d_syn[k])) for k in np.argsort(d_syn)[:16]]}
