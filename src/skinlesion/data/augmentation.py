"""Augmentation used by the paper.

* ``hflip_offline`` — Step 3: every image gets a horizontally-flipped copy,
  turning 100 images/class into 200 images/class (used by both pipelines).
* ``KerasStyleAugmenter`` — on-the-fly random transforms for the CNN, a
  re-implementation of Keras' ``ImageDataGenerator`` (rotation, width/height
  shift, shear, zoom, horizontal flip, ``fill_mode='nearest'``).
"""
from __future__ import annotations

import cv2
import numpy as np


def hflip(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, ::-1])


def hflip_offline(images: np.ndarray, labels: np.ndarray, groups: np.ndarray):
    """Return originals followed by their mirrored copies, plus an ``is_augmented`` mask."""
    flipped = np.stack([hflip(im) for im in images])
    images2 = np.concatenate([images, flipped])
    labels2 = np.concatenate([labels, labels])
    groups2 = np.concatenate([groups, groups])  # twin keeps the original's group id
    is_aug = np.concatenate([np.zeros(len(images), bool), np.ones(len(images), bool)])
    return images2, labels2, groups2, is_aug


class KerasStyleAugmenter:
    """Random affine transform matching ``keras.preprocessing.image.ImageDataGenerator``."""

    _BORDER = {"nearest": cv2.BORDER_REPLICATE, "constant": cv2.BORDER_CONSTANT,
               "reflect": cv2.BORDER_REFLECT, "wrap": cv2.BORDER_WRAP}

    def __init__(self, rotation_range=0.0, width_shift_range=0.0, height_shift_range=0.0,
                 shear_range=0.0, zoom_range=0.0, horizontal_flip=False, fill_mode="nearest", seed=None):
        self.rotation_range = float(rotation_range)
        self.width_shift_range = float(width_shift_range)
        self.height_shift_range = float(height_shift_range)
        self.shear_range = float(shear_range)
        self.zoom_range = float(zoom_range)
        self.horizontal_flip = bool(horizontal_flip)
        self.border = self._BORDER[fill_mode]
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, aug_cfg, seed=None) -> "KerasStyleAugmenter":
        return cls(**dict(aug_cfg), seed=seed)

    def _matrix(self, h: int, w: int) -> np.ndarray:
        r = self.rng
        m = np.eye(3)
        if self.rotation_range:
            t = np.deg2rad(r.uniform(-self.rotation_range, self.rotation_range))
            m = m @ np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
        tx = r.uniform(-self.width_shift_range, self.width_shift_range) * w if self.width_shift_range else 0.0
        ty = r.uniform(-self.height_shift_range, self.height_shift_range) * h if self.height_shift_range else 0.0
        if tx or ty:
            m = m @ np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        if self.shear_range:  # Keras: shear angle in degrees
            s = np.deg2rad(r.uniform(-self.shear_range, self.shear_range))
            m = m @ np.array([[1, -np.sin(s), 0], [0, np.cos(s), 0], [0, 0, 1]])
        if self.zoom_range:  # Keras draws x and y zoom independently
            zx, zy = r.uniform(1 - self.zoom_range, 1 + self.zoom_range, 2)
            m = m @ np.diag([zx, zy, 1.0])
        c = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]])
        return c @ m @ np.linalg.inv(c)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        m = self._matrix(h, w)
        out = cv2.warpAffine(image, m[:2], (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=self.border)
        if self.horizontal_flip and self.rng.random() < 0.5:
            out = hflip(out)
        return out
