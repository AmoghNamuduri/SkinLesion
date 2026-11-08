"""The three global feature descriptors named in the paper.

* colour  — HSV colour histogram (bins³ values, default 8×8×8 = 512)
* shape   — 7 Hu moments of the grayscale image
* texture — 13 Haralick features (mean over the 4 GLCM directions)

Each function takes a BGR uint8 image and returns a 1-D float vector. Add a new
descriptor by writing a function with the same signature and registering it in
``DESCRIPTORS``.
"""
from __future__ import annotations

import cv2
import mahotas
import numpy as np


def color_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def hu_moments(image: np.ndarray, **_) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(gray)).flatten()


def haralick_texture(image: np.ndarray, **_) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray).mean(axis=0)


# Order matches the reference implementation: histogram, haralick, hu
DESCRIPTORS = {
    "color_histogram": color_histogram,
    "haralick": haralick_texture,
    "hu_moments": hu_moments,
}
