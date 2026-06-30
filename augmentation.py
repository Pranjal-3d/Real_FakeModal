"""Lightweight image augmentation for small phone-captured datasets."""

import cv2
import numpy as np


def augment_image(img_bgr, seed=0):
    """Apply random photometric + geometric aug; deterministic per seed."""
    rng = np.random.default_rng(seed)
    out = img_bgr.copy()

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    angle = float(rng.uniform(-8, 8))
    h, w = out.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    alpha = float(rng.uniform(0.85, 1.15))
    beta = float(rng.uniform(-18, 18))
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)

    if rng.random() < 0.25:
        noise = rng.normal(0, 6, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return out


def augment_variants(img_bgr, count=2):
    """Return list of augmented copies (does not include original)."""
    return [augment_image(img_bgr, seed=i * 9973 + 42) for i in range(count)]
