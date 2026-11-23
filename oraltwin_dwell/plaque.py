"""Residual plaque analog from UV fluorescence images.

The plaque analog (Glo Germ) glows under UV light while a clean printed
model stays dark. A pixel counts as plaque when its brightness is above a
threshold. The threshold comes from Otsu's method on the tooth pixels, or
is fixed from calibration images.

This is a simple, fully transparent measure. It is meant as an independent
cross-check of the DeepLabV3+ segmentation, not a replacement for it.
"""
from __future__ import annotations

import numpy as np


def brightness(rgb: np.ndarray) -> np.ndarray:
    """Max channel, 0-255. Works for any emission colour."""
    return np.asarray(rgb, dtype=float).max(axis=-1)


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu's threshold for values in 0-255."""
    hist, edges = np.histogram(values, bins=256, range=(0, 256))
    p = hist / max(hist.sum(), 1)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    m0 = np.cumsum(p * mids)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0
    return float(edges[int(between.argmax()) + 1])


def plaque_mask(rgb: np.ndarray, region: np.ndarray,
                threshold: float | None = None,
                min_contrast: float = 30.0) -> np.ndarray:
    """Boolean plaque mask inside ``region``.

    threshold: fixed brightness cut-off. If None, Otsu is used.
    min_contrast: if the bright and dark groups differ by less than this,
    the region is treated as one class (all clean or all coated) and is
    classified against the 0-255 midpoint, so Otsu cannot invent plaque on
    a clean surface.
    """
    region = np.asarray(region, dtype=bool)
    if not region.any():
        raise ValueError("empty region mask")
    b = brightness(rgb)
    vals = b[region]
    if threshold is None:
        t = otsu_threshold(vals)
        hi, lo = vals[vals >= t], vals[vals < t]
        if hi.size == 0 or lo.size == 0 or hi.mean() - lo.mean() < min_contrast:
            t = 127.5
        threshold = t
    return region & (b >= threshold)


def residual_fraction(rgb: np.ndarray, region: np.ndarray, **kw) -> float:
    """Fraction of a region's pixels that still show plaque analog."""
    region = np.asarray(region, dtype=bool)
    return float(plaque_mask(rgb, region, **kw).sum() / region.sum())


def removal_percent(before: float, after: float) -> float:
    """Plaque removal (%) for one region; NaN if there was none before."""
    return float("nan") if before == 0 else 100.0 * (before - after) / before


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Agreement between two plaque masks (e.g. this method vs DeepLabV3+)."""
    a, b = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)
