"""Redistribute a fixed brushing-time budget toward high-risk regions.

Rule: every region gets at least ``min_s`` seconds and at most ``max_s``.
The rest of the budget is shared in proportion to risk. When a region hits
its cap, the excess is passed to the others (water-filling). Total time is
always exactly ``total_s``, so guided and uniform brushing are compared at
equal duration.
"""
from __future__ import annotations

import numpy as np


def uniform(n_regions: int, total_s: float = 120.0) -> np.ndarray:
    return np.full(n_regions, total_s / n_regions)


def guided(risk: np.ndarray, total_s: float = 120.0,
           min_s: float | None = None, max_s: float | None = None,
           baseline: float = 0.1) -> np.ndarray:
    """Return seconds per region.

    risk: non-negative risk per region.
    min_s / max_s: per-region bounds (default 50% and 250% of uniform time).
    baseline: added to every risk so zero-risk regions still share time.
    """
    risk = np.asarray(risk, dtype=float)
    n = risk.size
    if n == 0:
        raise ValueError("no regions")
    if np.any(risk < 0):
        raise ValueError("risk must be non-negative")
    u = total_s / n
    min_s = 0.5 * u if min_s is None else min_s
    max_s = 2.5 * u if max_s is None else max_s
    if min_s * n > total_s + 1e-9 or max_s * n < total_s - 1e-9:
        raise ValueError("bounds cannot meet the total time")

    w = risk + baseline
    t = np.full(n, min_s)
    free = np.ones(n, dtype=bool)
    remaining = total_s - t.sum()
    # Water-filling: share remaining time by weight, cap, repeat.
    while remaining > 1e-9 and free.any():
        share = remaining * w[free] / w[free].sum()
        room = max_s - t[free]
        add = np.minimum(share, room)
        idx = np.flatnonzero(free)
        t[idx] += add
        remaining -= add.sum()
        free[idx[add >= room - 1e-12]] = False
    return t


def region_risk(surface_risks: dict, how: str = "mean") -> dict:
    """Combine surface risks into region (e.g. sextant) risks.

    surface_risks: {region_id: [risk, ...]}.
    how: "mean" or "max".
    """
    f = np.mean if how == "mean" else np.max
    return {k: float(f(v)) for k, v in sorted(surface_risks.items())}
