"""Surface-level plaque-retention risk from dentition geometry.

Two routes, matching the two-step logic of the study:

1. ``risk_score`` - a transparent geometric score. Every value can be
   traced back to its inputs. It needs no training data, so it also serves
   as the pre-specified fallback for Aim 3 if the learned model (Aim 2)
   is not accurate enough.
2. ``oraltwin_dwell.model.HybridRisk`` - Random Forest + Support Vector
   Regression trained on measured residual plaque (Aim 2.3).

Features are the anatomical descriptors named in the Research Strategy:
crowding, angulation, interproximal spacing, crown width, surface curvature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# name -> (low, high, direction). Bounds are physical ranges, not dataset
# statistics, so scores are comparable across models.
# direction +1: larger value = harder to clean. -1: smaller value = harder.
FEATURES = {
    "crowding_mm": (0.0, 4.0, +1),            # local arch-length deficit
    "angulation_deg": (0.0, 45.0, +1),        # rotation / tip from ideal
    "interproximal_space_mm": (0.0, 1.5, -1), # narrow contact = poor access
    "surface_curvature": (0.0, 1.0, +1),      # normalized concavity, 0-1
}
# Crown width is recorded and passed to the learned model, but it has no
# agreed direction of effect, so it is left out of the transparent score.
MODEL_FEATURES = list(FEATURES) + ["crown_width_mm"]

DEFAULT_WEIGHTS = {
    "crowding_mm": 0.35,
    "angulation_deg": 0.25,
    "interproximal_space_mm": 0.25,
    "surface_curvature": 0.15,
}

# Value used when a feature was not measured: the "no added risk" end.
NEUTRAL = {k: (lo if d > 0 else hi) for k, (lo, hi, d) in FEATURES.items()}


@dataclass
class Surface:
    tooth: int            # FDI tooth number, e.g. 11, 36
    surface: str          # "buccal", "lingual", "mesial", "distal", "occlusal"
    features: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.tooth}-{self.surface}"

    @property
    def sextant(self) -> int:
        """Sextant 1-6 (upper right, upper anterior, upper left, lower left,
        lower anterior, lower right)."""
        quadrant, pos = divmod(self.tooth, 10)
        if not 1 <= pos <= 8 or quadrant not in (1, 2, 3, 4):
            raise ValueError(f"invalid permanent FDI tooth number: {self.tooth}")
        anterior = pos <= 3
        return {1: 2 if anterior else 1, 2: 2 if anterior else 3,
                3: 5 if anterior else 4, 4: 5 if anterior else 6}[quadrant]


def normalize(name: str, value: float) -> float:
    """Map a feature to 0-1, where 1 is the hardest-to-clean end."""
    lo, hi, direction = FEATURES[name]
    x = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    return x if direction > 0 else 1.0 - x


def risk_score(surface: Surface, weights: dict | None = None) -> float:
    """Geometric risk in [0, 1]. Unmeasured features add no risk."""
    w = weights or DEFAULT_WEIGHTS
    s = sum(w[k] * normalize(k, surface.features.get(k, NEUTRAL[k])) for k in w)
    return s / sum(w.values())


def fit_weights(X: np.ndarray, residual_plaque: np.ndarray,
                names: list[str] | None = None) -> dict:
    """Re-estimate the score weights from measured residual plaque.

    X: (n_surfaces, n_features) of *normalized* features (see ``normalize``).
    residual_plaque: (n_surfaces,) plaque fraction left after uniform brushing.
    Non-negative least squares keeps each weight interpretable: more of a
    risk feature can never lower the score.
    """
    from scipy.optimize import nnls

    names = names or list(DEFAULT_WEIGHTS)
    if X.shape[1] != len(names):
        raise ValueError("X columns must match feature names")
    coef, _ = nnls(X, residual_plaque)
    if coef.sum() == 0:
        raise ValueError("no feature explains residual plaque; check data")
    return {n: float(c) for n, c in zip(names, coef)}
