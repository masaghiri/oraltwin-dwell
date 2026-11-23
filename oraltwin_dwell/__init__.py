"""oraltwin-dwell: anatomy-guided brushing time allocation and evaluation."""
from .risk import Surface, risk_score, fit_weights
from .allocate import uniform, guided, region_risk
from .plaque import plaque_mask, residual_fraction, removal_percent, dice
from .stats import (randomize_order, paired_compare, power_paired,
                    n_for_paired, detectable_dz, mixed_model)

__version__ = "0.1.0"
