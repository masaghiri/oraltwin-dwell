# oraltwin-dwell

Methods code for anatomy-guided brushing: turn tooth geometry into a plaque-retention
risk map, turn the risk map into brushing time per region, and compare guided with
uniform brushing at equal total time.

**Status:** work in progress. This repository holds the methods code for the OralTwin
project. Study data and results will be added as the work proceeds.

## What it does

| Step | Module | In short |
|---|---|---|
| Risk from geometry | `risk.py` | Transparent score (0-1) from crowding, angulation, interproximal space and surface curvature. Needs no training data. |
| Learned risk | `model.py` | Random Forest + Support Vector Regression trained on measured residual plaque. Whole dental models are held out during validation, so there is no leakage between training and test surfaces. |
| Time allocation | `allocate.py` | Shares a fixed time budget in proportion to risk, with a floor and a cap per region. Total time never changes. |
| Plaque measurement | `plaque.py` | Fraction of a region still glowing under UV (fluorescent plaque analog). A simple cross-check for a deep-learning segmentation, with a Dice score to compare the two. |
| Design and analysis | `stats.py` | Balanced randomized order, paired t-test and Wilcoxon, effect size with 95% CI, power and sample size, mixed-effects model. |

## The allocation rule

Every region gets at least 50% and at most 250% of its uniform share (both adjustable).
The remaining time is shared in proportion to risk. If a region reaches its cap, the
extra time passes to the other regions. The total stays fixed (for example 120 s), so a
difference in plaque removal comes from where the time is spent, not from brushing longer.

## Pre-specified fallback

`model.accurate_enough()` is a go/no-go check on the learned model. If the learned model
does not predict held-out dental models well enough, the guided-brushing experiment runs
on the transparent geometric score instead. The experiment therefore does not depend on
the learned model succeeding.

## Sample size

For a two-sided paired t-test at alpha 0.05 and 80% power:

| Standardized paired effect (dz) | Pairs needed |
|---|---|
| 0.30 | 90 |
| 0.40 | 52 |
| 0.50 | 34 |
| 0.80 | 15 |

With 45 pairs, the smallest detectable effect is dz = 0.43.

## Use

```
pip install numpy scipy scikit-learn pandas statsmodels pytest
python -m pytest
python examples/demo.py
```

```python
import numpy as np
from oraltwin_dwell import guided

risk = np.array([0.05, 0.10, 0.05, 0.10, 0.80, 0.10])   # six sextants
print(guided(risk, total_s=120))                         # seconds per sextant
```

## Limits

- Feature ranges and default weights are starting values. They must be re-estimated
  from measured data (`risk.fit_weights`).
- The plaque threshold must be calibrated on the imaging setup used.
- Bench results on printed models do not predict clinical plaque removal.
