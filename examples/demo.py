"""End-to-end walk-through with simulated inputs.

Shows how the steps connect: geometry -> risk -> brushing time -> paired analysis.

Run from the repository root:  python examples/demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from oraltwin_dwell import (Surface, risk_score, region_risk, uniform, guided,
                            randomize_order, paired_compare, detectable_dz)

SEXTANTS = {1: "upper right", 2: "upper anterior", 3: "upper left",
            4: "lower left", 5: "lower anterior", 6: "lower right"}

# 1. A made-up arch: well aligned, except crowded lower front teeth.
surfaces = []
for quadrant in (1, 2, 3, 4):
    for pos in range(1, 8):
        tooth = quadrant * 10 + pos
        crowded = quadrant in (3, 4) and pos <= 3
        surfaces.append(Surface(tooth, "lingual", {
            "crowding_mm": 3.0 if crowded else 0.3,
            "angulation_deg": 25.0 if crowded else 3.0,
            "interproximal_space_mm": 0.1 if crowded else 0.9,
            "surface_curvature": 0.6 if crowded else 0.2,
        }))

# 2. Risk per sextant, then seconds per sextant for a 2-minute session.
by_sextant = {}
for s in surfaces:
    by_sextant.setdefault(s.sextant, []).append(risk_score(s))
risk = region_risk(by_sextant)
t_uniform = uniform(6, total_s=120)
t_guided = guided(np.array(list(risk.values())), total_s=120)

print("Example run with simulated inputs\n")
print(f"{'sextant':<16}{'risk':>6}{'uniform s':>11}{'guided s':>10}")
for (k, r), tu, tg in zip(risk.items(), t_uniform, t_guided):
    print(f"{SEXTANTS[k]:<16}{r:>6.2f}{tu:>11.1f}{tg:>10.1f}")
print(f"{'total':<16}{'':>6}{t_uniform.sum():>11.1f}{t_guided.sum():>10.1f}")

# 3. Aim 3 design: 45 paired models, order randomized with a recorded seed.
ids = [f"M{i:02d}" for i in range(1, 46)]
order = randomize_order(ids, seed=20260101)
print("\nfirst three models, brushing order:",
      {m: order[m] for m in ids[:3]})
print(f"smallest effect 45 pairs can detect (80% power): dz = {detectable_dz(45):.2f}")

# 4. Analysis on simulated outcomes (assumed 15% relative reduction).
rng = np.random.default_rng(0)
u = np.clip(rng.normal(0.30, 0.06, 45), 0, 1)
g = np.clip(u * 0.85 + rng.normal(0, 0.04, 45), 0, 1)
res = paired_compare(u, g)
print("\nsimulated paired result:")
for k, v in res.items():
    print(f"  {k}: {v}")
