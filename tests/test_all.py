import numpy as np
import pandas as pd
import pytest

from oraltwin_dwell import (Surface, risk_score, fit_weights, uniform, guided,
                            plaque_mask, residual_fraction, removal_percent,
                            dice, randomize_order, paired_compare,
                            power_paired, n_for_paired, detectable_dz,
                            mixed_model)
from oraltwin_dwell.risk import FEATURES, normalize
from oraltwin_dwell.model import HybridRisk, cross_validate, accurate_enough


# ---- risk ----
def test_sextants():
    got = [Surface(t, "buccal").sextant for t in (17, 11, 26, 36, 41, 47)]
    assert got == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        Surface(19, "buccal").sextant


def test_risk_bounds_and_direction():
    assert risk_score(Surface(11, "mesial")) == 0
    worst = {k: (hi * 2 if d > 0 else lo) for k, (lo, hi, d) in FEATURES.items()}
    assert risk_score(Surface(13, "mesial", worst)) == pytest.approx(1.0)
    # A narrower contact is harder to clean.
    tight = Surface(12, "mesial", {"interproximal_space_mm": 0.1})
    open_ = Surface(12, "mesial", {"interproximal_space_mm": 1.4})
    assert risk_score(tight) > risk_score(open_)
    assert normalize("crowding_mm", 2.0) == pytest.approx(0.5)


def test_fit_weights_recovers_signal():
    rng = np.random.default_rng(0)
    X = rng.random((300, 4))
    true = np.array([0.5, 0.0, 0.3, 0.2])
    y = X @ true + rng.normal(0, 0.01, 300)
    assert list(fit_weights(X, y).values()) == pytest.approx(true, abs=0.03)


# ---- allocation ----
def test_guided_keeps_total_and_bounds():
    t = guided(np.array([0.9, 0.1, 0.0, 0.5, 0.0, 0.2]), total_s=120)
    assert t.sum() == pytest.approx(120)
    assert t.min() >= 10 - 1e-9 and t.max() <= 50 + 1e-9
    assert t.argmax() == 0
    assert uniform(6).sum() == pytest.approx(120)


def test_guided_cap_redistributes():
    t = guided(np.array([100, 0, 0, 0, 0, 0.0]), total_s=120)
    assert t[0] == pytest.approx(50)
    assert t.sum() == pytest.approx(120)


def test_guided_equal_risk_is_uniform():
    assert guided(np.ones(6)) == pytest.approx(uniform(6))


def test_guided_many_surfaces_random():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(2, 140))
        t = guided(rng.random(n) ** 3, total_s=120)
        u = 120 / n
        assert t.sum() == pytest.approx(120)
        assert t.min() >= 0.5 * u - 1e-9 and t.max() <= 2.5 * u + 1e-9


def test_guided_bad_input():
    with pytest.raises(ValueError):
        guided(np.ones(6), total_s=120, min_s=30)
    with pytest.raises(ValueError):
        guided(np.array([0.2, -0.1]))


# ---- plaque ----
def _image(plaque_rows):
    img = np.full((20, 20, 3), 15.0)          # dark clean model under UV
    img[:plaque_rows] = (230, 240, 200)       # glowing analog
    return img


def test_plaque_fraction_otsu_and_fixed():
    region = np.ones((20, 20), bool)
    assert residual_fraction(_image(5), region) == pytest.approx(0.25)
    assert residual_fraction(_image(5), region, threshold=100) == pytest.approx(0.25)
    assert removal_percent(0.5, 0.1) == pytest.approx(80)
    assert np.isnan(removal_percent(0, 0))


def test_plaque_uniform_regions_not_split():
    region = np.ones((20, 20), bool)
    rng = np.random.default_rng(0)
    clean = np.full((20, 20, 3), 15.0) + rng.normal(0, 3, (20, 20, 3))
    assert residual_fraction(clean, region) == 0
    assert residual_fraction(_image(20), region) == 1


def test_region_restricts_and_dice():
    region = np.zeros((20, 20), bool)
    region[10:] = True                        # plaque rows lie outside
    assert residual_fraction(_image(5), region, threshold=100) == 0
    a = plaque_mask(_image(5), np.ones((20, 20), bool))
    assert dice(a, a) == 1
    assert dice(a, ~a) == 0


# ---- learned model ----
def _synthetic_surfaces(n_models=30, per_model=28, seed=0):
    rng = np.random.default_rng(seed)
    n = n_models * per_model
    X = rng.random((n, 5))
    y = np.clip(0.05 + 0.35 * X[:, 0] + 0.2 * X[:, 1] * X[:, 2]
                + rng.normal(0, 0.03, n), 0, 1)
    return X, y, np.repeat(np.arange(n_models), per_model)


def test_hybrid_model_learns_and_holds_out_models():
    X, y, ids = _synthetic_surfaces()
    cv = cross_validate(X, y, ids)
    assert cv["n_models"] == 30 and cv["r"] > 0.8 and cv["mae"] < 0.06
    assert accurate_enough(cv)
    p = HybridRisk().fit(X, y).predict(X)
    assert p.min() >= 0 and p.max() <= 1


def test_no_signal_triggers_fallback():
    X, _, ids = _synthetic_surfaces()
    y = np.random.default_rng(9).random(X.shape[0])
    assert not accurate_enough(cross_validate(X, y, ids))
    with pytest.raises(ValueError):
        cross_validate(X[:56], y[:56], ids[:56])


# ---- design and statistics ----
def test_randomize_order_balanced_and_reproducible():
    ids = [f"M{i:02d}" for i in range(45)]
    a, b = randomize_order(ids, seed=7), randomize_order(ids, seed=7)
    assert a == b
    first = sum(v[0] == "uniform" for v in a.values())
    assert first in (22, 23)


def test_paired_compare():
    rng = np.random.default_rng(1)
    u = rng.normal(0.30, 0.05, 45)
    g = u - rng.normal(0.05, 0.03, 45)
    r = paired_compare(u, g)
    assert r["mean_diff"] > 0 and r["t_p"] < 0.001 and r["wilcoxon_p"] < 0.001
    assert r["ci95"][0] < r["mean_diff"] < r["ci95"][1]
    assert r["relative_reduction_pct"] == pytest.approx(
        100 * (u.mean() - g.mean()) / u.mean())


def test_power_matches_published_values():
    # G*Power, paired t-test, two-sided, alpha .05, power .80: dz .5 -> n 34.
    assert n_for_paired(0.5) == 34
    assert power_paired(34, 0.5) == pytest.approx(0.80, abs=0.01)
    dz = detectable_dz(45)
    assert power_paired(45, dz) == pytest.approx(0.80, abs=1e-3)
    assert 0.42 < dz < 0.44


def test_mixed_model_finds_condition_effect():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(45):
        base = rng.normal(0.30, 0.05)
        ph = ("crowding", "crossbite", "open_bite")[i % 3]
        rows.append((f"M{i}", ph, "uniform", base + rng.normal(0, 0.02)))
        rows.append((f"M{i}", ph, "guided", base - 0.05 + rng.normal(0, 0.02)))
    df = pd.DataFrame(rows, columns=["model_id", "phenotype", "condition", "residual"])
    res = mixed_model(df)
    name = [n for n in res.params.index if "guided" in n and ":" not in n][0]
    assert res.params[name] == pytest.approx(-0.05, abs=0.02)
    assert res.pvalues[name] < 0.001
