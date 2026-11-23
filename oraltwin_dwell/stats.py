"""Design and analysis for Aim 3 (each dental model is its own control)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def randomize_order(model_ids, seed: int) -> dict:
    """Balanced random order of the two brushing conditions per model.

    Half of the models get uniform first, half guided first (difference of
    at most one). The seed is recorded so the schedule can be reproduced.
    """
    ids = list(model_ids)
    rng = np.random.default_rng(seed)
    first_uniform = np.arange(len(ids)) % 2 == 0
    rng.shuffle(first_uniform)
    return {m: ("uniform", "guided") if u else ("guided", "uniform")
            for m, u in zip(ids, first_uniform)}


def paired_compare(uniform_residual, guided_residual) -> dict:
    """Residual plaque after uniform vs guided brushing.

    Positive mean_diff means guided brushing left less plaque.
    """
    u = np.asarray(uniform_residual, float)
    g = np.asarray(guided_residual, float)
    if u.shape != g.shape or u.size < 3:
        raise ValueError("need matched pairs, n >= 3")
    d = u - g
    sd = d.std(ddof=1)
    if sd == 0:
        raise ValueError("all paired differences are identical")
    lo, hi = stats.t.interval(0.95, d.size - 1, loc=d.mean(),
                              scale=sd / np.sqrt(d.size))
    return {
        "n": int(d.size),
        "mean_uniform": float(u.mean()),
        "mean_guided": float(g.mean()),
        "mean_diff": float(d.mean()),
        "ci95": (float(lo), float(hi)),
        "relative_reduction_pct": float(100 * d.mean() / u.mean()),
        "cohen_dz": float(d.mean() / sd),
        "t_p": float(stats.ttest_rel(u, g).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(u, g).pvalue),
    }


def power_paired(n: int, dz: float, alpha: float = 0.05) -> float:
    """Power of a two-sided paired t-test with n pairs and effect dz."""
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    ncp = dz * np.sqrt(n)
    p = stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp)
    # The noncentral t is numerically unstable for very large effects,
    # where power is 1 to machine precision anyway.
    return 1.0 if np.isnan(p) else float(min(p, 1.0))


def n_for_paired(dz: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Pairs needed to detect effect dz."""
    if dz <= 0:
        raise ValueError("dz must be positive")
    for n in range(3, 100000):
        if power_paired(n, dz, alpha) >= power:
            return n
    raise ValueError("effect too small")


def detectable_dz(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest standardized paired effect detectable with n pairs."""
    from scipy.optimize import brentq
    return float(brentq(lambda dz: power_paired(n, dz, alpha) - power, 1e-4, 10))


def mixed_model(df):
    """Linear mixed model: residual ~ condition * phenotype + (1 | model_id).

    df columns: residual, condition ("uniform"/"guided"), phenotype, model_id.
    Returns the fitted statsmodels result.
    """
    import statsmodels.formula.api as smf

    return smf.mixedlm(
        "residual ~ C(condition, Treatment('uniform')) * C(phenotype)",
        df, groups=df["model_id"]).fit(reml=True)
