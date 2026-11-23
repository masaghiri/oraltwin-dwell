"""Hybrid plaque-risk model (Aim 2.3): Random Forest + Support Vector Regression.

The prediction is the mean of the two regressors. Evaluation always holds
out whole dental models, never single surfaces, so surfaces from one model
cannot appear in both training and test data (no leakage).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class HybridRisk:
    def __init__(self, seed: int = 0):
        self.rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                        random_state=seed)
        self.svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.02))

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.svr.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        p = (self.rf.predict(X) + self.svr.predict(X)) / 2.0
        return np.clip(p, 0.0, 1.0)   # residual plaque is a fraction


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.std() > 0 and y_pred.std() > 0 else float("nan")
    return {"mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "r": r}


def cross_validate(X, y, model_ids, n_splits: int = 5, seed: int = 0) -> dict:
    """Out-of-fold MAE, RMSE and correlation with whole models held out.

    model_ids: one id per row saying which dental model the surface is from.
    """
    X, y, model_ids = np.asarray(X, float), np.asarray(y, float), np.asarray(model_ids)
    n_groups = np.unique(model_ids).size
    if n_groups < n_splits:
        raise ValueError(f"need at least {n_splits} dental models, got {n_groups}")
    pred = np.empty_like(y)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, model_ids):
        assert not set(model_ids[train]) & set(model_ids[test])
        pred[test] = HybridRisk(seed).fit(X[train], y[train]).predict(X[test])
    out = metrics(y, pred)
    out["n_models"] = int(n_groups)
    out["n_surfaces"] = int(y.size)
    return out


def accurate_enough(cv: dict, min_r: float = 0.5) -> bool:
    """Go/no-go rule for using the learned model in Aim 3.

    If False, Aim 3 proceeds with the transparent geometric score
    (``oraltwin_dwell.risk.risk_score``) instead. Set ``min_r`` before
    looking at Aim 3 data.
    """
    return bool(cv["r"] >= min_r)
