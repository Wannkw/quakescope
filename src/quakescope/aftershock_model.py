"""Predict whether an M>=6 earthquake will produce a strong aftershock.

Framing (no look-ahead leakage): we stand 24 hours after a mainshock and use only
what is known at that moment to predict whether an M>=``target_mag`` event will
occur nearby between day 1 and day 30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .decluster import gk_window
from .stats import haversine_km

FEATURES = [
    "mag",
    "depth",
    "abs_lat",
    "n_aft_24h",
    "max_aft_mag_24h",
    "n_fore_30d",
    "bg_rate_365d",
]


def _neighbours(t, lat, lon, i, t_from, t_to, radius_km):
    lo = np.searchsorted(t, t[i] + t_from, side="left")
    hi = np.searchsorted(t, t[i] + t_to, side="right")
    idx = np.arange(lo, hi)
    idx = idx[idx != i]
    if len(idx) == 0:
        return idx
    return idx[haversine_km(lat[i], lon[i], lat[idx], lon[idx]) <= radius_km]


def build_dataset(
    df: pd.DataFrame,
    main_mag: float = 6.0,
    target_mag: float = 5.5,
    horizon_days: float = 30.0,
    bg_radius_km: float = 300.0,
) -> pd.DataFrame:
    t = df["t_days"].to_numpy()
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    mag = df["mag"].to_numpy()
    radius, _ = gk_window(mag)

    last_t = t[-1]
    rows = []
    for i in np.flatnonzero(mag >= main_mag):
        if t[i] + horizon_days > last_t:
            continue  # the outcome window is not fully observed yet
        r = radius[i]
        first_day = _neighbours(t, lat, lon, i, 0.0, 1.0, r)
        future = _neighbours(t, lat, lon, i, 1.0, horizon_days, r)
        fore = _neighbours(t, lat, lon, i, -30.0, 0.0, r)
        bg = _neighbours(t, lat, lon, i, -365.0, -30.0, bg_radius_km)
        rows.append(
            {
                "id": df["id"].iat[i],
                "time": df["time"].iat[i],
                "place": df["place"].iat[i],
                "latitude": lat[i],
                "longitude": lon[i],
                "mag": mag[i],
                "depth": df["depth"].iat[i],
                "abs_lat": abs(lat[i]),
                "n_aft_24h": len(first_day),
                "max_aft_mag_24h": mag[first_day].max() if len(first_day) else 0.0,
                "n_fore_30d": len(fore),
                "bg_rate_365d": len(bg) / 335.0,
                "target": int((mag[future] >= target_mag).any()),
            }
        )
    return pd.DataFrame(rows)


def temporal_split(ds: pd.DataFrame, test_from: str = "2019-01-01"):
    cutoff = pd.Timestamp(test_from, tz="UTC")
    return ds[ds["time"] < cutoff], ds[ds["time"] >= cutoff]


def candidate_models(seed: int = 42) -> dict:
    return {
        "Baseline: logistic on magnitude only": (
            ["mag"],
            make_pipeline(StandardScaler(), LogisticRegression()),
        ),
        "Logistic regression (all features)": (
            FEATURES,
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        ),
        "Gradient boosting (all features)": (
            FEATURES,
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=seed,
            ),
        ),
    }


def evaluate(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42):
    """Fit every candidate on ``train`` and score on the held-out future ``test``."""
    results, fitted = {}, {}
    for name, (cols, model) in candidate_models(seed).items():
        model.fit(train[cols], train["target"])
        proba = model.predict_proba(test[cols])[:, 1]
        results[name] = {
            "roc_auc": roc_auc_score(test["target"], proba),
            "pr_auc": average_precision_score(test["target"], proba),
            "brier": brier_score_loss(test["target"], proba),
        }
        fitted[name] = (cols, model, proba)
    return results, fitted
