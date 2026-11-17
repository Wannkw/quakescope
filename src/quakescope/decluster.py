"""Gardner-Knopoff (1974) window declustering.

Each event is assigned to the cluster of the largest earlier-or-later event whose
space-time window contains it. Events that belong to no larger event's window are
mainshocks (or background events).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import haversine_km


def gk_window(mag):
    """Gardner-Knopoff window: (distance km, duration days)."""
    mag = np.asarray(mag, dtype=float)
    dist = 10 ** (0.1238 * mag + 0.983)
    days = np.where(mag >= 6.5, 10 ** (0.032 * mag + 2.7389), 10 ** (0.5409 * mag - 0.547))
    return dist, days


def decluster(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``cluster_id`` and ``is_mainshock`` columns.

    ``df`` must be sorted by ``t_days``. Runs in O(N log N + sum of window sizes).
    """
    t = df["t_days"].to_numpy()
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    mag = df["mag"].to_numpy()
    dist_w, time_w = gk_window(mag)

    n = len(df)
    cluster = np.full(n, -1)
    # Process from largest to smallest so the biggest event claims its window first.
    for i in np.argsort(-mag, kind="stable"):
        if cluster[i] != -1:
            continue
        cluster[i] = i  # unclaimed: a mainshock
        lo = np.searchsorted(t, t[i] - time_w[i], side="left")
        hi = np.searchsorted(t, t[i] + time_w[i], side="right")
        idx = np.arange(lo, hi)
        idx = idx[cluster[idx] == -1]
        if len(idx) == 0:
            continue
        close = haversine_km(lat[i], lon[i], lat[idx], lon[idx]) <= dist_w[i]
        cluster[idx[close]] = i

    out = df.copy()
    out["cluster_id"] = cluster
    out["is_mainshock"] = cluster == np.arange(n)
    return out
