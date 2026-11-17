"""Classic statistical seismology: Gutenberg-Richter and Omori-Utsu."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

LOG10_E = np.log10(np.e)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; broadcasts over numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def rupture_length_km(mag):
    """Wells & Coppersmith (1994) subsurface rupture length, all slip types."""
    return 10 ** (-2.44 + 0.59 * np.asarray(mag, dtype=float))


# --- Gutenberg-Richter: log10 N(>=M) = a - b M -------------------------------

def completeness_maxc(mags: np.ndarray, bin_width: float = 0.1, correction: float = 0.2) -> float:
    """Magnitude of completeness via maximum curvature (+ the usual 0.2 correction)."""
    binned = np.round(np.asarray(mags) / bin_width) * bin_width
    values, counts = np.unique(np.round(binned, 2), return_counts=True)
    return float(values[np.argmax(counts)] + correction)


def b_value_mle(mags: np.ndarray, mc: float, bin_width: float = 0.1) -> float:
    """Aki (1965) maximum-likelihood b-value with Utsu's binning correction."""
    m = np.asarray(mags)
    m = m[m >= mc - 1e-9]
    if len(m) < 2:
        raise ValueError("need at least two events above Mc")
    return LOG10_E / (m.mean() - (mc - bin_width / 2))


def b_value_bootstrap(mags, mc, n_boot: int = 500, seed: int = 0, bin_width: float = 0.1):
    """Point estimate plus a 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    m = np.asarray(mags)
    m = m[m >= mc - 1e-9]
    boots = [b_value_mle(rng.choice(m, len(m)), mc, bin_width) for _ in range(n_boot)]
    return b_value_mle(m, mc, bin_width), tuple(np.percentile(boots, [2.5, 97.5]))


# --- Omori-Utsu: n(t) = K / (c + t)^p -----------------------------------------

def _omori_integral(t0, t1, c, p):
    if abs(p - 1) < 1e-6:
        return np.log((c + t1) / (c + t0))
    return ((c + t0) ** (1 - p) - (c + t1) ** (1 - p)) / (p - 1)


def fit_omori(t: np.ndarray, t_start: float, t_end: float) -> dict:
    """MLE fit of the Omori-Utsu law to aftershock times (days after mainshock).

    Uses the point-process likelihood of Ogata (1983); K is profiled out.
    """
    t = np.asarray(t, dtype=float)
    t = t[(t >= t_start) & (t <= t_end)]
    n = len(t)
    if n < 10:
        raise ValueError("need at least 10 aftershocks to fit Omori")

    def nll(params):
        log_c, p = params
        c = np.exp(log_c)
        integral = _omori_integral(t_start, t_end, c, p)
        k = n / integral
        return -(n * np.log(k) - p * np.log(c + t).sum() - n)

    best = min(
        (minimize(nll, x0=[np.log(c0), p0], method="Nelder-Mead") for c0 in (0.01, 0.1) for p0 in (0.9, 1.2)),
        key=lambda r: r.fun,
    )
    c, p = float(np.exp(best.x[0])), float(best.x[1])
    k = n / _omori_integral(t_start, t_end, c, p)
    return {"K": float(k), "c": c, "p": p, "n": n}
