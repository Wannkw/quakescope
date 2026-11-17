import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quakescope.decluster import decluster, gk_window  # noqa: E402
from quakescope.stats import b_value_mle, completeness_maxc, fit_omori, haversine_km  # noqa: E402


def test_haversine_known_distance():
    # Bangkok -> Tokyo is about 4,600 km.
    assert haversine_km(13.7563, 100.5018, 35.6762, 139.6503) == pytest.approx(4600, rel=0.02)
    assert haversine_km(0, 0, 0, 0) == 0


def test_b_value_recovers_synthetic_truth():
    rng = np.random.default_rng(1)
    true_b, mc = 1.0, 4.5
    # G-R magnitudes are exponential with rate b*ln(10); bin to 0.1 like a real catalog.
    raw = (mc - 0.05) + rng.exponential(1 / (true_b * np.log(10)), 50_000)
    mags = np.round(raw, 1)
    assert b_value_mle(mags, mc) == pytest.approx(true_b, abs=0.03)


def test_completeness_is_mode_plus_correction():
    mags = np.array([4.5] * 5 + [4.6] * 50 + [4.7] * 30 + [5.0] * 3)
    assert completeness_maxc(mags) == pytest.approx(4.8)


def test_omori_recovers_p():
    rng = np.random.default_rng(7)
    c, p, t_end = 0.05, 1.1, 365.0
    # Inverse-CDF sampling of the Omori-Utsu density on [0, t_end].
    u = rng.uniform(size=5000)
    a, b_ = c ** (1 - p), (c + t_end) ** (1 - p)
    t = (a - u * (a - b_)) ** (1 / (1 - p)) - c
    fit = fit_omori(t, 0.0, t_end)
    assert fit["p"] == pytest.approx(p, abs=0.05)


def test_gk_window_grows_with_magnitude():
    d, t = gk_window([5.0, 6.0, 7.0])
    assert np.all(np.diff(d) > 0) and np.all(np.diff(t) > 0)


def test_decluster_groups_aftershocks_but_not_distant_events():
    df = pd.DataFrame(
        {
            "t_days": [0.0, 0.5, 2.0, 3.0],
            "latitude": [0.0, 0.1, 0.2, 40.0],
            "longitude": [0.0, 0.1, 0.0, 40.0],
            "mag": [7.0, 5.0, 4.8, 5.5],
        }
    )
    out = decluster(df)
    assert out["is_mainshock"].tolist() == [True, False, False, True]
    assert out["cluster_id"].tolist()[:3] == [0, 0, 0]
