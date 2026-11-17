"""End-to-end pipeline: download -> statistics -> declustering -> model -> figures.

Usage:
    python scripts/run_pipeline.py [--start 2000] [--end 2025]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quakescope import aftershock_model as am  # noqa: E402
from quakescope import plots  # noqa: E402
from quakescope.data import load_catalog  # noqa: E402
from quakescope.decluster import decluster  # noqa: E402
from quakescope.stats import (  # noqa: E402
    b_value_bootstrap,
    completeness_maxc,
    fit_omori,
    haversine_km,
    rupture_length_km,
)

FIG = ROOT / "reports" / "figures"


def main(start: int, end: int) -> None:
    print("1/5 Loading catalog")
    df = load_catalog(start, end, cache_dir=ROOT / "data")
    print(f"    {len(df):,} events")

    print("2/5 Gutenberg-Richter")
    mc = completeness_maxc(df["mag"])
    b, ci = b_value_bootstrap(df["mag"], mc)
    plots.world_map(df, FIG / "world_map.png")
    plots.gutenberg_richter(df["mag"], mc, b, ci, FIG / "gutenberg_richter.png")

    print("3/5 Declustering (Gardner-Knopoff)")
    df = decluster(df)
    plots.declustering(df, FIG / "declustering.png")
    main_mags = df.loc[df["is_mainshock"], "mag"]
    b_main, ci_main = b_value_bootstrap(main_mags, mc)

    print("4/5 Omori law on the four largest sequences")
    # Gardner-Knopoff windows saturate near 125 km, far shorter than a great quake's
    # rupture, so aftershocks are selected within one Wells-Coppersmith rupture length.
    sequences, omori = [], {}
    for i in df.nlargest(4, "mag").index:
        dist = haversine_km(df.at[i, "latitude"], df.at[i, "longitude"], df["latitude"], df["longitude"])
        t_after = (df["t_days"] - df.at[i, "t_days"]).to_numpy()
        t_after = t_after[(t_after > 0) & (t_after <= 365) & (dist <= rupture_length_km(df.at[i, "mag"]))]
        fit = fit_omori(t_after, 0.0, 365.0)
        label = f"M{df.at[i, 'mag']:.1f} {df.at[i, 'place']}"
        sequences.append({"title": label, "t": t_after, "fit": fit})
        omori[label] = {k: round(v, 4) for k, v in fit.items()}
    plots.omori_panels(sequences, FIG / "omori.png")

    print("5/5 Aftershock model")
    ds = am.build_dataset(df)
    train, test = am.temporal_split(ds)
    results, fitted = am.evaluate(train, test)
    plots.model_report(test, fitted, results, FIG / "model_roc_calibration.png")

    best = "Gradient boosting (all features)"
    cols, model, _ = fitted[best]
    imp = permutation_importance(model, test[cols], test["target"], scoring="roc_auc", n_repeats=30, random_state=0)
    plots.importance(cols, imp.importances_mean, imp.importances_std, FIG / "feature_importance.png")

    metrics = {
        "catalog": {
            "events": int(len(df)),
            "years": [start, end],
            "mainshocks_after_declustering": int(df["is_mainshock"].sum()),
            "share_in_clusters": round(1 - df["is_mainshock"].mean(), 3),
        },
        "gutenberg_richter": {
            "mc": round(mc, 2),
            "b_all": round(b, 3), "b_all_ci95": [round(x, 3) for x in ci],
            "b_declustered": round(b_main, 3), "b_declustered_ci95": [round(x, 3) for x in ci_main],
        },
        "omori": omori,
        "aftershock_model": {
            "task": "P(M>=5.5 within GK radius, days 1-30 | M>=6 mainshock, first 24h observed)",
            "n_train": int(len(train)), "n_test": int(len(test)),
            "base_rate_train": round(train["target"].mean(), 3),
            "base_rate_test": round(test["target"].mean(), 3),
            "results": {k: {m: round(v, 3) for m, v in r.items()} for k, r in results.items()},
            "permutation_importance": dict(
                sorted(zip(cols, np.round(imp.importances_mean, 4).tolist()), key=lambda kv: -kv[1])
            ),
        },
    }
    out = ROOT / "reports" / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()
    main(args.start, args.end)
