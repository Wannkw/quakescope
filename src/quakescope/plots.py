"""Figures for the README. Every function writes one PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)
INK = "#1f2937"
ACCENT = "#e4572e"
MUTED = "#4c78a8"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def world_map(df: pd.DataFrame, path: Path) -> Path:
    """Epicentres coloured by depth: the plate boundaries draw themselves."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#0b1020")
    order = df.sort_values("depth", ascending=False)
    sc = ax.scatter(
        order["longitude"], order["latitude"], c=order["depth"].clip(upper=700),
        s=(order["mag"] - 4.0) ** 2.6, cmap="plasma_r", alpha=0.55, linewidths=0,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-75, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.set_title(f"{len(df):,} earthquakes (M≥4.5), {df['time'].dt.year.min()}–{df['time'].dt.year.max()}")
    cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label("Depth (km)")
    return _save(fig, path)


def gutenberg_richter(mags: np.ndarray, mc: float, b: float, ci, path: Path) -> Path:
    mags = np.asarray(mags)
    bins = np.round(np.arange(mags.min(), mags.max() + 0.1, 0.1), 1)
    cum = np.array([(mags >= m - 1e-9).sum() for m in bins])
    inc = np.array([((mags >= m - 0.05) & (mags < m + 0.05)).sum() for m in bins])
    n_mc = (mags >= mc - 1e-9).sum()
    a = np.log10(n_mc) + b * mc

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(bins, cum, "o", ms=4, color=INK, label="Cumulative N(≥M)")
    ax.semilogy(bins, np.where(inc > 0, inc, np.nan), "s", ms=3, color=MUTED, alpha=0.6, label="Per 0.1 bin")
    fit_m = np.linspace(mc, bins.max(), 50)
    ax.semilogy(fit_m, 10 ** (a - b * fit_m), "-", color=ACCENT, lw=2,
                label=f"b = {b:.2f} (95% CI {ci[0]:.2f}–{ci[1]:.2f})")
    ax.axvline(mc, ls="--", color="grey", lw=1)
    ax.text(mc - 0.05, 3, f"Mc = {mc:.1f}", color="grey", ha="right")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Number of events")
    ax.set_title("Gutenberg–Richter law: every +1 magnitude is ~10× rarer")
    ax.legend(frameon=False)
    return _save(fig, path)


def omori_panels(sequences: list[dict], path: Path) -> Path:
    """One log-log panel per sequence: dict(title=..., t=array of days, fit=dict)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, seq in zip(axes.flat, sequences):
        t, fit = seq["t"], seq["fit"]
        edges = np.logspace(-2, np.log10(t.max()), 30)
        counts, _ = np.histogram(t, edges)
        centres = np.sqrt(edges[:-1] * edges[1:])
        rate = counts / np.diff(edges)
        ax.loglog(centres[counts > 0], rate[counts > 0], "o", ms=4, color=INK, label=f"Observed ({len(t):,} events)")
        tt = np.logspace(-2, np.log10(t.max()), 200)
        ax.loglog(tt, fit["K"] / (fit["c"] + tt) ** fit["p"], color=ACCENT, lw=2,
                  label=f"Omori–Utsu: p = {fit['p']:.2f}, c = {fit['c']:.2f} d")
        ax.set_title(seq["title"], fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("Days after mainshock")
    for ax in axes[:, 0]:
        ax.set_ylabel("Aftershocks per day (M≥4.5)")
    fig.suptitle("Aftershock rates decay as a power law (~1/t) after the four largest quakes")
    return _save(fig, path)


def declustering(df: pd.DataFrame, path: Path) -> Path:
    yearly = df.groupby(df["time"].dt.year).agg(all=("id", "size"), main=("is_mainshock", "sum"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(yearly.index, yearly["all"], color="#cbd5e1", label="All events")
    ax.bar(yearly.index, yearly["main"], color=MUTED, label="Mainshocks (declustered)")
    for year in yearly["all"].nlargest(3).index:
        ax.annotate(str(year), (year, yearly.loc[year, "all"]), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Events per year (M≥4.5)")
    ax.set_title("Aftershocks cause the spikes; the declustered background is far steadier")
    ax.legend(frameon=False)
    return _save(fig, path)


def model_report(test: pd.DataFrame, fitted: dict, results: dict, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = ["#9ca3af", MUTED, ACCENT]
    for (name, (_, _, proba)), color in zip(fitted.items(), colors):
        fpr, tpr, _ = roc_curve(test["target"], proba)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {results[name]['roc_auc']:.2f})")
        frac, mean_pred = calibration_curve(test["target"], proba, n_bins=8, strategy="quantile")
        ax2.plot(mean_pred, frac, "o-", color=color, lw=1.5, ms=4, label=name)
    ax1.plot([0, 1], [0, 1], ":", color="grey")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC on held-out future (2019+)")
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.plot([0, 1], [0, 1], ":", color="grey")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Observed frequency")
    ax2.set_title("Calibration")
    return _save(fig, path)


def importance(names, means, stds, path: Path) -> Path:
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(np.array(names)[order], np.array(means)[order], xerr=np.array(stds)[order], color=MUTED)
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title("What the model relies on (permutation importance)")
    return _save(fig, path)
