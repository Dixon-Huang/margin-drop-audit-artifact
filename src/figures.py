from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .coordinates import add_collapse_columns
from .statistics import spearman_pair

BLUE = "#1f5fbf"
ORANGE = "#f28e2b"
RED = "#d62728"
GRAY = "#6b6b6b"
LIGHT_GRAY = "#e6e6e6"


def save_figure(fig, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if out.suffix.lower() == ".pdf":
        fig.savefig(out.with_suffix(".png"), dpi=400)
    plt.close(fig)


def _style_axes(ax) -> None:
    ax.grid(True, color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def collapse_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str = "delta_R_10_200",
    xlabel: str,
    ylabel: str = "collapse Delta R (pp)",
    out: str | Path,
    threshold_x: float | None = None,
    threshold_y: float | None = None,
    annotations: dict[str, tuple[float, float]] | None = None,
) -> None:
    work = add_collapse_columns(df)
    fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
    ax.scatter(work[x_col], work[y_col], s=28, color=BLUE, edgecolor="white", linewidth=0.3, alpha=0.95)
    if threshold_x is not None:
        ax.axvline(threshold_x, color=RED, linestyle=(0, (4, 3)), linewidth=1.0)
    if threshold_y is not None:
        ax.axhline(threshold_y, color=GRAY, linestyle=(0, (4, 3)), linewidth=0.9)
    rho, _, _ = spearman_pair(work[x_col], work[y_col])
    ax.text(0.03, 0.95, f"Spearman = {rho:+.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    if annotations:
        for name, offset in annotations.items():
            row = work[work["encoder"].astype(str).str.lower().eq(name.lower())]
            if row.empty:
                continue
            r = row.iloc[0]
            ax.annotate(
                name,
                xy=(r[x_col], r[y_col]),
                xytext=offset,
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.7},
                fontsize=7.5,
            )
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    _style_axes(ax)
    save_figure(fig, out)


def kappa_rho_plane(
    df: pd.DataFrame,
    *,
    out: str | Path,
    kappa_col: str = "bar_kappa",
    rho_col: str = "bar_rho10",
    collapse_col: str = "delta_R_10_200",
) -> None:
    work = add_collapse_columns(df)
    fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
    scatter = ax.scatter(work[kappa_col], work[rho_col], c=work[collapse_col], s=32, cmap="coolwarm", edgecolor="white", linewidth=0.3)
    xmin, xmax = ax.get_xlim()
    for c in np.linspace(-0.4, 0.6, 6):
        xs = np.array([xmin, xmax])
        ax.plot(xs, xs + c, color=GRAY, linestyle=(0, (4, 3)), linewidth=0.7, alpha=0.5)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("collapse (pp)", fontsize=8)
    ax.set_xlabel("median on-ray shortfall kappa", fontsize=9)
    ax.set_ylabel("median off-ray drift rho10", fontsize=9)
    _style_axes(ax)
    save_figure(fig, out)


def matched_survival_bar(
    labels: list[str],
    rates: list[float],
    counts: list[int],
    *,
    out: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
    colors = [BLUE, ORANGE]
    xs = np.arange(len(labels))
    ax.bar(xs, rates, color=colors[: len(labels)], width=0.58, edgecolor="white", linewidth=0.5)
    for x, rate, count in zip(xs, rates, counts):
        ax.text(x, rate + 2.0, f"{rate:.1f}%\nn={count}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PGD-200 pass rate", fontsize=9)
    ax.set_ylim(0, max(100.0, max(rates) + 10.0))
    _style_axes(ax)
    save_figure(fig, out)
