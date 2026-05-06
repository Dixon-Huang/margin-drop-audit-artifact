#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.figures import BLUE, GRAY, ORANGE, RED, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a coordinate ledger panel from one sample record.")
    parser.add_argument("--sample-table", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = pd.read_csv(args.sample_table, low_memory=False).iloc[args.row]
    lam = float(row["lambda"])
    kappa = float(row["kappa"])
    rho = float(row["rho"] if "rho" in row else row.get("rho_10", row.get("rho10")))
    d_fgsm = float(row.get("D_fgsm_over_Lp", row.get("D_FGSM_10_norm", 1.0 - kappa)))
    d_pgd = float(row.get("D_pgd_over_Lp", d_fgsm + rho))
    m = float(row.get("m", row.get("m_10", lam + kappa - 1.0 - rho)))

    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    ax.set_xlim(0, max(lam, d_pgd, 1.0) + 0.25)
    ax.set_ylim(-0.7, 4.2)
    ax.axis("off")
    rows = [
        ("clean margin", lam, BLUE),
        ("linear prediction", 1.0, GRAY),
        ("one-step drop", d_fgsm, ORANGE),
        ("iterative drop", d_pgd, RED),
    ]
    for idx, (label, value, color) in enumerate(rows):
        y = 3.4 - idx
        ax.plot([0, value], [y, y], color=color, linewidth=7, solid_capstyle="butt")
        ax.plot([0, max(lam, d_pgd, 1.0)], [y, y], color="#cccccc", linewidth=1)
        ax.text(-0.03, y, label, ha="right", va="center", fontsize=8)
        ax.text(value + 0.04, y, f"{value:.2f}", ha="left", va="center", fontsize=8)
    ax.annotate("kappa", xy=(d_fgsm, 1.4), xytext=(1.0, 1.4), arrowprops={"arrowstyle": "<->", "color": ORANGE}, color=ORANGE, ha="center")
    ax.annotate("rho", xy=(d_fgsm, 0.4), xytext=(d_pgd, 0.4), arrowprops={"arrowstyle": "<->", "color": RED}, color=RED, ha="center")
    ax.text(0, -0.15, f"m = lambda + kappa - 1 - rho = {m:.2f}", fontsize=10)
    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
