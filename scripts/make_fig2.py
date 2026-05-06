#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.figures import collapse_scatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create collapse scatter panels.")
    parser.add_argument("--encoder-table", required=True)
    parser.add_argument("--out-rho", required=True)
    parser.add_argument("--out-endpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.encoder_table)
    collapse_scatter(
        df,
        x_col="bar_rho10",
        xlabel="median off-ray drift rho10",
        out=args.out_rho,
        threshold_x=0.40,
    )
    collapse_scatter(
        df,
        x_col="R10",
        xlabel="PGD-10 robust retention R10 (%)",
        out=args.out_endpoint,
    )


if __name__ == "__main__":
    main()
