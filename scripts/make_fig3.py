#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.figures import kappa_rho_plane, matched_survival_bar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create directional-split panels.")
    parser.add_argument("--encoder-table", required=True)
    parser.add_argument("--matched-table", default=None)
    parser.add_argument("--out-plane", required=True)
    parser.add_argument("--out-bars", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.encoder_table)
    kappa_rho_plane(df, out=args.out_plane)
    if args.matched_table and args.out_bars:
        matched = pd.read_csv(args.matched_table)
        matched_survival_bar(
            matched["label"].astype(str).tolist(),
            matched["pass_rate"].astype(float).tolist(),
            matched["n"].astype(int).tolist(),
            out=args.out_bars,
        )


if __name__ == "__main__":
    main()
