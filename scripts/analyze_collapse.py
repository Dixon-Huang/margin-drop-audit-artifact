#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coordinates import add_collapse_columns
from src.statistics import family_cluster_bootstrap, partial_spearman, predictor_summary, spearman_pair
from src.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cross-budget collapse from an encoder-level table.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rho-col", default="bar_rho10")
    parser.add_argument("--target-col", default="delta_R_10_200")
    parser.add_argument("--family-col", default=None)
    parser.add_argument("--predictors", nargs="*", default=["bar_rho10", "R10", "bar_tau10", "bar_lambda", "bar_kappa"])
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = add_collapse_columns(pd.read_csv(args.table))
    target = args.target_col if args.target_col in df else "delta_R_10_200"
    rho, p_value, n = spearman_pair(df[args.rho_col], df[target])
    result = {
        "main_spearman": {"predictor": args.rho_col, "target": target, "rho": rho, "p_value": p_value, "n": n},
        "predictor_summary": predictor_summary(df, args.predictors, target).to_dict(orient="records"),
    }
    if args.family_col and args.family_col in df:
        result["family_cluster_bootstrap"] = family_cluster_bootstrap(
            df,
            args.rho_col,
            target,
            args.family_col,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    if "bar_kappa" in df.columns:
        partial, n_partial = partial_spearman(df, args.rho_col, target, ["bar_kappa"])
        result["partial_spearman_rho_control_kappa"] = {"rho": partial, "n": n_partial}
    if "bar_kappa" in df.columns and args.rho_col in df.columns:
        partial, n_partial = partial_spearman(df, "bar_kappa", target, [args.rho_col])
        result["partial_spearman_kappa_control_rho"] = {"rho": partial, "n": n_partial}
    write_json(result, args.out)


if __name__ == "__main__":
    main()
