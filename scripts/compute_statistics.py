#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.statistics import spearman_pair
from src.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute headline statistics from supplementary records.")
    parser.add_argument("--records-root", type=Path, default=Path("supplementary/records"))
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    root = args.records_root
    result: dict[str, object] = {}

    canonical = read_csv(root / "canonical_l2/per_encoder_summary.csv")
    if canonical is not None:
        target = "delta_R_10_200" if "delta_R_10_200" in canonical else "aggregate_delta_R_10_200"
        rho_col = "bar_rho10" if "bar_rho10" in canonical else "rho10"
        rho, p_value, n = spearman_pair(canonical[rho_col], canonical[target])
        result["canonical_l2_rho10_vs_collapse"] = {
            "rho": rho,
            "p_value": p_value,
            "n": n,
            "predictor": rho_col,
            "target": target,
        }

    heldout = read_csv(root / "heldout_l2/per_encoder_summary.csv")
    if heldout is not None:
        rho, p_value, n = spearman_pair(heldout["rho10"], heldout["delta_R_10_200"])
        result["heldout_l2_rho10_vs_collapse"] = {"rho": rho, "p_value": p_value, "n": n}

    linf = read_csv(root / "canonical_linf/per_encoder_summary.csv")
    if linf is not None:
        op = linf[linf["eps_tag"].astype(str).eq("eps038centi_255")]
        rho, p_value, n = spearman_pair(op["rho10"], op["DeltaR_10_100"])
        result["canonical_linf_rho10_vs_collapse_eps_0.375_255"] = {
            "rho": rho,
            "p_value": p_value,
            "n": n,
        }

    cross_norm = read_csv(root / "canonical_linf/sweep/cross_norm_by_eps.csv")
    if cross_norm is not None:
        result["cross_norm_rho_rank_consistency"] = cross_norm.to_dict(orient="records")

    ce = read_csv(root / "ce_apgd/per_encoder_summary.csv")
    if ce is not None and canonical is not None:
        rho_col = "bar_rho10" if "bar_rho10" in ce else "rho10"
        if rho_col in ce:
            merged = ce.copy()
        else:
            rho_col = "bar_rho10" if "bar_rho10" in canonical else "rho10"
            merged = ce.merge(canonical[["encoder", rho_col]], on="encoder", how="inner")
        rho, p_value, n = spearman_pair(merged[rho_col], merged["delta_R_top1_10_100"])
        result["ce_apgd_top1_collapse_vs_pairwise_rho10"] = {
            "rho": rho,
            "p_value": p_value,
            "n": n,
            "predictor": rho_col,
            "target": "delta_R_top1_10_100",
        }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is None:
        print(text)
    else:
        write_json(result, args.out)


if __name__ == "__main__":
    main()
