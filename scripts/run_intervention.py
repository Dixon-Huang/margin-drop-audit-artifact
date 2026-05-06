#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize packaged intervention records.")
    parser.add_argument("--records", default="supplementary/records/intervention/intervention_records.csv")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.records)
    closure = df[df["record_type"].eq("closure")].copy()
    coords = df[df["record_type"].eq("coordinate_delta")].copy()
    rows = []
    for encoder in sorted(set(df["encoder"])):
        enc_closure = closure[closure["encoder"].eq(encoder)]
        enc_coords = coords[coords["encoder"].eq(encoder)]
        for _, row in enc_closure.iterrows():
            rows.append(
                {
                    "encoder": encoder,
                    "method": row["method"],
                    "endpoint": "collapse_10_200",
                    "value": row.get("collapse_10_200"),
                }
            )
        for _, row in enc_coords.iterrows():
            rows.append(
                {
                    "encoder": encoder,
                    "method": row["method"],
                    "delta_lambda": row.get("delta_lambda"),
                    "delta_kappa": row.get("delta_kappa"),
                    "delta_rho50": row.get("delta_rho50"),
                    "delta_m": row.get("delta_m"),
                }
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    if out.suffix.lower() == ".json":
        records = json.loads(result.to_json(orient="records"))
        out.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        result.to_csv(out, index=False)


if __name__ == "__main__":
    main()
