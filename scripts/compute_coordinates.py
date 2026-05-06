#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coordinates import coordinate_from_record
from src.utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert stored audit records to coordinate rows.")
    parser.add_argument(
        "--input-json",
        required=True,
        help="Audit JSON containing records with M, Lp, D_fgsm, and D_pgd fields.",
    )
    parser.add_argument("--out", required=True, help="Output table path. CSV is recommended outside this package.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obj = load_json(args.input_json)
    records = obj.get("records", obj.get("results", []))
    rows = [coordinate_from_record(rec) for rec in records]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
