#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate example figure panels from supplementary records.")
    parser.add_argument("--records-root", type=Path, default=Path("supplementary/records"))
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent

    sample_table = args.records_root / "canonical_l2/per_sample_coordinates.csv.gz"
    encoder_table = args.records_root / "canonical_l2/per_encoder_summary.csv"
    matched_table = args.records_root / "matched_pairs/matched_band_summary.csv"

    if sample_table.exists():
        run([
            sys.executable,
            str(script_dir / "make_fig1.py"),
            "--sample-table",
            str(sample_table),
            "--out",
            str(args.out_dir / "fig1_ledger.pdf"),
        ])
    if encoder_table.exists():
        run([
            sys.executable,
            str(script_dir / "make_fig2.py"),
            "--encoder-table",
            str(encoder_table),
            "--out-rho",
            str(args.out_dir / "fig2_rho_scatter.pdf"),
            "--out-endpoint",
            str(args.out_dir / "fig2_endpoint_scatter.pdf"),
        ])
    if encoder_table.exists():
        cmd = [
            sys.executable,
            str(script_dir / "make_fig3.py"),
            "--encoder-table",
            str(encoder_table),
            "--out-plane",
            str(args.out_dir / "fig3_coordinate_plane.pdf"),
        ]
        if matched_table.exists():
            cmd.extend([
                "--matched-table",
                str(matched_table),
                "--out-bars",
                str(args.out_dir / "fig3_matched_bars.pdf"),
            ])
        run(cmd)


if __name__ == "__main__":
    main()
