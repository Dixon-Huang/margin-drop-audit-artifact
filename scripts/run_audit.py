#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import run_coordinate_audit
from src.utils import import_symbol, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-competitor margin-coordinate audit.")
    parser.add_argument("--model-builder", required=True, help="Import path such as module:function returning a torch model.")
    parser.add_argument("--data-builder", required=True, help="Import path such as module:function returning a dataloader.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--eps", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=10, help="Restarts per step size.")
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=None,
        help="L2 PGD step sizes. Defaults to eps/4 and eps/10.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model = import_symbol(args.model_builder)()
    dataloader = import_symbol(args.data_builder)()
    records = run_coordinate_audit(
        model,
        dataloader,
        eps=args.eps,
        pgd_steps=args.steps,
        pgd_restarts=args.restarts,
        step_sizes=args.step_sizes,
        device=args.device,
        max_batches=args.max_batches,
    )
    config = vars(args).copy()
    config["norm"] = "l2"
    if config["step_sizes"] is None:
        config["step_sizes"] = [args.eps / 4.0, args.eps / 10.0]
    write_json({"config": config, "records": records}, args.out)


if __name__ == "__main__":
    main()
