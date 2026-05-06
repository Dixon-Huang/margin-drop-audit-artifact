from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def coordinate_record(
    M: float,
    Lp: float,
    D_fgsm: float,
    D_pgd: float,
    *,
    sample_id: Any | None = None,
    label: int | None = None,
    competitor: int | None = None,
) -> dict[str, Any]:
    if Lp <= 0 or not math.isfinite(Lp):
        raise ValueError("Lp must be positive and finite.")
    lambda_value = M / Lp
    d_fgsm_norm = D_fgsm / Lp
    d_pgd_norm = D_pgd / Lp
    kappa = 1.0 - d_fgsm_norm
    rho = d_pgd_norm - d_fgsm_norm
    tau = rho - kappa
    m = lambda_value + kappa - 1.0 - rho
    rec: dict[str, Any] = {
        "M": float(M),
        "Lp": float(Lp),
        "D_fgsm": float(D_fgsm),
        "D_pgd": float(D_pgd),
        "lambda": float(lambda_value),
        "kappa": float(kappa),
        "rho": float(rho),
        "tau": float(tau),
        "m": float(m),
        "pass": bool(m > 0),
        "D_fgsm_over_Lp": float(d_fgsm_norm),
        "D_pgd_over_Lp": float(d_pgd_norm),
    }
    if sample_id is not None:
        try:
            rec["sample_id"] = int(sample_id)
        except (TypeError, ValueError):
            rec["sample_id"] = sample_id
    if label is not None:
        rec["label"] = int(label)
    if competitor is not None:
        rec["competitor"] = int(competitor)
    return rec


def coordinate_from_record(record: dict[str, Any]) -> dict[str, Any]:
    if {"M", "Lp", "D_fgsm", "D_pgd"}.issubset(record):
        M = float(record["M"])
        Lp = float(record["Lp"])
        D_fgsm = float(record["D_fgsm"])
        D_pgd = float(record["D_pgd"])
    elif {"M_k_star", "L_k_star", "D_lower_k_star", "fidelity_k_star"}.issubset(record):
        M = float(record["M_k_star"])
        Lp = float(record["L_k_star"])
        D_pgd = float(record["D_lower_k_star"])
        D_fgsm = float(record["fidelity_k_star"]) * D_pgd
    else:
        raise KeyError(
            "Record must contain either M/Lp/D_fgsm/D_pgd or "
            "M_k_star/L_k_star/D_lower_k_star/fidelity_k_star fields."
        )
    return coordinate_record(
        M,
        Lp,
        D_fgsm,
        D_pgd,
        sample_id=record.get("sample_id", record.get("sample_idx")),
        label=record.get("label"),
        competitor=record.get("competitor", record.get("hardest_k")),
    )


def robust_retention(records: list[dict[str, Any]], pass_key: str = "pass") -> float:
    if not records:
        return float("nan")
    return 100.0 * float(np.mean([bool(r[pass_key]) for r in records]))


def collapse(R_shallow: float, R_deep: float) -> float:
    return float(R_shallow) - float(R_deep)


def summarize_encoder(records: list[dict[str, Any]], encoder: str | None = None) -> dict[str, Any]:
    df = pd.DataFrame(records)
    out: dict[str, Any] = {"n": int(len(df))}
    if encoder is not None:
        out["encoder"] = encoder
    for col in ["lambda", "kappa", "rho", "tau", "m", "Lp"]:
        if col in df:
            out[f"median_{col}"] = float(df[col].median())
    if "pass" in df:
        out["R"] = robust_retention(records)
    return out


def add_collapse_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "collapse_10_200" not in out and {"R10", "R200"}.issubset(out.columns):
        out["collapse_10_200"] = out["R10"].astype(float) - out["R200"].astype(float)
    if "delta_R_10_200" not in out and "collapse_10_200" in out:
        out["delta_R_10_200"] = out["collapse_10_200"]
    return out
