from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def spearman_pair(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, int]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    if len(np.unique(x_arr[mask])) < 2 or len(np.unique(y_arr[mask])) < 2:
        return float("nan"), float("nan"), int(mask.sum())
    rho, p_value = spearmanr(x_arr[mask], y_arr[mask])
    return float(rho), float(p_value), int(mask.sum())


def binary_auc(scores: Iterable[float], labels: Iterable[bool]) -> float:
    scores_arr = np.asarray(list(scores), dtype=float)
    labels_arr = np.asarray(list(labels), dtype=bool)
    mask = np.isfinite(scores_arr)
    scores_arr = scores_arr[mask]
    labels_arr = labels_arr[mask]
    n_pos = int(labels_arr.sum())
    n_neg = int(len(labels_arr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores_arr)
    pos_ranks = ranks[labels_arr].sum()
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if controls.ndim == 1:
        design = np.column_stack([np.ones(len(controls)), controls])
    else:
        design = np.column_stack([np.ones(len(controls)), controls])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - np.matmul(design, coef)


def partial_spearman(df: pd.DataFrame, x_col: str, y_col: str, control_cols: list[str]) -> tuple[float, int]:
    cols = [x_col, y_col] + control_cols
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 6:
        return float("nan"), len(sub)
    x_rank = rankdata(sub[x_col].to_numpy(dtype=float))
    y_rank = rankdata(sub[y_col].to_numpy(dtype=float))
    controls = []
    for col in control_cols:
        if pd.api.types.is_numeric_dtype(sub[col]):
            controls.append(rankdata(sub[col].to_numpy(dtype=float)))
        else:
            dummies = pd.get_dummies(sub[col], drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                controls.extend([dummies[c].to_numpy(dtype=float) for c in dummies.columns])
    if not controls:
        return spearman_pair(sub[x_col], sub[y_col])[0], len(sub)
    control_matrix = np.column_stack(controls)
    x_res = _residualize(x_rank, control_matrix)
    y_res = _residualize(y_rank, control_matrix)
    if np.std(x_res) <= 0 or np.std(y_res) <= 0:
        return float("nan"), len(sub)
    return float(np.corrcoef(x_res, y_res)[0, 1]), len(sub)


def family_cluster_bootstrap(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    family_col: str,
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    families = sorted(df[family_col].dropna().unique())
    values = []
    for _ in range(n_boot):
        parts = []
        draws = rng.choice(families, size=len(families), replace=True)
        for family in draws:
            fam = df[df[family_col] == family]
            if fam.empty:
                continue
            idx = rng.choice(fam.index.to_numpy(), size=len(fam), replace=True)
            parts.append(df.loc[idx])
        boot = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if boot.empty:
            continue
        rho, _, _ = spearman_pair(boot[predictor], boot[target])
        if math.isfinite(rho):
            values.append(rho)
    point, p_value, n = spearman_pair(df[predictor], df[target])
    return {
        "spearman": point,
        "p_value": p_value,
        "n": n,
        "boot_median": float(np.median(values)) if values else float("nan"),
        "ci_low": float(np.quantile(values, 0.025)) if values else float("nan"),
        "ci_high": float(np.quantile(values, 0.975)) if values else float("nan"),
        "n_boot_valid": len(values),
    }


def threshold_metrics(scores: Iterable[float], labels: Iterable[bool], threshold: float) -> dict[str, float]:
    score_arr = np.asarray(list(scores), dtype=float)
    label_arr = np.asarray(list(labels), dtype=bool)
    pred = score_arr >= threshold
    tp = float(np.logical_and(pred, label_arr).sum())
    fp = float(np.logical_and(pred, ~label_arr).sum())
    fn = float(np.logical_and(~pred, label_arr).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def predictor_summary(df: pd.DataFrame, predictors: list[str], target: str, event_threshold: float = 20.0) -> pd.DataFrame:
    rows = []
    event = df[target].astype(float) >= event_threshold
    for predictor in predictors:
        if predictor not in df:
            continue
        rho, p_value, n = spearman_pair(df[predictor], df[target])
        rows.append(
            {
                "predictor": predictor,
                "n": n,
                "spearman": rho,
                "p_value": p_value,
                "roc_auc": binary_auc(df[predictor], event),
            }
        )
    return pd.DataFrame(rows)
