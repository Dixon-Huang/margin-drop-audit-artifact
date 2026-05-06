from __future__ import annotations

from typing import Any

import torch

from .attacks import margin_drop, one_step_l2, projected_gradient_descent
from .coordinates import coordinate_record
from .margins import clean_correct_mask, select_clean_competitor


def _as_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    if isinstance(batch, dict):
        inputs = batch["inputs"]
        labels = batch["labels"]
        sample_ids = batch.get("sample_ids", list(range(inputs.shape[0])))
        return inputs, labels, list(sample_ids)
    if len(batch) == 2:
        inputs, labels = batch
        return inputs, labels, list(range(inputs.shape[0]))
    if len(batch) >= 3:
        inputs, labels, sample_ids = batch[:3]
        return inputs, labels, list(sample_ids)
    raise ValueError("Dataloader batches must provide inputs and labels.")


def compute_batch_coordinates(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float,
    pgd_steps: int = 50,
    pgd_restarts: int = 10,
    step_sizes: list[float] | None = None,
    sample_ids: list[Any] | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    labels = labels.long()
    with torch.no_grad():
        logits = model(inputs)
        keep = clean_correct_mask(logits, labels)
        if not bool(keep.any()):
            return []
        inputs = inputs[keep]
        labels = labels[keep]
        logits = logits[keep]
        if sample_ids is not None:
            sample_ids = [sid for sid, keep_i in zip(sample_ids, keep.detach().cpu().tolist()) if keep_i]
        competitors, clean_margin = select_clean_competitor(logits, labels)

    delta_one, Lp = one_step_l2(model, inputs, labels, competitors, eps)
    D_fgsm = margin_drop(model, inputs, labels, competitors, delta_one, clean_margin)
    delta_pgd, D_pgd = projected_gradient_descent(
        model,
        inputs,
        labels,
        competitors,
        clean_margin,
        eps=eps,
        steps=pgd_steps,
        restarts=pgd_restarts,
        step_sizes=step_sizes,
        include_delta=delta_one,
    )
    del delta_pgd
    records: list[dict[str, Any]] = []
    if sample_ids is None:
        sample_ids = list(range(inputs.shape[0]))
    for i in range(inputs.shape[0]):
        if not bool(torch.isfinite(Lp[i])) or float(Lp[i].detach().cpu()) <= 0.0:
            continue
        rec = coordinate_record(
            float(clean_margin[i].detach().cpu()),
            float(Lp[i].detach().cpu()),
            float(D_fgsm[i].detach().cpu()),
            float(D_pgd[i].detach().cpu()),
            sample_id=int(sample_ids[i]) if isinstance(sample_ids[i], (int, float)) else None,
            label=int(labels[i].detach().cpu()),
            competitor=int(competitors[i].detach().cpu()),
        )
        records.append(rec)
    return records


def run_coordinate_audit(
    model,
    dataloader,
    *,
    eps: float,
    pgd_steps: int = 50,
    pgd_restarts: int = 10,
    step_sizes: list[float] | None = None,
    device: str | torch.device = "cuda",
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model = model.to(device).eval()
    out: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs, labels, sample_ids = _as_batch(batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        out.extend(
            compute_batch_coordinates(
                model,
                inputs,
                labels,
                eps=eps,
                pgd_steps=pgd_steps,
                pgd_restarts=pgd_restarts,
                step_sizes=step_sizes,
                sample_ids=sample_ids,
            )
        )
    return out


def evaluate_retention(model, dataloader, **kwargs) -> dict[str, Any]:
    records = run_coordinate_audit(model, dataloader, **kwargs)
    if not records:
        return {"records": records, "R": float("nan"), "n": 0}
    passed = sum(1 for rec in records if rec["pass"])
    return {"records": records, "R": 100.0 * passed / len(records), "n": len(records)}
