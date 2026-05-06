from __future__ import annotations

from collections.abc import Sequence

import torch

from .margins import pairwise_margin


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
    flat = _flatten(delta)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.minimum(torch.ones_like(norm), torch.tensor(eps, device=delta.device) / norm)
    return (flat * scale).reshape_as(delta)


def clip_delta(inputs: torch.Tensor, delta: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    return (inputs + delta).clamp(clip_min, clip_max) - inputs


def one_step_l2(model, inputs: torch.Tensor, labels: torch.Tensor, competitors: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = inputs.detach().clone().requires_grad_(True)
    logits = model(x)
    margin = pairwise_margin(logits, labels, competitors)
    margin.sum().backward()
    grad = x.grad.detach()
    grad_norm = _flatten(grad).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    delta = -eps * (_flatten(grad) / grad_norm).reshape_as(inputs)
    Lp = eps * grad_norm.squeeze(1)
    return delta.detach(), Lp.detach()


def margin_drop(model, inputs: torch.Tensor, labels: torch.Tensor, competitors: torch.Tensor, delta: torch.Tensor, clean_margin: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits_adv = model((inputs + delta).clamp(0.0, 1.0))
        adv_margin = pairwise_margin(logits_adv, labels, competitors)
    return clean_margin - adv_margin


def projected_gradient_descent(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    competitors: torch.Tensor,
    clean_margin: torch.Tensor,
    *,
    eps: float,
    steps: int = 50,
    restarts: int = 10,
    step_sizes: Sequence[float] | None = None,
    include_delta: torch.Tensor | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if step_sizes is None:
        step_sizes = (eps / 4.0, eps / 10.0)
    batch = inputs.shape[0]
    best_delta = torch.zeros_like(inputs)
    best_drop = torch.full((batch,), -float("inf"), device=inputs.device)
    if include_delta is not None:
        inc = clip_delta(inputs, include_delta.detach(), clip_min, clip_max)
        best_delta = inc.clone()
        best_drop = margin_drop(model, inputs, labels, competitors, inc, clean_margin)

    for step_index, step_size in enumerate(step_sizes):
        for restart in range(restarts):
            if step_index == 0 and restart == 0:
                delta = torch.zeros_like(inputs)
            else:
                noise = torch.randn_like(inputs)
                noise = project_l2(noise, eps)
                radii = torch.rand(batch, 1, 1, 1, device=inputs.device)
                delta = noise * radii
            delta = clip_delta(inputs, delta, clip_min, clip_max)

            for _ in range(steps):
                delta = delta.detach().requires_grad_(True)
                logits = model((inputs + delta).clamp(clip_min, clip_max))
                margin = pairwise_margin(logits, labels, competitors)
                loss = -margin.sum()
                grad = torch.autograd.grad(loss, delta)[0]
                with torch.no_grad():
                    grad_norm = _flatten(grad).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    delta = delta + float(step_size) * (_flatten(grad) / grad_norm).reshape_as(delta)
                    delta = project_l2(delta, eps)
                    delta = clip_delta(inputs, delta, clip_min, clip_max)

            drop = margin_drop(model, inputs, labels, competitors, delta.detach(), clean_margin)
            better = drop > best_drop
            best_drop = torch.where(better, drop, best_drop)
            best_delta = torch.where(better.reshape(batch, 1, 1, 1), delta.detach(), best_delta)

    return best_delta.detach(), best_drop.detach()
