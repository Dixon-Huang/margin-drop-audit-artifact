from __future__ import annotations

import torch


def pairwise_margin(logits: torch.Tensor, labels: torch.Tensor, competitors: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    competitors = competitors.long()
    batch = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch, labels] - logits[batch, competitors]


def all_pairwise_margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    batch = torch.arange(logits.shape[0], device=logits.device)
    true_logits = logits[batch, labels].unsqueeze(1)
    margins = true_logits - logits
    margins[batch, labels] = float("inf")
    return margins


def select_clean_competitor(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    margins = all_pairwise_margins(logits, labels)
    competitors = margins.argmin(dim=1)
    batch = torch.arange(logits.shape[0], device=logits.device)
    clean_margins = margins[batch, competitors]
    return competitors, clean_margins


def clean_correct_mask(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1).eq(labels.long())
