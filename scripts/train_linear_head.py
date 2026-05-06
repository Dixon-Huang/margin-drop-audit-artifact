#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear head from frozen feature NPZ files.")
    parser.add_argument("--train-features", required=True, help="NPZ with `features` and `labels` arrays.")
    parser.add_argument("--val-features", default=None, help="Optional NPZ with `features` and `labels` arrays.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_npz(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    obj = np.load(path)
    features = torch.as_tensor(obj["features"], dtype=torch.float32)
    labels = torch.as_tensor(obj["labels"], dtype=torch.long)
    return features, labels


@torch.no_grad()
def accuracy(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> float:
    logits = model(features.to(device))
    pred = logits.argmax(dim=1).cpu()
    return float((pred == labels).float().mean().item())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    x_train, y_train = load_npz(args.train_features)
    n_classes = int(y_train.max().item()) + 1
    model = nn.Linear(x_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.cpu().state_dict(), "n_features": x_train.shape[1], "n_classes": n_classes}
    if args.val_features:
        x_val, y_val = load_npz(args.val_features)
        payload["val_accuracy"] = accuracy(model.to(device), x_val, y_val, device)
    torch.save(payload, out)


if __name__ == "__main__":
    main()
