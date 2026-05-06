from __future__ import annotations

import importlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def load_json(path: str | Path) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_json_safe(obj), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def import_symbol(spec: str):
    if ":" not in spec:
        raise ValueError("Expected import spec in the form 'module:function'.")
    module_name, symbol_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
