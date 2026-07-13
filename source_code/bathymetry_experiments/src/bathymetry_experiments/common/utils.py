from __future__ import annotations

import csv
import random
import time
import uuid
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
    except Exception:
        pass


def make_run_dir(base_dir: str | Path, model: str) -> Path:
    run_id = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    path = Path(base_dir) / model / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
