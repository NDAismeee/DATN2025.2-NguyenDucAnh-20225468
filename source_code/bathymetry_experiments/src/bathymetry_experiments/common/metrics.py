from __future__ import annotations

import numpy as np


def regression_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    valid = np.isfinite(pred) & np.isfinite(target)
    if mask is not None:
        valid &= mask > 0
    if not np.any(valid):
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "bias": float("nan"), "p95ae": float("nan")}
    y_pred = pred[valid].astype(np.float64)
    y_true = target[valid].astype(np.float64)
    err = y_pred - y_true
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "r2": float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot),
        "bias": float(np.mean(err)),
        "p95ae": float(np.percentile(np.abs(err), 95)),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ["mae", "rmse", "r2", "bias", "p95ae"]
    out: dict[str, float] = {}
    for key in keys:
        vals = np.array([row[key] for row in rows if np.isfinite(row.get(key, float("nan")))], dtype=np.float64)
        out[key] = float(np.mean(vals)) if vals.size else float("nan")
    return out
