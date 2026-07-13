from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _valid_arrays(target: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return target[valid].astype(np.float64), pred[valid].astype(np.float64)


def compute_metrics(target: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    y, p = _valid_arrays(target, pred, valid_mask)
    if y.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "bias": 0.0, "p95ae": 0.0, "err_std": 0.0}
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if denom < 1e-12 else float(1.0 - np.sum(err ** 2) / denom)
    bias = float(np.mean(err))
    p95ae = float(np.percentile(np.abs(err), 95))
    err_std = float(np.std(np.abs(err)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias, "p95ae": p95ae, "err_std": err_std}


def expected_calibration_error(
    abs_err: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 15,
) -> float:
    abs_err = np.asarray(abs_err, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if abs_err.size == 0 or sigma.size == 0 or abs_err.size != sigma.size:
        return 0.0
    sigma = np.clip(sigma, 1e-6, None)
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    z = abs_err / sigma
    ece = 0.0
    n = float(z.size)
    for i in range(int(n_bins)):
        q_lo, q_hi = quantiles[i], quantiles[i + 1]
        lo_thr = float(np.quantile(z, q_lo))
        hi_thr = float(np.quantile(z, q_hi)) if q_hi < 1.0 else float(np.inf)
        in_bin = (z >= lo_thr) & (z < hi_thr) if q_hi < 1.0 else (z >= lo_thr)
        bin_n = float(in_bin.sum())
        if bin_n <= 0:
            continue
        empirical = bin_n / n
        expected = q_hi - q_lo
        ece += (bin_n / n) * abs(empirical - expected)
    return float(ece)


def quantile_mae(
    abs_err: np.ndarray,
    sigma: np.ndarray,
    quantile_low: float = 0.2,
    quantile_high: float = 0.8,
) -> Dict[str, float]:
    abs_err = np.asarray(abs_err, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if abs_err.size == 0:
        return {"low_q_mae": 0.0, "high_q_mae": 0.0}
    lo_thr = float(np.quantile(sigma, quantile_low))
    hi_thr = float(np.quantile(sigma, quantile_high))
    low_mask = sigma <= lo_thr
    high_mask = sigma >= hi_thr
    low_q_mae = float(np.mean(abs_err[low_mask])) if low_mask.any() else 0.0
    high_q_mae = float(np.mean(abs_err[high_mask])) if high_mask.any() else 0.0
    return {"low_q_mae": low_q_mae, "high_q_mae": high_q_mae}


def uncertainty_diagnostics(
    target: np.ndarray,
    pred: np.ndarray,
    variance: np.ndarray,
    valid_mask: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    y, p = _valid_arrays(target, pred, valid_mask)
    if y.size == 0:
        return {"ece": 0.0, "sigma_err_corr": 0.0, "low_q_mae": 0.0, "high_q_mae": 0.0}
    sigma = np.sqrt(np.maximum(variance.astype(np.float64), 1e-12))
    sigma_v = sigma[valid_mask.astype(bool)]
    abs_err = np.abs(p - y)
    if abs_err.size < 2 or sigma_v.size < 2:
        corr = 0.0
    else:
        denom = float(np.std(abs_err) * np.std(sigma_v))
        corr = 0.0 if denom < 1e-12 else float(np.corrcoef(abs_err, sigma_v)[0, 1])
    quantile_metrics = quantile_mae(abs_err, sigma_v)
    return {
        "ece": expected_calibration_error(abs_err, sigma_v, n_bins=n_bins),
        "sigma_err_corr": corr,
        **quantile_metrics,
    }


def out_of_range_rate(
    pred: np.ndarray,
    valid_mask: np.ndarray,
    domain_min: float,
    domain_max: float,
) -> float:
    mask = valid_mask.astype(bool)
    if mask.sum() == 0:
        return 0.0
    p = pred[mask].astype(np.float64)
    out = np.logical_or(p < float(domain_min), p > float(domain_max))
    return float(out.mean())


def slope_error(
    target: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    if target.ndim == 3:
        gt = target[0]
        pr = pred[0] if pred.ndim == 3 else pred
        vm = valid_mask[0] if valid_mask.ndim == 3 else valid_mask
    else:
        gt, pr, vm = target, pred, valid_mask
    if gt.shape != pr.shape:
        return 0.0
    gt = gt.astype(np.float64)
    pr = pr.astype(np.float64)
    mask = vm.astype(bool)
    if mask.sum() < 4:
        return 0.0
    gy, gx = np.gradient(gt)
    py, px = np.gradient(pr)
    dy = (py - gy)
    dx = (px - gx)
    grad_err = np.sqrt(dy ** 2 + dx ** 2)
    return float(np.mean(grad_err[mask]))


def unreliable_region_mae(
    target: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    unreliable_mask: np.ndarray,
) -> float:
    valid = valid_mask.astype(bool)
    unreliable = (unreliable_mask.astype(np.float64) > 0.5)
    combined = valid & unreliable
    if combined.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(pred[combined] - target[combined])))


def physical_consistency_metrics(
    target: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    unreliable_mask: Optional[np.ndarray] = None,
    domain_min: Optional[float] = None,
    domain_max: Optional[float] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {"slope_err": slope_error(target, pred, valid_mask)}
    if domain_min is not None and domain_max is not None:
        out["out_of_range"] = out_of_range_rate(pred, valid_mask, domain_min, domain_max)
    if unreliable_mask is not None:
        out["unreliable_mae"] = unreliable_region_mae(target, pred, valid_mask, unreliable_mask)
    return out


def mask_iou_f1(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> Dict[str, float]:
    p = (pred_mask.astype(np.float64) > 0.5)
    r = (ref_mask.astype(np.float64) > 0.5)
    if p.shape != r.shape:
        raise ValueError(f"mask shape mismatch: {p.shape} vs {r.shape}")
    inter = float(np.logical_and(p, r).sum())
    union = float(np.logical_or(p, r).sum())
    iou = 0.0 if union <= 0 else inter / union
    p_sum = float(p.sum())
    r_sum = float(r.sum())
    precision = 0.0 if p_sum <= 0 else inter / p_sum
    recall = 0.0 if r_sum <= 0 else inter / r_sum
    denom = precision + recall
    f1 = 0.0 if denom <= 0 else 2.0 * precision * recall / denom
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


def interval_coverage(
    target: np.ndarray,
    d_min: np.ndarray,
    d_max: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, float]:
    valid = valid_mask.astype(bool)
    if valid.sum() == 0:
        return {"coverage": 0.0, "mean_width": 0.0}
    y = target[valid].astype(np.float64)
    lo = d_min[valid].astype(np.float64)
    hi = d_max[valid].astype(np.float64)
    in_range = (y >= lo) & (y <= hi)
    width = np.maximum(hi - lo, 0.0)
    return {
        "coverage": float(in_range.mean()),
        "mean_width": float(width.mean()),
    }
