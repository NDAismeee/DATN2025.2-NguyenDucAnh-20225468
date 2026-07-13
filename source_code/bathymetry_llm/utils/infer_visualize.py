from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEPTH_CMAP = "turbo"
ALPHA_CMAP = "magma"
VAR_CMAP = "plasma"
ERR_CMAP = "magma"
DEPTH_INTERPOLATION = "bilinear"
AUX_INTERPOLATION = "nearest"
VIS_SMOOTH_SIGMA = 3.0
DISPLAY_DEPTH_MIN = -20.0
DISPLAY_DEPTH_MAX = 0.0

try:
    import cv2
except Exception:
    cv2 = None


def chw_to_display_rgb(image_chw: np.ndarray) -> np.ndarray:
    c, _, _ = image_chw.shape
    if c >= 3:
        rgb = np.transpose(image_chw[:3], (1, 2, 0))
    else:
        rgb = np.stack([image_chw[0], image_chw[0], image_chw[0]], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    if hi > lo:
        rgb = (rgb - lo) / (hi - lo)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)


def to_display_depth(depth_positive: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    dp = np.asarray(depth_positive, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=np.float64)
    if vm.ndim == 3 and vm.shape[0] == 1:
        vm = vm[0]
    if dp.ndim == 3 and dp.shape[0] == 1:
        dp = dp[0]
    return np.where(vm > 0, -dp, np.nan)


def smooth_masked_2d(arr: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    if cv2 is None or sigma_px <= 0:
        return arr.astype(np.float64, copy=False)
    m = (mask > 0).astype(np.float32) if mask.ndim == 2 else (mask[0] > 0).astype(np.float32)
    x = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    k = int(max(3, round(sigma_px * 6)))
    if k % 2 == 0:
        k += 1
    num = cv2.GaussianBlur(x * m, (k, k), float(sigma_px))
    den = cv2.GaussianBlur(m, (k, k), float(sigma_px))
    out = num / np.maximum(den, 1e-6)
    return np.where(den > 1e-3, out, np.nan).astype(np.float64)


def prepare_depth_for_display(
    depth_positive: np.ndarray,
    valid_mask: np.ndarray,
    sigma_px: float = VIS_SMOOTH_SIGMA,
) -> np.ndarray:
    d = depth_positive.squeeze()
    vm = valid_mask.squeeze() if valid_mask.ndim > 2 else valid_mask
    if vm.ndim > 2:
        vm = vm[0]
    arr = to_display_depth(d, vm)
    return smooth_masked_2d(arr, vm, sigma_px=sigma_px)


def prepare_aux_for_display(
    arr: np.ndarray,
    valid_mask: np.ndarray,
    smooth: bool = False,
    sigma_px: float = VIS_SMOOTH_SIGMA,
) -> np.ndarray:
    a = arr.squeeze()
    vm = valid_mask.squeeze() if valid_mask.ndim > 2 else valid_mask
    if vm.ndim > 2:
        vm = vm[0]
    out = np.where(vm > 0, a, np.nan)
    if smooth and cv2 is not None:
        out = smooth_masked_2d(out, vm, sigma_px=sigma_px)
    return out


def fmt_range(arr: np.ndarray) -> str:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "[nan, nan]"
    return f"[{finite.min():.2f}, {finite.max():.2f}]"


def save_llm_guided_infer_figure(
    image_chw: np.ndarray,
    gt_positive: np.ndarray,
    pred_positive: np.ndarray,
    mu_positive: np.ndarray,
    valid_mask: np.ndarray,
    d_phys: Optional[np.ndarray],
    alpha: Optional[np.ndarray],
    var: Optional[np.ndarray],
    out_path: str | Path,
    title_prefix: str = "",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = chw_to_display_rgb(image_chw)
    vm = valid_mask
    gt_d = prepare_depth_for_display(gt_positive, vm)
    pr_d = prepare_depth_for_display(pred_positive, vm)
    mu_d = prepare_depth_for_display(mu_positive, vm, sigma_px=VIS_SMOOTH_SIGMA)
    dphys_d = prepare_depth_for_display(d_phys, vm, sigma_px=VIS_SMOOTH_SIGMA) if d_phys is not None else None
    alpha_d = prepare_aux_for_display(alpha, vm, smooth=False) if alpha is not None else None
    var_d = prepare_aux_for_display(var, vm, smooth=False) if var is not None else None
    err = np.where(np.isfinite(gt_d) & np.isfinite(pr_d), np.abs(pr_d - gt_d), np.nan)

    vmin, vmax = DISPLAY_DEPTH_MIN, DISPLAY_DEPTH_MAX
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    im1 = axes[1].imshow(gt_d, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, interpolation=DEPTH_INTERPOLATION)
    axes[1].set_title(f"Ground truth\n{fmt_range(gt_d)} m")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pr_d, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, interpolation=DEPTH_INTERPOLATION)
    axes[2].set_title(f"Final prediction\n{fmt_range(pr_d)} m")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(mu_d, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, interpolation=DEPTH_INTERPOLATION)
    axes[3].set_title(f"Raw model μ\n{fmt_range(mu_d)} m")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    if dphys_d is not None:
        im4 = axes[4].imshow(dphys_d, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, interpolation=DEPTH_INTERPOLATION)
        axes[4].set_title(f"d_phys\n{fmt_range(dphys_d)} m")
        fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    else:
        axes[4].text(0.5, 0.5, "d_phys n/a", ha="center", va="center")
        axes[4].set_title("d_phys")
    axes[4].axis("off")

    if alpha_d is not None:
        im5 = axes[5].imshow(alpha_d, cmap=ALPHA_CMAP, vmin=0.0, vmax=1.0, interpolation=AUX_INTERPOLATION)
        axes[5].set_title(f"Gate α\n{fmt_range(alpha_d)}")
        fig.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)
    else:
        axes[5].text(0.5, 0.5, "α n/a", ha="center", va="center")
        axes[5].set_title("Gate α")
    axes[5].axis("off")

    if var_d is not None:
        fv = var_d[np.isfinite(var_d)]
        v0, v1 = (float(fv.min()), float(fv.max())) if fv.size > 0 else (0.0, 1.0)
        im6 = axes[6].imshow(var_d, cmap=VAR_CMAP, vmin=v0, vmax=v1, interpolation=AUX_INTERPOLATION)
        axes[6].set_title(f"Variance\n{fmt_range(var_d)}")
        fig.colorbar(im6, ax=axes[6], fraction=0.046, pad=0.04)
    else:
        axes[6].text(0.5, 0.5, "var n/a", ha="center", va="center")
        axes[6].set_title("Variance")
    axes[6].axis("off")

    fe = err[np.isfinite(err)]
    e0, e1 = (float(fe.min()), float(fe.max())) if fe.size > 0 else (0.0, 1.0)
    im7 = axes[7].imshow(err, cmap=ERR_CMAP, vmin=e0, vmax=max(e1, e0 + 1e-6), interpolation=AUX_INTERPOLATION)
    axes[7].set_title(f"|pred−gt| (display)\n{fmt_range(err)}")
    fig.colorbar(im7, ax=axes[7], fraction=0.046, pad=0.04)
    axes[7].axis("off")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
