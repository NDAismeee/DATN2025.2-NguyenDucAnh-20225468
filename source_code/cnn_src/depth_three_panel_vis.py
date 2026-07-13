from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.colors import ListedColormap


DISPLAY_DEPTH_MIN = -20.0
DISPLAY_DEPTH_MAX = 0.0
VIS_SMOOTH_SIGMA = 3.0


def _to_display_depth(depth_positive: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return np.where(valid_mask > 0, -depth_positive, np.nan).astype(np.float32)


def _safe_cmap(name: str) -> str:
    if isinstance(name, str) and name.strip().lower() == "turbo":
        try:
            return plt.get_cmap("turbo")
        except Exception:
            return _turbo_cmap()
    try:
        return plt.get_cmap(name)
    except Exception:
        return plt.get_cmap("viridis")


def _turbo_cmap(n: int = 256) -> ListedColormap:
    x = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
    r = (
        0.13572138
        + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    )
    g = (
        0.09140261
        + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    )
    b = (
        0.10667330
        + x * (12.64194608 + x * (-62.53311460 + x * (142.71343904 + x * (-150.86423692 + x * 53.01055690))))
    )
    rgb = np.stack([r, g, b], axis=1)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return ListedColormap(rgb, name="turbo_local")


def _smooth_masked_2d(arr: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return arr.astype(np.float64, copy=False)
    m = (mask > 0).astype(np.float32)
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
    arr = _to_display_depth(depth_positive, valid_mask)
    return _smooth_masked_2d(arr, valid_mask, sigma_px=sigma_px).astype(np.float32)


def save_three_panel_depth_figure(
    rgb_hwc: np.ndarray,
    gt_depth_positive: np.ndarray,
    pred_depth_positive: np.ndarray,
    valid_mask: np.ndarray,
    out_path: str | Path,
    suptitle: Optional[str] = None,
    cmap: str = "turbo",
    vmin: float = DISPLAY_DEPTH_MIN,
    vmax: float = DISPLAY_DEPTH_MAX,
    sigma_px: float = VIS_SMOOTH_SIGMA,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.clip(rgb_hwc.astype(np.float32), 0.0, 1.0)
    gt_disp = prepare_depth_for_display(gt_depth_positive, valid_mask, sigma_px=sigma_px)
    pr_disp = prepare_depth_for_display(pred_depth_positive, valid_mask, sigma_px=sigma_px)
    cmap = _safe_cmap(cmap)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    im1 = axes[1].imshow(
        gt_disp,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        interpolation="bilinear",
    )
    axes[1].set_title("GT depth (m, display negative)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        pr_disp,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        interpolation="bilinear",
    )
    axes[2].set_title("Pred depth (m, display negative)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

