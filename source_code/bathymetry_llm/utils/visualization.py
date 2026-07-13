from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_prediction_figure(
    image: np.ndarray,
    depth: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    uncertainty: np.ndarray,
    alpha: np.ndarray,
    unreliable_mask: np.ndarray,
    d_phys: np.ndarray,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.transpose(image[:3], (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)
    mask = valid_mask[0] > 0.5
    gt = np.where(mask, depth[0], np.nan)
    pd = np.where(mask, pred[0], np.nan)
    err = np.where(mask, np.abs(pd - gt), np.nan)
    panels = [
        (rgb, "RGB", None),
        (gt, "Ground Truth", "viridis"),
        (pd, "Predicted Depth", "viridis"),
        (err, "Absolute Error", "magma"),
        (uncertainty[0], "Uncertainty", "plasma"),
        (alpha[0], "Gate Alpha", "magma"),
        (unreliable_mask[0], "Unreliable Mask", "inferno"),
        (d_phys[0], "Physical Prior", "viridis"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (arr, title, cmap) in zip(axes.ravel(), panels):
        if cmap is None:
            ax.imshow(arr)
        else:
            im = ax.imshow(arr, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
