from __future__ import annotations

from pathlib import Path
from typing import FrozenSet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_EXCLUDED_IMG_NUMBERS: FrozenSet[int] = frozenset(
    int(x) for x in "339 340 341 351 352 362 371 381 380 390 401 411 412 418 419 421 420 430 431".split()
)


def _excluded(stem: str, excluded_nums: FrozenSet[int]) -> bool:
    if not excluded_nums:
        return False
    if stem.startswith("img_") and stem[4:].isdigit():
        return int(stem[4:]) in excluded_nums
    if stem.isdigit():
        return int(stem) in excluded_nums
    return False


def _stem(pred_path: Path) -> str:
    if not pred_path.name.endswith("_pred.npy"):
        raise ValueError(pred_path.name)
    return pred_path.name[:-len("_pred.npy")]


def load_prediction_pairs(pred_dir: str | Path, sample_id: str | None = None, exclude_default: bool = True) -> tuple[np.ndarray, np.ndarray]:
    root = Path(pred_dir)
    excluded = DEFAULT_EXCLUDED_IMG_NUMBERS if exclude_default else frozenset()
    pred_paths = sorted(root.glob("*_pred.npy"))
    if sample_id:
        pred_paths = [path for path in pred_paths if _stem(path) == sample_id]
    if not pred_paths:
        raise FileNotFoundError(f"No *_pred.npy files found in {root}")
    all_gt: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for pred_path in pred_paths:
        stem = _stem(pred_path)
        if sample_id is None and _excluded(stem, excluded):
            continue
        gt_path = root / f"{stem}_gt.npy"
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        pred = np.load(pred_path)
        gt = np.load(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {stem}: pred={pred.shape}, gt={gt.shape}")
        mask_path = root / f"{stem}_valid_mask.npy"
        valid = np.load(mask_path) > 0 if mask_path.is_file() else (np.isfinite(pred) & np.isfinite(gt))
        all_gt.append(gt[valid].ravel())
        all_pred.append(pred[valid].ravel())
    if not all_gt:
        raise ValueError("No samples left after exclusions")
    return np.concatenate(all_gt), np.concatenate(all_pred)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)


def subsample(x: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_points, replace=False)
    return x[idx], y[idx]


def create_scatter(
    pred_dir: str | Path,
    output: str | Path | None = None,
    sample_id: str | None = None,
    title: str = "Bathymetry Prediction Scatter",
    max_points: int = 200_000,
    seed: int = 0,
    exclude_default: bool = True,
) -> dict[str, float | str | int]:
    root = Path(pred_dir).resolve()
    y_true, y_pred = load_prediction_pairs(root, sample_id, exclude_default)
    r2 = r2_score(y_true, y_pred)
    x_plot, y_plot = subsample(y_true, y_pred, max_points, seed)
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_plot, y_plot, s=4, c="red", alpha=0.35, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color="blue", linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ground truth")
    ax.set_ylabel("prediction")
    ax.set_title(title)
    ax.text(0.03, 0.97, rf"$R^2$ = {100.0 * r2:.2f}\%", transform=ax.transAxes, fontsize=12, verticalalignment="top")
    fig.tight_layout()
    target = Path(output) if output else root / f"scatter_gt_pred{'_' + sample_id if sample_id else ''}.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return {"output": str(target), "n_pixels": int(y_true.size), "r2": float(r2)}


def run_scatter(args) -> None:
    result = create_scatter(
        pred_dir=args.pred_dir,
        output=args.out,
        sample_id=args.sample_id,
        title=args.title,
        max_points=args.max_points,
        seed=args.seed,
        exclude_default=not args.no_exclude_default,
    )
    print(f"saved: {result['output']}")
    print(f"n_pixels(valid): {result['n_pixels']}")
    print(f"R2: {result['r2']:.6f}")
