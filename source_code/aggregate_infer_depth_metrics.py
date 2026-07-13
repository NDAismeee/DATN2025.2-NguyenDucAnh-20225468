from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _metrics_flat(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    m = valid.astype(bool) & np.isfinite(gt) & np.isfinite(pred)
    if not np.any(m):
        return float("nan"), float("nan"), float("nan")
    y = gt[m].astype(np.float64, copy=False)
    p = pred[m].astype(np.float64, copy=False)
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if denom < 1e-12 else float(1.0 - float(np.sum(err**2)) / denom)
    return mae, rmse, r2


def _patch_ids_from_preds(model_dir: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(model_dir.glob("*_pred.npy")):
        stem = p.name[: -len("_pred.npy")]
        if stem:
            out.append(stem)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CSV path (default: source_code/figures/model_infer_depth_mae_rmse_r2.csv)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ref_dir = root / "cnn" / "cnn_infer_outputs"
    model_specs: list[tuple[str, str]] = [
        ("mlp/mlp_infer_depth", "MLP"),
        ("new_test/proposed_infer_depth", "BathyAgent"),
        ("rf/rf_infer_depth", "RF"),
        ("unet/unet_infer_depth", "UNet"),
        ("knn/knn_infer_depth", "KNN-RF"),
        ("dpt/dpt_infer_depth", "DPT"),
        ("depth_anythingv2/depth_anythingv2_infer_depth", "Depth Anything V2"),
        ("da-sdb/da_sdb_infer_depth", "DA-SDB"),
        ("cnn/cnn_infer_outputs", "CNN"),
    ]

    out_path = Path(args.out) if str(args.out).strip() else root / "figures" / "model_infer_depth_mae_rmse_r2.csv"

    rows: list[dict[str, str | float]] = []
    for rel, name in model_specs:
        model_dir = (root / rel).resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
        patch_ids = _patch_ids_from_preds(model_dir)
        if not patch_ids:
            raise FileNotFoundError(f"No *_pred.npy under {model_dir}")
        maes: list[float] = []
        rmses: list[float] = []
        r2s: list[float] = []
        for pid in patch_ids:
            pred_path = model_dir / f"{pid}_pred.npy"
            gt_path = ref_dir / f"{pid}_gt.npy"
            mask_path = ref_dir / f"{pid}_valid_mask.npy"
            if not pred_path.is_file():
                continue
            if not gt_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing ref for {pid}: {gt_path} / {mask_path}")
            pred = np.load(pred_path).astype(np.float32)
            gt = np.load(gt_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.float32)
            if pred.shape != gt.shape or pred.shape != mask.shape:
                raise ValueError(f"Shape mismatch {name} {pid}: pred {pred.shape} gt {gt.shape} mask {mask.shape}")
            mae, rmse, r2 = _metrics_flat(gt, pred, mask)
            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)
        rows.append(
            {
                "model": name,
                "mae": float(sum(maes) / len(maes)),
                "rmse": float(sum(rmses) / len(rmses)),
                "r2": float(sum(r2s) / len(r2s)),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "mae", "rmse", "r2"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(out_path.resolve())


if __name__ == "__main__":
    main()
