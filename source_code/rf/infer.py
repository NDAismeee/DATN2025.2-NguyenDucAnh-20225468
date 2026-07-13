import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from joblib import load

from dataset import build_pairs_magic, read_raster

_RF_ROOT = Path(__file__).resolve().parent
_CNN_SRC = _RF_ROOT.parent / "cnn_src"
if _CNN_SRC.is_dir():
    sys.path.insert(0, str(_CNN_SRC))


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        key = obj[2:-1]
        return os.environ.get(key, obj)
    return obj


def load_yaml_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _expand_env(cfg)


def _resolve_data_cfg(raw_d: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(raw_d)
    img = d.get("train_image_dir") or d.get("image_dir")
    if not img:
        raise KeyError("data.image_dir (or data.train_image_dir) is required.")
    d["image_dir"] = img
    return d


DISPLAY_DEPTH_MIN = -20.0
DISPLAY_DEPTH_MAX = 0.0
DEPTH_CMAP = "turbo"
DEPTH_INTERPOLATION = "bilinear"
VIS_SMOOTH_SIGMA = 3.0


def _valid_mask_from_depth(depth_1hw: np.ndarray, magic_negative_depth_valid: bool) -> np.ndarray:
    d = depth_1hw.astype(np.float32, copy=False)
    if magic_negative_depth_valid:
        valid = np.isfinite(d) & (d < 0)
    else:
        valid = np.isfinite(d)
    return valid[0].astype(np.uint8, copy=False)


def _to_positive_depth(depth_1hw: np.ndarray, magic_negative_depth_valid: bool) -> np.ndarray:
    d = depth_1hw.astype(np.float32, copy=False)[0]
    if magic_negative_depth_valid:
        return -d
    return d


def _rgb_hw3(img_chw: np.ndarray, reflectance_scale: float) -> np.ndarray:
    x = img_chw[:3].astype(np.float32, copy=False)
    if reflectance_scale and reflectance_scale != 1.0:
        x = x / float(reflectance_scale)
    return np.transpose(x, (1, 2, 0))


def _pixel_features(rgb: np.ndarray, include_xy: bool) -> np.ndarray:
    h, w, _ = rgb.shape
    feats = [rgb.reshape(-1, 3)]
    if include_xy:
        yy, xx = np.mgrid[0:h, 0:w]
        fx = (xx.astype(np.float32) / max(w - 1, 1)).reshape(-1, 1)
        fy = (yy.astype(np.float32) / max(h - 1, 1)).reshape(-1, 1)
        feats.extend([fx, fy])
    return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


def chw_rgb_preview(image_chw: np.ndarray) -> np.ndarray:
    x = np.transpose(np.nan_to_num(image_chw[:3], nan=0.0, posinf=0.0, neginf=0.0), (1, 2, 0)).astype(np.float32)
    lo = np.percentile(x, 2, axis=(0, 1))
    hi = np.percentile(x, 98, axis=(0, 1))
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(3):
        a, b = float(lo[c]), float(hi[c])
        if b > a:
            out[:, :, c] = np.clip((x[:, :, c] - a) / (b - a), 0.0, 1.0)
    return out


def smooth_masked_2d(arr: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
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


def _masked_mae(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    m = valid > 0
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(pred[m] - gt[m])))


def _masked_rmse(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    m = valid > 0
    if not np.any(m):
        return float("nan")
    r = pred[m] - gt[m]
    return float(np.sqrt(np.mean(r * r)))


def to_display_depth(depth_positive: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return np.where(valid_mask > 0, -depth_positive, np.nan)


def fmt_range(arr: np.ndarray) -> str:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return "[nan, nan]"
    return f"[{float(v.min()):.2f}, {float(v.max()):.2f}]"


def prepare_depth_for_display(depth_positive: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = to_display_depth(depth_positive, valid_mask)
    arr = smooth_masked_2d(arr, valid_mask, sigma_px=VIS_SMOOTH_SIGMA)
    return arr


def add_colorbar(fig, ax, im):
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _panel_na(ax, title: str, subtitle: str) -> None:
    ax.text(0.5, 0.55, subtitle, ha="center", va="center", fontsize=11)
    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal")


def _predict_in_chunks(model, feats: np.ndarray, chunk: int = 200000) -> np.ndarray:
    out = np.zeros((feats.shape[0],), dtype=np.float32)
    n = feats.shape[0]
    i = 0
    while i < n:
        j = min(n, i + int(chunk))
        out[i:j] = model.predict(feats[i:j]).astype(np.float32, copy=False)
        i = j
    return out


def make_vis(
    out_png: Path,
    rgb_preview: np.ndarray,
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    valid_mask: np.ndarray,
    mae: float,
    rmse: float,
    *,
    rgb_only: bool = False,
) -> None:
    vmin = DISPLAY_DEPTH_MIN
    vmax = DISPLAY_DEPTH_MAX

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    axes[0].imshow(rgb_preview, interpolation="nearest")
    axes[0].set_title("Input image")
    axes[0].axis("off")
    axes[0].set_aspect("equal")

    if rgb_only:
        axes[1].set_facecolor("0.15")
        axes[1].text(0.5, 0.5, "No ground truth\n(RGB-only)", ha="center", va="center", color="0.85", fontsize=12)
        axes[1].set_title("Ground truth")
        axes[1].axis("off")
        axes[1].set_aspect("equal")
        pred_vm = np.isfinite(pred_pos).astype(np.float32)
        pred_display = prepare_depth_for_display(pred_pos, pred_vm)
    else:
        vm = valid_mask.astype(np.float32)
        gt_display = prepare_depth_for_display(gt_pos, vm)
        pred_display = prepare_depth_for_display(pred_pos, vm)
        im1 = axes[1].imshow(
            gt_display,
            cmap=DEPTH_CMAP,
            vmin=vmin,
            vmax=vmax,
            interpolation=DEPTH_INTERPOLATION,
        )
        axes[1].set_title(f"Ground truth\n{fmt_range(gt_display)} m")
        axes[1].axis("off")
        axes[1].set_aspect("equal")
        add_colorbar(fig, axes[1], im1)

    im2 = axes[2].imshow(
        pred_display,
        cmap=DEPTH_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation=DEPTH_INTERPOLATION,
    )
    title_metrics = "n/a" if rgb_only else f"MAE={mae:.3f} RMSE={rmse:.3f}"
    axes[2].set_title(f"RF prediction\n{fmt_range(pred_display)} m\n{title_metrics}")
    axes[2].axis("off")
    axes[2].set_aspect("equal")
    add_colorbar(fig, axes[2], im2)

    _panel_na(axes[3], "Raw model μ", "N/A (RF)")
    _panel_na(axes[4], "Physical prior d_phys", "N/A (RF)")
    _panel_na(axes[5], "Gate α", "N/A (RF)")
    _panel_na(axes[6], "Uncertainty var", "N/A (RF)")
    _panel_na(axes[7], "Zone / distance map", "N/A (RF)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--rgb_only", action="store_true")
    ap.add_argument("--image_dir", type=str, default=None)
    ap.add_argument("--depth_dir", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--img_glob", type=str, default=None)
    args = ap.parse_args()

    load_dotenv(dotenv_path=_RF_ROOT / ".env", override=True)
    cfg = dict(load_yaml_config(args.config))
    data_cfg = _resolve_data_cfg(dict(cfg.get("data", {}) or {}))
    infer_cfg = dict(cfg.get("infer", {}) or {})
    train_cfg = cfg.get("train", {}) or {}

    if args.image_dir is not None:
        data_cfg["image_dir"] = str(args.image_dir)
    if args.depth_dir is not None:
        data_cfg["depth_dir"] = str(args.depth_dir)
    if args.img_glob is not None:
        data_cfg["image_suffix"] = str(args.img_glob)
    if args.output_dir is not None:
        infer_cfg["output_dir"] = str(args.output_dir)
    if args.checkpoint is not None:
        infer_cfg["checkpoint"] = str(args.checkpoint)
    cfg["data"] = data_cfg
    cfg["infer"] = infer_cfg

    img_dir = Path(str(data_cfg.get("image_dir", ""))).expanduser()
    depth_dir = Path(str(data_cfg.get("depth_dir", ""))).expanduser()
    image_suffix = str(data_cfg.get("image_suffix", "img_*.tif"))
    depth_suffixes_to_try = data_cfg.get("depth_suffixes_to_try", ["_depth", "_bathy", "_gt", "_label"])
    reflectance_scale = float(data_cfg.get("reflectance_scale", 1.0))
    magic_negative_depth_valid = bool(data_cfg.get("magic_negative_depth_valid", True))

    include_xy = bool(train_cfg.get("include_xy", True))
    rgb_only = bool(args.rgb_only) or bool(infer_cfg.get("rgb_only", False))
    if rgb_only:
        print(
            "[rf infer] rgb_only=True: no depth; MAE/RMSE are n/a. "
            "Set infer.rgb_only: false and data.depth_dir for GT + metrics."
        )

    ckpt_path = Path(str(infer_cfg.get("checkpoint", "checkpoints_rf_rgb/random_forest.joblib"))).expanduser()
    out_dir = Path(str(infer_cfg.get("output_dir", "rf_infer_outputs"))).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load(ckpt_path)
    model = bundle["model"]

    rows: List[Dict[str, Any]] = []

    if rgb_only:
        if not img_dir.is_dir():
            raise ValueError(f"image_dir is not a directory: {img_dir}")
        paths = sorted(img_dir.glob(image_suffix))
        if not paths:
            raise ValueError(f"No images matching {image_suffix!r} under {img_dir}")
        for img_path in paths:
            pid = img_path.stem
            img = read_raster(img_path)
            rgb_preview = chw_rgb_preview(img)
            rgb = _rgb_hw3(img, reflectance_scale=reflectance_scale)
            feats = _pixel_features(rgb, include_xy=include_xy)
            pred_pos = _predict_in_chunks(model, feats, chunk=200000).reshape(rgb.shape[:2])
            pred_path = out_dir / f"{pid}_pred.npy"
            np.save(pred_path, pred_pos.astype(np.float32))
            z = np.zeros_like(pred_pos, dtype=np.float32)
            vis_path = out_dir / f"{pid}_vis.png"
            make_vis(
                out_png=vis_path,
                rgb_preview=rgb_preview,
                gt_pos=z,
                pred_pos=pred_pos,
                valid_mask=z.astype(np.uint8),
                mae=float("nan"),
                rmse=float("nan"),
                rgb_only=True,
            )
            rows.append(
                {
                    "id": pid,
                    "image_path": str(img_path),
                    "depth_path": "",
                    "pred_path": str(pred_path),
                    "vis_path": str(vis_path),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "num_valid": 0,
                }
            )
    else:
        if not str(data_cfg.get("depth_dir", "")).strip():
            raise ValueError("data.depth_dir is required when infer.rgb_only is false.")
        pairs = build_pairs_magic(
            img_dir=img_dir,
            depth_dir=depth_dir,
            image_suffix=image_suffix,
            depth_suffixes_to_try=depth_suffixes_to_try,
        )
        if not pairs:
            raise ValueError("No matched image-depth pairs found. Check IMAGE_DIR/DEPTH_DIR and config.yaml.")

        for img_path, depth_path, pid in pairs:
            img = read_raster(img_path)
            depth = read_raster(depth_path)

            rgb_preview = chw_rgb_preview(img)
            rgb = _rgb_hw3(img, reflectance_scale=reflectance_scale)

            valid = _valid_mask_from_depth(depth, magic_negative_depth_valid=magic_negative_depth_valid)
            gt_pos = _to_positive_depth(depth, magic_negative_depth_valid=magic_negative_depth_valid)

            feats = _pixel_features(rgb, include_xy=include_xy)
            pred_pos = _predict_in_chunks(model, feats, chunk=200000).reshape(gt_pos.shape)

            pred_path = out_dir / f"{pid}_pred.npy"
            np.save(pred_path, pred_pos.astype(np.float32))

            mae = _masked_mae(pred_pos, gt_pos, valid)
            rmse = _masked_rmse(pred_pos, gt_pos, valid)

            vis_path = out_dir / f"{pid}_vis.png"
            make_vis(
                out_png=vis_path,
                rgb_preview=rgb_preview,
                gt_pos=gt_pos,
                pred_pos=pred_pos,
                valid_mask=valid,
                mae=mae,
                rmse=rmse,
                rgb_only=False,
            )

            rows.append(
                {
                    "id": pid,
                    "image_path": str(img_path),
                    "depth_path": str(depth_path),
                    "pred_path": str(pred_path),
                    "vis_path": str(vis_path),
                    "mae": mae,
                    "rmse": rmse,
                    "num_valid": int((valid > 0).sum()),
                }
            )

    out_csv = out_dir / "infer_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "image_path", "depth_path", "pred_path", "vis_path", "mae", "rmse", "num_valid"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

