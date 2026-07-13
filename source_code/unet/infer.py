import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

from common import load_yaml_config, pick_torch_device
from dataset import BathymetryDataset, build_pairs_magic, read_raster
from model import PretrainedAerialUNet


DISPLAY_DEPTH_MIN = -20.0
DISPLAY_DEPTH_MAX = 0.0
DEPTH_CMAP = "turbo"
DEPTH_INTERPOLATION = "bilinear"
VIS_SMOOTH_SIGMA = 3.0
VIS_FILL_NAN_FOR_DISPLAY = False


def chw_rgb_preview(image_chw: np.ndarray) -> np.ndarray:
    x = np.transpose(np.nan_to_num(image_chw[:3], nan=0.0, posinf=0.0, neginf=0.0), (1, 2, 0))
    lo = np.percentile(x, 2, axis=(0, 1))
    hi = np.percentile(x, 98, axis=(0, 1))
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(3):
        a, b = float(lo[c]), float(hi[c])
        if b > a:
            out[:, :, c] = np.clip((x[:, :, c] - a) / (b - a), 0.0, 1.0)
    return out


def fmt_range(arr: np.ndarray) -> str:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return "[nan, nan]"
    return f"[{float(v.min()):.2f}, {float(v.max()):.2f}]"


def to_display_depth(depth_positive: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return np.where(valid_mask > 0, -depth_positive, np.nan)


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


def fill_nan_with_local_mean(arr: np.ndarray, mask: np.ndarray, sigma_px: float = 3.0) -> np.ndarray:
    valid = np.isfinite(arr) & (mask > 0)
    if valid.sum() == 0:
        return arr
    base = np.where(valid, arr, 0.0).astype(np.float32)
    w = valid.astype(np.float32)
    k = int(max(3, round(sigma_px * 6)))
    if k % 2 == 0:
        k += 1
    num = cv2.GaussianBlur(base, (k, k), float(sigma_px))
    den = cv2.GaussianBlur(w, (k, k), float(sigma_px))
    filled = num / np.maximum(den, 1e-6)
    out = arr.copy()
    out[~np.isfinite(out) & (mask > 0)] = filled[~np.isfinite(out) & (mask > 0)]
    return out


def prepare_depth_for_display(
    depth_positive: np.ndarray,
    valid_mask: np.ndarray,
    sigma_px: float = VIS_SMOOTH_SIGMA,
    fill_nan: bool = VIS_FILL_NAN_FOR_DISPLAY,
) -> np.ndarray:
    arr = to_display_depth(depth_positive, valid_mask)
    arr = smooth_masked_2d(arr, valid_mask, sigma_px=sigma_px)
    if fill_nan:
        arr = fill_nan_with_local_mean(arr, valid_mask, sigma_px=max(1.0, sigma_px))
    return arr


def add_colorbar(fig, ax, im):
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _panel_na(ax, title: str, subtitle: str) -> None:
    ax.text(0.5, 0.55, subtitle, ha="center", va="center", fontsize=11)
    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal")


def save_cnn_figure(
    rgb: np.ndarray,
    gt_m: np.ndarray,
    pred_m: np.ndarray,
    valid_hw: np.ndarray,
    out_path: str,
    title: str,
    show: bool,
) -> None:
    vm = valid_hw.astype(np.float32)
    gt_display = prepare_depth_for_display(gt_m, vm, sigma_px=VIS_SMOOTH_SIGMA)
    pred_display = prepare_depth_for_display(pred_m, vm, sigma_px=VIS_SMOOTH_SIGMA)
    vmin = DISPLAY_DEPTH_MIN
    vmax = DISPLAY_DEPTH_MAX

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("Input image")
    axes[0].axis("off")
    axes[0].set_aspect("equal")

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
    axes[2].set_title(f"Final prediction\n{fmt_range(pred_display)} m")
    axes[2].axis("off")
    axes[2].set_aspect("equal")
    add_colorbar(fig, axes[2], im2)

    _panel_na(axes[3], "Raw model μ", "N/A (UNet)")
    _panel_na(axes[4], "Physical prior d_phys", "N/A (UNet)")
    _panel_na(axes[5], "Gate α", "N/A (UNet)")
    _panel_na(axes[6], "Uncertainty var", "N/A (UNet)")
    _panel_na(axes[7], "Zone / distance map", "N/A (UNet)")

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    p = Path(out_path).parent
    if str(p):
        p.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    if show and str(plt.get_backend()).lower() != "agg":
        plt.show()
    plt.close(fig)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[PretrainedAerialUNet, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    bilinear = bool(((cfg.get("model", {}) or {}).get("bilinear", False))) if isinstance(cfg, dict) else False
    model = PretrainedAerialUNet(n_channels=3, n_classes=1, bilinear=bilinear)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg if isinstance(cfg, dict) else {}


def masked_mae_np(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(pred[m] - gt[m])))


def masked_rmse_np(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0
    if not np.any(m):
        return float("nan")
    r = pred[m] - gt[m]
    return float(np.sqrt(np.mean(r * r)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=root / ".env", override=True)
    cfg = load_yaml_config(args.config)

    data_cfg = cfg.get("data", {}) or {}
    infer_cfg = cfg.get("infer", {}) or {}
    tr_cfg = cfg.get("train", {}) or {}

    device = pick_torch_device(str(tr_cfg.get("device", "auto")), gpu_id=int(tr_cfg.get("gpu_id", 0)))

    img_dir = Path(str(data_cfg.get("image_dir", ""))).expanduser()
    depth_dir = Path(str(data_cfg.get("depth_dir", ""))).expanduser()
    image_suffix = str(data_cfg.get("image_suffix", "img_*.tif"))
    depth_suffixes_to_try = data_cfg.get("depth_suffixes_to_try", ["_depth", "_bathy", "_gt", "_label"])
    reflectance_scale = float(data_cfg.get("reflectance_scale", 1.0))
    magic_negative_depth = bool(data_cfg.get("magic_negative_depth_valid", True))

    ckpt_path = Path(str(infer_cfg.get("checkpoint", ""))).expanduser()
    out_dir = Path(str(infer_cfg.get("output_dir", "unet_infer_outputs"))).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _ckpt_cfg = load_model(str(ckpt_path), device=device)

    pairs = build_pairs_magic(
        img_dir=img_dir,
        depth_dir=depth_dir,
        image_suffix=image_suffix,
        depth_suffixes_to_try=depth_suffixes_to_try,
    )
    if not pairs:
        raise ValueError("No matched image-depth pairs found. Check IMAGE_DIR/DEPTH_DIR and config.yaml.")

    dataset = BathymetryDataset(
        img_dir=str(img_dir),
        depth_dir=str(depth_dir),
        img_glob=image_suffix,
        depth_glob="depth_*.tif",
        selected_bands=None,
        normalize=False,
        mean=None,
        std=None,
        normalize_depth=False,
        pairing_mode="magic",
        depth_suffixes_to_try=depth_suffixes_to_try,
        image_mode=str(data_cfg.get("image_mode", "rgb")),
        reflectance_scale=reflectance_scale,
        magic_negative_depth=magic_negative_depth,
    )
    dataset.pairs = pairs

    rows: List[Dict[str, Any]] = []
    nb = device.type == "cuda"
    for idx in range(len(dataset)):
        sample = dataset[idx]
        img_path = dataset.pairs[idx][0]
        patch_id = str(sample["patch_id"])

        x = sample["image"].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=nb)
        gt_pos = sample["depth"].squeeze(0).numpy().astype(np.float32, copy=False)
        vm = sample["valid_mask"].squeeze(0).numpy().astype(np.float32, copy=False)

        raw = read_raster(Path(img_path)).astype(np.float32)
        rs = float(reflectance_scale) if reflectance_scale else 1.0
        rgb_preview = chw_rgb_preview(raw[:3] / rs)

        with torch.inference_mode():
            pred = model(x)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        pred_pos = pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

        pred_npy = out_dir / f"{patch_id}_pred.npy"
        gt_npy = out_dir / f"{patch_id}_gt.npy"
        vm_npy = out_dir / f"{patch_id}_valid_mask.npy"
        np.save(pred_npy, pred_pos.astype(np.float32))
        np.save(gt_npy, gt_pos.astype(np.float32))
        np.save(vm_npy, vm.astype(np.float32))

        mae = masked_mae_np(pred_pos, gt_pos, vm)
        rmse = masked_rmse_np(pred_pos, gt_pos, vm)

        fig_path = out_dir / f"{patch_id}_vis.png"
        title = f"{patch_id} | UNet | MAE={mae:.3f} RMSE={rmse:.3f}"
        save_cnn_figure(rgb_preview, gt_pos, pred_pos, vm, str(fig_path), title, show=False)

        rows.append(
            {
                "sample_idx": idx,
                "sample_id": patch_id,
                "mae": mae,
                "rmse": rmse,
                "fig_path": str(fig_path),
                "image_path": str(img_path),
                "pred_npy_path": str(pred_npy),
                "gt_npy_path": str(gt_npy),
                "valid_mask_npy_path": str(vm_npy),
            }
        )

    out_csv = out_dir / "infer_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "sample_id",
                "mae",
                "rmse",
                "fig_path",
                "image_path",
                "pred_npy_path",
                "gt_npy_path",
                "valid_mask_npy_path",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

