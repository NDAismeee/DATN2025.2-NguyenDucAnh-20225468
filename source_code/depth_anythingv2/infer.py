import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import BathymetryDataset, read_raster
from model import load_depth_anything_v2, pick_torch_device


DISPLAY_DEPTH_MIN = -20.0
DISPLAY_DEPTH_MAX = 0.0
DEPTH_CMAP = "jet"
DEPTH_INTERPOLATION = "bilinear"
VIS_SMOOTH_SIGMA = 3.0
VIS_FILL_NAN_FOR_DISPLAY = False


def load_yaml_config(path: str) -> Dict[str, Any]:
    import yaml
    import os
    import re

    env_pat = re.compile(r"\$\{([^}]+)\}")

    def _expand(v):
        if isinstance(v, str):
            return env_pat.sub(lambda m: os.environ.get(m.group(1), m.group(0)), v)
        if isinstance(v, list):
            return [_expand(x) for x in v]
        if isinstance(v, dict):
            return {k: _expand(x) for k, x in v.items()}
        return v

    with open(path, "r", encoding="utf-8") as f:
        return _expand(yaml.safe_load(f) or {})


def denormalize_depth(depth_arr: np.ndarray, depth_mean: Optional[float], depth_std: Optional[float], normalize_depth: bool) -> np.ndarray:
    out = depth_arr.astype(np.float32, copy=True)
    if normalize_depth and depth_mean is not None and depth_std is not None:
        out = out * float(depth_std) + float(depth_mean)
    return out


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


def prepare_depth_for_display(depth_positive: np.ndarray, valid_mask: np.ndarray, sigma_px: float = VIS_SMOOTH_SIGMA, fill_nan: bool = VIS_FILL_NAN_FOR_DISPLAY) -> np.ndarray:
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


def save_depth_anythingv2_figure(
    rgb: np.ndarray,
    gt_m: np.ndarray,
    pred_m: np.ndarray,
    valid_hw: np.ndarray,
    out_path: str,
    title: str,
    *,
    rgb_only: bool = False,
) -> None:
    vmin = DISPLAY_DEPTH_MIN
    vmax = DISPLAY_DEPTH_MAX

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("Input image")
    axes[0].axis("off")
    axes[0].set_aspect("equal")

    if rgb_only:
        axes[1].set_facecolor("0.15")
        axes[1].text(0.5, 0.5, "No ground truth\n(RGB-only)", ha="center", va="center", color="0.85", fontsize=12)
        axes[1].set_title("Ground truth")
        axes[1].axis("off")
        axes[1].set_aspect("equal")
        pred_vm = np.isfinite(pred_m).astype(np.float32)
        pred_display = prepare_depth_for_display(np.abs(pred_m), pred_vm, sigma_px=VIS_SMOOTH_SIGMA)
    else:
        vm = valid_hw.astype(np.float32)
        gt_display = prepare_depth_for_display(gt_m, vm, sigma_px=VIS_SMOOTH_SIGMA)
        pred_display = prepare_depth_for_display(pred_m, vm, sigma_px=VIS_SMOOTH_SIGMA)
        im1 = axes[1].imshow(gt_display, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, interpolation=DEPTH_INTERPOLATION)
        axes[1].set_title(f"Ground truth\n{fmt_range(gt_display)} m")
        axes[1].axis("off")
        axes[1].set_aspect("equal")
        add_colorbar(fig, axes[1], im1)

    im2 = axes[2].imshow(pred_display, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, interpolation=DEPTH_INTERPOLATION)
    axes[2].set_title(f"Final prediction\n{fmt_range(pred_display)} m")
    axes[2].axis("off")
    axes[2].set_aspect("equal")
    add_colorbar(fig, axes[2], im2)

    _panel_na(axes[3], "Raw model μ", "N/A (DepthAnythingV2)")
    _panel_na(axes[4], "Physical prior d_phys", "N/A (DepthAnythingV2)")
    _panel_na(axes[5], "Gate α", "N/A (DepthAnythingV2)")
    _panel_na(axes[6], "Uncertainty var", "N/A (DepthAnythingV2)")
    _panel_na(axes[7], "Zone / distance map", "N/A (DepthAnythingV2)")

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    p = Path(out_path).parent
    if str(p):
        p.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys = ["sample_idx", "sample_id", "mae", "rmse", "fig_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def infer_one(dataset: BathymetryDataset, idx: int, model: torch.nn.Module, cfg: Dict[str, Any], device: torch.device, output_dir: str) -> Dict[str, Any]:
    sample = dataset[idx]
    img_path = dataset.pairs[idx][0]
    patch_id = str(sample["patch_id"])

    raw_img = read_raster(img_path).astype(np.float32)
    rs = float(cfg["data"].get("reflectance_scale", 255.0)) or 1.0
    rgb = chw_rgb_preview(raw_img[:3] / rs)

    depth_n = sample["depth"]
    vm = sample["valid_mask"]

    raw_rgb_u8 = np.clip((raw_img[:3] / rs) * 255.0, 0, 255).astype(np.uint8)
    raw_hwc = np.transpose(raw_rgb_u8, (1, 2, 0))
    raw_bgr = cv2.cvtColor(raw_hwc, cv2.COLOR_RGB2BGR)

    inf = cfg.get("infer", {}) or {}
    input_size = int(inf.get("input_size", 518))

    pred_raw = model.infer_image(raw_bgr, input_size=input_size).astype(np.float32)
    gt_np_n = depth_n.squeeze(0).cpu().numpy().astype(np.float32)
    mask_np = vm.squeeze(0).cpu().numpy().astype(np.float32)

    nd = bool(inf.get("normalize_depth", False))
    dm = inf.get("depth_mean", None)
    ds = inf.get("depth_std", None)

    gt_m = denormalize_depth(gt_np_n, dm, ds, nd)

    m = (mask_np > 0) & np.isfinite(gt_m) & np.isfinite(pred_raw)
    if np.any(m):
        x = pred_raw[m].astype(np.float64)
        y = gt_m[m].astype(np.float64)
        vx = float(np.var(x))
        if vx > 1e-12:
            a = float(np.cov(x, y, bias=True)[0, 1] / vx)
        else:
            a = 0.0
        b = float(np.mean(y) - a * np.mean(x))
        pred_m = (a * pred_raw + b).astype(np.float32)
    else:
        pred_m = pred_raw.astype(np.float32)
    pred_m = np.clip(pred_m, 0.0, np.inf)

    mae = masked_mae_np(pred_m, gt_m, mask_np)
    rmse = masked_rmse_np(pred_m, gt_m, mask_np)

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{patch_id}_vis.png")
    title = f"{patch_id} | MAE={mae:.4f} m | RMSE={rmse:.4f} m | DepthAnythingV2 infer"
    save_depth_anythingv2_figure(rgb, gt_m, pred_m, mask_np, fig_path, title, rgb_only=False)

    np.save(os.path.join(output_dir, f"{patch_id}_pred.npy"), pred_m.astype(np.float32))
    np.save(os.path.join(output_dir, f"{patch_id}_gt.npy"), gt_m.astype(np.float32))
    np.save(os.path.join(output_dir, f"{patch_id}_valid_mask.npy"), mask_np.astype(np.float32))

    return {"sample_idx": idx, "sample_id": patch_id, "mae": mae, "rmse": rmse, "fig_path": fig_path}


def infer_one_rgb_only(
    img_path: Path,
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    output_dir: str,
    sample_idx: int,
) -> Dict[str, Any]:
    patch_id = img_path.stem
    raw_img = read_raster(img_path).astype(np.float32)
    rs = float(cfg["data"].get("reflectance_scale", 255.0)) or 1.0
    rgb = chw_rgb_preview(raw_img[:3] / rs)

    raw_rgb_u8 = np.clip((raw_img[:3] / rs) * 255.0, 0, 255).astype(np.uint8)
    raw_hwc = np.transpose(raw_rgb_u8, (1, 2, 0))
    raw_bgr = cv2.cvtColor(raw_hwc, cv2.COLOR_RGB2BGR)

    inf = cfg.get("infer", {}) or {}
    input_size = int(inf.get("input_size", 518))

    pred_raw = model.infer_image(raw_bgr, input_size=input_size).astype(np.float32)
    pred_m = np.clip(pred_raw, 0.0, np.inf)

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{patch_id}_vis.png")
    title = f"{patch_id} | RGB-only pretrained | DepthAnythingV2"
    z = np.zeros_like(pred_m, dtype=np.float32)
    save_depth_anythingv2_figure(rgb, z, pred_m, z, fig_path, title, rgb_only=True)

    np.save(os.path.join(output_dir, f"{patch_id}_pred.npy"), pred_m.astype(np.float32))

    return {
        "sample_idx": sample_idx,
        "sample_id": patch_id,
        "mae": float("nan"),
        "rmse": float("nan"),
        "fig_path": fig_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--rgb_only", action="store_true", help="Infer every RGB in image_dir without depth pairing (pretrained relative depth).")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--depth_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--img_glob", type=str, default=None)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env")
    except Exception:
        pass

    cfg = load_yaml_config(args.config)
    cfg = dict(cfg)
    data = dict(cfg.get("data", {}) or {})
    inf = dict(cfg.get("infer", {}) or {})

    if args.image_dir is not None:
        data["image_dir"] = str(args.image_dir)
    if args.depth_dir is not None:
        data["depth_dir"] = str(args.depth_dir)
    if args.img_glob is not None:
        data["image_suffix"] = str(args.img_glob)
    if args.output_dir is not None:
        inf["output_dir"] = str(args.output_dir)
    if args.checkpoint is not None:
        inf["checkpoint"] = str(args.checkpoint)
    cfg["data"] = data
    cfg["infer"] = inf

    rgb_only = bool(args.rgb_only) or bool(inf.get("rgb_only", False))

    device = pick_torch_device(str(inf.get("device", "cpu")), int(inf.get("gpu_id", 0)))
    print(f"[depth_anythingv2 infer] device={device} rgb_only={rgb_only}")
    if rgb_only:
        print(
            "[depth_anythingv2 infer] rgb_only=True: no depth is loaded; "
            "figures show 'No ground truth'. Set infer.rgb_only: false and data.depth_dir "
            "to use magic pairing (e.g. img_47.tif with depth_47.tif)."
        )

    checkpoint = str(inf.get("checkpoint", "depth_anything_v2_vitl.pth"))
    encoder = str(inf.get("encoder", "vitl"))
    model, _ = load_depth_anything_v2(checkpoint, encoder, device)

    output_dir = str(inf.get("output_dir", "depth_anythingv2_infer_outputs"))
    rows: List[Dict[str, Any]] = []

    if rgb_only:
        img_dir = Path(str(data.get("image_dir", ""))).expanduser()
        if not img_dir.is_dir():
            raise ValueError(f"image_dir is not a directory: {img_dir}")
        img_glob = str(data.get("image_suffix", "img_*.tif"))
        paths = sorted(img_dir.glob(img_glob))
        if not paths:
            raise ValueError(f"No images matching {img_glob!r} under {img_dir}")
        n = len(paths)
        start = max(0, int(args.start_idx))
        end = n if args.end_idx is None else int(args.end_idx)
        sub = paths[start:end]
        for k, img_path in enumerate(sub):
            if len(sub) > 1:
                print(f"[{k + 1}/{len(sub)}] {img_path.name} ...", flush=True)
            row = infer_one_rgb_only(img_path, model, cfg, output_dir, sample_idx=start + k)
            rows.append(row)
            if len(sub) > 1:
                print(f"    -> {row['sample_id']} -> {row['fig_path']}")
    else:
        dataset = BathymetryDataset(
            img_dir=str(data["image_dir"]),
            depth_dir=str(data["depth_dir"]),
            img_glob=str(data.get("image_suffix", "img_*.tif")),
            depth_suffixes_to_try=data.get("depth_suffixes_to_try"),
            image_mode=str(data.get("image_mode", "rgb")),
            reflectance_scale=float(data.get("reflectance_scale", 255.0)),
            magic_negative_depth=bool(data.get("magic_negative_depth_valid", True)),
            normalize_depth=bool(inf.get("normalize_depth", False)),
            depth_mean=inf.get("depth_mean"),
            depth_std=inf.get("depth_std"),
        )

        n = len(dataset)
        start = max(0, int(args.start_idx))
        end = n if args.end_idx is None else int(args.end_idx)
        indices = list(range(start, end)) if args.all else [start]

        for k, idx in enumerate(indices):
            if len(indices) > 1:
                print(f"[{k + 1}/{len(indices)}] idx={idx} ...", flush=True)
            row = infer_one(dataset, idx, model, cfg, device, output_dir)
            rows.append(row)
            if len(indices) > 1:
                print(f"    -> {row['sample_id']} MAE={row['mae']:.6f} RMSE={row['rmse']:.6f} -> {row['fig_path']}")

    if len(rows) > 1 or (rgb_only and len(rows) >= 1):
        sp = os.path.join(output_dir, "infer_summary.csv")
        write_summary_csv(rows, sp)
        print(f"[depth_anythingv2 infer] wrote {sp} ({len(rows)} rows).")


if __name__ == "__main__":
    main()

