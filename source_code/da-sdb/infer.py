import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import BathymetryDataset, read_raster
from model import DASDB


DISPLAY_DEPTH_MIN = -20.0
DISPLAY_DEPTH_MAX = 0.0
DEPTH_CMAP = "jet"
DEPTH_INTERPOLATION = "bilinear"
VIS_SMOOTH_SIGMA = 3.0
VIS_FILL_NAN_FOR_DISPLAY = False

_ROOT = Path(__file__).resolve().parent


def load_yaml_config(path: str) -> Dict[str, Any]:
    import yaml

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


def pick_torch_device(device_pref: str, gpu_id: int = 0):
    pref = (device_pref or "auto").strip().lower()
    if pref == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_id}")


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


def prepare_depth_for_display(depth_positive: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = to_display_depth(depth_positive, valid_mask)
    return smooth_masked_2d(arr, valid_mask, sigma_px=VIS_SMOOTH_SIGMA)


def add_colorbar(fig, ax, im):
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _panel_na(ax, title: str, subtitle: str) -> None:
    ax.text(0.5, 0.55, subtitle, ha="center", va="center", fontsize=11)
    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal")


def save_figure(rgb: np.ndarray, gt_m: np.ndarray, pred_m: np.ndarray, valid_hw: np.ndarray, out_path: str, title: str) -> None:
    vm = valid_hw.astype(np.float32)
    gt_display = prepare_depth_for_display(gt_m, vm)
    pred_display = prepare_depth_for_display(pred_m, vm)
    vmin = DISPLAY_DEPTH_MIN
    vmax = DISPLAY_DEPTH_MAX

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("Input image")
    axes[0].axis("off")
    axes[0].set_aspect("equal")

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

    _panel_na(axes[3], "Raw model μ", "N/A (DA-SDB)")
    _panel_na(axes[4], "Physical prior d_phys", "N/A (DA-SDB)")
    _panel_na(axes[5], "Gate α", "N/A (DA-SDB)")
    _panel_na(axes[6], "Uncertainty var", "N/A (DA-SDB)")
    _panel_na(axes[7], "Zone / distance map", "N/A (DA-SDB)")

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


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[DASDB, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    hidden = tuple(int(x) for x in (model_cfg.get("hidden_channels") or [32, 64, 64, 32]))
    model = DASDB(
        in_channels=3,
        hidden_channels=hidden,
        dropout=float(model_cfg.get("dropout", 0.05)),
        use_coordconv=bool(model_cfg.get("use_coordconv", True)),
        norm_type=str(model_cfg.get("norm_type", "group")),
        num_groups=int(model_cfg.get("num_groups", 8)),
        domain_hidden=int(model_cfg.get("domain_hidden", 128)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(_ROOT / "config.yaml"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--domain", type=str, default="target", choices=["source", "target"])
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except Exception:
        pass

    cfg = load_yaml_config(args.config)
    d = cfg.get("data", {}) or {}
    t = cfg.get("train", {}) or {}
    inf = cfg.get("infer", {}) or {}

    device = pick_torch_device(str(t.get("device", "cpu")), int(t.get("gpu_id", 0)))
    print(f"[da-sdb infer] device={device}")

    ckpt = str(inf.get("checkpoint", "checkpoints_da_sdb/best_model.pt"))
    model, ckpt_cfg = load_model(ckpt, device)

    stats = (ckpt_cfg.get("stats") or {})
    dm = stats.get("depth_mean")
    ds = stats.get("depth_std")
    nd = bool((ckpt_cfg.get("train") or {}).get("normalize_depth", True))

    if args.domain == "source":
        img_dir = str(d["source_image_dir"])
        depth_dir = str(d["source_depth_dir"])
    else:
        img_dir = str(d["target_image_dir"])
        depth_dir = str(d.get("target_depth_dir") or d["source_depth_dir"])

    dataset = BathymetryDataset(
        img_dir=img_dir,
        depth_dir=depth_dir,
        img_glob=str(d.get("image_suffix", "img_*.tif")),
        depth_glob="depth_*.tif",
        selected_bands=None,
        normalize=bool((ckpt_cfg.get("train") or {}).get("normalize", True)),
        mean=stats.get("mean"),
        std=stats.get("std"),
        normalize_depth=nd,
        depth_mean=dm,
        depth_std=ds,
        depth_min=None,
        depth_max=None,
        invalid_depth_values=[],
        return_metadata=False,
        pairing_mode="magic",
        depth_suffixes_to_try=d.get("depth_suffixes_to_try"),
        image_mode=str(d.get("image_mode", "rgb")),
        reflectance_scale=float(d.get("reflectance_scale", 255.0)),
        magic_negative_depth=bool(d.get("magic_negative_depth_valid", True)),
    )

    n = len(dataset)
    start = max(0, int(args.start_idx))
    end = n if args.end_idx is None else int(args.end_idx)
    indices = list(range(start, end)) if args.all else [start]

    out_dir = str(inf.get("output_dir", "da_sdb_infer_outputs"))
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for k, idx in enumerate(indices):
        if len(indices) > 1:
            print(f"[{k + 1}/{len(indices)}] idx={idx} ...", flush=True)
        sample = dataset[idx]
        img_path = dataset.pairs[idx][0]
        patch_id = str(sample["patch_id"])

        raw_img = read_raster(img_path).astype(np.float32)
        rs = float(d.get("reflectance_scale", 255.0)) or 1.0
        rgb = chw_rgb_preview(raw_img[:3] / rs)

        x = sample["image"].unsqueeze(0).to(device)
        depth_n = sample["depth"]
        vm = sample["valid_mask"]

        with torch.inference_mode():
            pred_n = model.forward_depth(x)
            pred_n = torch.nan_to_num(pred_n, nan=0.0, posinf=0.0, neginf=0.0)

        pred_np_n = pred_n.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        gt_np_n = depth_n.squeeze(0).cpu().numpy().astype(np.float32)
        mask_np = vm.squeeze(0).cpu().numpy().astype(np.float32)

        pred_m = denormalize_depth(pred_np_n, dm, ds, nd)
        gt_m = denormalize_depth(gt_np_n, dm, ds, nd)
        mae = masked_mae_np(pred_m, gt_m, mask_np)
        rmse = masked_rmse_np(pred_m, gt_m, mask_np)

        fig_path = os.path.join(out_dir, f"{patch_id}_vis.png")
        title = f"{patch_id} | MAE={mae:.4f} m | RMSE={rmse:.4f} m | DA-SDB infer ({args.domain})"
        save_figure(rgb, gt_m, pred_m, mask_np, fig_path, title)

        np.save(os.path.join(out_dir, f"{patch_id}_pred.npy"), pred_m.astype(np.float32))
        np.save(os.path.join(out_dir, f"{patch_id}_gt.npy"), gt_m.astype(np.float32))
        np.save(os.path.join(out_dir, f"{patch_id}_valid_mask.npy"), mask_np.astype(np.float32))

        row = {"sample_idx": idx, "sample_id": patch_id, "mae": mae, "rmse": rmse, "fig_path": fig_path}
        rows.append(row)
        if len(indices) > 1:
            print(f"    -> {patch_id} MAE={mae:.6f} RMSE={rmse:.6f} -> {fig_path}")

    if len(rows) > 1:
        sp = os.path.join(out_dir, "infer_summary.csv")
        write_summary_csv(rows, sp)
        print(f"[da-sdb infer] wrote {sp} ({len(rows)} rows).")


if __name__ == "__main__":
    main()

