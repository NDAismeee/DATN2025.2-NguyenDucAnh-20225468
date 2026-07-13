import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import BathymetryDataset, read_raster, S2_BAND_TO_INDEX
from model import DensePredictionTransformer


DISPLAY_DEPTH_MIN = -20.0
DISPLAY_DEPTH_MAX = 0.0
DEPTH_CMAP = "jet"
DEPTH_INTERPOLATION = "bilinear"
VIS_SMOOTH_SIGMA = 3.0

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


def _resolve_data_cfg(raw_d: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(raw_d)
    img = d.get("train_image_dir") or d.get("image_dir")
    if not img:
        raise KeyError("data.image_dir (or data.train_image_dir) is required.")
    d["image_dir"] = img
    return d


def _band_indices(image_mode: str) -> List[int]:
    mode = (image_mode or "rgb").lower().strip()
    if mode == "rgb":
        return [0, 1, 2]
    return [S2_BAND_TO_INDEX[b] for b in S2_BAND_TO_INDEX.keys()]


def _stats_mean_std_arr(stats: Dict[str, Any], n_bands: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        return None, None
    m = np.asarray(mean, dtype=np.float32).reshape(-1)
    s = np.asarray(std, dtype=np.float32).reshape(-1)
    if m.size != n_bands or s.size != n_bands:
        return None, None
    s = np.maximum(s, 1e-6)
    return m, s


def image_tensor_from_path(
    img_path: Path,
    band_indices: List[int],
    reflectance_scale: float,
    normalize: bool,
    mean_np: Optional[np.ndarray],
    std_np: Optional[np.ndarray],
) -> torch.Tensor:
    img = read_raster(img_path).astype(np.float32)
    img = img[band_indices]
    rs = float(reflectance_scale) if reflectance_scale else 1.0
    img = img / rs
    img = np.clip(img, 0.0, 1.5)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize and mean_np is not None and std_np is not None:
        img = (img - mean_np[:, None, None]) / std_np[:, None, None]
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(img).float()


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


def save_figure(
    rgb: np.ndarray,
    gt_m: np.ndarray,
    pred_m: np.ndarray,
    valid_hw: np.ndarray,
    out_path: str,
    *,
    rgb_only: bool = False,
    patch_id: str = "",
    mae: float = float("nan"),
    rmse: float = float("nan"),
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
        pred_display = prepare_depth_for_display(pred_m, pred_vm)
    else:
        vm = valid_hw.astype(np.float32)
        gt_display = prepare_depth_for_display(gt_m, vm)
        pred_display = prepare_depth_for_display(pred_m, vm)
        im1 = axes[1].imshow(gt_display, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, interpolation=DEPTH_INTERPOLATION)
        axes[1].set_title(f"Ground truth\n{fmt_range(gt_display)} m")
        axes[1].axis("off")
        axes[1].set_aspect("equal")
        add_colorbar(fig, axes[1], im1)

    im2 = axes[2].imshow(pred_display, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, interpolation=DEPTH_INTERPOLATION)
    title_metrics = "n/a" if rgb_only else f"MAE={mae:.3f} RMSE={rmse:.3f}"
    axes[2].set_title(f"DPT prediction\n{fmt_range(pred_display)} m\n{title_metrics}")
    axes[2].axis("off")
    axes[2].set_aspect("equal")
    add_colorbar(fig, axes[2], im2)

    _panel_na(axes[3], "Raw model μ", "N/A (DPT)")
    _panel_na(axes[4], "Physical prior d_phys", "N/A (DPT)")
    _panel_na(axes[5], "Gate α", "N/A (DPT)")
    _panel_na(axes[6], "Uncertainty var", "N/A (DPT)")
    _panel_na(axes[7], "Zone / distance map", "N/A (DPT)")

    if patch_id:
        st = f"{patch_id} | DPT infer (RGB-only, metrics n/a)" if rgb_only else f"{patch_id} | MAE={mae:.4f} m | RMSE={rmse:.4f} m | DPT infer"
        fig.suptitle(st, fontsize=13)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys = ["id", "image_path", "depth_path", "pred_path", "vis_path", "mae", "rmse", "num_valid"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[DensePredictionTransformer, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {}) or {}
    m = cfg.get("model", {}) or {}
    model = DensePredictionTransformer(
        in_channels=3,
        width=int(m.get("width", 128)),
        patch_size=int(m.get("patch_size", 8)),
        layers=int(m.get("layers", 2)),
        heads=int(m.get("heads", 4)),
        dropout=float(m.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--rgb_only", action="store_true")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--depth_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints_dpt_depth/best_model.pt")
    parser.add_argument("--img_glob", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env", override=True)
    except Exception:
        pass

    cfg = dict(load_yaml_config(args.config))
    data_cfg = _resolve_data_cfg(dict(cfg.get("data", {}) or {}))
    infer_cfg = dict(cfg.get("infer", {}) or {})
    t = cfg.get("train", {}) or {}

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

    d = data_cfg
    inf = infer_cfg
    dev_pref = str(infer_cfg.get("device") or t.get("device", "cpu"))
    gpu_id = int(infer_cfg["gpu_id"]) if "gpu_id" in infer_cfg else int(t.get("gpu_id", 0))
    device = pick_torch_device(dev_pref, gpu_id)
    print(f"[dpt infer] device={device}")

    ckpt_path = str(inf.get("checkpoint", "checkpoints_dpt_rgb/best_model.pt"))
    model, ckpt_cfg = load_model(ckpt_path, device)
    stats = (ckpt_cfg.get("stats") or {})
    dm = stats.get("depth_mean")
    ds = stats.get("depth_std")
    nd = bool((ckpt_cfg.get("train") or {}).get("normalize_depth", True))
    normalize_img = bool((ckpt_cfg.get("train") or {}).get("normalize", True))

    rgb_only = bool(args.rgb_only) or bool(infer_cfg.get("rgb_only", False))
    if rgb_only:
        print(
            "[dpt infer] rgb_only=True: no depth; MAE/RMSE are n/a. "
            "Set infer.rgb_only: false and data.depth_dir for GT + metrics."
        )

    out_dir = str(inf.get("output_dir", "dpt_infer_outputs"))
    os.makedirs(out_dir, exist_ok=True)
    out_p = Path(out_dir)

    img_dir = Path(str(d.get("image_dir", ""))).expanduser()
    image_suffix = str(d.get("image_suffix", "img_*.tif"))
    reflectance_scale = float(d.get("reflectance_scale", 255.0))
    image_mode = str(d.get("image_mode", "rgb"))
    band_indices = _band_indices(image_mode)
    mean_np, std_np = _stats_mean_std_arr(stats, len(band_indices))

    rows: List[Dict[str, Any]] = []

    if rgb_only:
        if not img_dir.is_dir():
            raise ValueError(f"image_dir is not a directory: {img_dir}")
        paths_all = sorted(img_dir.glob(image_suffix))
        if not paths_all:
            raise ValueError(f"No images matching {image_suffix!r} under {img_dir}")
        start = max(0, int(args.start_idx))
        end = len(paths_all) if args.end_idx is None else int(args.end_idx)
        indices_paths = list(range(start, min(end, len(paths_all)))) if args.all else ([start] if start < len(paths_all) else [])
        if not indices_paths:
            raise ValueError(f"No samples in range start_idx={start} end_idx={end} (n_images={len(paths_all)}).")
        if normalize_img and (mean_np is None or std_np is None):
            raise ValueError(
                "RGB-only infer needs checkpoint stats.mean / stats.std with length matching image bands "
                "(same as training), or disable image normalize in the saved train config."
            )
        for k, pi in enumerate(indices_paths):
            img_path = paths_all[pi]
            patch_id = img_path.stem
            if len(indices_paths) > 1:
                print(f"[{k + 1}/{len(indices_paths)}] {patch_id} ...", flush=True)
            raw_img = read_raster(img_path).astype(np.float32)
            rs = reflectance_scale or 1.0
            rgb = chw_rgb_preview(raw_img[:3] / rs)
            x = image_tensor_from_path(
                img_path,
                band_indices,
                reflectance_scale,
                normalize_img,
                mean_np,
                std_np,
            ).unsqueeze(0).to(device)
            with torch.inference_mode():
                pred_n = model(x)
                pred_n = torch.nan_to_num(pred_n, nan=0.0, posinf=0.0, neginf=0.0)
            pred_np_n = pred_n.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            pred_m = denormalize_depth(pred_np_n, dm, ds, nd)
            z = np.zeros_like(pred_m, dtype=np.float32)
            fig_path = str(out_p / f"{patch_id}_vis.png")
            save_figure(rgb, z, pred_m, z, fig_path, rgb_only=True, patch_id=patch_id, mae=float("nan"), rmse=float("nan"))
            pred_path = str(out_p / f"{patch_id}_pred.npy")
            np.save(pred_path, pred_m.astype(np.float32))
            rows.append(
                {
                    "id": patch_id,
                    "image_path": str(img_path),
                    "depth_path": "",
                    "pred_path": pred_path,
                    "vis_path": fig_path,
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "num_valid": 0,
                }
            )
            if len(indices_paths) > 1:
                print(f"    -> {patch_id} -> {fig_path}")
    else:
        if not str(d.get("depth_dir", "")).strip():
            raise ValueError("data.depth_dir is required when infer.rgb_only is false.")
        dataset = BathymetryDataset(
            img_dir=str(d["image_dir"]),
            depth_dir=str(d["depth_dir"]),
            img_glob=str(d.get("image_suffix", "img_*.tif")),
            depth_glob="depth_*.tif",
            selected_bands=None,
            normalize=normalize_img,
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
            image_mode=image_mode,
            reflectance_scale=reflectance_scale,
            magic_negative_depth=bool(d.get("magic_negative_depth_valid", True)),
        )

        n = len(dataset)
        start = max(0, int(args.start_idx))
        end = n if args.end_idx is None else int(args.end_idx)
        indices = list(range(start, end)) if args.all else [start]

        for k, idx in enumerate(indices):
            if len(indices) > 1:
                print(f"[{k + 1}/{len(indices)}] idx={idx} ...", flush=True)
            sample = dataset[idx]
            img_path = dataset.pairs[idx][0]
            depth_path = dataset.pairs[idx][1]
            patch_id = str(sample["patch_id"])

            raw_img = read_raster(img_path).astype(np.float32)
            rs = float(d.get("reflectance_scale", 255.0)) or 1.0
            rgb = chw_rgb_preview(raw_img[:3] / rs)

            x = sample["image"].unsqueeze(0).to(device)
            depth_n = sample["depth"]
            vm = sample["valid_mask"]

            with torch.inference_mode():
                pred_n = model(x)
                pred_n = torch.nan_to_num(pred_n, nan=0.0, posinf=0.0, neginf=0.0)

            pred_np_n = pred_n.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            gt_np_n = depth_n.squeeze(0).cpu().numpy().astype(np.float32)
            mask_np = vm.squeeze(0).cpu().numpy().astype(np.float32)

            pred_m = denormalize_depth(pred_np_n, dm, ds, nd)
            gt_m = denormalize_depth(gt_np_n, dm, ds, nd)
            mae = masked_mae_np(pred_m, gt_m, mask_np)
            rmse = masked_rmse_np(pred_m, gt_m, mask_np)

            fig_path = str(out_p / f"{patch_id}_vis.png")
            save_figure(
                rgb,
                gt_m,
                pred_m,
                mask_np,
                fig_path,
                rgb_only=False,
                patch_id=patch_id,
                mae=mae,
                rmse=rmse,
            )

            pred_path = str(out_p / f"{patch_id}_pred.npy")
            np.save(pred_path, pred_m.astype(np.float32))
            np.save(str(out_p / f"{patch_id}_gt.npy"), gt_m.astype(np.float32))
            np.save(str(out_p / f"{patch_id}_valid_mask.npy"), mask_np.astype(np.float32))

            rows.append(
                {
                    "id": patch_id,
                    "image_path": str(img_path),
                    "depth_path": str(depth_path),
                    "pred_path": pred_path,
                    "vis_path": fig_path,
                    "mae": mae,
                    "rmse": rmse,
                    "num_valid": int((mask_np > 0).sum()),
                }
            )
            if len(indices) > 1:
                print(f"    -> {patch_id} MAE={mae:.6f} RMSE={rmse:.6f} -> {fig_path}")

    if rows:
        sp = str(out_p / "infer_summary.csv")
        write_summary_csv(rows, sp)
        print(f"[dpt infer] wrote {sp} ({len(rows)} rows).")


if __name__ == "__main__":
    main()

