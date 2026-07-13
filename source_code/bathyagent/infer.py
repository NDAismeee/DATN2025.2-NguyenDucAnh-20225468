import argparse
import csv
import os
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    load_yaml_config,
    masked_mae,
    masked_rmse,
    pick_torch_device,
    set_seed,
)
from dataset import MagicBathyDataset, resolve_selected_bands, S2_BAND_TO_INDEX
from model import LLMGuidedBathymetryModel


DISPLAY_DEPTH_MIN = -20.0
DISPLAY_DEPTH_MAX = 0.0

DEPTH_CMAP = "turbo"
ALPHA_CMAP = "magma"
VAR_CMAP = "plasma"
ZONE_CMAP = "tab10"
DIST_CMAP = "cividis"

DEPTH_INTERPOLATION = "bilinear"
AUX_INTERPOLATION = "nearest"

VIS_SMOOTH_SIGMA = 3.0
VIS_SMOOTH_DPHYS = True
VIS_SMOOTH_RAW = True
VIS_FILL_NAN_FOR_DISPLAY = False


def infer_num_input_channels(selected_bands, image_mode: str = "rgb") -> int:
    resolved = resolve_selected_bands(selected_bands, image_mode=str(image_mode))
    if resolved is None:
        return 13
    return len(resolved)


def build_model(config: Dict[str, Any], device: torch.device) -> LLMGuidedBathymetryModel:
    return LLMGuidedBathymetryModel(config).to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)


def get_rgb_indices_for_display(num_channels: int, selected_bands) -> list[int]:
    resolved = resolve_selected_bands(selected_bands)

    if resolved is None:
        if num_channels >= 4:
            return [3, 2, 1]  # B4,B3,B2
        if num_channels >= 3:
            return [0, 1, 2]
        return [0, 0, 0]

    rgb_raw = [S2_BAND_TO_INDEX["B4"], S2_BAND_TO_INDEX["B3"], S2_BAND_TO_INDEX["B2"]]
    if all(b in resolved for b in rgb_raw):
        return [resolved.index(b) for b in rgb_raw]

    if num_channels >= 3:
        return [0, 1, 2]
    return [0, 0, 0]


def chw_to_display_rgb(image_chw: np.ndarray, selected_bands) -> np.ndarray:
    c, h, w = image_chw.shape
    rgb_idx = get_rgb_indices_for_display(c, selected_bands)

    rgb = np.stack(
        [
            image_chw[rgb_idx[0]],
            image_chw[rgb_idx[1]],
            image_chw[rgb_idx[2]],
        ],
        axis=-1,
    )

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    if hi > lo:
        rgb = (rgb - lo) / (hi - lo)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def to_display_depth(depth_positive: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Convert positive depth to negative-for-display convention:
        2.41 -> -2.41
        invalid -> NaN
    """
    return np.where(valid_mask > 0, -depth_positive, np.nan)


def apply_nan_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, arr, np.nan)


def smooth_masked_2d(
    arr: np.ndarray,
    mask: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    """
    Gaussian smoothing that respects the valid mask.
    Invalid locations do not bleed strongly into valid regions.
    """
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
    """
    Optional helper for display only.
    Fills NaNs using nearby valid values. Kept separate because sometimes
    users prefer preserving the exact valid support shape.
    """
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


def prepare_aux_for_display(
    arr: np.ndarray | None,
    valid_mask: np.ndarray,
    smooth: bool = False,
    sigma_px: float = VIS_SMOOTH_SIGMA,
) -> np.ndarray | None:
    if arr is None:
        return None
    out = apply_nan_mask(arr, valid_mask)
    if smooth:
        out = smooth_masked_2d(out, valid_mask, sigma_px=sigma_px)
    return out


def fmt_range(arr: np.ndarray) -> str:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "[nan, nan]"
    return f"[{finite.min():.2f}, {finite.max():.2f}]"


def to_numpy_info(info: Dict[str, Any], key: str):
    value = info.get(key, None)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value[0, 0].detach().cpu().numpy()
    return value


def add_colorbar(fig, ax, im):
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_visualization(
    rgb: np.ndarray,
    gt_positive: np.ndarray,
    pred_positive: np.ndarray,
    raw_model_output: np.ndarray,
    valid_mask: np.ndarray,
    d_phys: np.ndarray | None,
    alpha: np.ndarray | None,
    var: np.ndarray | None,
    zone_map: np.ndarray | None,
    dist_map: np.ndarray | None,
    out_path: str,
    title_prefix: str = "",
    show: bool = True,
) -> None:
    gt_display = prepare_depth_for_display(
        gt_positive,
        valid_mask,
        sigma_px=VIS_SMOOTH_SIGMA,
        fill_nan=VIS_FILL_NAN_FOR_DISPLAY,
    )
    pred_display = prepare_depth_for_display(
        pred_positive,
        valid_mask,
        sigma_px=VIS_SMOOTH_SIGMA,
        fill_nan=VIS_FILL_NAN_FOR_DISPLAY,
    )
    raw_display = prepare_depth_for_display(
        raw_model_output,
        valid_mask,
        sigma_px=VIS_SMOOTH_SIGMA if VIS_SMOOTH_RAW else 0.0,
        fill_nan=VIS_FILL_NAN_FOR_DISPLAY,
    )

    d_phys_display = None
    if d_phys is not None:
        d_phys_display = prepare_depth_for_display(
            d_phys,
            valid_mask,
            sigma_px=VIS_SMOOTH_SIGMA if VIS_SMOOTH_DPHYS else 0.0,
            fill_nan=VIS_FILL_NAN_FOR_DISPLAY,
        )

    alpha_display = prepare_aux_for_display(alpha, valid_mask, smooth=False)
    var_display = prepare_aux_for_display(var, valid_mask, smooth=False)
    zone_display = prepare_aux_for_display(zone_map, valid_mask, smooth=False)
    dist_display = prepare_aux_for_display(dist_map, valid_mask, smooth=False)

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

    im3 = axes[3].imshow(
        raw_display,
        cmap=DEPTH_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation=DEPTH_INTERPOLATION,
    )
    axes[3].set_title(f"Raw model μ\n{fmt_range(raw_display)} m")
    axes[3].axis("off")
    axes[3].set_aspect("equal")
    add_colorbar(fig, axes[3], im3)

    if d_phys_display is not None:
        im4 = axes[4].imshow(
            d_phys_display,
            cmap=DEPTH_CMAP,
            vmin=vmin,
            vmax=vmax,
            interpolation=DEPTH_INTERPOLATION,
        )
        axes[4].set_title(f"Physical prior d_phys\n{fmt_range(d_phys_display)} m")
        add_colorbar(fig, axes[4], im4)
    else:
        axes[4].text(0.5, 0.5, "d_phys unavailable", ha="center", va="center")
        axes[4].set_title("Physical prior d_phys")
    axes[4].axis("off")
    axes[4].set_aspect("equal")

    if alpha_display is not None:
        im5 = axes[5].imshow(
            alpha_display,
            cmap=ALPHA_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation=AUX_INTERPOLATION,
        )
        axes[5].set_title(f"Gate α\n{fmt_range(alpha_display)}")
        add_colorbar(fig, axes[5], im5)
    else:
        axes[5].text(0.5, 0.5, "alpha unavailable", ha="center", va="center")
        axes[5].set_title("Gate α")
    axes[5].axis("off")
    axes[5].set_aspect("equal")

    if var_display is not None:
        finite_var = var_display[np.isfinite(var_display)]
        var_vmin = float(finite_var.min()) if finite_var.size > 0 else 0.0
        var_vmax = float(finite_var.max()) if finite_var.size > 0 else 1.0
        im6 = axes[6].imshow(
            var_display,
            cmap=VAR_CMAP,
            vmin=var_vmin,
            vmax=var_vmax,
            interpolation=AUX_INTERPOLATION,
        )
        axes[6].set_title(f"Uncertainty var\n{fmt_range(var_display)}")
        add_colorbar(fig, axes[6], im6)
    else:
        axes[6].text(0.5, 0.5, "var unavailable", ha="center", va="center")
        axes[6].set_title("Uncertainty var")
    axes[6].axis("off")
    axes[6].set_aspect("equal")

    if zone_display is not None:
        im7 = axes[7].imshow(
            zone_display,
            cmap=ZONE_CMAP,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        axes[7].set_title("Zone map\n0=bg, 1=near, 2=trans, 3=off")
        add_colorbar(fig, axes[7], im7)
    elif dist_display is not None:
        finite_dist = dist_display[np.isfinite(dist_display)]
        dist_vmin = float(finite_dist.min()) if finite_dist.size > 0 else 0.0
        dist_vmax = float(finite_dist.max()) if finite_dist.size > 0 else 1.0
        im7 = axes[7].imshow(
            dist_display,
            cmap=DIST_CMAP,
            vmin=dist_vmin,
            vmax=dist_vmax,
            interpolation=AUX_INTERPOLATION,
        )
        axes[7].set_title(f"Distance-to-shore\n{fmt_range(dist_display)}")
        add_colorbar(fig, axes[7], im7)
    else:
        axes[7].text(0.5, 0.5, "zone/dist unavailable", ha="center", va="center")
        axes[7].set_title("Zone / distance map")
    axes[7].axis("off")
    axes[7].set_aspect("equal")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    if show and str(plt.get_backend()).lower() != "agg":
        plt.show()
    plt.close(fig)


def infer_one_sample(
    dataset: MagicBathyDataset,
    sample_idx: int,
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    selected_bands,
    output_dir: str,
    show_figure: bool,
    quiet: bool,
) -> Dict[str, Any]:
    sample = dataset[sample_idx]

    x = sample["image"].unsqueeze(0).to(device)
    depth_gt = sample["depth"].unsqueeze(0).to(device)
    valid_mask = sample["valid_mask"].unsqueeze(0).to(device)
    water_mask = sample["water_mask"].unsqueeze(0).to(device)
    reliability_mask = sample["reliability_mask"].unsqueeze(0).to(device)
    disturbance_masks = sample["disturbance_masks"].unsqueeze(0).to(device)
    text_embeddings = sample["text_embeddings"].unsqueeze(0).to(device)
    region_valid_mask = torch.ones((1, disturbance_masks.shape[1]), device=device)
    prior_depth = sample["prior_depth_map"].unsqueeze(0).to(device)
    prior_valid = sample["prior_valid_mask"].unsqueeze(0).to(device)
    prior_conf = sample["prior_confidence"].unsqueeze(0).to(device)

    with torch.inference_mode():
        pred, info = model(
            x,
            reliability_mask=reliability_mask,
            disturbance_masks=disturbance_masks,
            text_embeddings=text_embeddings,
            region_valid_mask=region_valid_mask,
            depth_gt=depth_gt,
            valid_mask=valid_mask,
            water_mask=water_mask,
            prior_depth_map=prior_depth,
            prior_valid_mask=prior_valid,
            prior_confidence=prior_conf,
        )

    image_np = sample["image"].cpu().numpy()
    gt_np = sample["depth"][0].cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()
    mask_np = sample["valid_mask"][0].cpu().numpy()
    
    # Reconstruction output
    reconstruction_chw = info["reconstruction"][0].detach().cpu().numpy()

    raw_model_np = to_numpy_info(info, "mu")
    if raw_model_np is None:
        raw_model_np = pred_np.copy()

    d_phys_np = to_numpy_info(info, "d_phys")
    prior_valid_np = to_numpy_info(info, "prior_valid_mask")
    if d_phys_np is not None and prior_valid_np is not None:
        d_phys_np = np.where(prior_valid_np > 0.5, d_phys_np, np.nan)
    alpha_np = to_numpy_info(info, "alpha")
    var_np = to_numpy_info(info, "var")
    zone_map_np = to_numpy_info(info, "zone_map")
    dist_map_np = to_numpy_info(info, "distance_to_shore")

    rgb = chw_to_display_rgb(image_np, selected_bands)

    mae = masked_mae(gt_np, pred_np, mask_np)
    rmse = masked_rmse(gt_np, pred_np, mask_np)

    os.makedirs(output_dir, exist_ok=True)

    sample_id = str(sample["sample_id"])
    fig_path = os.path.join(output_dir, f"{sample_id}_vis.png")
    pred_path = os.path.join(output_dir, f"{sample_id}_pred.npy")
    mu_path = os.path.join(output_dir, f"{sample_id}_mu.npy")
    gt_path = os.path.join(output_dir, f"{sample_id}_gt.npy")
    rgb_path = os.path.join(output_dir, f"{sample_id}_rgb.npy")
    recon_path = os.path.join(output_dir, f"{sample_id}_reconstruction.npy")

    title = (
        f"{sample_id} | "
        f"MAE={mae:.4f} | RMSE={rmse:.4f} | "
        f"use_llm_prior={config.get('model', {}).get('use_llm_prior', False)} | "
        f"use_real_llm={config.get('model', {}).get('use_real_llm', False)}"
    )

    save_visualization(
        rgb=rgb,
        gt_positive=gt_np,
        pred_positive=pred_np,
        raw_model_output=raw_model_np,
        valid_mask=mask_np,
        d_phys=d_phys_np,
        alpha=alpha_np,
        var=var_np,
        zone_map=zone_map_np,
        dist_map=dist_map_np,
        out_path=fig_path,
        title_prefix=title,
        show=show_figure,
    )

    np.save(pred_path, pred_np)
    np.save(mu_path, raw_model_np)
    np.save(gt_path, gt_np)
    np.save(rgb_path, rgb)
    np.save(recon_path, reconstruction_chw)
    np.save(os.path.join(output_dir, f"{sample_id}_M.npy"), sample["reliability_mask"].cpu().numpy())
    np.save(os.path.join(output_dir, f"{sample_id}_valid_mask.npy"), sample["valid_mask"].cpu().numpy())
    np.save(os.path.join(output_dir, f"{sample_id}_water_mask.npy"), sample["water_mask"].cpu().numpy())

    if d_phys_np is not None:
        np.save(os.path.join(output_dir, f"{sample_id}_d_phys.npy"), d_phys_np)
    if prior_valid_np is not None:
        np.save(os.path.join(output_dir, f"{sample_id}_prior_valid_mask.npy"), prior_valid_np)
    if alpha_np is not None:
        np.save(os.path.join(output_dir, f"{sample_id}_alpha.npy"), alpha_np)
    if var_np is not None:
        np.save(os.path.join(output_dir, f"{sample_id}_var.npy"), var_np)
    if zone_map_np is not None:
        np.save(os.path.join(output_dir, f"{sample_id}_zone.npy"), zone_map_np)
    if dist_map_np is not None:
        np.save(os.path.join(output_dir, f"{sample_id}_dist.npy"), dist_map_np)

    if not quiet:
        print(f"Sample ID      : {sample_id}")
        print(f"MAE            : {mae:.6f}")
        print(f"RMSE           : {rmse:.6f}")
        print(f"GT display     : {fmt_range(prepare_depth_for_display(gt_np, mask_np, sigma_px=VIS_SMOOTH_SIGMA))}")
        print(f"Pred display   : {fmt_range(prepare_depth_for_display(pred_np, mask_np, sigma_px=VIS_SMOOTH_SIGMA))}")
        print(
            f"Raw model mu   : "
            f"{fmt_range(prepare_depth_for_display(raw_model_np, mask_np, sigma_px=VIS_SMOOTH_SIGMA if VIS_SMOOTH_RAW else 0.0))}"
        )
        if d_phys_np is not None:
            print(
                f"d_phys         : "
                f"{fmt_range(prepare_depth_for_display(d_phys_np, mask_np, sigma_px=VIS_SMOOTH_SIGMA if VIS_SMOOTH_DPHYS else 0.0))}"
            )
        if alpha_np is not None:
            print(f"alpha          : {fmt_range(apply_nan_mask(alpha_np, mask_np))}")
        if var_np is not None:
            print(f"var            : {fmt_range(apply_nan_mask(var_np, mask_np))}")
        print(f"Fixed scale    : [{DISPLAY_DEPTH_MIN:.1f}, {DISPLAY_DEPTH_MAX:.1f}] m")
        print(f"Figure saved   : {fig_path}")
        print(f"Pred saved     : {pred_path}")
        print(f"Mu saved       : {mu_path}")
        print(f"GT saved       : {gt_path}")
        print(f"RGB saved      : {rgb_path}")
        print(f"Reconstruction saved : {recon_path}")

    return {
        "sample_id": sample_id,
        "sample_idx": sample_idx,
        "mae": mae,
        "rmse": rmse,
        "fig_path": fig_path,
    }


def _write_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames = ["sample_idx", "sample_id", "mae", "rmse", "fig_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/depth_model_3band_rgb_sem_fuseenc_rgb_issue/2026-04-19_020340_0b208d72/best_model.pt",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index when not using --all (same order as sorted images in data.image_dir).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on every sample in the dataset (entire image folder from config).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="With --all: first index (inclusive).",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="With --all: end index (exclusive); default = dataset length.",
    )
    parser.add_argument("--output_dir", type=str, default="infer_outputs")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu | cuda | auto (default: train.device from config)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open figure windows (recommended with --all).",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    seed = int(config.get("train", {}).get("seed", 42))
    set_seed(seed)

    train_cfg = config.get("train", {})
    device_pref = args.device if args.device is not None else train_cfg.get("device", "auto")
    device = pick_torch_device(str(device_pref), int(args.gpu_id))
    print(f"[infer] device={device}")

    selected_bands = config.get("data", {}).get("selected_bands", None)
    image_mode = str(config.get("data", {}).get("image_mode", "rgb"))
    config.setdefault("model", {})
    config["model"]["image_channels"] = infer_num_input_channels(selected_bands, image_mode=image_mode)
    config["model"]["encoder_in_channels"] = config["model"].get("image_channels", 3) + 1

    data_cfg = config.get("data", {})
    semantic_cfg = config.get("semantic", {})
    dataset = MagicBathyDataset(
        image_dir=data_cfg["image_dir"],
        depth_dir=data_cfg["depth_dir"],
        modality=data_cfg.get("modality", "rgb"),
        image_mode=image_mode,
        image_size=data_cfg.get("image_size", None),
        image_suffix=data_cfg.get("image_suffix", "*.tif"),
        selected_bands=data_cfg.get("selected_bands", None),
        reflectance_scale=float(data_cfg.get("reflectance_scale", 255.0)),
        depth_suffixes_to_try=data_cfg.get("depth_suffixes_to_try", None),
        allow_empty_mask=False,
        verbose=True,
        semantic_dir=semantic_cfg.get("semantic_dir", None),
        require_semantic_if_enabled=bool(semantic_cfg.get("require_semantic_if_enabled", True)),
        reliability_suffix=semantic_cfg.get("reliability_suffix", "_M.npy"),
        disturbance_masks_suffix=semantic_cfg.get("disturbance_masks_suffix", "_R.npy"),
        depth_prior_suffix=semantic_cfg.get("depth_prior_suffix", "_prior.npy"),
        depth_prior_valid_suffix=semantic_cfg.get("depth_prior_valid_suffix", "_prior_valid.npy"),
        depth_prior_conf_suffix=semantic_cfg.get("depth_prior_conf_suffix", "_prior_conf.npy"),
        text_embeddings_suffix=semantic_cfg.get("text_embeddings_suffix", "_text_embeddings.npy"),
        region_texts_suffix=semantic_cfg.get("region_texts_suffix", "_region_texts.json"),
        water_suffix=semantic_cfg.get("water_suffix", "_water.npy"),
        text_dim=int(config.get("text_encoder", {}).get("output_dim", config.get("model", {}).get("text_dim", 384))),
    )

    n = len(dataset)
    if n == 0:
        raise RuntimeError("Dataset is empty; check data.image_dir / depth pairing in config.")

    if args.all:
        start = max(0, int(args.start_idx))
        end = n if args.end_idx is None else int(args.end_idx)
        if start < 0 or start > n or end < start or end > n:
            raise IndexError(
                f"Invalid range start_idx={start} end_idx={end} for dataset size {n}"
            )
        indices: List[int] = list(range(start, end))
    else:
        if args.sample_idx < 0 or args.sample_idx >= n:
            raise IndexError(f"sample_idx={args.sample_idx} out of range for dataset size {n}")
        indices = [int(args.sample_idx)]

    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    show_fig = (not args.no_show) and (len(indices) == 1) and (not args.all)
    quiet_batch = len(indices) > 1

    summary_rows: List[Dict[str, Any]] = []
    for k, idx in enumerate(indices):
        if quiet_batch:
            print(f"[{k + 1}/{len(indices)}] sample_idx={idx} ...", flush=True)
        row = infer_one_sample(
            dataset=dataset,
            sample_idx=idx,
            model=model,
            config=config,
            device=device,
            selected_bands=selected_bands,
            output_dir=args.output_dir,
            show_figure=show_fig,
            quiet=quiet_batch,
        )
        summary_rows.append(row)
        if quiet_batch:
            print(f"    -> {row['sample_id']} MAE={row['mae']:.6f} RMSE={row['rmse']:.6f} saved {row['fig_path']}")

    if len(summary_rows) > 1:
        summary_path = os.path.join(args.output_dir, "infer_summary.csv")
        _write_summary_csv(summary_rows, summary_path)
        print(f"[infer] Wrote {summary_path} ({len(summary_rows)} rows).")


if __name__ == "__main__":
    main()