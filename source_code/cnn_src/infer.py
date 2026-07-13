import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from dataset import BathymetryDataset, read_raster
from model import SimpleBathymetryCNN

_CNN_ROOT = Path(__file__).resolve().parent
_NEW_TEST = _CNN_ROOT.parent / "new_test"
if _NEW_TEST.is_dir():
    sys.path.insert(0, str(_NEW_TEST))
try:
    from depth_three_panel_vis import save_three_panel_depth_figure
except ImportError:
    _SRC = _CNN_ROOT.parent
    sys.path.insert(0, str(_SRC))
    from depth_three_panel_vis import save_three_panel_depth_figure
try:
    from common import load_yaml_config, pick_torch_device
except ImportError:

    def load_yaml_config(path: str) -> Dict[str, Any]:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def pick_torch_device(device_pref: str, gpu_id: int = 0):
        pref = (device_pref or "auto").strip().lower()
        if pref == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        if pref in ("cuda", "gpu"):
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cpu") if not torch.cuda.is_available() else torch.device(f"cuda:{gpu_id}")


DISPLAY_DEPTH_MIN = -20.0
DISPLAY_DEPTH_MAX = 0.0
DEPTH_CMAP = "turbo"
DEPTH_INTERPOLATION = "bilinear"
VIS_SMOOTH_SIGMA = 3.0
VIS_FILL_NAN_FOR_DISPLAY = False


def denormalize_depth(
    depth_arr: np.ndarray,
    depth_mean: Optional[float],
    depth_std: Optional[float],
    normalize_depth: bool,
) -> np.ndarray:
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


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[SimpleBathymetryCNN, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    tr = ckpt.get("config", {}) or {}

    imode = str(tr.get("image_mode", "")).lower().strip()
    mean = tr.get("mean", None)
    if imode == "rgb":
        in_channels = 3
    elif isinstance(mean, list) and len(mean) == 3:
        in_channels = 3
    elif tr.get("selected_bands") is None:
        in_channels = 13
    else:
        in_channels = len(tr["selected_bands"])

    hc = tr.get("hidden_channels", (32, 64, 64, 32))
    if isinstance(hc, list):
        hc = tuple(int(x) for x in hc)
    else:
        hc = tuple(hc)

    model = SimpleBathymetryCNN(
        in_channels=in_channels,
        hidden_channels=hc,
        use_batchnorm=bool(tr.get("use_batchnorm", False)),
        dropout=float(tr.get("dropout", 0.0)),
        use_coordconv=bool(tr.get("use_coordconv", True)),
        norm_type=str(tr.get("norm_type", "group")),
        num_groups=int(tr.get("num_groups", 8)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tr


def merge_infer_paths(tr: Dict[str, Any], data_yaml: Dict[str, Any]) -> Dict[str, Any]:
    d = data_yaml or {}
    out = dict(tr)
    if d.get("image_dir"):
        out["img_dir"] = d["image_dir"]
    if d.get("depth_dir"):
        out["depth_dir"] = d["depth_dir"]
    if d.get("image_suffix"):
        out["img_glob"] = d["image_suffix"]
    if d.get("depth_glob"):
        out["depth_glob"] = d["depth_glob"]
    if d.get("depth_suffixes_to_try") is not None:
        out["depth_suffixes_to_try"] = d["depth_suffixes_to_try"]
    if d.get("pairing"):
        out["pairing_mode"] = str(d["pairing"]).lower().strip()
    if d.get("image_mode"):
        out["image_mode"] = str(d["image_mode"]).lower().strip()
    if d.get("reflectance_scale") is not None:
        out["reflectance_scale"] = float(d["reflectance_scale"])
    if d.get("magic_negative_depth_valid") is not None:
        out["magic_negative_depth"] = bool(d["magic_negative_depth_valid"])
    return out


def resolve_patch_index(dataset: BathymetryDataset, patch_stem: str) -> int:
    stem = str(patch_stem).strip()
    for idx, (_, _, pid) in enumerate(dataset.pairs):
        if pid == stem:
            return idx
    for idx, (img_path, _, _) in enumerate(dataset.pairs):
        if img_path.stem == stem:
            return idx
    available = [p[2] for p in dataset.pairs[:12]]
    raise ValueError(f"No patch matching {stem!r}. First ids: {available} ... (total {len(dataset.pairs)})")


def build_infer_dataset(tr: Dict[str, Any]) -> BathymetryDataset:
    return BathymetryDataset(
        img_dir=str(tr["img_dir"]),
        depth_dir=str(tr["depth_dir"]),
        img_glob=str(tr.get("img_glob", "img_*.tif")),
        depth_glob=str(tr.get("depth_glob", "depth_*.tif")),
        selected_bands=tr.get("selected_bands"),
        normalize=bool(tr.get("normalize", True)),
        mean=tr.get("mean"),
        std=tr.get("std"),
        normalize_depth=bool(tr.get("normalize_depth", True)),
        depth_mean=tr.get("depth_mean"),
        depth_std=tr.get("depth_std"),
        depth_min=tr.get("depth_min"),
        depth_max=tr.get("depth_max"),
        invalid_depth_values=tr.get("invalid_depth_values") or [],
        return_metadata=False,
        pairing_mode=str(tr.get("pairing_mode", "magic")).lower(),
        depth_suffixes_to_try=tr.get("depth_suffixes_to_try"),
        image_mode=str(tr.get("image_mode", "rgb")).lower(),
        reflectance_scale=float(tr.get("reflectance_scale", 255.0)),
        magic_negative_depth=bool(tr.get("magic_negative_depth", True)),
    )


def infer_one_patch(
    dataset: BathymetryDataset,
    idx: int,
    model: torch.nn.Module,
    tr: Dict[str, Any],
    device: torch.device,
    output_dir: str,
    quiet: bool,
) -> Dict[str, Any]:
    sample = dataset[idx]
    img_path = dataset.pairs[idx][0]
    patch_id = str(sample["patch_id"])

    raw_img = read_raster(img_path).astype(np.float32)
    raw_img = raw_img[dataset.band_indices]
    rs = float(tr.get("reflectance_scale", 255.0)) or 1.0
    rgb = chw_rgb_preview(raw_img / rs)

    x = sample["image"].unsqueeze(0).to(device)
    depth_n = sample["depth"]
    vm = sample["valid_mask"]

    with torch.inference_mode():
        pred_n = model(x)
        pred_n = torch.nan_to_num(pred_n, nan=0.0, posinf=0.0, neginf=0.0)

    pred_np_n = pred_n.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    gt_np_n = depth_n.squeeze(0).cpu().numpy().astype(np.float32)
    mask_np = vm.squeeze(0).cpu().numpy().astype(np.float32)

    nd = bool(tr.get("normalize_depth", True))
    dm = tr.get("depth_mean", None)
    ds = tr.get("depth_std", None)

    pred_m = denormalize_depth(pred_np_n, dm, ds, nd)
    gt_m = denormalize_depth(gt_np_n, dm, ds, nd)

    mae = masked_mae_np(pred_m, gt_m, mask_np)
    rmse = masked_rmse_np(pred_m, gt_m, mask_np)

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{patch_id}_vis.png")
    title = f"{patch_id} | MAE={mae:.4f} m | RMSE={rmse:.4f} m | CNN infer"
    save_three_panel_depth_figure(
        rgb,
        gt_m,
        pred_m,
        mask_np,
        fig_path,
        suptitle=title,
        cmap=DEPTH_CMAP,
        vmin=DISPLAY_DEPTH_MIN,
        vmax=DISPLAY_DEPTH_MAX,
        sigma_px=VIS_SMOOTH_SIGMA,
    )

    np.save(os.path.join(output_dir, f"{patch_id}_pred.npy"), pred_m.astype(np.float32))
    np.save(os.path.join(output_dir, f"{patch_id}_gt.npy"), gt_m.astype(np.float32))
    np.save(os.path.join(output_dir, f"{patch_id}_valid_mask.npy"), mask_np.astype(np.float32))

    if not quiet:
        print(f"patch_id       : {patch_id}")
        print(f"MAE            : {mae:.6f}")
        print(f"RMSE           : {rmse:.6f}")
        print(f"GT display     : {fmt_range(prepare_depth_for_display(gt_m, mask_np, sigma_px=VIS_SMOOTH_SIGMA))}")
        print(f"Pred display   : {fmt_range(prepare_depth_for_display(pred_m, mask_np, sigma_px=VIS_SMOOTH_SIGMA))}")
        print(f"Fixed scale    : [{DISPLAY_DEPTH_MIN:.1f}, {DISPLAY_DEPTH_MAX:.1f}] m")
        print(f"Figure saved   : {fig_path}")

    return {
        "sample_idx": idx,
        "sample_id": patch_id,
        "mae": mae,
        "rmse": rmse,
        "fig_path": fig_path,
    }


def write_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys = ["sample_idx", "sample_id", "mae", "rmse", "fig_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(_CNN_ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", type=str, default="checkpoints_cnn_depth/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="cnn_infer_depth")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument(
        "--patch_stem",
        type=str,
        default=None,
        help="Infer this patch id (e.g. img_379); overrides --sample_idx when set.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Override data.image_dir for this run.",
    )
    parser.add_argument(
        "--depth_dir",
        type=str,
        default=None,
        help="Override data.depth_dir for this run.",
    )
    parser.add_argument(
        "--img_glob",
        type=str,
        default=None,
        help="Override image glob for dataset listing (default from config).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on every paired sample (same folder logic as training).",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_CNN_ROOT / ".env")
    except ImportError:
        pass

    yaml_cfg = load_yaml_config(args.config)
    data_y = yaml_cfg.get("data", {}) or {}
    train_y = yaml_cfg.get("train", {}) or {}

    device_pref = args.device if args.device is not None else train_y.get("device", "cpu")
    device = pick_torch_device(str(device_pref), int(args.gpu_id))
    print(f"[cnn infer] device={device}")

    model, tr = load_model(args.checkpoint, device)
    tr = merge_infer_paths(tr, data_y)
    if args.image_dir is not None:
        tr["img_dir"] = str(args.image_dir)
    if args.depth_dir is not None:
        tr["depth_dir"] = str(args.depth_dir)
    if args.img_glob is not None:
        tr["img_glob"] = str(args.img_glob)
    if not str(tr.get("image_mode", "")).strip():
        m = tr.get("mean", None)
        if isinstance(m, list) and len(m) == 3:
            tr["image_mode"] = "rgb"
    dataset = build_infer_dataset(tr)
    n = len(dataset)
    if n == 0:
        raise RuntimeError("No samples in dataset; check paths and pairing in config / checkpoint.")

    if args.all:
        start = max(0, int(args.start_idx))
        end = n if args.end_idx is None else int(args.end_idx)
        if start > n or end < start or end > n:
            raise IndexError(f"Invalid start_idx={start} end_idx={end} for n={n}")
        indices = list(range(start, end))
    else:
        if args.patch_stem is not None:
            indices = [resolve_patch_index(dataset, args.patch_stem)]
        else:
            if args.sample_idx < 0 or args.sample_idx >= n:
                raise IndexError(f"sample_idx={args.sample_idx} out of range for n={n}")
            indices = [int(args.sample_idx)]

    quiet = len(indices) > 1
    rows: List[Dict[str, Any]] = []

    for k, idx in enumerate(indices):
        if quiet:
            print(f"[{k + 1}/{len(indices)}] idx={idx} ...", flush=True)
        row = infer_one_patch(
            dataset=dataset,
            idx=idx,
            model=model,
            tr=tr,
            device=device,
            output_dir=args.output_dir,
            quiet=quiet,
        )
        rows.append(row)
        if quiet:
            print(f"    -> {row['sample_id']} MAE={row['mae']:.6f} RMSE={row['rmse']:.6f} -> {row['fig_path']}")

    if len(rows) > 1:
        sp = os.path.join(args.output_dir, "infer_summary.csv")
        write_summary_csv(rows, sp)
        print(f"[cnn infer] wrote {sp} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
