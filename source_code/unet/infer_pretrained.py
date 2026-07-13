import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_DIR = Path(__file__).resolve().parent
_CNN_SRC = _REPO_DIR.parent / "cnn_src"
sys.path.insert(0, str(_CNN_SRC))

from dataset import BathymetryDataset, read_raster

import infer as cnn_infer


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PretrainedAerialUNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.encoder = nn.ModuleList(
            [
                DoubleConv(n_channels, 32),
                Down(32, 64),
                Down(64, 128),
                Down(128, 256),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                Up(256, 128, bilinear),
                Up(128, 64, bilinear),
                Up(64, 32, bilinear),
                nn.Conv2d(32, n_classes, kernel_size=1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x = self.decoder[0](x4, x3)
        x = self.decoder[1](x, x2)
        x = self.decoder[2](x, x1)
        return self.decoder[3](x)


def load_pretrained_unet(weights_path: str, device: torch.device) -> PretrainedAerialUNet:
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model = PretrainedAerialUNet(n_channels=3, n_classes=1, bilinear=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def compute_channel_stats_from_images(
    image_paths: List[Path],
    reflectance_scale: float,
    max_images: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    paths = image_paths if max_images is None else image_paths[: int(max_images)]
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    channel_count = np.zeros(3, dtype=np.float64)

    rs = float(reflectance_scale) if reflectance_scale and reflectance_scale > 0 else 1.0

    for p in paths:
        img = read_raster(p).astype(np.float32)
        if img.ndim != 3 or img.shape[0] < 3:
            continue
        rgb = (img[:3] / rs).reshape(3, -1)
        valid = np.isfinite(rgb)
        for c in range(3):
            vals = rgb[c][valid[c]]
            if vals.size == 0:
                continue
            channel_sum[c] += vals.sum(dtype=np.float64)
            channel_sq_sum[c] += np.square(vals, dtype=np.float64).sum(dtype=np.float64)
            channel_count[c] += float(vals.size)

    if np.any(channel_count == 0):
        raise ValueError("Could not compute mean/std from images (no finite pixels found).")

    mean = channel_sum / channel_count
    var = channel_sq_sum / channel_count - mean**2
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def write_pred_geotiff_like(src_path: Path, pred_hw: np.ndarray, out_path: Path) -> None:
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=None)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(pred_hw.astype(np.float32), 1)


def infer_one_image_file(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    reflectance_scale: float,
    normalize: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    pred_is_negative: bool,
    save_tif: bool,
) -> Dict[str, Any]:
    img = read_raster(image_path).astype(np.float32)
    if img.ndim != 3 or img.shape[0] < 3:
        raise ValueError(f"Expected CHW raster with >=3 bands, got shape {img.shape} for {image_path}")

    rs = float(reflectance_scale) if reflectance_scale and reflectance_scale > 0 else 1.0
    rgb_chw = img[:3] / rs
    rgb_preview = cnn_infer.chw_rgb_preview(rgb_chw)

    x = torch.from_numpy(rgb_chw).unsqueeze(0).to(device=device, dtype=torch.float32)
    if normalize:
        if mean is None or std is None:
            raise ValueError("normalize=True requires mean/std (pass via --mean/--std or allow auto stats).")
        m = torch.from_numpy(mean.reshape(1, 3, 1, 1)).to(device=device, dtype=torch.float32)
        s = torch.from_numpy(std.reshape(1, 3, 1, 1)).to(device=device, dtype=torch.float32)
        x = (x - m) / torch.clamp(s, min=1e-6)

    with torch.inference_mode():
        pred = model(x)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    pred_hw = pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    if pred_is_negative:
        pred_hw = -pred_hw
    else:
        med = float(np.nanmedian(pred_hw)) if np.isfinite(pred_hw).any() else 0.0
        if med < 0.0:
            pred_hw = -pred_hw

    valid_mask = np.isfinite(pred_hw).astype(np.float32)
    pred_display = cnn_infer.prepare_depth_for_display(pred_hw, valid_mask, sigma_px=cnn_infer.VIS_SMOOTH_SIGMA)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = image_path.stem
    fig_path = output_dir / f"{sample_id}_vis.png"
    pred_npy_path = output_dir / f"{sample_id}_pred.npy"
    mask_npy_path = output_dir / f"{sample_id}_valid_mask.npy"
    np.save(pred_npy_path, pred_hw.astype(np.float32))
    np.save(mask_npy_path, valid_mask.astype(np.float32))

    gt_dummy = np.full_like(pred_hw, np.nan, dtype=np.float32)
    title = f"{sample_id} | Pretrained UNet | pred_range={cnn_infer.fmt_range(pred_display)} m"
    cnn_infer.save_cnn_figure(rgb_preview, gt_dummy, pred_hw, valid_mask, str(fig_path), title, show=False)

    pred_tif_path = None
    if save_tif:
        pred_tif_path = output_dir / f"{sample_id}_pred.tif"
        write_pred_geotiff_like(image_path, pred_hw, pred_tif_path)

    return {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "pred_npy_path": str(pred_npy_path),
        "valid_mask_npy_path": str(mask_npy_path),
        "fig_path": str(fig_path),
        "pred_tif_path": str(pred_tif_path) if pred_tif_path is not None else "",
    }


def infer_one_paired_sample(
    dataset: BathymetryDataset,
    idx: int,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    pred_is_negative: bool,
    save_tif: bool,
) -> Dict[str, Any]:
    sample = dataset[idx]
    img_path = dataset.pairs[idx][0]
    patch_id = str(sample["patch_id"])

    x = sample["image"].unsqueeze(0).to(device)
    depth = sample["depth"]
    vm = sample["valid_mask"]

    raw_img = read_raster(img_path).astype(np.float32)
    raw_img = raw_img[dataset.band_indices]
    rs = float(dataset.reflectance_scale) if dataset.reflectance_scale else 1.0
    rgb_preview = cnn_infer.chw_rgb_preview(raw_img / rs)

    with torch.inference_mode():
        pred = model(x)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    pred_hw = pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    if pred_is_negative:
        pred_hw = -pred_hw
    else:
        med = float(np.nanmedian(pred_hw)) if np.isfinite(pred_hw).any() else 0.0
        if med < 0.0:
            pred_hw = -pred_hw

    gt_hw = depth.squeeze(0).cpu().numpy().astype(np.float32)
    mask_hw = vm.squeeze(0).cpu().numpy().astype(np.float32)

    mae = cnn_infer.masked_mae_np(pred_hw, gt_hw, mask_hw)
    rmse = cnn_infer.masked_rmse_np(pred_hw, gt_hw, mask_hw)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"{patch_id}_vis.png"
    title = f"{patch_id} | MAE={mae:.4f} m | RMSE={rmse:.4f} m | Pretrained UNet"
    cnn_infer.save_cnn_figure(rgb_preview, gt_hw, pred_hw, mask_hw, str(fig_path), title, show=False)

    pred_npy_path = output_dir / f"{patch_id}_pred.npy"
    gt_npy_path = output_dir / f"{patch_id}_gt.npy"
    mask_npy_path = output_dir / f"{patch_id}_valid_mask.npy"
    np.save(pred_npy_path, pred_hw.astype(np.float32))
    np.save(gt_npy_path, gt_hw.astype(np.float32))
    np.save(mask_npy_path, mask_hw.astype(np.float32))

    pred_tif_path = None
    if save_tif:
        pred_tif_path = output_dir / f"{patch_id}_pred.tif"
        write_pred_geotiff_like(Path(img_path), pred_hw, pred_tif_path)

    return {
        "sample_idx": idx,
        "sample_id": patch_id,
        "mae": mae,
        "rmse": rmse,
        "fig_path": str(fig_path),
        "image_path": str(img_path),
        "pred_npy_path": str(pred_npy_path),
        "gt_npy_path": str(gt_npy_path),
        "valid_mask_npy_path": str(mask_npy_path),
        "pred_tif_path": str(pred_tif_path) if pred_tif_path is not None else "",
    }


def write_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys = [
        "sample_idx",
        "sample_id",
        "mae",
        "rmse",
        "fig_path",
        "image_path",
        "pred_npy_path",
        "gt_npy_path",
        "valid_mask_npy_path",
        "pred_tif_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="bathymetry_aerial_an",
        help="Raw state_dict or full checkpoint containing model_state_dict.",
    )
    parser.add_argument("--output_dir", type=str, default='pretrained_infer_outputs')
    parser.add_argument("--img_dir", type=str, default="D:/AnhHieu/new_bathymetric/MagicBathyNet/agia_napa/img/aerial", help="Folder of input images (same as cnn_src/infer.py).")
    parser.add_argument(
        "--depth_dir",
        type=str,
        default="D:/AnhHieu/new_bathymetric/MagicBathyNet/agia_napa/depth/aerial",
        help="Optional: if provided, will pair images with GT depths and compute MAE/RMSE (cnn_src/infer.py style).",
    )
    parser.add_argument("--img_glob", type=str, default="img_*.tif")
    parser.add_argument("--depth_glob", type=str, default="depth_*.tif")
    parser.add_argument(
        "--pairing_mode",
        type=str,
        default="magic",
        help="magic | prefix (same semantics as cnn_src/dataset.py).",
    )
    parser.add_argument(
        "--magic_negative_depth",
        action="store_true",
        help="If depth rasters store valid depths as negative values, convert to positive and mask (matches cnn_src/config.yaml).",
    )
    parser.add_argument(
        "--no_magic_negative_depth",
        action="store_true",
        help="Disable negative-depth conversion.",
    )
    parser.add_argument(
        "--depth_suffixes_to_try",
        type=str,
        default="_depth,_bathy,_gt,_label",
        help="Comma-separated suffixes (used only when pairing_mode=magic).",
    )
    parser.add_argument("--image_mode", type=str, default="rgb")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--reflectance_scale", type=float, default=255.0)
    parser.add_argument("--normalize", action="store_true", help="Normalize inputs using mean/std (recommended).")
    parser.add_argument("--no_normalize", action="store_true", help="Disable normalization.")
    parser.add_argument("--stats_max_images", type=int, default=50)
    parser.add_argument("--mean", type=str, default=None, help="Comma-separated RGB mean, e.g. 0.1,0.2,0.3")
    parser.add_argument("--std", type=str, default=None, help="Comma-separated RGB std, e.g. 0.05,0.06,0.07")
    parser.add_argument("--pred_is_negative", action="store_true", help="If model outputs negative depths, flip sign.")
    parser.add_argument("--save_tif", action="store_true", help="Also save GeoTIFF predictions next to PNG/NPY.")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"[pretrained unet] device={device}")

    model = load_pretrained_unet(args.weights, device)

    normalize = bool(args.normalize) and (not bool(args.no_normalize))
    reflectance_scale = float(args.reflectance_scale) if args.reflectance_scale else 1.0

    mean = None
    std = None
    if args.mean is not None and args.std is not None:
        mean = np.asarray([float(x) for x in str(args.mean).split(",")], dtype=np.float32)
        std = np.asarray([float(x) for x in str(args.std).split(",")], dtype=np.float32)
        if mean.size != 3 or std.size != 3:
            raise ValueError("--mean/--std must have exactly 3 comma-separated values.")

    out_dir = Path(args.output_dir)
    rows: List[Dict[str, Any]] = []

    if args.depth_dir is not None:
        depth_suffixes = [s.strip() for s in str(args.depth_suffixes_to_try).split(",") if s.strip()]
        magic_neg = bool(args.magic_negative_depth) and (not bool(args.no_magic_negative_depth))
        dataset = BathymetryDataset(
            img_dir=str(args.img_dir),
            depth_dir=str(args.depth_dir),
            img_glob=str(args.img_glob),
            depth_glob=str(args.depth_glob),
            selected_bands=None,
            normalize=normalize,
            mean=mean.tolist() if mean is not None else None,
            std=std.tolist() if std is not None else None,
            normalize_depth=False,
            pairing_mode=str(args.pairing_mode).lower().strip(),
            depth_suffixes_to_try=depth_suffixes,
            image_mode=str(args.image_mode).lower().strip(),
            reflectance_scale=reflectance_scale,
            magic_negative_depth=magic_neg,
        )
        n = len(dataset)
        start = max(0, int(args.start_idx))
        end = n if args.end_idx is None else int(args.end_idx)
        if start > n or end < start or end > n:
            raise IndexError(f"Invalid start_idx={start} end_idx={end} for n={n}")
        indices = list(range(start, end))
        for k, idx in enumerate(indices):
            print(f"[{k + 1}/{len(indices)}] idx={idx} ...", flush=True)
            row = infer_one_paired_sample(
                dataset=dataset,
                idx=idx,
                model=model,
                device=device,
                output_dir=out_dir,
                pred_is_negative=bool(args.pred_is_negative),
                save_tif=bool(args.save_tif),
            )
            rows.append(row)
            print(f"    -> saved {row['fig_path']}", flush=True)
    else:
        root = Path(args.img_dir)
        if args.recursive:
            image_paths = sorted(root.rglob(args.img_glob))
        else:
            image_paths = sorted(root.glob(args.img_glob))
        if not image_paths:
            raise RuntimeError(
                f"No images found in {root} with glob {args.img_glob} (recursive={args.recursive})"
            )
        if normalize and (mean is None or std is None):
            mean, std = compute_channel_stats_from_images(
                image_paths=image_paths,
                reflectance_scale=reflectance_scale,
                max_images=int(args.stats_max_images) if args.stats_max_images is not None else None,
            )

        n = len(image_paths)
        start = max(0, int(args.start_idx))
        end = n if args.end_idx is None else int(args.end_idx)
        if start > n or end < start or end > n:
            raise IndexError(f"Invalid start_idx={start} end_idx={end} for n={n}")
        paths = image_paths[start:end]
        for k, p in enumerate(paths):
            print(f"[{k + 1}/{len(paths)}] {p.name} ...", flush=True)
            row = infer_one_image_file(
                image_path=p,
                model=model,
                device=device,
                output_dir=out_dir,
                reflectance_scale=reflectance_scale,
                normalize=normalize,
                mean=mean,
                std=std,
                pred_is_negative=bool(args.pred_is_negative),
                save_tif=bool(args.save_tif),
            )
            rows.append(row)
            print(f"    -> saved {row['fig_path']}", flush=True)

    sp = out_dir / "infer_summary.csv"
    write_summary_csv(rows, str(sp))
    print(f"[pretrained unet] wrote {sp} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
