import argparse
from pathlib import Path
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


DISPLAY_DEPTH_MIN = -20.0
DISPLAY_DEPTH_MAX = 0.0
DEPTH_INTERPOLATION = "bilinear"
VIS_SMOOTH_SIGMA = 3.0


def depth_colormap():
    u = np.arange(256, dtype=np.uint8).reshape(1, -1)
    bgr = cv2.applyColorMap(u, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[0] / 255.0
    return ListedColormap(rgb, name="cv2_turbo")


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


def load_rgb_tif(path: Path, reflectance_scale: float) -> np.ndarray:
    with rasterio.open(path) as src:
        raw = src.read().astype(np.float32)
    return chw_rgb_preview(raw / float(reflectance_scale or 1.0))


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


def prepare_depth_for_display(
    depth_positive: np.ndarray,
    valid_mask: np.ndarray,
    sigma_px: float = VIS_SMOOTH_SIGMA,
) -> np.ndarray:
    arr = to_display_depth(depth_positive, valid_mask)
    return smooth_masked_2d(arr, valid_mask, sigma_px=sigma_px)


def depth_panel(ax, depth_display: np.ndarray, title: str) -> None:
    im = ax.imshow(
        depth_display,
        cmap=depth_colormap(),
        vmin=DISPLAY_DEPTH_MIN,
        vmax=DISPLAY_DEPTH_MAX,
        interpolation=DEPTH_INTERPOLATION,
    )
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    ax.set_aspect("equal")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4.2%", pad=0.06)
    cb = plt.colorbar(im, cax=cax)
    ticks = [0.0, -2.5, -5.0, -7.5, -10.0, -12.5, -15.0, -17.5, -20.0]
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=8)


def input_panel(ax, rgb: np.ndarray, title: str) -> None:
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    ax.set_aspect("equal")


def default_layout(patch_id: str) -> list:
    root = Path(__file__).resolve().parent
    ref = root / "cnn" / "cnn_infer_outputs"
    return [
        ("input", None, "Input image", (0, 0)),
        ("depth", root / "mlp" / "mlp_infer_depth", "MLP", (0, 1)),
        ("depth", root / "unet" / "unet_depth", "UNet", (0, 2)),
        ("depth", root / "cnn" / "cnn_infer_outputs", "CNN", (0, 3)),
        ("depth_gt", ref, "Groundtruth", (1, 0)),
        ("depth", root / "depth_anythingv2" / "depth_anythingv2_infer_depth", "Depth Anything", (1, 1)),
        ("depth", root / "dpt" / "dpt_infer_depth", "DPT", (1, 2)),
        ("depth", root / "knn" / "knn_infer_depth", "KNN-RF", (1, 3)),
        ("input", None, "Deep-water Regions", (2, 0)),
        ("depth", root / "rf" / "rf_infer_depth", "RF", (2, 1)),
        ("depth", root / "da-sdb" / "da_sdb_infer_depth", "DA-SDB", (2, 2)),
        ("depth", root / "new_test" / "proposed_infer_depth", "BathyAgent", (2, 3)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_id", type=str, default="img_379")
    parser.add_argument("--input_tif", type=str, default="")
    parser.add_argument("--reflectance_scale", type=float, default=255.0)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    patch_id = args.patch_id.strip()
    out_default = Path(__file__).resolve().parent / "figures" / f"{patch_id}_depth_maps_all_models.png"
    out_path = Path(args.out) if str(args.out).strip() else out_default
    subplot_title_pt = 10
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "axes.titlesize": subplot_title_pt,
            "figure.titlesize": 15,
        }
    )

    if str(args.input_tif).strip():
        input_path = Path(args.input_tif)
    else:
        input_path = Path("/mnt/disk3/anhnd2468/MagicBathyNet/agia_napa/img/depth_map") / f"{patch_id}.tif"
    if not input_path.is_file():
        alt = Path(__file__).resolve().parents[2] / "agia_napa" / "img" / "depth_map" / f"{patch_id}.tif"
        if alt.is_file():
            input_path = alt
        else:
            raise FileNotFoundError(f"Missing input tif: {input_path}")

    rgb = load_rgb_tif(input_path, args.reflectance_scale)

    ref_dir = Path(__file__).resolve().parent / "cnn" / "cnn_infer_outputs"
    mask_path = ref_dir / f"{patch_id}_valid_mask.npy"
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    mask = np.load(mask_path).astype(np.float32)

    fig = plt.figure(figsize=(14.5, 10.2))
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.28, hspace=0.12)

    layout = default_layout(patch_id)
    for kind, folder, title, (r, c) in layout:
        ax = fig.add_subplot(gs[r, c])
        if kind == "input":
            input_panel(ax, rgb, title)
            continue
        if kind == "depth_gt":
            gt_path = Path(folder) / f"{patch_id}_gt.npy"
            if not gt_path.is_file():
                raise FileNotFoundError(gt_path)
            gt = np.load(gt_path).astype(np.float32)
            disp = prepare_depth_for_display(gt, mask)
            depth_panel(ax, disp, title)
            continue
        if kind == "depth":
            pred_path = Path(folder) / f"{patch_id}_pred.npy"
            if not pred_path.is_file():
                raise FileNotFoundError(pred_path)
            pred = np.load(pred_path).astype(np.float32)
            disp = prepare_depth_for_display(pred, mask)
            depth_panel(ax, disp, title)
            continue

    fig.suptitle(
        f"{patch_id}: Input, Ground Truth, and Final Prediction Depth Maps",
        fontweight="bold",
        y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(out_path.resolve())


if __name__ == "__main__":
    main()
