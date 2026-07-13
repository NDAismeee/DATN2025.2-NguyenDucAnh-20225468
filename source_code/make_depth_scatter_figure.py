import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


def scatter_layout(root: Path) -> list:
    return [
        (root / "mlp" / "mlp_infer_depth", "MLP", (0, 0)),
        (root / "unet" / "unet_infer_depth", "UNet", (0, 1)),
        (root / "cnn" / "cnn_infer_outputs", "CNN", (0, 2)),
        (root / "depth_anythingv2" / "depth_anythingv2_infer_depth", "Depth Anything", (1, 0)),
        (root / "dpt" / "dpt_infer_depth", "DPT", (1, 1)),
        (root / "knn" / "knn_infer_depth", "KNN-RF", (1, 2)),
        (root / "rf" / "rf_infer_depth", "RF", (2, 0)),
        (root / "da-sdb" / "da_sdb_infer_depth", "DA-SDB", (2, 1)),
        (root / "new_test" / "proposed_infer_depth", "BathyAgent", (2, 2)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_id", type=str, default="img_379")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max_points", type=int, default=25000)
    args = parser.parse_args()

    patch_id = args.patch_id.strip()
    root = Path(__file__).resolve().parent
    ref = root / "cnn" / "cnn_infer_outputs"
    out_default = root / "figures" / f"{patch_id}_depth_scatter_models.png"
    out_path = Path(args.out) if str(args.out).strip() else out_default

    mask_path = ref / f"{patch_id}_valid_mask.npy"
    gt_path = ref / f"{patch_id}_gt.npy"
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)

    mask = np.load(mask_path).astype(np.float32) > 0
    gt = np.load(gt_path).astype(np.float32)
    gt_flat = gt[mask]

    preds_list = []
    for folder, _title, _pos in scatter_layout(root):
        p = folder / f"{patch_id}_pred.npy"
        if not p.is_file():
            raise FileNotFoundError(p)
        pr = np.load(p).astype(np.float32)[mask]
        preds_list.append(pr)

    lo = float(np.min(gt_flat))
    hi = float(np.max(gt_flat))
    for pr in preds_list:
        lo = min(lo, float(np.min(pr)))
        hi = max(hi, float(np.max(pr)))
    pad = 0.05 * max(hi - lo, 1e-6)
    lim_lo = lo - pad
    lim_hi = hi + pad

    axis_label_pt = 9
    tick_pt = 8
    model_title_pt = axis_label_pt + 1
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "axes.labelsize": axis_label_pt,
            "axes.titlesize": model_title_pt,
            "xtick.labelsize": tick_pt,
            "ytick.labelsize": tick_pt,
            "figure.titlesize": 14,
            "legend.fontsize": tick_pt,
        }
    )

    fig = plt.figure(figsize=(10.2, 10.2))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.28, hspace=0.28)
    fig.suptitle(
        f"{patch_id}: predicted vs ground-truth depth (valid pixels)",
        fontweight="bold",
        y=0.98,
    )

    rng = np.random.default_rng(0)
    max_pts = int(max(1000, args.max_points))

    for (folder, title, (r, c)), pr_flat in zip(scatter_layout(root), preds_list):
        ax = fig.add_subplot(gs[r, c])
        n = int(gt_flat.shape[0])
        if n > max_pts:
            idx = rng.choice(n, size=max_pts, replace=False)
            x = gt_flat[idx]
            y = pr_flat[idx]
        else:
            x = gt_flat
            y = pr_flat
        ax.scatter(x, y, s=2, alpha=0.35, c="#1f77b4", edgecolors="none", rasterized=True)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1.0, alpha=0.85)
        ax.set_title(title, fontweight="bold", fontsize=model_title_pt)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35, linestyle=":", linewidth=0.6)
        ax.tick_params(labelsize=tick_pt)
        if r == 2:
            ax.set_xlabel("Ground truth depth (m)", fontsize=axis_label_pt)
        if c == 0:
            ax.set_ylabel("Predicted depth (m)", fontsize=axis_label_pt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(out_path.resolve())


if __name__ == "__main__":
    main()
