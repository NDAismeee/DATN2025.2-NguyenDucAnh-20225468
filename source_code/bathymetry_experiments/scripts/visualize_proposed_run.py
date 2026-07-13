from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            os.environ.setdefault(key, value)


def _zones_to_maps(image_chw01: np.ndarray, zones: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    img_hw = np.moveaxis(image_chw01, 0, -1)
    intensity = img_hw.mean(axis=2).astype(np.float32)
    centers = np.array([0.15, 0.45, 0.75], dtype=np.float32).reshape(1, 1, 3)
    d2 = (intensity[..., None] - centers) ** 2
    w = np.exp(-d2 / 0.02)
    w = w / (w.sum(axis=2, keepdims=True) + 1e-6)
    zone_map = (np.argmax(w, axis=2) + 1).astype(np.uint8)

    mids = {"nearshore": 1.0, "mid": 4.0, "offshore": 8.0}
    for item in zones:
        if not isinstance(item, dict):
            continue
        zone = str(item.get("zone") or "").strip()
        if zone in mids and item.get("d_min_m") is not None and item.get("d_max_m") is not None:
            mids[zone] = 0.5 * (float(item.get("d_min_m")) + float(item.get("d_max_m")))
    midv = np.array([mids["nearshore"], mids["mid"], mids["offshore"]], dtype=np.float32).reshape(1, 1, 3)
    d_phys = (w * midv).sum(axis=2).astype(np.float32)
    return d_phys, zone_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default="/mnt/disk3/anhnd2468/MagicBathyNet/hao-chapter1-depth-prediction/source_code/bathymetry_experiments/runs_agia_napa/proposed/2026-04-28_162716_642b0b84",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default="/mnt/disk3/anhnd2468/MagicBathyNet/hao-chapter1-depth-prediction/source_code/bathymetry_experiments/.env",
    )
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    _load_dotenv(Path(args.env_file))

    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import torch
    import cv2
    import matplotlib.pyplot as plt

    from bathymetry_experiments.common.metrics import regression_metrics
    from bathymetry_experiments.common.utils import write_csv
    from bathymetry_experiments.data.io import find_pairs, load_sample
    from bathymetry_experiments.models.torch_models import ProposedPaperRegressor

    cfg = json.loads("{}")
    import yaml

    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8")) or {}
    data = cfg["data"]
    train = cfg["train"]
    expert = cfg.get("expert") or {}

    pairs = find_pairs(data["image_dir"], data["depth_dir"], data.get("image_glob", "img_*.tif"))
    if args.max_items and args.max_items > 0:
        pairs = pairs[: args.max_items]

    device = torch.device("cuda" if str(train.get("device", "cpu")).startswith("cuda") and torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = ProposedPaperRegressor(in_channels=4).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    semantic_dir = Path(str(expert.get("semantic_dir", project_root / "semantic")))
    annotation_dir = Path(str(expert.get("annotation_dir", project_root / "annotation")))
    out_dir = run_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = float(data.get("reflectance_scale", 255.0))
    size = int(train.get("image_size", 128))

    rows: list[dict] = []
    for pair in pairs:
        image, target, valid, sample_id = load_sample(pair, scale)

        rel = np.load(semantic_dir / f"{sample_id}_reliability.npy").astype(np.float32)
        ann = json.loads((annotation_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
        zones = ann.get("depth_zones") or []
        emb = np.load(semantic_dir / f"{sample_id}_expert_embedding.npy").astype(np.float32)

        rel_rs = cv2.resize(rel, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        image_hw = np.moveaxis(image, 0, -1)
        image_rs = cv2.resize(image_hw, (size, size), interpolation=cv2.INTER_AREA)
        image_chw = np.moveaxis(image_rs, -1, 0).astype(np.float32)

        d_phys_rs, zone_map_rs = _zones_to_maps(np.moveaxis(image_rs, -1, 0), zones)
        x = np.concatenate([image_chw[None], rel_rs[None, None]], axis=1).astype(np.float32)

        with torch.no_grad():
            d_hat, sigma2, alpha, _, mu = model(
                torch.from_numpy(x).to(device),
                reliability=torch.from_numpy(rel_rs[None, None]).to(device),
                d_phys=torch.from_numpy(d_phys_rs[None, None]).to(device),
                expert_embedding=torch.from_numpy(emb[None]).to(device),
                return_mu=True,
            )
        pred_rs = d_hat.detach().cpu().numpy()[0, 0]
        mu_rs = mu.detach().cpu().numpy()[0, 0]
        alpha_rs = alpha.detach().cpu().numpy()[0, 0]
        sig_rs = sigma2.detach().cpu().numpy()[0, 0]

        pred = cv2.resize(pred_rs, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mu_full = cv2.resize(mu_rs, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        alpha_full = cv2.resize(alpha_rs, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        sig_full = cv2.resize(sig_rs, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        d_phys_full, zone_full = _zones_to_maps(image, zones)
        rel_full = rel.astype(np.float32)

        metrics = regression_metrics(pred, target, valid)
        rows.append({"sample_id": sample_id, **metrics})

        vmin = float(np.nanmin(target)) if np.isfinite(target).any() else 0.0
        vmax = float(np.nanmax(target)) if np.isfinite(target).any() else 1.0

        fig = plt.figure(figsize=(14, 7), dpi=150)
        fig.suptitle(
            f"{sample_id} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}",
            fontsize=10,
        )

        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(image_hw)
        ax.set_title("Input image")
        ax.axis("off")

        ax = fig.add_subplot(2, 4, 2)
        im = ax.imshow(-target, cmap="turbo", vmin=-vmax, vmax=-vmin)
        ax.set_title(f"Ground truth\n[{np.nanmin(-target):.2f}, {np.nanmax(-target):.2f}] m")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(2, 4, 3)
        im = ax.imshow(-pred, cmap="turbo", vmin=-vmax, vmax=-vmin)
        ax.set_title(f"Final prediction\n[{np.nanmin(-pred):.2f}, {np.nanmax(-pred):.2f}] m")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(2, 4, 4)
        im = ax.imshow(-mu_full, cmap="turbo", vmin=-vmax, vmax=-vmin)
        ax.set_title(f"Raw model μ\n[{np.nanmin(-mu_full):.2f}, {np.nanmax(-mu_full):.2f}] m")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(2, 4, 5)
        im = ax.imshow(-d_phys_full, cmap="turbo", vmin=-vmax, vmax=-vmin)
        ax.set_title(f"Physical prior d_phys\n[{np.nanmin(-d_phys_full):.2f}, {np.nanmax(-d_phys_full):.2f}] m")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(2, 4, 6)
        im = ax.imshow(alpha_full, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"Gate α\n[{np.nanmin(alpha_full):.2f}, {np.nanmax(alpha_full):.2f}]")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(2, 4, 7)
        im = ax.imshow(sig_full, cmap="viridis")
        ax.set_title(f"Uncertainty var\n[{np.nanmin(sig_full):.2f}, {np.nanmax(sig_full):.2f}]")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(2, 4, 8)
        im = ax.imshow(zone_full, cmap="tab20", vmin=0, vmax=3)
        ax.set_title("Zone map\n0=bg,1=near,2=mid,3=off")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_dir / f"{sample_id}_vis.png")
        plt.close(fig)

    write_csv(out_dir / "visualization_metrics.csv", rows)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

