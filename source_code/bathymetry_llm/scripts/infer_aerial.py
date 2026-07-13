from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from bathymetry_llm.data.dataset import build_bathymetry_dataset
from bathymetry_llm.models.llm_guided_bathymetry import LLMGuidedBathymetryModel
from bathymetry_llm.utils.infer_visualize import save_llm_guided_infer_figure
from bathymetry_llm.utils.io import load_config, package_root, pick_device, resolve_path
from bathymetry_llm.utils.metrics import compute_metrics


def _forward(model: LLMGuidedBathymetryModel, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return model(
        batch["image"],
        batch["unreliable_mask"],
        batch["d_phys"],
        batch["region_masks"],
        batch["text_embeddings"],
        batch["gamma_map"],
        batch["w_phys"],
        batch["region_valid_mask"],
    )


def infer_one(
    sample: Dict[str, Any],
    model: LLMGuidedBathymetryModel,
    device: torch.device,
    out_dir: Path,
    quiet: bool,
) -> Dict[str, Any]:
    batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    with torch.inference_mode():
        outputs = _forward(model, batch)
    scene_id = str(sample["scene_id"])
    depth = outputs["depth"][0].detach().cpu().numpy()
    mu = outputs["mu"][0].detach().cpu().numpy()
    var = outputs["var"][0].detach().cpu().numpy()
    alpha = outputs["alpha"][0].detach().cpu().numpy()
    gt = sample["depth"].numpy()
    vm = sample["valid_mask"].numpy()
    d_phys = sample["d_phys"].numpy()
    image = sample["image"].numpy()
    metrics = compute_metrics(gt, depth, vm)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{scene_id}_pred.npy", depth.astype(np.float32))
    np.save(out_dir / f"{scene_id}_gt.npy", gt.astype(np.float32))
    np.save(out_dir / f"{scene_id}_mu.npy", mu.astype(np.float32))
    np.save(out_dir / f"{scene_id}_var.npy", var.astype(np.float32))
    np.save(out_dir / f"{scene_id}_alpha.npy", alpha.astype(np.float32))
    np.save(out_dir / f"{scene_id}_d_phys.npy", d_phys.astype(np.float32))
    fig_path = out_dir / f"{scene_id}_vis.png"
    title = f"{scene_id} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}"
    save_llm_guided_infer_figure(
        image_chw=image,
        gt_positive=gt,
        pred_positive=depth,
        mu_positive=mu,
        valid_mask=vm,
        d_phys=d_phys,
        alpha=alpha,
        var=var,
        out_path=fig_path,
        title_prefix=title,
    )
    if not quiet:
        print(f"sample={scene_id} MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} -> {fig_path}")
    return {
        "scene_id": scene_id,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "fig_path": str(fig_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference + infer.py-style visualization for paired aerial config (agia_napa_aerial_train.yaml)."
    )
    parser.add_argument("--config", default="configs/agia_napa_aerial_train.yaml")
    parser.add_argument("--checkpoint", default="/mnt/disk3/anhnd2468/MagicBathyNet/hao-chapter1-depth-prediction/source_code/new_test/bathymetry_llm/outputs/agia_napa_aerial/checkpoints/best.pt")
    parser.add_argument("--output_dir", default="/mnt/disk3/anhnd2468/MagicBathyNet/hao-chapter1-depth-prediction/source_code/new_test/bathymetry_llm/outputs/predictions/aerial_train")
    parser.add_argument("--all", action="store_true", help="Run on every tile in the dataset (split=all).")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = package_root()
    cfg = load_config(resolve_path(args.config, root))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("output", {})
    if str(data_cfg.get("layout", "")) != "paired_aerial_folders":
        print("infer_aerial.py expects data.layout: paired_aerial_folders in the config.", file=sys.stderr)
        sys.exit(1)
    device = pick_device(str(args.device))
    ckpt_path = resolve_path(args.checkpoint, root)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", cfg)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    text_dim = int(data_cfg.get("text_dim", model_cfg.get("text_dim", 384)))
    k_max = int(data_cfg.get("k_max", 8))
    image_size = data_cfg.get("image_size")
    ds = build_bathymetry_dataset(data_cfg, "all", image_size, k_max, text_dim, config_root=root)
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset is empty.")
    if args.all:
        start = max(0, int(args.start_idx))
        end = n if args.end_idx is None else int(args.end_idx)
        if start > n or end > n or end < start:
            raise IndexError(f"Invalid range start={start} end={end} for size {n}")
        indices = list(range(start, end))
    else:
        if args.sample_idx < 0 or args.sample_idx >= n:
            raise IndexError(f"sample_idx={args.sample_idx} out of range for size {n}")
        indices = [int(args.sample_idx)]
    out_dir = Path(args.output_dir) if args.output_dir else resolve_path(out_cfg.get("prediction_dir", "outputs/predictions"), root) / "infer_aerial"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = LLMGuidedBathymetryModel(
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        text_dim=text_dim,
        gate_hidden=int(model_cfg.get("gate_hidden", 128)),
        gate_dropout=float(model_cfg.get("gate_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    quiet = len(indices) > 1
    rows: List[Dict[str, Any]] = []
    for k, idx in enumerate(indices):
        if quiet:
            print(f"[{k + 1}/{len(indices)}] idx={idx} ...", flush=True)
        row = infer_one(ds[idx], model, device, out_dir, quiet=quiet)
        rows.append({**row, "sample_idx": idx})
    if len(rows) > 1:
        summary = out_dir / "infer_summary.csv"
        with open(summary, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sample_idx", "scene_id", "mae", "rmse", "fig_path"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {summary} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
