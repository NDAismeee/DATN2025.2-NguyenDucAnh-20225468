from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from bathymetry_llm.data.dataset import bathymetry_collate_fn, build_bathymetry_dataset
from bathymetry_llm.models.llm_guided_bathymetry import LLMGuidedBathymetryModel
from bathymetry_llm.utils.io import load_config, package_root, pick_device, resolve_path, save_json
from bathymetry_llm.utils.metrics import (
    compute_metrics,
    interval_coverage,
    physical_consistency_metrics,
    uncertainty_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--domain_min", type=float, default=None, help="Lower bound of evaluation domain (meters)")
    parser.add_argument("--domain_max", type=float, default=None, help="Upper bound of evaluation domain (meters)")
    args = parser.parse_args()
    root = package_root()
    cfg = load_config(resolve_path(args.config, root))
    device = pick_device(cfg.get("training", {}).get("device", "auto"))
    ckpt = torch.load(resolve_path(args.checkpoint, root), map_location=device)
    cfg = ckpt.get("config", cfg)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("output", {})
    split = args.split or data_cfg.get("test_split", "test")
    dataset = build_bathymetry_dataset(
        data_cfg,
        split,
        data_cfg.get("image_size"),
        int(data_cfg.get("k_max", 8)),
        int(data_cfg.get("text_dim", model_cfg.get("text_dim", 384))),
        config_root=root,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=bathymetry_collate_fn,
    )
    model = LLMGuidedBathymetryModel(
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        text_dim=int(data_cfg.get("text_dim", model_cfg.get("text_dim", 384))),
        gate_hidden=int(model_cfg.get("gate_hidden", 128)),
        gate_dropout=float(model_cfg.get("gate_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    targets, preds, masks = [], [], []
    variances, unreliable, d_mins, d_maxs = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(
                tensor_batch["image"],
                tensor_batch["unreliable_mask"],
                tensor_batch["d_phys"],
                tensor_batch["region_masks"],
                tensor_batch["text_embeddings"],
                tensor_batch["gamma_map"],
                tensor_batch["w_phys"],
                tensor_batch["region_valid_mask"],
            )
            targets.append(tensor_batch["depth"].cpu().numpy())
            preds.append(outputs["depth"].cpu().numpy())
            masks.append(tensor_batch["valid_mask"].cpu().numpy())
            variances.append(outputs["var"].cpu().numpy())
            unreliable.append(tensor_batch["unreliable_mask"].cpu().numpy())
            d_mins.append(tensor_batch["d_min"].cpu().numpy())
            d_maxs.append(tensor_batch["d_max"].cpu().numpy())
    target_arr = np.concatenate(targets)
    pred_arr = np.concatenate(preds)
    mask_arr = np.concatenate(masks)
    var_arr = np.concatenate(variances)
    unreliable_arr = np.concatenate(unreliable)
    d_min_arr = np.concatenate(d_mins)
    d_max_arr = np.concatenate(d_maxs)
    metrics: Dict[str, Any] = {}
    metrics.update(compute_metrics(target_arr, pred_arr, mask_arr))
    metrics.update(uncertainty_diagnostics(target_arr, pred_arr, var_arr, mask_arr))
    metrics.update(
        physical_consistency_metrics(
            target_arr, pred_arr, mask_arr,
            unreliable_mask=unreliable_arr,
            domain_min=args.domain_min,
            domain_max=args.domain_max,
        )
    )
    metrics.update(interval_coverage(target_arr, d_min_arr, d_max_arr, mask_arr))
    metric_dir = resolve_path(out_cfg.get("metric_dir", "outputs/metrics"), root)
    save_json(metrics, metric_dir / f"{split}_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
