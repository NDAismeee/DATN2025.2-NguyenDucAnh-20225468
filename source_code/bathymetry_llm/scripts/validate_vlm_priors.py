from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bathymetry_llm.data.dataset import _read_depth, _read_optional_npy, _resolve_depth_path_for_image
from bathymetry_llm.llm_pipeline.aerial_paths import (
    iter_tiles_from_dirs,
    load_paired_dirs_from_config,
)
from bathymetry_llm.utils.io import load_config, package_root, resolve_path, save_json
from bathymetry_llm.utils.metrics import interval_coverage, mask_iou_f1


def _load_reference_mask(ref_dir: Optional[Path], stem: str, h: int, w: int) -> Optional[np.ndarray]:
    if ref_dir is None:
        return None
    for name in (f"{stem}.npy", f"{stem}_unreliable.npy"):
        path = ref_dir / name
        if path.exists():
            arr = np.load(path).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.shape != (h, w):
                raise ValueError(f"reference mask shape {arr.shape} != {(h, w)} for {stem}")
            return (arr > 0.5).astype(np.float32)
    return None


def evaluate_one(scene: Path, image_path: Path, depth_path: Path, ref_mask_dir: Optional[Path]) -> Dict[str, Any]:
    depth, valid = _read_depth(depth_path)
    h, w = depth.shape[-2:]
    valid_mask = (valid[0] > 0.5).astype(np.float32)
    target = depth[0]
    pred_mask = _read_optional_npy(scene / "unreliable_mask.npy", (1, h, w), default=0.0)[0]
    d_min = _read_optional_npy(scene / "d_min.npy", (1, h, w), default=0.0)[0]
    d_max = _read_optional_npy(scene / "d_max.npy", (1, h, w), default=0.0)[0]
    cov = interval_coverage(target, d_min, d_max, valid_mask)
    out: Dict[str, Any] = {
        "scene": scene.name,
        "image": image_path.name,
        "depth": depth_path.name,
        "interval_coverage": cov["coverage"],
        "interval_mean_width": cov["mean_width"],
    }
    ref = _load_reference_mask(ref_mask_dir, image_path.stem, h, w)
    if ref is not None:
        out.update({f"mask_{k}": v for k, v in mask_iou_f1(pred_mask, ref).items()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VLM priors: mask IoU/F1 (if reference available) + interval coverage.")
    parser.add_argument("--config", required=True, help="Config providing data.aerial_train_image_dir / depth_dir / pair_aux_root.")
    parser.add_argument("--reference-mask-dir", default=None, help="Folder with reference unreliable masks named <stem>.npy")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <metric_dir>/vlm_prior_validation.json)")
    args = parser.parse_args()
    root = package_root()
    cfg = load_config(resolve_path(args.config, root))
    data_cfg = cfg.get("data", {}) or {}
    out_cfg = cfg.get("output", {}) or {}
    image_dir, depth_dir, pair_cfg = load_paired_dirs_from_config(data_cfg, root)
    ref_dir = Path(resolve_path(args.reference_mask_dir, root)) if args.reference_mask_dir else None
    rows: List[Dict[str, Any]] = []
    for stem, image_path, scene, depth_path in iter_tiles_from_dirs(image_dir, depth_dir, pair_cfg):
        if not (scene / "unreliable_mask.npy").exists():
            continue
        rows.append(evaluate_one(scene, image_path, depth_path, ref_dir))
    summary: Dict[str, Any] = {
        "n_scenes": len(rows),
        "interval_coverage_mean": float(np.mean([r["interval_coverage"] for r in rows])) if rows else 0.0,
        "interval_mean_width_mean": float(np.mean([r["interval_mean_width"] for r in rows])) if rows else 0.0,
    }
    iou_vals = [r["mask_iou"] for r in rows if "mask_iou" in r]
    f1_vals = [r["mask_f1"] for r in rows if "mask_f1" in r]
    if iou_vals:
        summary["mask_iou_mean"] = float(np.mean(iou_vals))
    if f1_vals:
        summary["mask_f1_mean"] = float(np.mean(f1_vals))
    out_path = (
        Path(args.output)
        if args.output
        else resolve_path(out_cfg.get("metric_dir", "outputs/metrics"), root) / "vlm_prior_validation.json"
    )
    save_json({"summary": summary, "per_scene": rows}, out_path)
    print(json.dumps(summary, indent=2))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
