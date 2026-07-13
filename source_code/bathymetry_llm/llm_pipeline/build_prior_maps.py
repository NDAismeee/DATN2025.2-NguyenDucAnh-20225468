from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from bathymetry_llm.data.dataset import _resolve_depth_path_for_image
from bathymetry_llm.data.polygon_to_mask import polygon_to_mask
from bathymetry_llm.llm_pipeline.aerial_paths import find_image_by_stem, iter_tiles_from_dirs, load_paired_dirs_from_config, resolve_sidecar_dir
from bathymetry_llm.llm_pipeline.validate_llm_output import fallback_llm_output, validate_llm_output
from bathymetry_llm.utils.io import load_config, package_root, resolve_path


def _raster_hw(path: Path) -> Tuple[int, int]:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])
        return int(arr.shape[-2]), int(arr.shape[-1])
    try:
        import rasterio

        with rasterio.open(path) as src:
            return int(src.height), int(src.width)
    except Exception:
        with Image.open(path) as img:
            w, h = img.size
            return int(h), int(w)


def _scene_hw(
    scene: Path,
    data: Dict[str, Any],
    ref_image: Optional[Path] = None,
    ref_depth: Optional[Path] = None,
) -> Tuple[int, int]:
    meta = data.get("_meta", {}) if isinstance(data, dict) else {}
    if "height" in meta and "width" in meta:
        return int(meta["height"]), int(meta["width"])
    for name in ("image.png", "image.jpg", "image.jpeg", "image.tif", "image.tiff"):
        path = scene / name
        if path.exists():
            return _raster_hw(path)
    for name in ("depth.npy", "valid_mask.npy"):
        path = scene / name
        if path.exists():
            return _raster_hw(path)
    if ref_image is not None and Path(ref_image).exists():
        return _raster_hw(ref_image)
    if ref_depth is not None and Path(ref_depth).exists():
        return _raster_hw(ref_depth)
    raise ValueError(
        f"Cannot infer height/width for {scene}. Add an image under the scene folder, or pass --image-dir/--stem (and optional --depth-dir), or --reference-image / --reference-depth."
    )


def _as_region_list(val: Any) -> List[Any]:
    if isinstance(val, list):
        return val
    return []


def _as_global_depth_range(val: Any) -> Dict[str, float]:
    lo, hi = 0.0, 6.0
    if isinstance(val, dict):
        try:
            return {
                "depth_min": float(val.get("depth_min", lo)),
                "depth_max": float(val.get("depth_max", hi)),
            }
        except (TypeError, ValueError):
            return {"depth_min": lo, "depth_max": hi}
    return {"depth_min": lo, "depth_max": hi}


def build_maps(
    scene: Path,
    use_fallback: bool = True,
    ref_image: Optional[Path] = None,
    ref_depth: Optional[Path] = None,
) -> Dict[str, Path]:
    scene = Path(scene)
    llm_path = scene / "llm_output.json"
    if not llm_path.exists():
        raise FileNotFoundError(f"Missing {llm_path}")
    with open(llm_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    height, width = _scene_hw(scene, data, ref_image=ref_image, ref_depth=ref_depth)
    if ref_image is not None and ref_depth is not None:
        hi, wi = _raster_hw(ref_image)
        hd, wd = _raster_hw(ref_depth)
        if (hi, wi) != (hd, wd):
            raise ValueError(f"Image shape {(hi, wi)} != depth shape {(hd, wd)} for {ref_image.name}")
    ok, errors = validate_llm_output(data, height=height, width=width, raise_on_error=False)
    if not ok:
        if not use_fallback:
            raise ValueError("; ".join(errors))
        data = fallback_llm_output(str(data.get("scene_id", scene.name)), height, width)
        validate_llm_output(data, height=height, width=width, raise_on_error=True)
        with open(scene / "llm_output_invalid_errors.json", "w", encoding="utf-8") as f:
            json.dump({"errors": errors}, f, ensure_ascii=False, indent=2)

    disturbance_regions = _as_region_list(data.get("disturbance_regions", []))
    prior_regions = _as_region_list(data.get("depth_prior_regions", []))

    region_masks = []
    unreliable_union = np.zeros((height, width), dtype=np.float32)
    descriptions = []
    disturbance_meta = []
    for region in disturbance_regions:
        mask = polygon_to_mask(region.get("polygon", []), height, width)
        severity = float(region.get("severity", 1.0))
        unreliable_union = np.maximum(unreliable_union, mask)
        region_masks.append(mask)
        description = str(region.get("description", "")).strip()
        if not description:
            description = str(region.get("type", "other"))
        descriptions.append(description)
        disturbance_meta.append({
            "region_id": region.get("region_id"),
            "type": region.get("type"),
            "severity": severity,
            "description": description,
            "failure_direction": region.get("failure_direction", ""),
            "expected_effect": region.get("expected_effect", ""),
        })

    if region_masks:
        region_masks_arr = np.stack(region_masks, axis=0).astype(np.float32)
    else:
        region_masks_arr = np.zeros((0, height, width), dtype=np.float32)

    weighted_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    d_min_sum = np.zeros((height, width), dtype=np.float32)
    d_max_sum = np.zeros((height, width), dtype=np.float32)
    gamma_num = np.zeros((height, width), dtype=np.float32)
    prior_meta = []
    for region in prior_regions:
        mask = polygon_to_mask(region.get("polygon", []), height, width)
        dmin = float(region.get("depth_min"))
        dmax = float(region.get("depth_max"))
        conf = float(region.get("confidence", 1.0))
        gamma_z = float(np.clip(conf, 0.0, 1.0))
        midpoint = 0.5 * (dmin + dmax)
        w = mask * max(conf, 1e-6)
        weighted_sum += w * midpoint
        d_min_sum += w * dmin
        d_max_sum += w * dmax
        gamma_num += w * gamma_z
        weight_sum += w
        prior_meta.append({
            "region_id": region.get("region_id"),
            "region_name": region.get("region_name", ""),
            "depth_min": dmin,
            "depth_max": dmax,
            "confidence": conf,
        })

    global_range = _as_global_depth_range(data.get("global_depth_range"))
    fallback_mid = 0.5 * (global_range["depth_min"] + global_range["depth_max"])
    fallback_min = global_range["depth_min"]
    fallback_max = global_range["depth_max"]
    assess = data.get("image_assessment", {}) if isinstance(data.get("image_assessment"), dict) else {}
    fallback_gamma = float(np.clip(float(assess.get("confidence", 0.5)), 0.0, 1.0))
    covered = weight_sum > 0
    d_phys = np.full((height, width), fallback_mid, dtype=np.float32)
    d_min = np.full((height, width), fallback_min, dtype=np.float32)
    d_max = np.full((height, width), fallback_max, dtype=np.float32)
    gamma_map = np.full((height, width), fallback_gamma, dtype=np.float32)
    d_phys[covered] = weighted_sum[covered] / weight_sum[covered]
    d_min[covered] = d_min_sum[covered] / weight_sum[covered]
    d_max[covered] = d_max_sum[covered] / weight_sum[covered]
    gamma_map[covered] = gamma_num[covered] / weight_sum[covered]
    w_phys = np.maximum(d_max - d_min, 0.0).astype(np.float32)

    outputs = {
        "unreliable_mask": scene / "unreliable_mask.npy",
        "region_masks": scene / "region_masks.npy",
        "d_phys": scene / "d_phys.npy",
        "d_min": scene / "d_min.npy",
        "d_max": scene / "d_max.npy",
        "gamma_map": scene / "gamma_map.npy",
        "w_phys": scene / "w_phys.npy",
        "region_metadata": scene / "region_metadata.json",
    }
    np.save(outputs["unreliable_mask"], np.clip(unreliable_union, 0.0, 1.0)[None, ...].astype(np.float32))
    np.save(outputs["region_masks"], region_masks_arr)
    np.save(outputs["d_phys"], d_phys[None, ...].astype(np.float32))
    np.save(outputs["d_min"], d_min[None, ...].astype(np.float32))
    np.save(outputs["d_max"], d_max[None, ...].astype(np.float32))
    np.save(outputs["gamma_map"], gamma_map[None, ...].astype(np.float32))
    np.save(outputs["w_phys"], w_phys[None, ...].astype(np.float32))
    with open(outputs["region_metadata"], "w", encoding="utf-8") as f:
        json.dump({"disturbance_regions": disturbance_meta, "depth_prior_regions": prior_meta, "descriptions": descriptions}, f, ensure_ascii=False, indent=2)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build prior maps from llm_output.json. Use --scene, or --image-dir + --stem with sidecar layout."
    )
    parser.add_argument("--scene", default=None, help="Folder containing llm_output.json (classic scene or sidecar dir)")
    parser.add_argument("--image-dir", default=None, help="With --stem: aerial RGB folder (e.g. agia_napa/img/aerial_train)")
    parser.add_argument("--stem", default=None, help="With --image-dir: image stem, e.g. img_409")
    parser.add_argument(
        "--depth-dir",
        default=None,
        help="With --image-dir: ground-truth depth folder (e.g. agia_napa/depth/aerial) to match image size",
    )
    parser.add_argument("--pair-aux-root", default=None, help="Override data.pair_aux_root; sidecar is <root>/<stem>/")
    parser.add_argument("--reference-image", default=None, help="Explicit image path for H×W if scene folder has no raster")
    parser.add_argument("--reference-depth", default=None, help="Explicit depth path for H×W if scene folder has no raster")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--all-tiles",
        action="store_true",
        help="Run for every image/depth pair from config data.aerial_train_* paths (requires --config; needs llm_output.json per sidecar).",
    )
    parser.add_argument("--no_fallback", action="store_true")
    args = parser.parse_args()
    root = package_root()
    data_cfg: Dict[str, Any] = {}
    if args.config:
        data_cfg = (load_config(resolve_path(args.config, root)) or {}).get("data") or {}
    pair_aux = args.pair_aux_root or data_cfg.get("pair_aux_root")
    pair_aux_path = Path(resolve_path(pair_aux, root)) if pair_aux else None

    ref_image = Path(resolve_path(args.reference_image, root)) if args.reference_image else None
    ref_depth = Path(resolve_path(args.reference_depth, root)) if args.reference_depth else None

    if args.all_tiles:
        if not args.config:
            parser.error("--all-tiles requires --config.")
        if args.scene or args.image_dir or args.stem or args.reference_image or args.reference_depth:
            parser.error("--all-tiles cannot be combined with --scene, --image-dir/--stem, or reference paths.")
        image_dir, depth_dir, pair_cfg = load_paired_dirs_from_config(data_cfg, root)
        pair_use = Path(resolve_path(args.pair_aux_root, root)) if args.pair_aux_root else pair_cfg
        for stem, ref_img, scene, ref_dep in iter_tiles_from_dirs(image_dir, depth_dir, pair_use):
            if not (scene / "llm_output.json").exists():
                print(f"skip_no_llm_output {stem}")
                continue
            scene.mkdir(parents=True, exist_ok=True)
            outputs = build_maps(scene, use_fallback=not args.no_fallback, ref_image=ref_img, ref_depth=ref_dep)
            for key, path in outputs.items():
                print(f"{stem} {key}: {path}")
        return

    if args.scene:
        if args.image_dir or args.stem:
            parser.error("Use either --scene alone, or --image-dir + --stem (not both).")
        scene = Path(args.scene)
        if not scene.is_absolute():
            scene = root / scene
    elif args.image_dir and args.stem:
        image_dir = Path(resolve_path(args.image_dir, root))
        stem = str(args.stem)
        ref_img = find_image_by_stem(image_dir, stem)
        if ref_image is None:
            ref_image = ref_img
        depth_key = args.depth_dir or data_cfg.get("aerial_train_depth_dir")
        depth_dir_path = Path(resolve_path(depth_key, root)) if depth_key else None
        if depth_dir_path is not None:
            if ref_depth is None:
                ref_depth = _resolve_depth_path_for_image(ref_img, depth_dir_path)
        scene = resolve_sidecar_dir(stem, pair_aux_path, image_dir)
    else:
        parser.error("Provide --scene, or both --image-dir and --stem.")

    if not scene.is_dir():
        raise FileNotFoundError(f"Scene / sidecar directory not found: {scene}")

    outputs = build_maps(scene, use_fallback=not args.no_fallback, ref_image=ref_image, ref_depth=ref_depth)
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
