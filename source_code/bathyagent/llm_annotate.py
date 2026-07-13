#!/usr/bin/env python3
"""
Offline coastal-image annotation via OpenAI vision API (OPENAI_API_KEY).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from common import load_yaml_config
from dataset import (
    S2_BAND_TO_INDEX,
    _match_depth_file,
    _normalize_s2_image,
    _read_tif_depth,
    _read_tif_image,
    resolve_selected_bands,
)
from vlm_utils import GenerationConfig, OpenAIVisionAnnotator, run_openai_annotation


FULL_PROMPT = """
You are a coastal bathymetry expert and remote-sensing assistant. You receive one aerial RGB image.

Output valid JSON only. Do not use markdown, code fences, or text outside JSON.
Use pixel coordinates in the ORIGINAL full image resolution (origin top-left, x right, y down).
Keep all depth intervals inside the site-specific valid depth range supplied by the user.
Use conservative, image-dependent depth intervals, not fixed default depth ranges.
Assign low confidence to uncertain areas instead of inventing precise depths.
Problem categories must be one of: sun_glint, shadow, turbidity, foam, ambiguous_bottom, other.

Do not partition the whole water area into broad bathymetric zones.
Only identify unreliable or visually ambiguous water regions where the neural model may make errors.
For each such region, return a conservative plausible depth interval.
If there are no problematic regions, return "problem_regions": [].

Task:
1. Detect land, water, and shoreline. Define water.polygons as the full visible open-water area only.
2. Identify only visually unreliable or ambiguous WATER regions where a neural bathymetry model is likely to make mistakes.
3. Do NOT partition the whole water area.
4. Do NOT create nearshore/transition/offshore zones unless those zones are themselves visibly unreliable.
5. For each unreliable region, return a polygon and a conservative plausible depth interval.
6. If the image has no obvious unreliable water regions, return an empty problem_regions list.
7. Use free-form polygons that follow the actual problematic region shape. Do not use large rectangular demo boxes.

Return exactly this schema:
{
  "water": {
    "polygons": [[[x, y], [x, y], [x, y]]]
  },
  "problem_regions": [
    {
      "id": "prob_0",
      "category": "sun_glint | shadow | turbidity | foam | ambiguous_bottom | other",
      "polygons": [[[x, y], [x, y], [x, y]]],
      "severity": 0.0,
      "description": "short explanation of why this region is visually unreliable",
      "depth_min": 0.0,
      "depth_max": 2.0,
      "depth_confidence": 0.6,
      "rationale": "physical rationale for this local conservative interval"
    }
  ],
  "scene_summary": "short auditing summary"
}
""".strip()


# =========================================================
# Utilities
# =========================================================
def list_images(input_dir: Path, exts: Optional[set[str]] = None) -> List[Path]:
    if exts is None:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])


def _resolve_vlm_band_indices(
    selected_bands: Optional[object], image_mode: str
) -> List[int]:
    if str(image_mode).lower() == "rgb":
        r = resolve_selected_bands(selected_bands, image_mode="rgb")
        assert r is not None
        if len(r) < 3:
            raise ValueError(
                f"VLM visualization needs 3 channels for rgb; got {len(r)} from selected_bands."
            )
        return r[:3]

    resolved = resolve_selected_bands(selected_bands, image_mode="s2")
    if resolved is None:
        return [
            S2_BAND_TO_INDEX["B8"],
            S2_BAND_TO_INDEX["B3"],
            S2_BAND_TO_INDEX["B2"],
        ]
    if len(resolved) < 3:
        raise ValueError(
            f"VLM visualization needs at least 3 bands; selected_bands has {len(resolved)}."
        )
    return resolved[:3]


def open_image_rgb(
    path: Path,
    max_side: Optional[int] = None,
    min_side: Optional[int] = None,
    selected_bands: Optional[object] = None,
    reflectance_scale: float = 255.0,
    image_mode: str = "rgb",
) -> Image.Image:
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"} and str(image_mode).lower() == "rgb":
        cube = _read_tif_image(path)
        if cube.shape[0] != 3:
            raise ValueError(
                f"Expected 3 RGB channels for image_mode rgb, got {cube.shape[0]} for {path}"
            )
        orig_h, orig_w = int(cube.shape[1]), int(cube.shape[2])
        idx = _resolve_vlm_band_indices(selected_bands, image_mode="rgb")
        cube = cube[idx, :, :]
        cube = _normalize_s2_image(cube, scale=reflectance_scale)
        rgb = np.transpose(cube, (1, 2, 0))
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_u8 = (rgb * 255.0).astype(np.uint8)
        img = Image.fromarray(rgb_u8, mode="RGB")

    elif suffix in {".tif", ".tiff"} and str(image_mode).lower() == "s2":
        cube = _read_tif_image(path)
        if cube.shape[0] < 3:
            raise ValueError(f"Expected at least 3 bands, got {cube.shape[0]} for {path}")
        orig_h, orig_w = int(cube.shape[1]), int(cube.shape[2])
        idx = _resolve_vlm_band_indices(selected_bands, image_mode="s2")
        cube = cube[idx, :, :]
        cube = _normalize_s2_image(cube, scale=reflectance_scale)
        rgb = np.transpose(cube, (1, 2, 0))
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_u8 = (rgb * 255.0).astype(np.uint8)
        img = Image.fromarray(rgb_u8, mode="RGB")

    else:
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size

    w, h = img.size
    if min_side is not None and max(w, h) < int(min_side):
        scale = float(min_side) / float(max(w, h))
        new_size = (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        )
        img = img.resize(new_size, Image.Resampling.BILINEAR)
        w, h = img.size

    if max_side is not None and max(w, h) > int(max_side):
        scale = float(max_side) / float(max(w, h))
        new_size = (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        )
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    img.info["orig_size"] = (int(orig_w), int(orig_h))
    return img


def build_output_paths(output_dir: Path, image_path: Path) -> Tuple[Path, Path]:
    stem = image_path.stem
    json_path = output_dir / f"{stem}.json"
    raw_path = output_dir / f"{stem}_raw.txt"
    return json_path, raw_path


def select_images_with_depth(
    images: List[Path],
    depth_dir: Path,
    depth_suffixes_to_try: Optional[Sequence[str]],
    allow_empty_depth: bool,
) -> Tuple[List[Path], Dict[str, object]]:
    kept: List[Path] = []
    skipped_no_depth: List[str] = []
    skipped_empty_depth: List[str] = []

    for img_path in images:
        depth_path = _match_depth_file(
            image_path=img_path,
            depth_dir=depth_dir,
            depth_suffixes_to_try=depth_suffixes_to_try,
        )
        if depth_path is None:
            skipped_no_depth.append(img_path.name)
            continue

        raw_depth = _read_tif_depth(depth_path)
        raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = raw_depth < 0
        if valid_mask.sum() == 0:
            skipped_empty_depth.append(img_path.name)
            if not allow_empty_depth:
                continue
        kept.append(img_path)

    stats: Dict[str, object] = {
        "num_candidates": len(images),
        "num_kept": len(kept),
        "num_skipped_no_depth_file": len(skipped_no_depth),
        "num_skipped_empty_valid_depth": len(skipped_empty_depth),
        "skipped_no_depth": skipped_no_depth,
        "skipped_empty_depth": skipped_empty_depth,
    }
    return kept, stats


# =========================================================
# JSON normalization
# =========================================================
def _clip_point_xy(pt, width: int, height: int) -> List[int]:
    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
        return []
    try:
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
    except Exception:
        return []
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return [x, y]


def _normalize_polygon_list(polygons, width: int, height: int) -> List[List[List[int]]]:
    if not isinstance(polygons, list):
        return []

    out: List[List[List[int]]] = []
    for poly in polygons:
        if not isinstance(poly, list):
            continue
        pts = [_clip_point_xy(p, width, height) for p in poly]

        cleaned: List[List[int]] = []
        for p in pts:
            if not p:
                continue
            if len(cleaned) == 0 or cleaned[-1] != p:
                cleaned.append(p)

        if len(cleaned) >= 3:
            out.append(cleaned)

    return out


def _count_polygons(polygons) -> int:
    if not isinstance(polygons, list):
        return 0
    return sum(1 for p in polygons if isinstance(p, list) and len(p) >= 3)


def _infer_site_from_image_path(path: Path) -> str:
    name = str(path).lower()
    if "agia" in name:
        return "agia_napa"
    if "puck" in name:
        return "puck_lagoon"
    return "unknown"


def _site_depth_range(config: Dict, image_path: Path) -> Tuple[float, float]:
    ranges = config.get("llm", {}).get("valid_depth_ranges", {})
    site = _infer_site_from_image_path(image_path)
    value = ranges.get(site, ranges.get("unknown", [0.0, 30.29]))
    return float(value[0]), float(value[1])


def _is_placeholder_triangle(polygons: List[List[List[int]]], width: int, height: int) -> bool:
    if _count_polygons(polygons) != 1:
        return False
    poly = polygons[0]
    if not isinstance(poly, list) or len(poly) != 3:
        return False

    w1 = int(width - 1)
    h1 = int(height - 1)
    pts = {(int(p[0]), int(p[1])) for p in poly if isinstance(p, list) and len(p) == 2}
    if len(pts) != 3:
        return False

    candidates = [
        {(0, 0), (w1, 0), (w1, h1)},
        {(0, 0), (0, h1), (w1, h1)},
        {(0, 0), (w1, 0), (0, h1)},
        {(w1, 0), (w1, h1), (0, h1)},
    ]
    return any(pts == c for c in candidates)


def normalize_annotation_json(
    data: Dict,
    image_path: Path,
    width: int,
    height: int,
    model_name: str,
    config: Optional[Dict] = None,
    allow_legacy_schema: bool = False,
) -> Dict:
    out: Dict[str, object] = {}

    if not isinstance(data, dict):
        raise ValueError("Annotation root must be a JSON object.")

    if not allow_legacy_schema:
        for key in ("uncertainty_regions", "disturbance_regions", "depth_regions"):
            if key in data:
                raise ValueError(
                    f"Legacy field {key} is not allowed in strict local-prior mode. "
                    "Rerun llm_annotate.py with the new prompt."
                )
        if "problem_regions" not in data:
            raise ValueError(
                "Missing problem_regions. This annotation is not compatible with the "
                "disturbance-localized prior design. Rerun llm_annotate.py with the new prompt."
            )
        problem_in = data["problem_regions"]
    else:
        problem_in = data.get("problem_regions", [])
        if not problem_in:
            problem_in = data.get("disturbance_regions", [])
        if not problem_in:
            problem_in = data.get("uncertainty_regions", [])

    water = data.get("water", {}) if isinstance(data, dict) else {}
    site_min, site_max = _site_depth_range(config or {}, image_path)

    out["water"] = {
        "polygons": _normalize_polygon_list(water.get("polygons", []), width, height)
    }

    if _count_polygons(out["water"]["polygons"]) == 0:
        raise ValueError("Invalid annotation: water.polygons is empty after normalization")
    if _is_placeholder_triangle(out["water"]["polygons"], width, height):
        raise ValueError("Invalid annotation: detected placeholder water triangle")

    allowed_categories = {
        "sun_glint",
        "shadow",
        "turbidity",
        "foam",
        "ambiguous_bottom",
        "other",
    }
    legacy_category = {
        "wave_roughness": "other",
        "bottom_confusion": "ambiguous_bottom",
        "color_ambiguity": "ambiguous_bottom",
        "sensor_artifact": "other",
    }

    cleaned_problem_regions: List[Dict[str, object]] = []
    if isinstance(problem_in, list):
        for idx, item in enumerate(problem_in):
            if not isinstance(item, dict):
                continue

            category = str(item.get("category", item.get("issue_type", "other"))).strip().lower()
            category = legacy_category.get(category, category)
            if category not in allowed_categories:
                category = "other"

            polygons = _normalize_polygon_list(item.get("polygons", []), width, height)
            if len(polygons) == 0 and isinstance(item.get("bbox"), list) and len(item.get("bbox")) == 4:
                x0, y0, x1, y1 = [int(round(float(v))) for v in item["bbox"]]
                polygons = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                polygons = _normalize_polygon_list([polygons], width, height)
            if len(polygons) == 0:
                continue

            if "depth_min" not in item or "depth_max" not in item:
                raise ValueError(
                    f"problem_regions[{idx}] missing depth_min/depth_max. "
                    "Do not use site-wide defaults. Rerun LLM annotation."
                )
            if "depth_confidence" not in item:
                raise ValueError(f"problem_regions[{idx}] missing depth_confidence.")
            for req_key in ("description", "rationale"):
                if req_key not in item or not str(item.get(req_key, "")).strip():
                    raise ValueError(f"problem_regions[{idx}] missing required field: {req_key}")

            severity = float(item.get("severity", 1.0))
            if "risk_level" in item:
                severity = {"low": 0.33, "medium": 0.66, "high": 1.0}.get(
                    str(item["risk_level"]).lower(),
                    severity,
                )
            d_min = float(item["depth_min"])
            d_max = float(item["depth_max"])
            d_conf = float(item["depth_confidence"])
            if d_min > d_max:
                raise ValueError(f"problem_regions[{idx}] has depth_min > depth_max.")
            if not (site_min <= d_min <= d_max <= site_max):
                raise ValueError(
                    f"Invalid local depth interval [{d_min}, {d_max}] outside [{site_min}, {site_max}]"
                )

            cleaned_problem_regions.append(
                {
                    "id": str(item.get("id", f"prob_{idx}")),
                    "category": category,
                    "polygons": polygons,
                    "severity": float(np.clip(severity, 0.0, 1.0)),
                    "description": str(item["description"]).strip(),
                    "depth_min": d_min,
                    "depth_max": d_max,
                    "depth_confidence": float(np.clip(d_conf, 0.0, 1.0)),
                    "rationale": str(item["rationale"]).strip(),
                }
            )

    out["problem_regions"] = cleaned_problem_regions
    out["scene_summary"] = str(data.get("scene_summary", "") if isinstance(data, dict) else "").strip()

    out["_meta"] = {
        "source_image": str(image_path),
        "sample_id": image_path.stem,
        "width": int(width),
        "height": int(height),
        "model_name": model_name,
        "site": _infer_site_from_image_path(image_path),
        "valid_depth_range": [site_min, site_max],
        "num_water_polygons": _count_polygons(out["water"]["polygons"]),
        "num_problem_regions": len(out["problem_regions"]),
        "local_prior_only": True,
        "schema": "disturbance_local_prior",
    }

    return out


def _scale_xy_inplace(obj, sx: float, sy: float) -> None:
    if isinstance(obj, list):
        for it in obj:
            _scale_xy_inplace(it, sx, sy)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _scale_xy_inplace(v, sx, sy)
        return
    if isinstance(obj, tuple):
        for it in obj:
            _scale_xy_inplace(it, sx, sy)


def scale_annotation_to_original(data: Dict, resized_wh: Tuple[int, int], orig_wh: Tuple[int, int]) -> Dict:
    rw, rh = resized_wh
    ow, oh = orig_wh
    if rw <= 0 or rh <= 0:
        return data
    sx = float(ow) / float(rw)
    sy = float(oh) / float(rh)

    def _scale_points(x):
        if isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
            return [float(x[0]) * sx, float(x[1]) * sy]
        if isinstance(x, list):
            return [_scale_points(v) for v in x]
        if isinstance(x, dict):
            return {k: _scale_points(v) for k, v in x.items()}
        return x

    return _scale_points(data)

# =========================================================
# Args
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Image folder (default: data.image_dir from config / ${IMAGE_DIR} in .env).",
    )
    parser.add_argument(
        "--depth_dir",
        type=str,
        default=None,
        help="Depth tile directory (default: data.depth_dir from config / ${DEPTH_DIR} in .env).",
    )
    parser.add_argument(
        "--no_depth_filter",
        action="store_true",
        help="Annotate every candidate image without requiring a depth pair.",
    )
    parser.add_argument(
        "--allow_empty_depth",
        action="store_true",
        help="Keep images whose depth file exists but has no valid water pixels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Annotation JSON folder (default: vlm.annotation_json_dir / ${ANNOTATION_JSON_DIR}).",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default=None,
        help="Override OpenAI vision model (default: vlm.openai_model in config).",
    )
    parser.add_argument("--max_side", type=int, default=1024)
    parser.add_argument("--min_side", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow_legacy_schema",
        action="store_true",
        help="Accept legacy disturbance_regions/uncertainty_regions (not recommended).",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--image_suffix",
        type=str,
        default=None,
        help="Optional glob pattern, e.g. img_*.tif. If omitted, all common image types are used.",
    )
    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    data_cfg = config.get("data", {})
    vlm_cfg = config.get("vlm", {})

    raw_image_dir = (args.input_dir or data_cfg.get("image_dir") or "").strip()
    if not raw_image_dir:
        raise ValueError(
            "No image directory: set data.image_dir in config (e.g. ${IMAGE_DIR} via .env) or pass --input_dir."
        )
    input_dir = Path(raw_image_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            f"Fix data.image_dir / IMAGE_DIR in .env, or pass --input_dir with a valid path."
        )

    raw_out = (args.output_dir or vlm_cfg.get("annotation_json_dir") or "").strip()
    if raw_out:
        output_dir = Path(raw_out)
    else:
        output_dir = input_dir.parent / "annotations"

    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.openai_model or vlm_cfg.get("openai_model") or "gpt-4o"
    min_side = int(getattr(args, "min_side", 512) or 512)
    if isinstance(vlm_cfg, dict) and "min_side" in vlm_cfg:
        try:
            min_side = int(vlm_cfg.get("min_side"))
        except Exception:
            pass

    # image selection
    if args.image_suffix is not None:
        images = sorted(input_dir.glob(args.image_suffix))
    else:
        default_suffix = data_cfg.get("image_suffix", None)
        if isinstance(default_suffix, str) and "*" in default_suffix:
            images = sorted(input_dir.glob(default_suffix))
        else:
            images = list_images(input_dir)

    if len(images) == 0:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    depth_filter_stats: Optional[Dict[str, object]] = None
    if not args.no_depth_filter:
        raw_depth_dir = args.depth_dir or data_cfg.get("depth_dir")
        if not raw_depth_dir or not str(raw_depth_dir).strip():
            raise ValueError(
                "Depth filtering is on but depth_dir is missing: set data.depth_dir in config or pass --depth_dir."
            )
        depth_dir = Path(raw_depth_dir)
        if not depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
        suffixes = data_cfg.get("depth_suffixes_to_try", None)
        images, depth_filter_stats = select_images_with_depth(
            images,
            depth_dir=depth_dir,
            depth_suffixes_to_try=suffixes,
            allow_empty_depth=bool(args.allow_empty_depth),
        )
        print(
            f"[depth filter] depth_dir={depth_dir} "
            f"candidates={depth_filter_stats['num_candidates']} "
            f"kept={depth_filter_stats['num_kept']} "
            f"skipped_no_depth={depth_filter_stats['num_skipped_no_depth_file']} "
            f"skipped_empty_valid={depth_filter_stats['num_skipped_empty_valid_depth']}"
        )
        if len(images) == 0:
            raise FileNotFoundError(
                "No images left after depth filtering. Check depth_dir naming and valid depth pixels."
            )

    if args.limit > 0:
        images = images[: args.limit]

    print(f"OpenAI model: {model_name}")
    print(f"Found {len(images)} image(s).")
    print(f"Saving annotations to: {output_dir}")

    client = OpenAIVisionAnnotator(model=model_name)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    allow_legacy = bool(
        args.allow_legacy_schema or vlm_cfg.get("allow_legacy_schema", False)
    )

    success = 0
    failed: List[Tuple[str, str]] = []

    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] Annotating {image_path.name} ...")

        json_path, raw_path = build_output_paths(output_dir, image_path)

        if json_path.exists() and not args.overwrite:
            print(f"  Skipped (exists): {json_path.name}")
            success += 1
            continue

        try:
            img = open_image_rgb(
                image_path,
                max_side=args.max_side,
                min_side=min_side,
                selected_bands=data_cfg.get("selected_bands"),
                reflectance_scale=float(data_cfg.get("reflectance_scale", 255.0)),
                image_mode=str(data_cfg.get("image_mode", "rgb")),
            )
            orig_w, orig_h = img.info.get("orig_size", img.size)
            resized_w, resized_h = img.size
            site_min, site_max = _site_depth_range(config, image_path)
            site = _infer_site_from_image_path(image_path)
            prompt = (
                FULL_PROMPT
                + f"\n\nImage size: width={int(orig_w)} pixels, height={int(orig_h)} pixels."
                + f"\nSite: {site}"
                + f"\nValid depth range in meters: [{site_min}, {site_max}]"
                + "\nReminder: return problem_regions only for visually unreliable water regions; [] is valid for clean images."
            )

            data, raw_text = run_openai_annotation(
                client=client,
                image=img,
                prompt=prompt,
                gen_cfg=gen_cfg,
                max_retry=2,
                site=site,
                depth_range=(site_min, site_max),
                allow_legacy_schema=allow_legacy,
            )

            data = scale_annotation_to_original(
                data,
                resized_wh=(int(resized_w), int(resized_h)),
                orig_wh=(int(orig_w), int(orig_h)),
            )
            data = normalize_annotation_json(
                data=data,
                image_path=image_path,
                width=int(orig_w),
                height=int(orig_h),
                model_name=model_name,
                config=config,
                allow_legacy_schema=allow_legacy,
            )

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if args.save_raw:
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(raw_text)

            success += 1

        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            if e.__cause__ is not None:
                detail += f" | cause: {type(e.__cause__).__name__}: {e.__cause__}"
            failed.append((image_path.name, detail))
            print(f"  FAILED: {detail}")

    summary = {
        "config": args.config,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "depth_filter_disabled": bool(args.no_depth_filter),
        "depth_filter_stats": depth_filter_stats,
        "num_images": len(images),
        "num_success": success,
        "num_failed": len(failed),
        "failed": [{"image": name, "error": err} for name, err in failed],
    }

    with open(output_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Done. Success: {success} / {len(images)}")
    if failed:
        print("Failures:")
        for name, err in failed:
            print(f"- {name}: {err}")


if __name__ == "__main__":
    main()