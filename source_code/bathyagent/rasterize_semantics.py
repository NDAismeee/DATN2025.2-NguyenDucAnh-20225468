#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from common import load_yaml_config


DISTURBANCE_CATEGORIES = {
    "sun_glint",
    "shadow",
    "turbidity",
    "foam",
    "ambiguous_bottom",
    "other",
}


def polygon_to_mask(polygons: List[List[List[float]]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    for poly in polygons or []:
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1.0)
    return mask


def bbox_to_polygon(bbox) -> List[List[int]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return []
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def region_polygons(region: Dict) -> List[List[List[float]]]:
    polygons = region.get("polygons", [])
    if polygons:
        return polygons
    bbox_poly = bbox_to_polygon(region.get("bbox", []))
    return [bbox_poly] if bbox_poly else []


def resolve_raster_hw(json_data: Dict, height: Optional[int], width: Optional[int]) -> Tuple[int, int]:
    if height is not None and width is not None:
        return int(height), int(width)
    meta = json_data.get("_meta", {})
    if isinstance(meta, dict) and meta.get("height") is not None and meta.get("width") is not None:
        return int(meta["height"]), int(meta["width"])
    raise ValueError("Missing raster size. Pass --height/--width or keep _meta.height/_meta.width in JSON.")


def infer_site(json_data: Dict, json_path: Path) -> str:
    meta = json_data.get("_meta", {})
    text = f"{json_path} {meta.get('source_image', '') if isinstance(meta, dict) else ''}".lower()
    if "agia" in text:
        return "agia_napa"
    if "puck" in text:
        return "puck_lagoon"
    return "unknown"


def site_depth_range(config: Dict, site: str) -> Tuple[float, float]:
    ranges = config.get("llm", {}).get("valid_depth_ranges", {})
    value = ranges.get(site, ranges.get("unknown", [0.0, 30.29]))
    return float(value[0]), float(value[1])


def normalize_problem_regions(
    json_data: Dict,
    site_min: float,
    site_max: float,
    allow_legacy_schema: bool = False,
) -> List[Dict]:
    if not allow_legacy_schema:
        for key in ("uncertainty_regions", "disturbance_regions", "depth_regions"):
            if key in json_data:
                raise ValueError(
                    f"Legacy field {key} is not allowed in strict local-prior mode. "
                    "Rerun llm_annotate.py with the new local-prior schema."
                )
        if "problem_regions" not in json_data:
            raise ValueError(
                "Missing problem_regions. Refusing to rasterize legacy annotation. "
                "Rerun llm_annotate.py with the new local-prior schema."
            )
        raw = json_data["problem_regions"]
    else:
        raw = json_data.get("problem_regions", [])
        if not raw:
            raw = json_data.get("disturbance_regions", [])
        if not raw:
            raw = json_data.get("uncertainty_regions", [])

    out: List[Dict] = []
    legacy_map = {
        "wave_roughness": "other",
        "bottom_confusion": "ambiguous_bottom",
        "color_ambiguity": "ambiguous_bottom",
        "sensor_artifact": "other",
    }
    required = ["depth_min", "depth_max", "depth_confidence", "description", "rationale"]
    for idx, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        if not allow_legacy_schema:
            for key in required:
                if key not in item:
                    raise ValueError(f"problem_regions[{idx}] missing required field: {key}")

        category = str(item.get("category", item.get("issue_type", "other"))).strip().lower()
        category = legacy_map.get(category, category)
        if category not in DISTURBANCE_CATEGORIES:
            category = "other"
        polygons = region_polygons(item)
        if not polygons:
            continue
        severity = float(item.get("severity", 1.0))
        if "risk_level" in item:
            severity = {"low": 0.33, "medium": 0.66, "high": 1.0}.get(
                str(item["risk_level"]).lower(),
                severity,
            )
        if allow_legacy_schema:
            d_min = float(item.get("depth_min", item.get("d_min", site_min)))
            d_max = float(item.get("depth_max", item.get("d_max", site_max)))
            d_conf = float(item.get("depth_confidence", item.get("confidence", item.get("gamma", 0.5))))
        else:
            d_min = float(item["depth_min"])
            d_max = float(item["depth_max"])
            d_conf = float(item["depth_confidence"])
        if d_min > d_max:
            raise ValueError(f"problem_regions[{idx}] has depth_min > depth_max.")
        if not (site_min <= d_min <= d_max <= site_max):
            raise ValueError(f"Invalid local prior interval [{d_min}, {d_max}] outside [{site_min}, {site_max}]")
        description = str(item.get("description", item.get("model_hint", ""))).strip()
        out.append(
            {
                "id": str(item.get("id", f"prob_{idx}")),
                "category": category,
                "polygons": polygons,
                "severity": float(np.clip(severity, 0.0, 1.0)),
                "description": description,
                "depth_min": d_min,
                "depth_max": d_max,
                "depth_confidence": float(np.clip(d_conf, 0.0, 1.0)),
                "rationale": str(item.get("rationale", "")).strip(),
            }
        )
    return out


def _assert_local_prior_maps(M: np.ndarray, prior: np.ndarray, prior_valid: np.ndarray, prior_conf: np.ndarray) -> None:
    m = M.reshape(-1)
    pv = prior_valid.reshape(-1)
    pr = prior.reshape(-1)
    pc = prior_conf.reshape(-1)
    if np.max(pv * (1.0 - m)) > 1e-6:
        raise ValueError("prior_valid must be a subset of M.")
    if np.any(pr[pv < 0.5] != 0):
        raise ValueError("prior must be zero outside prior_valid.")
    if np.any(pc[pv < 0.5] != 0):
        raise ValueError("prior_conf must be zero outside prior_valid.")


def rasterize_annotation(
    json_data: Dict,
    height: int,
    width: int,
    site_min: float,
    site_max: float,
    allow_legacy_schema: bool = False,
):
    water = polygon_to_mask(json_data.get("water", {}).get("polygons", []), height, width)
    water = (water > 0.5).astype(np.float32)
    if water.sum() == 0:
        raise ValueError("water mask is empty")

    problem_regions = normalize_problem_regions(
        json_data, site_min, site_max, allow_legacy_schema=allow_legacy_schema
    )
    masks: List[np.ndarray] = []
    midpoints: List[float] = []
    confidences: List[float] = []
    severity = np.zeros((height, width), dtype=np.float32)
    region_texts: List[Dict[str, object]] = []
    for region in problem_regions:
        mask = polygon_to_mask(region["polygons"], height, width) * water
        if mask.sum() == 0:
            continue
        masks.append(mask.astype(np.float32))
        midpoint = 0.5 * (float(region["depth_min"]) + float(region["depth_max"]))
        midpoints.append(midpoint)
        confidences.append(float(region["depth_confidence"]))
        severity = np.maximum(severity, mask * float(region["severity"]))
        text_parts = [
            region.get("category", ""),
            region.get("description", ""),
            f"local conservative depth interval: {region['depth_min']:.2f}-{region['depth_max']:.2f} m",
            f"depth_confidence: {region['depth_confidence']:.2f}",
            region.get("rationale", ""),
        ]
        region_texts.append(
            {
                "id": region["id"],
                "category": region["category"],
                "severity": region["severity"],
                "depth_min": region["depth_min"],
                "depth_max": region["depth_max"],
                "depth_confidence": region["depth_confidence"],
                "text": ". ".join([str(part) for part in text_parts if part]),
                "rationale": region["rationale"],
            }
        )

    if masks:
        R = np.stack(masks, axis=0).astype(np.float32)
        M = (R.sum(axis=0, keepdims=True) > 0).astype(np.float32)
        masks_arr = R
        midpoints_arr = np.asarray(midpoints, dtype=np.float32)
        confidences_arr = np.asarray(confidences, dtype=np.float32)
        weights = masks_arr * confidences_arr[:, None, None]
        denom = weights.sum(axis=0, keepdims=True)
        prior_valid = (denom > 1.0e-6).astype(np.float32) * M
        omega = weights / np.maximum(denom, 1.0e-6)
        prior = (omega * midpoints_arr[:, None, None]).sum(axis=0, keepdims=True).astype(np.float32)
        prior = np.clip(prior, site_min, site_max) * prior_valid
        prior_conf = np.clip(denom, 0.0, 1.0).astype(np.float32) * prior_valid
    else:
        R = np.zeros((1, height, width), dtype=np.float32)
        M = np.zeros((1, height, width), dtype=np.float32)
        prior = np.zeros((1, height, width), dtype=np.float32)
        prior_valid = np.zeros((1, height, width), dtype=np.float32)
        prior_conf = np.zeros((1, height, width), dtype=np.float32)

    _assert_local_prior_maps(M, prior, prior_valid, prior_conf)

    return {
        "M": M,
        "R": R,
        "region_texts": region_texts,
        "prior": prior,
        "prior_valid": prior_valid,
        "prior_conf": prior_conf,
        "problem_regions": problem_regions,
        "water": water[None, :, :].astype(np.float32),
        "severity": severity[None, :, :].astype(np.float32),
    }


def process_file(
    json_path: Path,
    output_dir: Path,
    config: Dict,
    height: Optional[int],
    width: Optional[int],
    allow_legacy_schema: bool = False,
) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raster_h, raster_w = resolve_raster_hw(data, height, width)
    site = infer_site(data, json_path)
    site_min, site_max = site_depth_range(config, site)
    result = rasterize_annotation(
        data, raster_h, raster_w, site_min, site_max, allow_legacy_schema=allow_legacy_schema
    )

    stem = json_path.stem
    np.save(output_dir / f"{stem}_M.npy", result["M"])
    np.save(output_dir / f"{stem}_R.npy", result["R"])
    np.save(output_dir / f"{stem}_prior.npy", result["prior"])
    np.save(output_dir / f"{stem}_prior_valid.npy", result["prior_valid"])
    np.save(output_dir / f"{stem}_prior_conf.npy", result["prior_conf"])
    np.save(output_dir / f"{stem}_water.npy", result["water"])
    np.save(output_dir / f"{stem}_severity.npy", result["severity"])
    with open(output_dir / f"{stem}_region_texts.json", "w", encoding="utf-8") as f:
        json.dump(result["region_texts"], f, ensure_ascii=False, indent=2)
    with open(output_dir / f"{stem}_problem_regions.json", "w", encoding="utf-8") as f:
        json.dump(result["problem_regions"], f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--json_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--allow_legacy_schema",
        action="store_true",
        help="Rasterize legacy disturbance_regions/uncertainty_regions (not recommended).",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    allow_legacy = bool(
        args.allow_legacy_schema or config.get("vlm", {}).get("allow_legacy_schema", False)
    )
    json_dir = Path(args.json_dir or config.get("vlm", {}).get("annotation_json_dir", ""))
    output_dir = Path(args.output_dir or config.get("semantic", {}).get("semantic_dir", ""))
    if not str(json_dir) or not str(output_dir):
        parser.error("Set --json_dir/--output_dir or configure vlm.annotation_json_dir and semantic.semantic_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in json_dir.glob("*.json") if not path.stem.startswith("_"))
    print(f"Found {len(files)} annotation files")
    failed = []
    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {path.name}")
        try:
            process_file(path, output_dir, config, args.height, args.width, allow_legacy_schema=allow_legacy)
        except Exception as exc:
            failed.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED: {failed[-1]['error']}")

    with open(output_dir / "_rasterize_summary.json", "w", encoding="utf-8") as f:
        json.dump({"num_files": len(files), "num_failed": len(failed), "failed": failed}, f, indent=2)
    if failed:
        raise SystemExit(f"Rasterization failed for {len(failed)} file(s)")
    print("Done.")


if __name__ == "__main__":
    main()
