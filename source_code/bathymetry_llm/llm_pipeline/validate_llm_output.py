from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ALLOWED_DISTURBANCE_TYPES = {
    "sun_glint",
    "shadow",
    "turbidity",
    "foam",
    "ambiguous_bottom",
    "other",
}

ALLOWED_FAILURE_DIRECTIONS = {
    "artificially_shallow",
    "artificially_deep",
    "ambiguous",
}

REQUIRED_KEYS = {
    "scene_id",
    "image_assessment",
    "disturbance_regions",
    "depth_prior_regions",
    "global_depth_range",
    "warnings",
}


def load_llm_output(value: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    path = Path(value)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _polygon_errors(name: str, polygon: Any, height: int | None, width: int | None) -> List[str]:
    errors: List[str] = []
    if not isinstance(polygon, list) or len(polygon) < 3:
        return [f"{name}.polygon must contain at least 3 points"]
    for idx, point in enumerate(polygon):
        if not isinstance(point, list) or len(point) != 2:
            errors.append(f"{name}.polygon[{idx}] must be [x, y]")
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            errors.append(f"{name}.polygon[{idx}] coordinates must be numeric")
            continue
        if width is not None and not (0 <= x < width):
            errors.append(f"{name}.polygon[{idx}].x out of bounds")
        if height is not None and not (0 <= y < height):
            errors.append(f"{name}.polygon[{idx}].y out of bounds")
    return errors


def _unique_ids(regions: Iterable[Dict[str, Any]], label: str) -> List[str]:
    errors: List[str] = []
    seen = set()
    for idx, region in enumerate(regions):
        rid = str(region.get("region_id", "")).strip()
        if not rid:
            errors.append(f"{label}[{idx}].region_id is missing")
            continue
        if rid in seen:
            errors.append(f"Duplicate region_id: {rid}")
        seen.add(rid)
    return errors


def validate_llm_output(
    value: str | Path | Dict[str, Any],
    height: int | None = None,
    width: int | None = None,
    raise_on_error: bool = True,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    try:
        data = load_llm_output(value)
    except Exception as exc:
        errors.append(f"Invalid JSON: {exc}")
        if raise_on_error:
            raise ValueError("; ".join(errors))
        return False, errors

    missing = sorted(REQUIRED_KEYS - set(data.keys()))
    if missing:
        errors.append(f"Missing required keys: {missing}")

    disturbance_regions = data.get("disturbance_regions", [])
    depth_prior_regions = data.get("depth_prior_regions", [])
    if not isinstance(disturbance_regions, list):
        errors.append("disturbance_regions must be a list")
        disturbance_regions = []
    if not isinstance(depth_prior_regions, list):
        errors.append("depth_prior_regions must be a list")
        depth_prior_regions = []
    if len(depth_prior_regions) == 0:
        errors.append("At least one depth_prior_region is required")

    errors.extend(_unique_ids(disturbance_regions, "disturbance_regions"))
    errors.extend(_unique_ids(depth_prior_regions, "depth_prior_regions"))

    for idx, region in enumerate(disturbance_regions):
        name = f"disturbance_regions[{idx}]"
        if not isinstance(region, dict):
            errors.append(f"{name} must be an object")
            continue
        rtype = region.get("type")
        if rtype not in ALLOWED_DISTURBANCE_TYPES:
            errors.append(f"{name}.type must be one of {sorted(ALLOWED_DISTURBANCE_TYPES)}")
        severity = region.get("severity")
        try:
            sev = float(severity)
            if not (0.0 <= sev <= 1.0):
                errors.append(f"{name}.severity must be in [0, 1]")
        except Exception:
            errors.append(f"{name}.severity must be numeric")
        fdir = region.get("failure_direction")
        if fdir is not None and fdir not in ALLOWED_FAILURE_DIRECTIONS:
            errors.append(
                f"{name}.failure_direction must be one of {sorted(ALLOWED_FAILURE_DIRECTIONS)}"
            )
        errors.extend(_polygon_errors(name, region.get("polygon"), height, width))

    for idx, region in enumerate(depth_prior_regions):
        name = f"depth_prior_regions[{idx}]"
        if not isinstance(region, dict):
            errors.append(f"{name} must be an object")
            continue
        try:
            dmin = float(region.get("depth_min"))
            dmax = float(region.get("depth_max"))
            if dmin < 0:
                errors.append(f"{name}.depth_min must be >= 0")
            if not dmin < dmax:
                errors.append(f"{name}.depth_min must be < depth_max")
        except Exception:
            errors.append(f"{name}.depth_min/depth_max must be numeric")
        try:
            conf = float(region.get("confidence", 1.0))
            if not (0.0 <= conf <= 1.0):
                errors.append(f"{name}.confidence must be in [0, 1]")
        except Exception:
            errors.append(f"{name}.confidence must be numeric")
        errors.extend(_polygon_errors(name, region.get("polygon"), height, width))

    ok = len(errors) == 0
    if not ok and raise_on_error:
        raise ValueError("; ".join(errors))
    return ok, errors


def fallback_llm_output(scene_id: str, height: int, width: int) -> Dict[str, Any]:
    h1 = int(height) - 1
    w1 = int(width) - 1
    y1 = max(0, int(round(height / 3)))
    y2 = max(y1 + 1, int(round(2 * height / 3)))
    return {
        "scene_id": scene_id,
        "image_assessment": {"overall_condition": "fallback conservative prior", "confidence": 0.0},
        "disturbance_regions": [],
        "depth_prior_regions": [
            {"region_id": "prior_nearshore", "region_name": "nearshore", "polygon": [[0, 0], [w1, 0], [w1, y1], [0, y1]], "depth_min": 0.0, "depth_max": 1.5, "rationale": "fallback horizontal zone", "confidence": 0.5},
            {"region_id": "prior_transition", "region_name": "transition", "polygon": [[0, y1], [w1, y1], [w1, y2], [0, y2]], "depth_min": 1.5, "depth_max": 3.0, "rationale": "fallback horizontal zone", "confidence": 0.5},
            {"region_id": "prior_offshore", "region_name": "offshore", "polygon": [[0, y2], [w1, y2], [w1, h1], [0, h1]], "depth_min": 3.0, "depth_max": 6.0, "rationale": "fallback horizontal zone", "confidence": 0.5},
        ],
        "global_depth_range": {"depth_min": 0.0, "depth_max": 6.0, "unit": "meters"},
        "warnings": ["Fallback prior generated because no valid LLM prior was available."],
    }
