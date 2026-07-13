#!/usr/bin/env python3

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class GenerationConfig:
    max_new_tokens: int = 2048
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None


ALLOWED_DISTURBANCE_CATEGORIES = {
    "sun_glint",
    "shadow",
    "turbidity",
    "foam",
    "ambiguous_bottom",
    "other",
}

LEGACY_ISSUE_TO_CATEGORY = {
    "wave_roughness": "other",
    "bottom_confusion": "ambiguous_bottom",
    "color_ambiguity": "ambiguous_bottom",
    "sensor_artifact": "other",
}

LEGACY_REGION_KEYS = ("uncertainty_regions", "disturbance_regions", "depth_regions")

DEPTH_COVERAGE_RULES = """
Local prior checklist before answering:
1. water.polygons covers all visible open water, excludes land, and follows the shoreline geometry.
2. problem_regions contains only visually unreliable water regions; [] is valid for clean images.
3. Do not partition the whole water area into depth zones.
4. Each problem region has a conservative local depth interval and confidence.
5. Do not use large demo boxes, squares, rectangles, or straight bands for irregular regions.
6. Use free-form polygons with enough points to trace the actual problematic shape.
7. All coordinates must stay inside [0, width-1] and [0, height-1].
"""


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _image_url_data(img: Image.Image) -> str:
    raw = _pil_to_png_bytes(img.convert("RGB"))
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def extract_json(text: str) -> Dict[str, Any]:
    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    def _balance_square_brackets(json_candidate: str) -> str:
        s = json_candidate
        opens = s.count("[")
        closes = s.count("]")
        if opens <= closes:
            return s
        missing = opens - closes
        last_brace = s.rfind("}")
        if last_brace == -1:
            return s + ("]" * missing)
        return s[:last_brace] + ("]" * missing) + s[last_brace:]

    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        cand = _strip_code_fences(match.group(0))
        try:
            return json.loads(cand)
        except Exception:
            cand2 = _balance_square_brackets(cand)
            return json.loads(cand2)
    raise ValueError("No valid JSON found in output.")


def _polygon_count(polygons: Any) -> int:
    if not isinstance(polygons, list):
        return 0
    count = 0
    for poly in polygons:
        if isinstance(poly, list) and len(poly) >= 3:
            count += 1
    return count


def _region_has_geometry(region: Dict[str, Any]) -> bool:
    if _polygon_count(region.get("polygons", [])) > 0:
        return True
    bbox = region.get("bbox", region.get("box", []))
    return isinstance(bbox, list) and len(bbox) == 4


def _legacy_problem_regions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("disturbance_regions", "uncertainty_regions"):
        regions = data.get(key, [])
        if isinstance(regions, list) and regions:
            return [r for r in regions if isinstance(r, dict)]
    return []


def validate_annotation_schema(
    data: Dict[str, Any],
    depth_range: Tuple[float, float] = (0.0, 30.29),
    *,
    strict_local_prior: bool = True,
    allow_legacy_schema: bool = False,
) -> Tuple[bool, str]:
    site_min, site_max = float(depth_range[0]), float(depth_range[1])
    try:
        if not isinstance(data, dict):
            return False, "Annotation must be a JSON object."

        if strict_local_prior and not allow_legacy_schema:
            for key in LEGACY_REGION_KEYS:
                if key in data:
                    return False, f"contains legacy field {key}"

        if "water" not in data:
            return False, "Missing required field: water."

        water = data.get("water")
        if not isinstance(water, dict):
            return False, "water must be an object."
        if _polygon_count(water.get("polygons", [])) == 0:
            return False, "water.polygons must contain at least one polygon with >=3 points."

        if "problem_regions" not in data:
            if allow_legacy_schema and _legacy_problem_regions(data):
                return False, (
                    "Missing required field: problem_regions. "
                    "Rerun LLM annotation with the disturbance-localized prior schema."
                )
            return False, (
                "Missing required field: problem_regions. "
                "Rerun LLM annotation with the disturbance-localized prior schema."
            )

        regions = data["problem_regions"]
        if not isinstance(regions, list):
            return False, "problem_regions must be a list."

        required_keys = (
            "category",
            "severity",
            "description",
            "depth_min",
            "depth_max",
            "depth_confidence",
            "rationale",
        )
        for idx, region in enumerate(regions):
            if not isinstance(region, dict):
                return False, f"problem_regions[{idx}] must be an object."
            for key in required_keys:
                if key not in region:
                    return False, f"problem_regions[{idx}] missing required field: {key}"
            if not _region_has_geometry(region):
                return False, f"problem_regions[{idx}] must contain polygons or bbox."

            category = str(region.get("category", "")).strip().lower()
            category = LEGACY_ISSUE_TO_CATEGORY.get(category, category)
            if category not in ALLOWED_DISTURBANCE_CATEGORIES:
                return False, f"problem_regions[{idx}] has invalid category: {category}"

            d_min = float(region["depth_min"])
            d_max = float(region["depth_max"])
            conf = float(region["depth_confidence"])
            if d_min > d_max:
                return False, f"problem_regions[{idx}] has depth_min > depth_max."
            if not (site_min <= d_min <= d_max <= site_max):
                return False, (
                    f"problem_regions[{idx}] interval [{d_min}, {d_max}] outside [{site_min}, {site_max}]"
                )
            if not (0.0 <= conf <= 1.0):
                return False, f"problem_regions[{idx}] depth_confidence must be in [0, 1]."

        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def repair_json_with_openai(
    client: "OpenAIVisionAnnotator",
    image: Image.Image,
    raw_text: str,
    gen_cfg: GenerationConfig,
    site: str = "unknown",
    depth_range: Tuple[float, float] = (0.0, 30.29),
) -> Dict[str, Any]:
    site_min, site_max = float(depth_range[0]), float(depth_range[1])
    repair_prompt = f"""The previous model output is invalid JSON or fails schema checks.
Return ONLY valid JSON (no markdown fences).

Site: {site}
Valid depth range in meters: [{site_min}, {site_max}]

Required schema:
{{
  "water": {{ "polygons": [[[x, y], [x, y], [x, y]]] }},
  "problem_regions": [
    {{
      "id": "prob_0",
      "category": "sun_glint",
      "polygons": [[[x, y], [x, y], [x, y]]],
      "severity": 0.0,
      "description": "short explanation of why this region is visually unreliable",
      "depth_min": 0.0,
      "depth_max": 2.0,
      "depth_confidence": 0.6,
      "rationale": "physical rationale for this local conservative interval"
    }}
  ],
  "scene_summary": "short auditing summary"
}}

Rules:
- problem_regions may be [] if no unreliable water region is visible.
- Do not partition the whole water area.
- Each problem region must include depth_min, depth_max, depth_confidence, description, and rationale.
- Water and problem polygons must follow the actual shoreline/problem geometry.
- Do not use squares, rectangles, bounding boxes, or straight bands unless the real visible boundary has that shape.
- Use free-form polygons with enough vertices to redraw irregular coastal shapes.
- category must be one of: {sorted(ALLOWED_DISTURBANCE_CATEGORIES)}
- depth intervals must stay inside [{site_min}, {site_max}].

{DEPTH_COVERAGE_RULES.strip()}

Broken output:
{raw_text}
"""
    repaired = client.generate(image, repair_prompt, gen_cfg)
    return extract_json(repaired)


class OpenAIVisionAnnotator:
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key or not str(key).strip():
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install openai package: pip install openai") from e
        self._client = OpenAI(
            api_key=key,
            timeout=120.0,
            max_retries=2,
        )
        self._model = str(model)

    def generate(self, image: Image.Image, prompt: str, gen_cfg: GenerationConfig) -> str:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": _image_url_data(image)},
            },
        ]
        req: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(gen_cfg.max_new_tokens),
        }
        if gen_cfg.do_sample and gen_cfg.temperature is not None:
            req["temperature"] = float(gen_cfg.temperature)
        if gen_cfg.do_sample and gen_cfg.top_p is not None:
            req["top_p"] = float(gen_cfg.top_p)
        if not gen_cfg.do_sample:
            req["temperature"] = 0.0
        resp = self._client.chat.completions.create(**req)
        choice = resp.choices[0]
        return (choice.message.content or "").strip()


def run_openai_annotation(
    client: OpenAIVisionAnnotator,
    image: Image.Image,
    prompt: str,
    gen_cfg: GenerationConfig,
    max_retry: int = 2,
    site: str = "unknown",
    depth_range: Tuple[float, float] = (0.0, 30.29),
    allow_legacy_schema: bool = False,
) -> Tuple[Dict[str, Any], str]:
    last_text: Optional[str] = None
    last_error: Optional[str] = None
    for attempt in range(max_retry + 1):
        text = client.generate(image, prompt, gen_cfg)
        last_text = text
        try:
            data = extract_json(text)
            ok, msg = validate_annotation_schema(
                data,
                depth_range=depth_range,
                allow_legacy_schema=allow_legacy_schema,
            )
            if ok:
                return data, text
            last_error = msg
            raise ValueError(msg)
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retry:
                try:
                    data = repair_json_with_openai(
                        client,
                        image,
                        text,
                        gen_cfg,
                        site=site,
                        depth_range=depth_range,
                    )
                    ok, msg = validate_annotation_schema(
                        data,
                        depth_range=depth_range,
                        allow_legacy_schema=allow_legacy_schema,
                    )
                    if ok:
                        return data, text
                    last_error = msg
                except Exception as repair_exc:
                    last_error = str(repair_exc)
                    continue
    raise RuntimeError(
        f"Failed after retries. Last error: {last_error}\nLast output:\n{last_text}"
    )
