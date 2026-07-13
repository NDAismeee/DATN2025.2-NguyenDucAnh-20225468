#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import rasterio

from common import load_yaml_config
from llm_annotate import _site_depth_range, normalize_annotation_json
from vlm_utils import validate_annotation_schema


def read_hand_file(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty annotation file: {path}")
    return json.loads(text)


def resolve_image_path(sample_id: str, config: Dict) -> Path:
    data_cfg = config.get("data", {})
    image_dir = Path(str(data_cfg.get("image_dir", "")))
    candidates = [
        image_dir / f"{sample_id}.tif",
        image_dir / f"{sample_id}.tiff",
    ]
    if not candidates[0].exists():
        hits = sorted(image_dir.glob(str(data_cfg.get("image_suffix", "img_*.tif"))))
        match = [h for h in hits if h.stem == sample_id]
        if match:
            return match[0]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Image not found for {sample_id} in {image_dir}")


def image_size(path: Path) -> Tuple[int, int]:
    with rasterio.open(path) as ds:
        return int(ds.width), int(ds.height)


def convert_one(
    hand_path: Path,
    config: Dict,
    annotations_dir: Path,
    hand_json_dir: Path | None,
) -> Path:
    sample_id = hand_path.stem
    raw = read_hand_file(hand_path)
    image_path = resolve_image_path(sample_id, config)
    width, height = image_size(image_path)
    site_min, site_max = _site_depth_range(config, image_path)

    normalized = normalize_annotation_json(
        data=raw,
        image_path=image_path,
        width=width,
        height=height,
        model_name="hand_annotation",
        config=config,
        allow_legacy_schema=False,
    )
    normalized["_meta"]["source"] = "hand_annotation"
    normalized["_meta"]["source_file"] = str(hand_path)

    ok, msg = validate_annotation_schema(normalized, depth_range=(float(site_min), float(site_max)))
    if not ok:
        raise ValueError(f"{sample_id}: schema validation failed: {msg}")

    out_path = annotations_dir / f"{sample_id}.json"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    if hand_json_dir is not None:
        hand_json_dir.mkdir(parents=True, exist_ok=True)
        hand_out = hand_json_dir / f"{sample_id}.json"
        with open(hand_out, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--hand_dir",
        type=str,
        default="annotation_hand",
        help="Folder containing hand annotation .txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Pipeline annotation folder (default: vlm.annotation_json_dir / annotations).",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    hand_dir = Path(args.hand_dir)
    if not hand_dir.exists():
        raise FileNotFoundError(f"Hand annotation folder not found: {hand_dir}")

    output_dir = Path(
        args.output_dir or config.get("vlm", {}).get("annotation_json_dir", "annotations")
    )
    hand_json_dir = hand_dir

    files = sorted(hand_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {hand_dir}")

    ok_paths: List[Path] = []
    failed: List[Tuple[str, str]] = []
    for path in files:
        try:
            out = convert_one(path, config, output_dir, hand_json_dir)
            ok_paths.append(out)
            print(f"OK {path.name} -> {out}")
        except Exception as exc:
            failed.append((path.name, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {path.name}: {failed[-1][1]}")

    summary = {
        "num_input": len(files),
        "num_ok": len(ok_paths),
        "num_failed": len(failed),
        "failed": [{"file": n, "error": e} for n, e in failed],
    }
    with open(output_dir / "_hand_import_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if failed:
        raise SystemExit(f"Conversion failed for {len(failed)} file(s)")
    print(f"Done. Wrote {len(ok_paths)} annotation JSON files to {output_dir}")


if __name__ == "__main__":
    main()
