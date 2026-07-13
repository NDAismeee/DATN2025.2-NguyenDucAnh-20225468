#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_yaml_config
from vlm_utils import LEGACY_REGION_KEYS, validate_annotation_schema


def check_annotation_file(path: Path, allow_legacy_schema: bool) -> list[str]:
    errors: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not allow_legacy_schema:
        for key in LEGACY_REGION_KEYS:
            if key in data:
                errors.append(f"FAIL {path.name}: contains legacy field {key}")

    ok, msg = validate_annotation_schema(
        data,
        allow_legacy_schema=allow_legacy_schema,
    )
    if not ok:
        errors.append(f"FAIL {path.name}: {msg}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--allow_legacy_schema",
        action="store_true",
        help="Allow legacy disturbance_regions/uncertainty_regions.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    vlm_cfg = config.get("vlm", {})
    allow_legacy = bool(args.allow_legacy_schema or vlm_cfg.get("allow_legacy_schema", False))
    anno_dir = Path(str(vlm_cfg.get("annotation_json_dir", "")))
    if not anno_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {anno_dir}")

    files = sorted(p for p in anno_dir.glob("*.json") if not p.stem.startswith("_"))
    all_errors: list[str] = []
    for path in files:
        all_errors.extend(check_annotation_file(path, allow_legacy_schema=allow_legacy))

    if all_errors:
        for line in all_errors:
            print(line)
        raise SystemExit(f"Schema check failed for {len(all_errors)} issue(s) across {len(files)} file(s).")

    print(f"PASS: all {len(files)} annotation files use strict disturbance-localized prior schema.")


if __name__ == "__main__":
    main()
