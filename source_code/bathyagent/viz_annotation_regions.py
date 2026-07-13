#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio

from common import load_yaml_config
from rasterize_semantics import polygon_to_mask, region_polygons


DISTURBANCE_COLORS = {
    "sun_glint": (1.0, 0.92, 0.20),
    "shadow": (0.35, 0.35, 0.35),
    "turbidity": (0.72, 0.45, 0.18),
    "foam": (0.95, 0.95, 0.95),
    "ambiguous_bottom": (0.85, 0.35, 0.85),
    "other": (1.0, 0.45, 0.10),
}


def resolve_selected_bands(selected_bands, image_mode: str = "rgb") -> List[int]:
    if str(image_mode).lower() == "rgb":
        if selected_bands is None:
            return [0, 1, 2]
        if isinstance(selected_bands, str) and selected_bands.strip().lower() in {"all", "rgb"}:
            return [0, 1, 2]
        return [int(x) for x in selected_bands][:3]
    raise ValueError("Only image_mode rgb is supported for this visualizer.")


def load_rgb_image(path: Path, selected_bands, reflectance_scale: float) -> np.ndarray:
    with rasterio.open(path) as ds:
        cube = ds.read().astype(np.float32)
    idx = resolve_selected_bands(selected_bands, image_mode="rgb")
    rgb = cube[idx, :, :]
    rgb = np.clip(rgb / float(reflectance_scale), 0.0, 1.0)
    return np.transpose(rgb, (1, 2, 0))


def load_annotation(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def problem_regions(anno: Dict[str, Any]) -> List[Dict[str, Any]]:
    regions = anno.get("problem_regions", [])
    if isinstance(regions, list) and regions:
        return [r for r in regions if isinstance(r, dict)]
    legacy = anno.get("disturbance_regions", anno.get("uncertainty_regions", []))
    if isinstance(legacy, list):
        return [r for r in legacy if isinstance(r, dict)]
    return []


def _put_label(img: np.ndarray, text: str, cx: int, cy: int, color: Tuple[float, float, float]) -> None:
    x = max(4, min(cx - 80, img.shape[1] - 180))
    y = max(18, min(cy, img.shape[0] - 8))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0, 1.0, 1.0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def overlay_problem_regions(
    rgb: np.ndarray,
    anno: Dict[str, Any],
) -> Tuple[np.ndarray, List[mpatches.Patch], int]:
    height, width = rgb.shape[:2]
    out = rgb.copy()
    legend: List[mpatches.Patch] = []
    regions = problem_regions(anno)
    if not regions:
        return out, legend, 0

    drawn = 0
    for idx, region in enumerate(regions):
        category = str(region.get("category", region.get("issue_type", "other"))).strip().lower()
        color = DISTURBANCE_COLORS.get(category, DISTURBANCE_COLORS["other"])
        mask = polygon_to_mask(region_polygons(region), height, width)
        if mask.sum() == 0:
            continue

        alpha = 0.48
        active = mask > 0.5
        for c in range(3):
            out[:, :, c] = np.where(
                active,
                (1.0 - alpha) * out[:, :, c] + alpha * color[c],
                out[:, :, c],
            )

        contours, _ = cv2.findContours(
            active.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, color, 2)

        d_min = region.get("depth_min")
        d_max = region.get("depth_max")
        d_conf = region.get("depth_confidence", region.get("confidence"))
        severity = region.get("severity", "?")

        if d_min is not None and d_max is not None:
            depth_text = f"{float(d_min):.1f}-{float(d_max):.1f}m"
            if d_conf is not None:
                depth_text += f" conf={float(d_conf):.2f}"
        else:
            depth_text = "depth n/a"

        ys, xs = np.where(active)
        if xs.size > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label = f"{category}: {depth_text}"
            _put_label(out, label, cx, cy, color)

        legend.append(
            mpatches.Patch(
                color=color,
                label=(
                    f"{region.get('id', f'prob_{idx}')} | {category} | sev={severity} | "
                    f"{depth_text}"
                ),
            )
        )
        drawn += 1

    return out, legend, drawn


def resolve_paths(
    sample_id: str,
    config: Dict[str, Any],
    image_path: Optional[str],
    annotation_path: Optional[str],
) -> Tuple[Path, Path]:
    data_cfg = config.get("data", {})
    vlm_cfg = config.get("vlm", {})
    image_dir = Path(str(data_cfg.get("image_dir", "")))
    anno_dir = Path(str(vlm_cfg.get("annotation_json_dir", "")))

    if image_path:
        img_path = Path(image_path)
    else:
        img_path = image_dir / f"{sample_id}.tif"
        if not img_path.exists():
            hits = sorted(image_dir.glob(str(data_cfg.get("image_suffix", "img_*.tif"))))
            match = [h for h in hits if h.stem == sample_id]
            if not match:
                raise FileNotFoundError(f"Image not found for sample_id={sample_id}")
            img_path = match[0]

    if annotation_path:
        anno_path = Path(annotation_path)
    else:
        anno_path = anno_dir / f"{sample_id}.json"

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")
    if not anno_path.exists():
        raise FileNotFoundError(f"Missing annotation: {anno_path}")
    return img_path, anno_path


def save_dual_panel(
    sample_id: str,
    rgb: np.ndarray,
    problem_img: np.ndarray,
    legend: List[mpatches.Patch],
    num_regions: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(np.clip(rgb, 0.0, 1.0))
    axes[0].set_title(f"Original RGB\n{sample_id}")
    axes[0].axis("off")

    axes[1].imshow(np.clip(problem_img, 0.0, 1.0))
    if num_regions == 0:
        axes[1].set_title("LLM problem regions\n(no unreliable regions detected)")
    else:
        axes[1].set_title(f"LLM problem regions + depth intervals\n{num_regions} region(s)")
    axes[1].axis("off")
    if legend:
        axes[1].legend(handles=legend, loc="lower left", fontsize=7, framealpha=0.9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def visualize_sample(
    sample_id: str,
    config: Dict[str, Any],
    image_path: Optional[str] = None,
    annotation_path: Optional[str] = None,
    out_dir: Path = Path("annotation_viz"),
) -> Path:
    img_path, anno_path = resolve_paths(sample_id, config, image_path, annotation_path)
    anno = load_annotation(anno_path)
    data_cfg = config.get("data", {})
    rgb = load_rgb_image(
        img_path,
        selected_bands=data_cfg.get("selected_bands"),
        reflectance_scale=float(data_cfg.get("reflectance_scale", 255.0)),
    )
    problem_img, legend, num_regions = overlay_problem_regions(rgb, anno)
    out_path = out_dir / f"{sample_id}_annotation_regions.png"
    save_dual_panel(
        sample_id=sample_id,
        rgb=rgb,
        problem_img=problem_img,
        legend=legend,
        num_regions=num_regions,
        out_path=out_path,
    )
    print(f"Saved: {out_path}")
    print(f"problem_regions drawn: {num_regions}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--annotation_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="annotation_viz")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    out_dir = Path(args.out_dir)

    if args.all:
        anno_dir = Path(str(config.get("vlm", {}).get("annotation_json_dir", "")))
        files = sorted(p for p in anno_dir.glob("*.json") if not p.stem.startswith("_"))
        for path in files:
            visualize_sample(
                sample_id=path.stem,
                config=config,
                out_dir=out_dir,
            )
        return

    if not args.sample_id:
        parser.error("--sample_id is required unless --all is set")

    visualize_sample(
        sample_id=args.sample_id,
        config=config,
        image_path=args.image_path,
        annotation_path=args.annotation_path,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
