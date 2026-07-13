#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from common import load_yaml_config

S2_BAND_TO_INDEX = {
    "B1": 0,
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B5": 4,
    "B6": 5,
    "B7": 6,
    "B8": 7,
    "B8A": 8,
    "B9": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}


def resolve_selected_bands(selected_bands, image_mode: str = "rgb"):
    im = str(image_mode).strip().lower()
    if im == "rgb":
        if selected_bands is None:
            return [0, 1, 2]
        if isinstance(selected_bands, str):
            key = selected_bands.strip().lower()
            if key in ("all", "rgb"):
                return [0, 1, 2]
            raise ValueError(f"Unknown selected_bands preset for image_mode rgb: {selected_bands}")
        out = []
        for b in selected_bands:
            if isinstance(b, int):
                out.append(int(b))
            else:
                raise ValueError(f"RGB selected_bands must be ints, got {b!r}")
        return out
    if selected_bands is None:
        return None
    if isinstance(selected_bands, str):
        key = selected_bands.strip().upper()
        if key == "ALL":
            return None
        raise ValueError(f"Unknown selected_bands preset for image_mode s2: {selected_bands}")
    out = []
    for b in selected_bands:
        if isinstance(b, int):
            out.append(int(b))
        else:
            name = str(b).strip().upper()
            if name not in S2_BAND_TO_INDEX:
                raise ValueError(f"Unknown S2 band name: {b!r}")
            out.append(int(S2_BAND_TO_INDEX[name]))
    return out


def _read_tif_as_chw(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)
    return arr


def _normalize_rgb01(chw: np.ndarray, reflectance_scale: float) -> np.ndarray:
    x = chw.astype(np.float32)
    if reflectance_scale <= 0:
        reflectance_scale = 255.0
    x = x / float(reflectance_scale)
    x = np.clip(x, 0.0, 1.0)
    return x


def _rgb_for_display(image_path: Path, image_mode: str, selected_bands, reflectance_scale: float) -> np.ndarray:
    cube = _read_tif_as_chw(image_path)
    mode = str(image_mode).lower().strip()
    if mode == "rgb":
        if cube.shape[0] < 3:
            raise ValueError(f"Expected >=3 bands for rgb display, got {cube.shape} for {image_path}")
        idx = resolve_selected_bands(selected_bands, image_mode="rgb") or [0, 1, 2]
        idx = idx[:3]
    else:
        resolved = resolve_selected_bands(selected_bands, image_mode="s2")
        if resolved is None:
            idx = [S2_BAND_TO_INDEX["B8"], S2_BAND_TO_INDEX["B3"], S2_BAND_TO_INDEX["B2"]]
        else:
            if len(resolved) < 3:
                raise ValueError("selected_bands must have >=3 for visualization")
            idx = resolved[:3]
    rgb = _normalize_rgb01(cube[idx, :, :], reflectance_scale=reflectance_scale)
    rgb = np.transpose(rgb, (1, 2, 0))
    return rgb


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    return arr.astype(np.float32)


def _maybe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _panel(ax, img, title: str, cmap=None, vmin=None, vmax=None):
    if cmap is None:
        ax.imshow(img)
    else:
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--sample_id", type=str, required=True)
    p.add_argument("--image_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="llm_viz")
    args = p.parse_args()

    cfg = load_yaml_config(args.config)
    data_cfg = cfg.get("data", {}) or {}
    sem_cfg = cfg.get("semantic", {}) or {}
    vlm_cfg = cfg.get("vlm", {}) or {}

    image_dir = Path(str(data_cfg.get("image_dir", "")))
    semantic_dir = Path(str(sem_cfg.get("semantic_dir", "")))
    anno_dir = Path(str(vlm_cfg.get("annotation_json_dir", "")))

    sample_id = str(args.sample_id).strip()
    if args.image_path:
        image_path = Path(args.image_path)
    else:
        pattern = data_cfg.get("image_suffix", "*.tif")
        cand = image_dir / f"{sample_id}.tif"
        if cand.exists():
            image_path = cand
        else:
            hits = sorted(image_dir.glob(pattern))
            match = [h for h in hits if h.stem == sample_id]
            if not match:
                raise FileNotFoundError(f"Cannot resolve image for sample_id={sample_id} in {image_dir}")
            image_path = match[0]

    semantic_path = semantic_dir / f"{sample_id}{sem_cfg.get('reliability_suffix', '_M.npy')}"
    prior_path = semantic_dir / f"{sample_id}{sem_cfg.get('depth_prior_suffix', '_prior.npy')}"
    prior_valid_path = semantic_dir / f"{sample_id}{sem_cfg.get('depth_prior_valid_suffix', '_prior_valid.npy')}"
    prior_conf_path = semantic_dir / f"{sample_id}{sem_cfg.get('depth_prior_conf_suffix', '_prior_conf.npy')}"
    water_path = semantic_dir / f"{sample_id}{sem_cfg.get('water_suffix', '_water.npy')}"
    anno_path = anno_dir / f"{sample_id}.json"

    rgb = _rgb_for_display(
        image_path=image_path,
        image_mode=str(data_cfg.get("image_mode", "rgb")),
        selected_bands=data_cfg.get("selected_bands", None),
        reflectance_scale=float(data_cfg.get("reflectance_scale", 255.0)),
    )

    semantic = _load_npy(semantic_path) if semantic_path.exists() else None
    prior = _load_npy(prior_path) if prior_path.exists() else None
    prior_valid = _load_npy(prior_valid_path) if prior_valid_path.exists() else None
    prior_conf = _load_npy(prior_conf_path) if prior_conf_path.exists() else None
    water = _load_npy(water_path) if water_path.exists() else None
    anno = _maybe_read_json(anno_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}_llm.png"

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    _panel(axes[0], rgb, f"RGB\n{sample_id}")

    if water is not None:
        wm = water[0] if water.ndim == 3 else water
        _panel(axes[1], wm, "Water mask", cmap="Blues", vmin=0.0, vmax=1.0)
    else:
        axes[1].text(0.5, 0.5, "missing *_water.npy", ha="center", va="center")
        axes[1].axis("off")

    if semantic is not None and semantic.shape[0] >= 1:
        issue = semantic[0] if semantic.ndim == 3 else semantic
        _panel(axes[2], issue, "problem region mask M", cmap="inferno", vmin=0.0, vmax=1.0)
    else:
        axes[2].text(0.5, 0.5, "missing *_M.npy", ha="center", va="center")
        axes[2].axis("off")

    if prior is not None:
        pm = prior[0] if prior.ndim == 3 else prior
        if prior_valid is not None:
            vm = prior_valid[0] if prior_valid.ndim == 3 else prior_valid
            pm = np.where(vm > 0.5, pm, np.nan)
        _panel(axes[3], pm, "local d_phys map", cmap="viridis")
    else:
        axes[3].text(0.5, 0.5, "no *_prior.npy", ha="center", va="center")
        axes[3].axis("off")

    if anno is not None and isinstance(anno, dict):
        meta = anno.get("_meta", {})
        local_prior_coverage = None
        mean_prior_conf = None
        if prior_valid is not None:
            vm = prior_valid[0] if prior_valid.ndim == 3 else prior_valid
            local_prior_coverage = float((vm > 0.5).mean())
            if prior_conf is not None and (vm > 0.5).any():
                cm = prior_conf[0] if prior_conf.ndim == 3 else prior_conf
                mean_prior_conf = float(cm[vm > 0.5].mean())
        s = json.dumps(
            {
                "model_name": meta.get("model_name", None),
                "width": meta.get("width", None),
                "height": meta.get("height", None),
                "num_water_polygons": meta.get("num_water_polygons", None),
                "num_problem_regions": meta.get("num_problem_regions", None),
                "local_prior_only": meta.get("local_prior_only", None),
                "local_prior_coverage": local_prior_coverage,
                "mean_prior_conf": mean_prior_conf,
            },
            ensure_ascii=False,
            indent=2,
        )
        axes[4].text(0.0, 1.0, s, ha="left", va="top", family="monospace", fontsize=9)
        axes[4].set_title("annotation meta")
        axes[4].axis("off")
    else:
        axes[4].text(0.5, 0.5, "missing annotation json", ha="center", va="center")
        axes[4].axis("off")

    if anno is not None and isinstance(anno, dict):
        regions = anno.get("problem_regions", anno.get("disturbance_regions", anno.get("uncertainty_regions", [])))
        nreg = len(regions) if isinstance(regions, list) else 0
        lines = [f"problem_regions: {nreg}"]
        if isinstance(regions, list):
            for j, r in enumerate(regions[:6]):
                if isinstance(r, dict):
                    lines.append(
                        f"  prob[{j}] {r.get('category', r.get('issue_type', '?'))} "
                        f"sev={r.get('severity', r.get('risk_level', '?'))} "
                        f"depth=[{r.get('depth_min', '?')}, {r.get('depth_max', '?')}] "
                        f"conf={r.get('depth_confidence', r.get('confidence', '?'))}"
                    )
            if nreg > 6:
                lines.append(f"  ... +{nreg - 6} more")
        axes[5].text(0.0, 1.0, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=9)
        axes[5].set_title("annotation regions (from JSON)")
    else:
        axes[5].text(0.5, 0.5, "no JSON", ha="center", va="center")
        axes[5].set_title("uncertainty list")
    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
