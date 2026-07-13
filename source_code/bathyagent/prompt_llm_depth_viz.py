#!/usr/bin/env python3
"""Optional debug tool: OpenAI text depth grid vs training priors (not used in default training)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from common import load_yaml_config
from llm_annotate import open_image_rgb
from vlm_utils import GenerationConfig, OpenAIVisionAnnotator


DEFAULT_PROMPT = """
You are a coastal bathymetry expert.

Task:
Given one coastal image, output ONLY a depth grid (no JSON, no markdown).

Output format (strict):
- First line: "Gh Gw" (two integers, 16..96). Prefer "32 32".
- Then Gh lines, each has Gw values separated by spaces
- Each value is a POSITIVE depth in meters (float), or "nan" if unknown

Rules:
- Gh and Gw must be between 16 and 96.
- Predict depth ONLY for WATER pixels/cells. For LAND (and anything that is not water), output "nan".
- Depth should be plausible for shallow coastal water: typically 0.0 to 20.0 meters (use larger only if clearly deep).
- Do not output a constant grid. Use spatial variation that matches the image.
- If the whole tile is land or you cannot see water, output all "nan".
""".strip()


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _parse_depth_grid_text(raw_text: str) -> np.ndarray:
    s = _strip_fences(raw_text)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip() != ""]
    if len(lines) < 2:
        raise ValueError("Depth grid output too short")
    header_idx = None
    Gh = Gw = None
    for i, ln in enumerate(lines[:20]):
        parts = ln.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                a = int(float(parts[0]))
                b = int(float(parts[1]))
            except Exception:
                continue
            if 16 <= a <= 96 and 16 <= b <= 96:
                header_idx = i
                Gh, Gw = a, b
                break
    if header_idx is None or Gh is None or Gw is None:
        raise ValueError("Cannot find 'Gh Gw' header line in model output")
    body = lines[header_idx + 1 :]
    g = np.full((Gh, Gw), np.nan, dtype=np.float32)
    rows_to_parse = min(Gh, len(body))
    for r in range(rows_to_parse):
        parts = body[r].replace(",", " ").split()
        cols_to_parse = min(Gw, len(parts))
        for c in range(cols_to_parse):
            tok = parts[c].strip().lower()
            if tok in {"nan", "none", "null", "na"}:
                continue
            try:
                v = float(tok)
            except Exception:
                continue
            if v < 0:
                v = 0.0
            g[r, c] = v
    return g


def _interp_nan_grid_to_hw(grid: np.ndarray, H: int, W: int) -> np.ndarray:
    import cv2

    g = grid.astype(np.float32)
    mask = np.isfinite(g)
    if mask.any():
        mean = float(np.nanmean(g))
    else:
        mean = 0.0
    g = np.where(mask, g, mean).astype(np.float32)
    out = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return out


def _read_tif_chw(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float32)


def _rgb_for_display(path: Path, image_mode: str, selected_bands, reflectance_scale: float) -> np.ndarray:
    cube = _read_tif_chw(path)
    mode = str(image_mode).lower().strip()
    if mode == "rgb":
        idx = [0, 1, 2]
    else:
        idx = [7, 2, 1]
    idx = idx[:3]
    x = cube[idx] / float(reflectance_scale if reflectance_scale > 0 else 255.0)
    x = np.clip(x, 0.0, 1.0)
    return np.transpose(x, (1, 2, 0))


def _save_viz(rgb: np.ndarray, depth_m: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = axes.ravel()
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    dm = depth_m.astype(np.float32)
    dm_vis = dm.copy()
    dm_vis[~np.isfinite(dm_vis)] = np.nan
    im1 = axes[1].imshow(-dm_vis, cmap="viridis")
    axes[1].set_title("OpenAI predicted depth (m)\n(water only; land=nan)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="llm_depth_viz")
    p.add_argument("--openai_model", type=str, default=None)
    p.add_argument("--max_side", type=int, default=None)
    p.add_argument("--min_side", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--save_raw", action="store_true")
    args = p.parse_args()

    cfg = load_yaml_config(args.config)
    data_cfg = cfg.get("data", {}) or {}
    vlm_cfg = cfg.get("vlm", {}) or {}

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_name = args.openai_model or vlm_cfg.get("openai_model") or "gpt-4o"
    max_side = args.max_side if args.max_side is not None else vlm_cfg.get("max_side", 1024)
    min_side = args.min_side if args.min_side is not None else vlm_cfg.get("min_side", None)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else vlm_cfg.get("max_new_tokens", 2048)

    img = open_image_rgb(
        image_path,
        max_side=int(max_side) if max_side is not None else None,
        min_side=int(min_side) if min_side is not None else None,
        selected_bands=data_cfg.get("selected_bands"),
        reflectance_scale=float(data_cfg.get("reflectance_scale", 255.0)),
        image_mode=str(data_cfg.get("image_mode", "rgb")),
    )
    orig_w, orig_h = img.info.get("orig_size", img.size)

    prompt = args.prompt or DEFAULT_PROMPT
    client = OpenAIVisionAnnotator(model=model_name)
    gen_cfg = GenerationConfig(max_new_tokens=int(max_new_tokens), do_sample=False)
    raw_text = client.generate(img, prompt, gen_cfg)
    g = _parse_depth_grid_text(raw_text)
    depth_m = _interp_nan_grid_to_hw(g, H=int(orig_h), W=int(orig_w))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    out_png = out_dir / f"{stem}_llm_depth.png"
    out_raw = out_dir / f"{stem}_llm_depth_raw.txt"

    rgb = _rgb_for_display(
        path=image_path,
        image_mode=str(data_cfg.get("image_mode", "rgb")),
        selected_bands=data_cfg.get("selected_bands", None),
        reflectance_scale=float(data_cfg.get("reflectance_scale", 255.0)),
    )

    _save_viz(rgb, depth_m, out_png)

    if args.save_raw:
        with open(out_raw, "w", encoding="utf-8") as f:
            f.write(raw_text)

    print(f"Saved: {out_png}")
    if args.save_raw:
        print(f"Saved: {out_raw}")


if __name__ == "__main__":
    main()
