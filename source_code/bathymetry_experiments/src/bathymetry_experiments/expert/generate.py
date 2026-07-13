from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from bathymetry_experiments.expert.openai_client import OpenAIClient
from bathymetry_experiments.data.io import Pair, load_sample


def _to_png_b64(image_chw01: np.ndarray) -> str:
    import cv2
    img = np.moveaxis(image_chw01, 0, -1)
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _heuristic_noise_masks(image_chw01: np.ndarray) -> dict[str, np.ndarray]:
    import cv2
    img = np.moveaxis(image_chw01, 0, -1)
    hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    glint = (v.astype(np.float32) > 235).astype(np.float32)
    shadow = (v.astype(np.float32) < 35).astype(np.float32)
    turbidity = ((img[..., 0] > img[..., 2]) & (s.astype(np.float32) > 70)).astype(np.float32)
    gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture = (np.abs(lap) > 25).astype(np.float32)
    return {
        "sun_glint": glint,
        "shadow": shadow,
        "turbidity": turbidity,
        "underwater_objects": texture,
    }


def build_reliability_map(masks: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    keys = list(masks.keys())
    w = np.array([float(weights.get(k, 0.0)) for k in keys], dtype=np.float32)
    if w.sum() <= 0:
        w = np.ones_like(w) / max(w.size, 1)
    else:
        w = w / w.sum()
    out = np.zeros_like(next(iter(masks.values())), dtype=np.float32)
    for key, ww in zip(keys, w):
        out += ww * masks[key].astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def ensure_expert_artifacts(
    pairs: list[Pair],
    config: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    expert = config.get("expert") or {}
    if not expert.get("enabled"):
        return
    semantic_dir = Path(str(expert.get("semantic_dir", "semantic")))
    annotation_dir = Path(str(expert.get("annotation_dir", "annotation")))
    semantic_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    model = str(expert.get("openai_model") or "gpt-4o")
    base_url = str(expert.get("openai_base_url") or "")
    timeout = float(expert.get("openai_timeout_sec") or 60)
    client = OpenAIClient(base_url=base_url, timeout_sec=timeout)

    embed_model = str(expert.get("embedding_model") or "text-embedding-3-small")

    scene_cache_path = annotation_dir / "_scene_expert.json"
    scene_emb_path = semantic_dir / "_scene_expert_embedding.npy"

    scale = float(config["data"].get("reflectance_scale", 255.0))
    if not pairs:
        return

    if overwrite or (not scene_cache_path.exists()) or (not scene_emb_path.exists()):
        image0, _, _, _ = load_sample(pairs[0], scale)
        png_b64 = _to_png_b64(image0)
        prompt = (
            "You are a coastal bathymetry domain expert. Analyze the aerial RGB image and return JSON with:\n"
            "1) noise_weights: mapping for keys sun_glint, shadow, turbidity, underwater_objects with values in [0,1] that sum to 1.\n"
            "2) noise_descriptions: short sentence per noise key.\n"
            "3) depth_zones: list of 3 objects with keys zone (nearshore|mid|offshore), d_min_m, d_max_m.\n"
            "Use conservative ranges for optically shallow coastal waters.\n"
        )
        result_scene = client.chat_json(model=model, prompt=prompt, image_png_b64=png_b64)
        noise_desc = result_scene.get("noise_descriptions") or {}
        desc_text = "\n".join([f"{k}: {noise_desc.get(k, '')}".strip() for k in sorted(noise_desc.keys()) if k])
        if not desc_text.strip():
            desc_text = json.dumps(result_scene, ensure_ascii=False)
        embedding_scene = np.asarray(client.embed(embed_model, desc_text), dtype=np.float32)
        scene_cache_path.write_text(json.dumps(result_scene, indent=2, ensure_ascii=False), encoding="utf-8")
        np.save(scene_emb_path, embedding_scene.astype(np.float32))

    result_scene = json.loads(scene_cache_path.read_text(encoding="utf-8"))
    embedding_scene = np.load(scene_emb_path).astype(np.float32)
    noise_weights_scene = result_scene.get("noise_weights") or {}
    noise_desc_scene = result_scene.get("noise_descriptions") or {}
    zones_scene = result_scene.get("depth_zones") or []

    for pair in pairs:
        image, _, _, sample_id = load_sample(pair, scale)
        ann_path = annotation_dir / f"{sample_id}.json"
        mask_path = semantic_dir / f"{sample_id}_reliability.npy"
        emb_path = semantic_dir / f"{sample_id}_expert_embedding.npy"

        if not overwrite and ann_path.exists() and mask_path.exists() and emb_path.exists():
            continue

        masks = _heuristic_noise_masks(image)
        reliability = build_reliability_map(masks, noise_weights_scene)

        ann = {
            "sample_id": sample_id,
            "noise_weights": noise_weights_scene,
            "noise_descriptions": noise_desc_scene,
            "depth_zones": zones_scene,
        }
        ann_path.write_text(json.dumps(ann, indent=2, ensure_ascii=False), encoding="utf-8")
        np.save(mask_path, reliability.astype(np.float32))
        np.save(emb_path, embedding_scene.astype(np.float32))

