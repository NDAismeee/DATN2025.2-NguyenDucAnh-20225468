#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from common import load_yaml_config


def load_region_texts(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts: List[str] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict):
            text = str(item.get("text", item.get("description", ""))).strip()
            category = str(item.get("category", "")).strip()
            effect = str(item.get("expected_depth_effect", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            dmin = item.get("depth_min", None)
            dmax = item.get("depth_max", None)
            interval = ""
            if dmin is not None and dmax is not None:
                interval = f"conservative local depth interval {float(dmin):.2f}-{float(dmax):.2f} meters"
            parts = [part for part in [category, text, effect, interval, rationale] if part]
            texts.append(". ".join(parts))
    return texts


def encode_texts(texts: List[str], model_name: str, output_dim: int) -> np.ndarray:
    if not texts:
        return np.zeros((1, output_dim), dtype=np.float32)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers to precompute text embeddings.") from exc
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=False)
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Encoder returned invalid embedding shape: {emb.shape}")
    return emb


def process_file(path: Path, output_suffix: str, model_name: str, output_dim: int) -> Path:
    texts = load_region_texts(path)
    emb = encode_texts(texts, model_name, output_dim)
    out_path = path.with_name(path.name.replace("_region_texts.json", output_suffix))
    np.save(out_path, emb.astype(np.float32))
    if emb.shape[0] != len(texts):
        raise ValueError(f"{path.name}: embedding/text count mismatch")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--semantic_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config: Dict = load_yaml_config(args.config)
    semantic_cfg = config.get("semantic", {})
    text_cfg = config.get("text_encoder", {})
    semantic_dir = Path(args.semantic_dir or semantic_cfg.get("semantic_dir", ""))
    if not semantic_dir.exists():
        raise FileNotFoundError(f"Semantic directory not found: {semantic_dir}")

    model_name = str(text_cfg.get("name", "sentence-transformers/all-MiniLM-L6-v2"))
    output_dim = int(text_cfg.get("output_dim", 384))
    output_suffix = str(semantic_cfg.get("text_embeddings_suffix", "_text_embeddings.npy"))
    files = sorted(semantic_dir.glob("*_region_texts.json"))
    if args.limit > 0:
        files = files[: args.limit]

    for idx, path in enumerate(files, start=1):
        out_path = process_file(path, output_suffix, model_name, output_dim)
        print(f"[{idx}/{len(files)}] {path.name} -> {out_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
