from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bathymetry_llm.llm_pipeline.aerial_paths import iter_tiles_from_dirs, load_paired_dirs_from_config, resolve_sidecar_dir
from bathymetry_llm.utils.io import load_config, package_root, resolve_path


def _load_descriptions(scene: Path) -> List[str]:
    meta_path = scene / "region_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return [str(x) for x in meta.get("descriptions", [])]
    llm_path = scene / "llm_output.json"
    if not llm_path.exists():
        return []
    with open(llm_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for region in data.get("disturbance_regions", []):
        text = str(region.get("description") or region.get("type") or "").strip()
        if text:
            out.append(text)
    return out


def _hash_embedding(text: str, dim: int) -> np.ndarray:
    values = np.zeros((dim,), dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        tokens = [text.lower() or "empty"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        values += rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(values)
    if norm > 0:
        values /= norm
    return values.astype(np.float32)


def encode_descriptions(descriptions: Iterable[str], text_dim: int = 384, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    descriptions = list(descriptions)
    if len(descriptions) == 0:
        return np.zeros((0, int(text_dim)), dtype=np.float32)
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        arr = model.encode(descriptions, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return np.stack([_hash_embedding(text, int(text_dim)) for text in descriptions], axis=0)


def encode_scene(scene: Path, text_dim: int = 384, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Path:
    scene = Path(scene)
    descriptions = _load_descriptions(scene)
    embeddings = encode_descriptions(descriptions, text_dim=text_dim, model_name=model_name)
    out_path = scene / "text_embeddings.npy"
    np.save(out_path, embeddings.astype(np.float32))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode disturbance descriptions to text_embeddings.npy")
    parser.add_argument("--scene", default=None, help="Scene or sidecar folder with region_metadata.json or llm_output.json")
    parser.add_argument("--image-dir", default=None, help="With --stem: aerial image folder (for sidecar path resolution)")
    parser.add_argument("--stem", default=None, help="Image stem, e.g. img_409")
    parser.add_argument("--pair-aux-root", default=None, help="Override data.pair_aux_root")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--all-tiles",
        action="store_true",
        help="Encode text for every tile sidecar from config aerial_train dirs (requires --config).",
    )
    parser.add_argument("--text_dim", type=int, default=384)
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    root = package_root()
    data_cfg: dict = {}
    if args.config:
        data_cfg = (load_config(resolve_path(args.config, root)) or {}).get("data") or {}
    pair_aux = args.pair_aux_root or data_cfg.get("pair_aux_root")
    pair_aux_path = Path(resolve_path(pair_aux, root)) if pair_aux else None

    if args.all_tiles:
        if not args.config:
            parser.error("--all-tiles requires --config.")
        if args.scene or args.image_dir or args.stem:
            parser.error("--all-tiles cannot be combined with --scene or --image-dir/--stem.")
        image_dir, depth_dir, pair_cfg = load_paired_dirs_from_config(data_cfg, root)
        pair_use = Path(resolve_path(args.pair_aux_root, root)) if args.pair_aux_root else pair_cfg
        for stem, _ip, scene, _dp in iter_tiles_from_dirs(image_dir, depth_dir, pair_use):
            if not (scene / "llm_output.json").exists() and not (scene / "region_metadata.json").exists():
                print(f"skip_no_text_source {stem}")
                continue
            scene.mkdir(parents=True, exist_ok=True)
            path = encode_scene(scene, text_dim=args.text_dim, model_name=args.model_name)
            print(path)
        return

    if args.scene:
        if args.image_dir or args.stem:
            parser.error("Use either --scene alone, or --image-dir + --stem (not both).")
        scene = Path(args.scene)
        if not scene.is_absolute():
            scene = root / scene
    elif args.image_dir and args.stem:
        image_dir = Path(resolve_path(args.image_dir, root))
        scene = resolve_sidecar_dir(str(args.stem), pair_aux_path, image_dir)
    else:
        parser.error("Provide --scene, or both --image-dir and --stem.")

    path = encode_scene(scene, text_dim=args.text_dim, model_name=args.model_name)
    print(path)


if __name__ == "__main__":
    main()
