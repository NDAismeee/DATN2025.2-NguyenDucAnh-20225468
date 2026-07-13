from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bathymetry_llm.llm_pipeline.aerial_paths import iter_tiles_from_dirs, load_paired_dirs_from_config, resolve_scene_or_sidecar
from bathymetry_llm.utils.io import load_config, package_root, resolve_path


def _normalize_openai_base_url(url: Optional[str]) -> Optional[str]:
    if url is None:
        return None
    s = str(url).strip()
    if not s or s.lower() in ("none", "null", "~"):
        return None
    if not (s.startswith("http://") or s.startswith("https://")):
        return None
    return s.rstrip("/")


def _strip_empty_openai_env_vars() -> None:
    for key in ("OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_ORGANIZATION"):
        if key not in os.environ:
            continue
        raw = os.environ.get(key, "")
        if raw is None or not str(raw).strip() or str(raw).strip().lower() in ("none", "null"):
            del os.environ[key]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidates = [
        PACKAGE_DIR / ".env",
        PACKAGE_DIR.parent / ".env",
        Path.cwd() / ".env",
    ]
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            return
    load_dotenv(override=False)


def _load_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _image_to_data_url(path: Path, max_side: int) -> tuple[str, int, int]:
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}", w0, h0


def _scene_image(scene: Path) -> Path:
    for name in ("image.png", "image.jpg", "image.jpeg", "image.tif", "image.tiff"):
        path = scene / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No image file found in {scene}")


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def query_openai(
    image_path: Path,
    metadata_dir: Path,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_side: int,
    include_metadata: bool,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai to query the VLM: pip install openai") from exc
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file in bathymetry_llm/ or export it in the shell."
        )
    _strip_empty_openai_env_vars()
    base_url = _normalize_openai_base_url(base_url)
    if organization is not None and not str(organization).strip():
        organization = None
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    data_url, width_px, height_px = _image_to_data_url(image_path, max_side=max_side)
    size_hint = (
        f"\n\nThe aerial image pixel size is width={width_px}, height={height_px}. "
        f"All polygon coordinates must be valid pixel indices with x in [0, {width_px - 1}] and y in [0, {height_px - 1}]."
    )
    full_prompt = prompt + size_hint
    if include_metadata:
        metadata_path = Path(metadata_dir) / "metadata.json"
        if metadata_path.exists():
            full_prompt = full_prompt + "\n\nOptional scene metadata (may be empty):\n" + metadata_path.read_text(encoding="utf-8")
    client_kwargs: Dict[str, Any] = {"api_key": key, "timeout": 180.0, "max_retries": 2}
    if base_url:
        client_kwargs["base_url"] = base_url
    if organization:
        client_kwargs["organization"] = organization
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = response.choices[0].message.content or ""
    return _parse_json_object(text)


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        description="Query OpenAI vision (gpt-4o) for bathymetry JSON. Use one of: --scene | --image | --image-dir+--stem."
    )
    parser.add_argument("--scene", default=None, help="Scene folder containing image.png or image.tif")
    parser.add_argument("--image", default=None, help="Single aerial image path (writes under --output-dir or pair_aux_root/<stem>)")
    parser.add_argument("--image-dir", default=None, help="Folder with img_*.tif (e.g. agia_napa/img/aerial_train)")
    parser.add_argument("--stem", default=None, help="Image stem without extension, e.g. img_409 (requires --image-dir)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for llm_output.json when using --image (default: llm_sidecars/<stem> next to image parent)",
    )
    parser.add_argument("--pair-aux-root", default=None, help="Override data.pair_aux_root from config for output location")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_side", type=int, default=None)
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Append metadata.json from metadata_dir to the prompt (off by default).",
    )
    parser.add_argument("--no-metadata", action="store_true", help="Force image-only prompt (overrides config).")
    parser.add_argument(
        "--all-tiles",
        action="store_true",
        help="Process every image in data.aerial_train_image_dir that has a matching depth in data.aerial_train_depth_dir (requires --config).",
    )
    parser.add_argument("--force", action="store_true", help="With --all-tiles, overwrite existing llm_output.json.")
    args = parser.parse_args()
    root = package_root()
    data_cfg: Dict[str, Any] = {}
    llm_cfg: Dict[str, Any] = {}
    if args.config:
        full = load_config(resolve_path(args.config, root)) or {}
        data_cfg = full.get("data") or {}
        llm_cfg = full.get("llm") or {}

    if args.all_tiles:
        if not args.config:
            parser.error("--all-tiles requires --config.")
        if args.scene or args.image or args.image_dir or args.stem:
            parser.error("--all-tiles cannot be combined with --scene, --image, --image-dir, or --stem.")
    else:
        n_modes = sum([bool(args.scene), bool(args.image), bool(args.image_dir)])
        if n_modes != 1:
            parser.error("Specify exactly one of: --scene, --image, or --image-dir (with --stem).")
        if args.image_dir and not args.stem:
            parser.error("--stem is required when using --image-dir.")
        if args.stem and not args.image_dir:
            parser.error("--image-dir is required when using --stem.")
    model = args.model or llm_cfg.get("model") or "gpt-4o"
    max_tokens = int(args.max_tokens if args.max_tokens is not None else llm_cfg.get("max_tokens", 4096))
    temperature = float(args.temperature if args.temperature is not None else llm_cfg.get("temperature", 0.0))
    max_side = int(args.max_side if args.max_side is not None else llm_cfg.get("image_max_side", 1024))
    include_metadata = bool(llm_cfg.get("include_metadata", False))
    if args.include_metadata:
        include_metadata = True
    if args.no_metadata:
        include_metadata = False
    base_url = _normalize_openai_base_url(os.environ.get("OPENAI_BASE_URL")) or _normalize_openai_base_url(
        llm_cfg.get("base_url")
    )
    organization = os.environ.get("OPENAI_ORG_ID") or llm_cfg.get("organization") or None
    if organization is not None and not str(organization).strip():
        organization = None
    pair_aux = args.pair_aux_root or data_cfg.get("pair_aux_root")
    pair_aux_path = Path(resolve_path(pair_aux, root)) if pair_aux else None
    prompt_path = Path(args.prompt) if args.prompt else Path(__file__).with_name("prompt_template.txt")
    prompt_text = _load_prompt(prompt_path)
    sleep_s = float(os.environ.get("BATCH_LLM_SLEEP_SECONDS", "0"))

    if args.all_tiles:
        image_dir, depth_dir, pair_cfg = load_paired_dirs_from_config(data_cfg, root)
        pair_use = Path(resolve_path(args.pair_aux_root, root)) if args.pair_aux_root else pair_cfg
        for stem, image_path, out_dir, _depth in iter_tiles_from_dirs(image_dir, depth_dir, pair_use):
            target = out_dir / "llm_output.json"
            if target.exists() and not args.force:
                print(f"skip_existing {stem}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            data = query_openai(
                image_path,
                out_dir,
                prompt_text,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                max_side=max_side,
                include_metadata=include_metadata,
                base_url=base_url if base_url else None,
                organization=organization if organization else None,
            )
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(target)
            if sleep_s > 0:
                time.sleep(sleep_s)
        return

    if args.scene:
        scene = Path(args.scene)
        if not scene.is_absolute():
            scene = root / scene
        image_path = _scene_image(scene)
        out_dir = scene
        metadata_dir = scene
    elif args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = root / image_path
        stem = image_path.stem
        if args.output_dir:
            out_dir = Path(args.output_dir)
            if not out_dir.is_absolute():
                out_dir = root / out_dir
        elif pair_aux_path is not None:
            out_dir = pair_aux_path / stem
        else:
            out_dir = image_path.parent / "llm_sidecars" / stem
        metadata_dir = out_dir
    else:
        if not args.image_dir or not args.stem:
            parser.error("--image-dir and --stem are required when not using --scene or --image")
        image_dir = Path(args.image_dir)
        if not image_dir.is_absolute():
            image_dir = root / image_dir
        out_dir, image_path = resolve_scene_or_sidecar(None, image_dir, args.stem, pair_aux_path)
        metadata_dir = out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    data = query_openai(
        image_path,
        metadata_dir,
        prompt_text,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        max_side=max_side,
        include_metadata=include_metadata,
        base_url=base_url if base_url else None,
        organization=organization if organization else None,
    )
    out_path = out_dir / "llm_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
