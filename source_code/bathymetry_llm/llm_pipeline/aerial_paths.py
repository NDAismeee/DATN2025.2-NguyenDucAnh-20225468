from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from bathymetry_llm.data.dataset import _resolve_depth_path_for_image
from bathymetry_llm.utils.io import resolve_path

_IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def find_image_by_stem(image_dir: Path, stem: str) -> Path:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    for ext in _IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.is_file():
            return p
        p2 = image_dir / f"{stem}{ext.upper()}"
        if p2.is_file():
            return p2
    raise FileNotFoundError(f"No image file named {stem}.* in {image_dir}")


def resolve_sidecar_dir(
    stem: str,
    pair_aux_root: Optional[Path],
    image_dir: Optional[Path],
) -> Path:
    if pair_aux_root is not None:
        return Path(pair_aux_root) / stem
    if image_dir is not None:
        return Path(image_dir).parent / "llm_sidecars" / stem
    raise ValueError("pair_aux_root or image_dir is required to resolve sidecar directory")


def resolve_scene_or_sidecar(
    scene: Optional[Path],
    image_dir: Optional[Path],
    stem: Optional[str],
    pair_aux_root: Optional[Path],
) -> tuple[Path, Path]:
    if scene is not None:
        p = Path(scene)
        return p, _scene_image_path(p)
    if not stem:
        raise ValueError("When --scene is omitted, --stem is required")
    if image_dir is None:
        raise ValueError("When --scene is omitted, --image-dir is required")
    img_dir = Path(image_dir)
    ref_image = find_image_by_stem(img_dir, stem)
    out_dir = resolve_sidecar_dir(stem, pair_aux_root, img_dir)
    return out_dir, ref_image


def load_paired_dirs_from_config(data_cfg: Dict[str, Any], root: Path) -> tuple[Path, Path, Optional[Path]]:
    im = data_cfg.get("aerial_train_image_dir")
    dd = data_cfg.get("aerial_train_depth_dir")
    if not im or not dd:
        raise ValueError("Set data.aerial_train_image_dir and data.aerial_train_depth_dir in the config.")
    image_dir = Path(resolve_path(im, root))
    depth_dir = Path(resolve_path(dd, root))
    pair = data_cfg.get("pair_aux_root")
    pair_path = Path(resolve_path(pair, root)) if pair else None
    return image_dir, depth_dir, pair_path


def list_stems_with_depth(image_dir: Path, depth_dir: Path) -> List[str]:
    image_dir = Path(image_dir)
    depth_dir = Path(depth_dir)
    stems: List[str] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
            continue
        try:
            _resolve_depth_path_for_image(path, depth_dir)
        except FileNotFoundError:
            continue
        stems.append(path.stem)
    return stems


def iter_tiles_from_dirs(
    image_dir: Path,
    depth_dir: Path,
    pair_aux_root: Optional[Path],
) -> Iterator[Tuple[str, Path, Path, Path]]:
    image_dir = Path(image_dir)
    depth_dir = Path(depth_dir)
    for stem in list_stems_with_depth(image_dir, depth_dir):
        image_path = find_image_by_stem(image_dir, stem)
        depth_path = _resolve_depth_path_for_image(image_path, depth_dir)
        sidecar = resolve_sidecar_dir(stem, pair_aux_root, image_dir)
        yield stem, image_path, sidecar, depth_path


def _scene_image_path(scene: Path) -> Path:
    for name in ("image.png", "image.jpg", "image.jpeg", "image.tif", "image.tiff"):
        path = Path(scene) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No image file found in {scene}")
