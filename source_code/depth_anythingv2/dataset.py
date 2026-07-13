import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def extract_patch_id(filename: str, prefix: str) -> Optional[str]:
    pattern = rf"{prefix}_(\d+)"
    m = re.search(pattern, filename)
    if m is None:
        return None
    return m.group(1)


def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    return arr


def build_pairs_magic(
    img_dir: Path,
    depth_dir: Path,
    image_suffix: str = "img_*.tif",
    depth_suffixes_to_try: Optional[Sequence[str]] = None,
) -> List[Tuple[Path, Path, str]]:
    suffixes = list(depth_suffixes_to_try or ["_depth", "_bathy", "_gt", "_label"])
    pairs: List[Tuple[Path, Path, str]] = []
    for img_path in sorted(img_dir.glob(image_suffix)):
        stem = img_path.stem
        depth_path: Optional[Path] = None
        exact = depth_dir / f"{stem}.tif"
        if exact.exists():
            depth_path = exact
        else:
            for s in suffixes:
                cand = depth_dir / f"{stem}{s}.tif"
                if cand.exists():
                    depth_path = cand
                    break
        if depth_path is None and stem.startswith("img_"):
            mapped = stem.replace("img_", "depth_", 1)
            cand = depth_dir / f"{mapped}.tif"
            if cand.exists():
                depth_path = cand
        if depth_path is not None:
            pairs.append((img_path, depth_path, stem))
    return pairs


class BathymetryDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        depth_dir: str,
        img_glob: str = "img_*.tif",
        depth_suffixes_to_try: Optional[Sequence[str]] = None,
        image_mode: str = "rgb",
        reflectance_scale: float = 255.0,
        magic_negative_depth: bool = True,
        normalize_depth: bool = False,
        depth_mean: Optional[float] = None,
        depth_std: Optional[float] = None,
    ):
        self.img_dir = Path(img_dir)
        self.depth_dir = Path(depth_dir)
        self.reflectance_scale = float(reflectance_scale) if reflectance_scale else 1.0
        self.magic_negative_depth = bool(magic_negative_depth)
        self.image_mode = str(image_mode).lower().strip()
        if self.image_mode != "rgb":
            raise ValueError("depth_anythingv2 runner currently supports only image_mode=rgb")

        self.pairs = build_pairs_magic(
            img_dir=self.img_dir,
            depth_dir=self.depth_dir,
            image_suffix=img_glob,
            depth_suffixes_to_try=depth_suffixes_to_try,
        )
        if len(self.pairs) == 0:
            raise ValueError("No matched image-depth pairs found.")

        self.normalize_depth = bool(normalize_depth)
        self.depth_mean = float(depth_mean) if depth_mean is not None else None
        self.depth_std = float(depth_std) if depth_std is not None else None
        if self.normalize_depth and (self.depth_mean is None or self.depth_std is None):
            raise ValueError("normalize_depth=true requires depth_mean and depth_std in config")
        if self.depth_std is not None:
            self.depth_std = max(self.depth_std, 1e-6)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, depth_path, patch_id = self.pairs[idx]

        img = read_raster(img_path).astype(np.float32)
        depth_raw = read_raster(depth_path).astype(np.float32)

        if img.shape[0] > 3:
            img = img[:3]
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)

        rs = self.reflectance_scale if self.reflectance_scale > 0 else 1.0
        img = img / rs
        img = np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

        d0 = depth_raw[0] if depth_raw.ndim == 3 else depth_raw
        if self.magic_negative_depth:
            valid_hw = (np.isfinite(d0) & (d0 < 0)).astype(np.float32)
            depth_hw = np.where(valid_hw > 0, -d0, 0.0).astype(np.float32)
        else:
            valid_hw = np.isfinite(d0).astype(np.float32)
            depth_hw = np.where(valid_hw > 0, d0, 0.0).astype(np.float32)

        depth = depth_hw[np.newaxis, ...]
        valid_mask = valid_hw[np.newaxis, ...]

        if self.normalize_depth and self.depth_mean is not None and self.depth_std is not None:
            depth = (depth - self.depth_mean) / self.depth_std

        return {
            "image": torch.from_numpy(img).float(),
            "depth": torch.from_numpy(depth).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "patch_id": patch_id,
        }

