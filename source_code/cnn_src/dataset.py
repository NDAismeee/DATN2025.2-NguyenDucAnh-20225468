import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


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


def build_pairs(
    img_dir: Path,
    depth_dir: Path,
    img_glob: str = "img_*.tif",
    depth_glob: str = "depth_*.tif",
) -> List[Tuple[Path, Path, str]]:
    img_files = sorted(img_dir.glob(img_glob))
    depth_files = sorted(depth_dir.glob(depth_glob))

    img_map: Dict[str, Path] = {}
    depth_map: Dict[str, Path] = {}

    for p in img_files:
        pid = extract_patch_id(p.name, "img")
        if pid is not None:
            img_map[pid] = p

    for p in depth_files:
        pid = extract_patch_id(p.name, "depth")
        if pid is not None:
            depth_map[pid] = p

    common_ids = sorted(set(img_map.keys()) & set(depth_map.keys()))
    pairs = [(img_map[pid], depth_map[pid], pid) for pid in common_ids]
    return pairs


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


def compute_channel_stats(
    pairs: List[Tuple[Path, Path, str]],
    band_indices: Sequence[int],
    max_samples: Optional[int] = None,
    reflectance_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean/std while ignoring NaN/Inf.
    """
    if max_samples is not None:
        pairs = pairs[:max_samples]

    channel_sum = None
    channel_sq_sum = None
    channel_count = None

    rs = float(reflectance_scale) if reflectance_scale and reflectance_scale > 0 else 1.0

    for img_path, _, _ in pairs:
        img = read_raster(img_path).astype(np.float32)
        img = img[list(band_indices)]
        img = img / rs

        c, _, _ = img.shape
        flat = img.reshape(c, -1)
        valid = np.isfinite(flat)

        if channel_sum is None:
            channel_sum = np.zeros(c, dtype=np.float64)
            channel_sq_sum = np.zeros(c, dtype=np.float64)
            channel_count = np.zeros(c, dtype=np.float64)

        for i in range(c):
            vals = flat[i][valid[i]]
            if vals.size == 0:
                continue
            channel_sum[i] += vals.sum(dtype=np.float64)
            channel_sq_sum[i] += np.square(vals, dtype=np.float64).sum(dtype=np.float64)
            channel_count[i] += vals.size

    if np.any(channel_count == 0):
        bad_idx = np.where(channel_count == 0)[0].tolist()
        raise ValueError(f"Some selected bands have no finite pixels: indices={bad_idx}")

    mean = channel_sum / channel_count
    var = channel_sq_sum / channel_count - mean ** 2
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)


def compute_depth_stats(
    pairs: List[Tuple[Path, Path, str]],
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    invalid_depth_values: Optional[Sequence[float]] = None,
    max_samples: Optional[int] = None,
    magic_negative_valid: bool = False,
) -> Tuple[float, float]:
    """
    Compute depth mean/std using only valid depth pixels.
    """
    if max_samples is not None:
        pairs = pairs[:max_samples]

    invalid_depth_values = list(invalid_depth_values) if invalid_depth_values is not None else []
    vals_all = []

    for _, depth_path, _ in pairs:
        depth = read_raster(depth_path).astype(np.float32)
        if depth.ndim == 3:
            d0 = depth[0]
        else:
            d0 = depth

        if magic_negative_valid:
            mask = np.isfinite(d0) & (d0 < 0)
            vals = (-d0[mask]).astype(np.float64)
            if vals.size > 0:
                vals_all.append(vals)
            continue

        mask = np.isfinite(d0)

        if depth_min is not None:
            mask &= d0 >= depth_min

        if depth_max is not None:
            mask &= d0 <= depth_max

        for val in invalid_depth_values:
            mask &= d0 != val

        vals = d0[mask]
        if vals.size > 0:
            vals_all.append(vals)

    if len(vals_all) == 0:
        raise ValueError("No valid depth pixels found for computing depth statistics.")

    vals_all = np.concatenate(vals_all, axis=0)
    mean = float(vals_all.mean())
    std = float(vals_all.std())
    std = max(std, 1e-6)

    return mean, std


class BathymetryDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        depth_dir: str,
        img_glob: str = "img_*.tif",
        depth_glob: str = "depth_*.tif",
        selected_bands: Optional[Sequence[str]] = None,
        normalize: bool = True,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        normalize_depth: bool = False,
        depth_mean: Optional[float] = None,
        depth_std: Optional[float] = None,
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
        invalid_depth_values: Optional[Sequence[float]] = None,
        return_metadata: bool = False,
        pairing_mode: str = "prefix",
        depth_suffixes_to_try: Optional[Sequence[str]] = None,
        image_mode: str = "s2",
        reflectance_scale: float = 1.0,
        magic_negative_depth: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.depth_dir = Path(depth_dir)
        self.return_metadata = return_metadata
        self.pairing_mode = str(pairing_mode).lower().strip()
        self.reflectance_scale = float(reflectance_scale) if reflectance_scale else 1.0
        self.magic_negative_depth = bool(magic_negative_depth)
        self.image_mode = str(image_mode).lower().strip()

        if self.pairing_mode == "magic":
            self.pairs = build_pairs_magic(
                img_dir=self.img_dir,
                depth_dir=self.depth_dir,
                image_suffix=img_glob,
                depth_suffixes_to_try=depth_suffixes_to_try,
            )
        else:
            self.pairs = build_pairs(
                img_dir=self.img_dir,
                depth_dir=self.depth_dir,
                img_glob=img_glob,
                depth_glob=depth_glob,
            )

        if len(self.pairs) == 0:
            raise ValueError("No matched image-depth pairs found.")

        if self.image_mode == "rgb":
            self.selected_bands = ["R", "G", "B"]
            self.band_indices = [0, 1, 2]
        elif selected_bands is None:
            self.selected_bands = list(S2_BAND_TO_INDEX.keys())
            self.band_indices = [S2_BAND_TO_INDEX[b] for b in self.selected_bands]
        else:
            self.selected_bands = list(selected_bands)
            self.band_indices = [S2_BAND_TO_INDEX[b] for b in self.selected_bands]

        self.normalize = normalize
        self.normalize_depth = normalize_depth

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.invalid_depth_values = (
            list(invalid_depth_values) if invalid_depth_values is not None else []
        )

        if self.normalize:
            if mean is None or std is None:
                mean_arr, std_arr = compute_channel_stats(
                    self.pairs,
                    band_indices=self.band_indices,
                    reflectance_scale=self.reflectance_scale,
                )
            else:
                mean_arr = np.asarray(mean, dtype=np.float32)
                std_arr = np.asarray(std, dtype=np.float32)

            if len(mean_arr) != len(self.band_indices):
                raise ValueError("Length of mean does not match number of selected bands.")
            if len(std_arr) != len(self.band_indices):
                raise ValueError("Length of std does not match number of selected bands.")

            std_arr = np.maximum(std_arr, 1e-6)
            self.mean = mean_arr
            self.std = std_arr
        else:
            self.mean = None
            self.std = None

        if self.normalize_depth:
            if depth_mean is None or depth_std is None:
                depth_mean, depth_std = compute_depth_stats(
                    self.pairs,
                    depth_min=self.depth_min,
                    depth_max=self.depth_max,
                    invalid_depth_values=self.invalid_depth_values,
                    magic_negative_valid=self.magic_negative_depth,
                )

            self.depth_mean = float(depth_mean)
            self.depth_std = max(float(depth_std), 1e-6)
        else:
            self.depth_mean = None
            self.depth_std = None

    def __len__(self) -> int:
        return len(self.pairs)

    def _build_valid_mask(self, depth_chw: np.ndarray) -> np.ndarray:
        d = depth_chw[0] if depth_chw.ndim == 3 else depth_chw
        mask = np.isfinite(d)

        if self.depth_min is not None:
            mask &= d >= self.depth_min

        if self.depth_max is not None:
            mask &= d <= self.depth_max

        for val in self.invalid_depth_values:
            mask &= d != val

        return mask.astype(np.float32)[np.newaxis, ...]

    def __getitem__(self, idx: int):
        img_path, depth_path, patch_id = self.pairs[idx]

        img = read_raster(img_path).astype(np.float32)
        depth_raw = read_raster(depth_path).astype(np.float32)

        if depth_raw.ndim == 2:
            depth_raw = depth_raw[np.newaxis, ...]

        img = img[self.band_indices]

        rs = self.reflectance_scale if self.reflectance_scale > 0 else 1.0
        img = img / rs
        img = np.clip(img, 0.0, 1.5)

        d0 = depth_raw[0]

        if self.magic_negative_depth:
            valid_hw = (np.isfinite(d0) & (d0 < 0)).astype(np.float32)
            depth_hw = np.where(valid_hw > 0, -d0, 0.0).astype(np.float32)
            depth = depth_hw[np.newaxis, ...]
            valid_mask = valid_hw[np.newaxis, ...]
        else:
            depth = depth_raw
            if img.shape[1:] != depth.shape[1:]:
                raise ValueError(
                    f"Shape mismatch for patch {patch_id}: "
                    f"image {img.shape} vs depth {depth.shape}"
                )
            valid_mask = self._build_valid_mask(depth)

        if img.shape[1:] != depth.shape[1:]:
            raise ValueError(
                f"Shape mismatch for patch {patch_id}: "
                f"image {img.shape} vs depth {depth.shape}"
            )

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize:
            img = (img - self.mean[:, None, None]) / self.std[:, None, None]

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize_depth and self.depth_mean is not None and self.depth_std is not None:
            depth = (depth - self.depth_mean) / self.depth_std

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        sample = {
            "image": torch.from_numpy(img).float(),
            "depth": torch.from_numpy(depth).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
            "patch_id": patch_id,
        }

        if self.return_metadata:
            sample["img_path"] = str(img_path)
            sample["depth_path"] = str(depth_path)

        return sample