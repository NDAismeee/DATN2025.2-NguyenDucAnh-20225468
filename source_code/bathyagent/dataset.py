from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


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
S2_INDEX_TO_BAND = {v: k for k, v in S2_BAND_TO_INDEX.items()}


@dataclass
class MagicBathySample:
    image_path: Path
    depth_path: Path
    sample_id: str
    site: str
    modality: str


def resolve_selected_bands(
    selected_bands: Optional[Union[str, Sequence[Union[int, str]]]],
    image_mode: str = "rgb",
) -> Optional[List[int]]:
    mode = image_mode.strip().lower()
    if mode == "rgb":
        if selected_bands is None:
            return [0, 1, 2]
        if isinstance(selected_bands, str):
            if selected_bands.strip().lower() in {"all", "rgb"}:
                return [0, 1, 2]
            raise ValueError(f"Unknown RGB selected_bands preset: {selected_bands}")
        return [int(x) for x in selected_bands]

    if mode != "s2":
        raise ValueError(f"Unknown image_mode: {image_mode}")
    if selected_bands is None:
        return None
    if isinstance(selected_bands, str):
        key = selected_bands.strip().lower()
        if key == "all":
            return None
        if key == "rgb":
            return [S2_BAND_TO_INDEX["B4"], S2_BAND_TO_INDEX["B3"], S2_BAND_TO_INDEX["B2"]]
        raise ValueError(f"Unknown S2 selected_bands preset: {selected_bands}")

    out: List[int] = []
    for band in selected_bands:
        if isinstance(band, int):
            out.append(band)
        else:
            name = str(band).strip().upper()
            if name not in S2_BAND_TO_INDEX:
                raise ValueError(f"Unknown Sentinel-2 band name: {band}")
            out.append(S2_BAND_TO_INDEX[name])
    return out


def selected_band_names(
    selected_bands: Optional[Union[str, Sequence[Union[int, str]]]],
    image_mode: str = "rgb",
    total_bands: int = 13,
) -> List[str]:
    resolved = resolve_selected_bands(selected_bands, image_mode=image_mode)
    if image_mode.strip().lower() == "rgb":
        labels = ("R", "G", "B")
        return [labels[i] for i in (resolved or [0, 1, 2])]
    if resolved is None:
        return [S2_INDEX_TO_BAND[i] for i in range(total_bands)]
    return [S2_INDEX_TO_BAND[i] for i in resolved]


def _read_tif_image(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def _read_tif_depth(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _read_npy(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _normalize_s2_image(image: np.ndarray, scale: float = 255.0) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(image / float(scale), 0.0, 1.5).astype(np.float32)


def _infer_sample_id(path: Path) -> str:
    return path.stem


def _infer_site_from_path(path: Path) -> str:
    text = str(path).lower()
    if "agia" in text:
        return "agia_napa"
    if "puck" in text:
        return "puck_lagoon"
    return "unknown"


def _default_site_note(site: str) -> str:
    if site == "agia_napa":
        return "Clear shallow coastal water with possible glint and bottom texture ambiguity."
    if site == "puck_lagoon":
        return "Lagoon/coastal scene with possible turbidity and complex optical conditions."
    return "Shallow-water bathymetry scene."


def _match_depth_file(
    image_path: Path,
    depth_dir: Path,
    depth_suffixes_to_try: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    stem = image_path.stem
    candidates = [depth_dir / f"{stem}.tif"]
    for suffix in depth_suffixes_to_try or ["_depth", "_bathy", "_gt", "_label"]:
        candidates.append(depth_dir / f"{stem}{suffix}.tif")
    if stem.startswith("img_"):
        candidates.append(depth_dir / f"{stem.replace('img_', 'depth_', 1)}.tif")
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _ensure_chw(name: str, arr: np.ndarray, channels: Optional[int] = None) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"{name} must be CHW or HW, got {arr.shape}")
    if channels is not None and arr.shape[0] != channels:
        raise ValueError(f"{name} channel count {arr.shape[0]} != {channels}")
    return arr.astype(np.float32)


def _resize_chw_tensor(x: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(x.unsqueeze(0), size=size, mode=mode, align_corners=False).squeeze(0)
    return F.interpolate(x.unsqueeze(0), size=size, mode=mode).squeeze(0)


def _as_image_size(value) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"image_size must be null or [H, W], got {value}")


class MagicBathyDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        depth_dir: str | Path,
        modality: str = "rgb",
        image_mode: str = "rgb",
        image_size: Optional[Tuple[int, int]] = None,
        image_suffix: str = "img_*.tif",
        selected_bands: Optional[Union[str, Sequence[Union[int, str]]]] = None,
        reflectance_scale: float = 255.0,
        depth_suffixes_to_try: Optional[Sequence[str]] = None,
        allow_empty_mask: bool = False,
        verbose: bool = True,
        semantic_dir: Optional[str | Path] = None,
        require_semantic_if_enabled: bool = False,
        reliability_suffix: str = "_M.npy",
        disturbance_masks_suffix: str = "_R.npy",
        depth_prior_suffix: str = "_prior.npy",
        depth_prior_valid_suffix: str = "_prior_valid.npy",
        depth_prior_conf_suffix: str = "_prior_conf.npy",
        text_embeddings_suffix: str = "_text_embeddings.npy",
        region_texts_suffix: str = "_region_texts.json",
        water_suffix: str = "_water.npy",
        text_dim: int = 384,
        use_semantic_channels: bool = False,
        use_prior_depth_map: bool = False,
        semantic_channel_order: Optional[Sequence[str]] = None,
        semantic_suffix: str = "_semantic.npy",
        prior_suffix: str = "_prior.npy",
    ):
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir)
        self.modality = modality
        self.image_mode = image_mode.strip().lower()
        self.image_size = _as_image_size(image_size)
        self.image_suffix = image_suffix
        self.selected_bands = resolve_selected_bands(selected_bands, self.image_mode)
        self.reflectance_scale = float(reflectance_scale)
        self.depth_suffixes_to_try = depth_suffixes_to_try
        self.allow_empty_mask = bool(allow_empty_mask)
        self.verbose = bool(verbose)
        self.semantic_dir = Path(semantic_dir) if semantic_dir is not None else None
        self.require_semantic = bool(require_semantic_if_enabled)
        self.reliability_suffix = reliability_suffix
        self.disturbance_masks_suffix = disturbance_masks_suffix
        self.depth_prior_suffix = depth_prior_suffix or prior_suffix
        self.depth_prior_valid_suffix = depth_prior_valid_suffix
        self.depth_prior_conf_suffix = depth_prior_conf_suffix
        self.text_embeddings_suffix = text_embeddings_suffix
        self.region_texts_suffix = region_texts_suffix
        self.water_suffix = water_suffix
        self.text_dim = int(text_dim)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {self.depth_dir}")
        if self.semantic_dir is not None and not self.semantic_dir.exists():
            raise FileNotFoundError(f"Semantic directory not found: {self.semantic_dir}")

        image_paths = sorted(self.image_dir.glob(self.image_suffix))
        if not image_paths:
            raise FileNotFoundError(f"No image files found in {self.image_dir} with {self.image_suffix}")

        self.samples: List[MagicBathySample] = []
        skipped_no_depth = 0
        skipped_empty = 0
        skipped_semantic = 0
        for image_path in image_paths:
            depth_path = _match_depth_file(image_path, self.depth_dir, self.depth_suffixes_to_try)
            if depth_path is None:
                skipped_no_depth += 1
                continue
            raw_depth = np.nan_to_num(_read_tif_depth(depth_path), nan=0.0, posinf=0.0, neginf=0.0)
            if (raw_depth < 0).sum() == 0 and not self.allow_empty_mask:
                skipped_empty += 1
                continue
            sample_id = _infer_sample_id(image_path)
            if self.require_semantic and not self._has_required_semantics(sample_id):
                skipped_semantic += 1
                continue
            self.samples.append(
                MagicBathySample(
                    image_path=image_path,
                    depth_path=depth_path,
                    sample_id=sample_id,
                    site=_infer_site_from_path(image_path),
                    modality=modality,
                )
            )

        if not self.samples:
            raise FileNotFoundError("No usable image-depth-semantic samples found.")
        if self.verbose:
            print(
                f"[MagicBathyDataset] matched={len(self.samples)} "
                f"skipped_no_depth={skipped_no_depth} skipped_empty={skipped_empty} "
                f"skipped_semantic={skipped_semantic}"
            )

    def _semantic_path(self, sample_id: str, suffix: str) -> Path:
        if self.semantic_dir is None:
            raise ValueError("semantic_dir is required")
        return self.semantic_dir / f"{sample_id}{suffix}"

    def _has_required_semantics(self, sample_id: str) -> bool:
        if self.semantic_dir is None:
            return False
        suffixes = [
            self.reliability_suffix,
            self.disturbance_masks_suffix,
            self.depth_prior_suffix,
            self.depth_prior_valid_suffix,
            self.depth_prior_conf_suffix,
            self.text_embeddings_suffix,
            self.water_suffix,
        ]
        return all(self._semantic_path(sample_id, suffix).exists() for suffix in suffixes)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_semantics(self, sample_id: str, hw: Tuple[int, int]):
        height, width = hw
        zero_m = np.zeros((1, height, width), dtype=np.float32)
        zero_r = np.zeros((1, height, width), dtype=np.float32)
        zero_prior = np.zeros((1, height, width), dtype=np.float32)
        zero_valid = np.zeros((1, height, width), dtype=np.float32)
        zero_conf = np.zeros((1, height, width), dtype=np.float32)
        zero_text = np.zeros((1, self.text_dim), dtype=np.float32)
        zero_water = np.zeros((1, height, width), dtype=np.float32)
        region_texts: List[Dict[str, object]] = []

        if self.semantic_dir is None:
            return zero_m, zero_r, zero_text, zero_prior, zero_valid, zero_conf, zero_water, region_texts

        def load_map(suffix: str, name: str, fallback: np.ndarray, channels: Optional[int] = None):
            path = self._semantic_path(sample_id, suffix)
            if not path.exists():
                if self.require_semantic:
                    raise FileNotFoundError(f"Missing {name}: {path}")
                return fallback
            arr = _ensure_chw(name, _read_npy(path), channels)
            if arr.shape[-2:] != (height, width):
                raise ValueError(f"{name} spatial shape {arr.shape[-2:]} != {(height, width)}")
            return arr

        reliability = load_map(self.reliability_suffix, "reliability_mask", zero_m, 1)
        disturbances = load_map(self.disturbance_masks_suffix, "disturbance_masks", zero_r, None)
        prior = load_map(self.depth_prior_suffix, "prior_depth_map", zero_prior, 1)
        prior_valid = load_map(self.depth_prior_valid_suffix, "prior_valid_mask", zero_valid, 1)
        prior_conf = load_map(self.depth_prior_conf_suffix, "prior_confidence", zero_conf, 1)
        water = load_map(self.water_suffix, "water_mask", zero_water, 1)

        emb_path = self._semantic_path(sample_id, self.text_embeddings_suffix)
        if emb_path.exists():
            text_embeddings = _read_npy(emb_path)
            if text_embeddings.ndim != 2:
                raise ValueError(f"text_embeddings must be 2D, got {text_embeddings.shape}")
        elif self.require_semantic:
            raise FileNotFoundError(f"Missing text embeddings: {emb_path}")
        else:
            text_embeddings = zero_text

        text_path = self._semantic_path(sample_id, self.region_texts_suffix)
        if text_path.exists():
            with open(text_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            region_texts = loaded if isinstance(loaded, list) else []

        if text_embeddings.shape[0] != disturbances.shape[0]:
            raise ValueError(
                f"{sample_id}: K mismatch R={disturbances.shape[0]} text={text_embeddings.shape[0]}"
            )
        if disturbances.shape[0] == 0:
            disturbances = np.zeros((1, height, width), dtype=np.float32)
            text_embeddings = np.zeros((1, self.text_dim), dtype=np.float32)
        return reliability, disturbances, text_embeddings, prior, prior_valid, prior_conf, water, region_texts

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        image = _read_tif_image(sample.image_path)
        if self.selected_bands is not None:
            image = image[self.selected_bands]
        image = _normalize_s2_image(image, self.reflectance_scale)

        raw_depth = np.nan_to_num(_read_tif_depth(sample.depth_path), nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = (raw_depth < 0).astype(np.float32)
        depth = np.where(valid_mask > 0, -raw_depth, 0.0).astype(np.float32)
        height, width = depth.shape
        if image.shape[-2:] != (height, width):
            raise ValueError(f"{sample.sample_id}: image/depth spatial mismatch")

        reliability, disturbances, text_embeddings, prior, prior_valid, prior_conf, semantic_water, region_texts = (
            self._load_semantics(sample.sample_id, (height, width))
        )
        water_mask = (semantic_water[0] > 0.5).astype(np.float32) if semantic_water.sum() > 0 else valid_mask.copy()

        image_t = torch.from_numpy(image).float()
        depth_t = torch.from_numpy(depth * valid_mask).float().unsqueeze(0)
        valid_t = torch.from_numpy(valid_mask).float().unsqueeze(0)
        water_t = torch.from_numpy(water_mask).float().unsqueeze(0)
        reliability_t = torch.from_numpy((reliability > 0.5).astype(np.float32)).float()
        disturbances_t = torch.from_numpy((disturbances > 0.5).astype(np.float32)).float()
        text_t = torch.from_numpy(text_embeddings).float()
        prior_valid_t = torch.from_numpy((prior_valid > 0.5).astype(np.float32)).float() * water_t
        prior_t = torch.from_numpy(prior).float() * prior_valid_t
        prior_conf_t = torch.from_numpy(prior_conf).float() * prior_valid_t

        if torch.any(prior_valid_t > reliability_t + 1e-6):
            raise ValueError(f"{sample.sample_id}: prior_valid_mask must be a subset of reliability_mask.")
        if torch.any((prior_valid_t <= 0) & (prior_t != 0)):
            raise ValueError(f"{sample.sample_id}: prior_depth_map must be zero outside prior_valid_mask.")

        if self.image_size is not None:
            image_t = _resize_chw_tensor(image_t, self.image_size, "bilinear")
            depth_t = _resize_chw_tensor(depth_t, self.image_size, "nearest")
            valid_t = (_resize_chw_tensor(valid_t, self.image_size, "nearest") > 0.5).float()
            water_t = (_resize_chw_tensor(water_t, self.image_size, "nearest") > 0.5).float()
            reliability_t = (_resize_chw_tensor(reliability_t, self.image_size, "nearest") > 0.5).float()
            if disturbances_t.shape[0] > 0:
                disturbances_t = (_resize_chw_tensor(disturbances_t, self.image_size, "nearest") > 0.5).float()
            prior_valid_t = (_resize_chw_tensor(prior_valid_t, self.image_size, "nearest") > 0.5).float() * water_t
            prior_t = _resize_chw_tensor(prior_t, self.image_size, "nearest") * prior_valid_t
            prior_conf_t = _resize_chw_tensor(prior_conf_t, self.image_size, "nearest") * prior_valid_t
            depth_t = depth_t * valid_t

        metadata = {
            "sample_id": sample.sample_id,
            "site": sample.site,
            "modality": sample.modality,
            "image_path": str(sample.image_path),
            "depth_path": str(sample.depth_path),
            "selected_band_names": selected_band_names(self.selected_bands, self.image_mode),
            "note": _default_site_note(sample.site),
            "region_texts": region_texts,
        }

        return {
            "image": image_t,
            "reliability_mask": reliability_t,
            "disturbance_masks": disturbances_t,
            "text_embeddings": text_t,
            "prior_depth_map": prior_t,
            "prior_valid_mask": prior_valid_t,
            "prior_confidence": prior_conf_t,
            "depth": depth_t,
            "valid_mask": valid_t,
            "water_mask": water_t,
            "semantic_channels": reliability_t,
            "llm_valid_mask": prior_valid_t,
            "sample_id": sample.sample_id,
            "site": sample.site,
            "modality": sample.modality,
            "image_path": str(sample.image_path),
            "depth_path": str(sample.depth_path),
            "metadata": metadata,
        }


def magicbathy_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    dense_keys = [
        "image",
        "reliability_mask",
        "prior_depth_map",
        "prior_valid_mask",
        "prior_confidence",
        "depth",
        "valid_mask",
        "water_mask",
        "semantic_channels",
        "llm_valid_mask",
    ]
    for key in dense_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)

    max_k = max(int(item["disturbance_masks"].shape[0]) for item in batch)
    text_dim = max(
        [int(item["text_embeddings"].shape[1]) for item in batch if item["text_embeddings"].numel() > 0]
        or [384]
    )
    bsz, _, height, width = out["image"].shape
    disturbance_batch = torch.zeros((bsz, max_k, height, width), dtype=out["image"].dtype)
    text_batch = torch.zeros((bsz, max_k, text_dim), dtype=out["image"].dtype)
    valid_batch = torch.zeros((bsz, max_k), dtype=out["image"].dtype)

    for bidx, item in enumerate(batch):
        masks = item["disturbance_masks"]
        texts = item["text_embeddings"]
        k = int(masks.shape[0])
        if k > 0:
            disturbance_batch[bidx, :k] = masks
            text_batch[bidx, :k, : texts.shape[1]] = texts
            region_area = masks.reshape(k, -1).sum(dim=1)
            valid_batch[bidx, :k] = (region_area > 0).float()

    out["disturbance_masks"] = disturbance_batch
    out["text_embeddings"] = text_batch
    out["region_valid_mask"] = valid_batch

    for key in ["sample_id", "site", "modality", "image_path", "depth_path", "metadata"]:
        out[key] = [item[key] for item in batch]
    return out


def create_dataloaders(
    image_dir: str | Path,
    depth_dir: str | Path,
    modality: str = "rgb",
    image_mode: str = "rgb",
    image_size: Optional[Tuple[int, int]] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    selected_bands: Optional[Union[str, Sequence[Union[int, str]]]] = None,
    reflectance_scale: float = 255.0,
    image_suffix: str = "img_*.tif",
    depth_suffixes_to_try: Optional[Sequence[str]] = None,
    allow_empty_mask: bool = False,
    verbose: bool = True,
    semantic_dir: Optional[str | Path] = None,
    require_semantic_if_enabled: bool = False,
    reliability_suffix: str = "_M.npy",
    disturbance_masks_suffix: str = "_R.npy",
    depth_prior_suffix: str = "_prior.npy",
    depth_prior_valid_suffix: str = "_prior_valid.npy",
    depth_prior_conf_suffix: str = "_prior_conf.npy",
    text_embeddings_suffix: str = "_text_embeddings.npy",
    region_texts_suffix: str = "_region_texts.json",
    water_suffix: str = "_water.npy",
    text_dim: int = 384,
    **legacy_kwargs,
):
    dataset = MagicBathyDataset(
        image_dir=image_dir,
        depth_dir=depth_dir,
        modality=modality,
        image_mode=image_mode,
        image_size=image_size,
        image_suffix=image_suffix,
        selected_bands=selected_bands,
        reflectance_scale=reflectance_scale,
        depth_suffixes_to_try=depth_suffixes_to_try,
        allow_empty_mask=allow_empty_mask,
        verbose=verbose,
        semantic_dir=semantic_dir,
        require_semantic_if_enabled=require_semantic_if_enabled,
        reliability_suffix=reliability_suffix,
        disturbance_masks_suffix=disturbance_masks_suffix,
        depth_prior_suffix=depth_prior_suffix,
        depth_prior_valid_suffix=depth_prior_valid_suffix,
        depth_prior_conf_suffix=depth_prior_conf_suffix,
        text_embeddings_suffix=text_embeddings_suffix,
        region_texts_suffix=region_texts_suffix,
        water_suffix=water_suffix,
        text_dim=text_dim,
        **legacy_kwargs,
    )

    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total_ratio - 1.0) > 1.0e-6:
        raise ValueError(f"train/val/test ratios must sum to 1, got {total_ratio}")

    n_total = len(dataset)
    if n_total < 3:
        raise ValueError(f"Dataset needs at least 3 samples for train/val/test split, got {n_total}")

    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_total - n_val - n_test)

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": magicbathy_collate_fn,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
