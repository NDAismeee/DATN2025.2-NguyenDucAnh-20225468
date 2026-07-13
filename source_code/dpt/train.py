import argparse
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import (
    BathymetryDataset,
    build_pairs_magic,
    compute_channel_stats,
    compute_depth_stats,
)
from eval import evaluate, masked_mse
from model import DensePredictionTransformer


PairType = Tuple[Path, Path, str]
_ROOT = Path(__file__).resolve().parent


def _resolve_data_section(raw_d: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(raw_d)
    img_dir = d.get("train_image_dir") or d.get("image_dir")
    if not img_dir:
        raise KeyError("data.image_dir (or data.train_image_dir) is required in config YAML.")
    d["image_dir"] = img_dir
    if not d.get("depth_dir"):
        raise KeyError("data.depth_dir is required in config YAML.")
    return d


def load_yaml_config(path: str) -> Dict[str, Any]:
    import yaml

    env_pat = re.compile(r"\$\{([^}]+)\}")

    def _expand(v):
        if isinstance(v, str):
            return env_pat.sub(lambda m: os.environ.get(m.group(1), m.group(0)), v)
        if isinstance(v, list):
            return [_expand(x) for x in v]
        if isinstance(v, dict):
            return {k: _expand(x) for k, x in v.items()}
        return v

    with open(path, "r", encoding="utf-8") as f:
        return _expand(yaml.safe_load(f) or {})


def pick_torch_device(device_pref: str, gpu_id: int = 0) -> torch.device:
    pref = (device_pref or "auto").strip().lower()
    if pref == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_id}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_pairs(pairs: List[PairType], val_ratio: float, seed: int) -> Tuple[List[PairType], List[PairType]]:
    if len(pairs) < 2:
        return list(pairs), list(pairs)
    idx = list(range(len(pairs)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_size = max(1, int(round(len(idx) * float(val_ratio))))
    train_idx = idx[:-val_size]
    val_idx = idx[-val_size:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


def masked_l1(pred, target, valid_mask, eps: float = 1e-8):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = torch.nan_to_num(valid_mask, nan=0.0, posinf=0.0, neginf=0.0)
    abs_error = torch.abs(pred - target) * valid_mask
    return abs_error.sum() / (valid_mask.sum() + eps)


def masked_combined_loss(pred, target, valid_mask, alpha: float = 0.3):
    return alpha * masked_mse(pred, target, valid_mask) + (1.0 - alpha) * masked_l1(pred, target, valid_mask)


def save_checkpoint(path, model, optimizer, epoch, best_val_rmse, config):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_rmse": best_val_rmse,
        "config": config,
    }
    torch.save(ckpt, path)


def _build_dataset_from_pairs(
    pairs_subset: List[PairType],
    *,
    img_dir: str,
    depth_dir: str,
    img_glob: str,
    depth_suffixes_to_try: Optional[Sequence[str]],
    reflectance_scale: float,
    magic_negative_depth: bool,
    normalize: bool,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
    normalize_depth: bool,
    depth_mean: Optional[float],
    depth_std: Optional[float],
) -> BathymetryDataset:
    ds = BathymetryDataset(
        img_dir=img_dir,
        depth_dir=depth_dir,
        img_glob=img_glob,
        depth_glob="depth_*.tif",
        selected_bands=None,
        normalize=normalize,
        mean=mean,
        std=std,
        normalize_depth=normalize_depth,
        depth_mean=depth_mean,
        depth_std=depth_std,
        depth_min=None,
        depth_max=None,
        invalid_depth_values=[],
        return_metadata=False,
        pairing_mode="magic",
        depth_suffixes_to_try=depth_suffixes_to_try,
        image_mode="rgb",
        reflectance_scale=reflectance_scale,
        magic_negative_depth=magic_negative_depth,
    )
    ds.pairs = list(pairs_subset)
    return ds


def train_one_epoch(model, dataloader, optimizer, device, depth_std=1.0):
    model.train()
    total_loss = 0.0
    total_valid = 0.0
    total_sq_error = 0.0
    total_abs_error = 0.0

    nb = device.type == "cuda"
    for batch in dataloader:
        image = batch["image"].to(device, non_blocking=nb)
        depth = batch["depth"].to(device, non_blocking=nb)
        valid_mask = batch["valid_mask"].to(device, non_blocking=nb)

        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = torch.nan_to_num(valid_mask, nan=0.0, posinf=0.0, neginf=0.0)

        optimizer.zero_grad(set_to_none=True)
        pred = model(image)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        loss = masked_combined_loss(pred, depth, valid_mask, alpha=0.3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            residual = pred - depth
            sq_error = ((residual ** 2) * valid_mask).sum().item()
            abs_error = (residual.abs() * valid_mask).sum().item()
            num_valid = valid_mask.sum().item()

        total_loss += float(loss.detach().cpu())
        total_sq_error += sq_error
        total_abs_error += abs_error
        total_valid += num_valid

    mse_norm = total_sq_error / max(total_valid, 1.0)
    rmse_norm = mse_norm ** 0.5
    mae_norm = total_abs_error / max(total_valid, 1.0)
    return {
        "loss": total_loss / max(len(dataloader), 1),
        "rmse": rmse_norm * depth_std,
        "mae": mae_norm * depth_std,
        "num_valid_pixels": total_valid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--train_image_dir",
        type=str,
        default=None,
        help="Folder of RGB training images; depth is paired only when a match exists in depth_dir (magic pairing, image-first).",
    )
    parser.add_argument("--image_dir", type=str, default=None, help="Override data.image_dir.")
    parser.add_argument("--depth_dir", type=str, default=None, help="Override data.depth_dir.")
    parser.add_argument("--image_suffix", type=str, default=None, help="Override data.image_suffix (e.g. *.tif, img_*.tif).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except Exception:
        pass

    cfg = load_yaml_config(args.config)
    d = _resolve_data_section(cfg.get("data", {}) or {})
    if args.train_image_dir is not None:
        d["image_dir"] = str(args.train_image_dir)
    elif args.image_dir is not None:
        d["image_dir"] = str(args.image_dir)
    if args.depth_dir is not None:
        d["depth_dir"] = str(args.depth_dir)
    if args.image_suffix is not None:
        d["image_suffix"] = str(args.image_suffix)
    cfg = dict(cfg)
    cfg["data"] = d
    t = dict(cfg.get("train", {}) or {})
    if args.epochs is not None:
        t["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        t["batch_size"] = int(args.batch_size)
    cfg["train"] = t
    m = cfg.get("model", {}) or {}
    l = cfg.get("logging", {}) or {}

    set_seed(int(t.get("seed", 42)))
    device_pref = args.device if args.device is not None else str(t.get("device", "cpu"))
    device = pick_torch_device(device_pref, int(t.get("gpu_id", 0)))
    print(f"[dpt train] device={device}")

    save_dir = Path(str(l.get("save_dir", "checkpoints_dpt_rgb")))
    save_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs_magic(
        img_dir=Path(d["image_dir"]),
        depth_dir=Path(d["depth_dir"]),
        image_suffix=str(d.get("image_suffix", "img_*.tif")),
        depth_suffixes_to_try=d.get("depth_suffixes_to_try"),
    )
    if len(pairs) == 0:
        raise ValueError("No matched image-depth pairs found.")
    print(f"[dpt train] magic pairs (image-first): {len(pairs)} | image_dir={d['image_dir']} | depth_dir={d['depth_dir']}")

    train_pairs, val_pairs = split_pairs(pairs, float(t.get("val_ratio", 0.2)), int(t.get("seed", 42)))

    reflectance_scale = float(d.get("reflectance_scale", 255.0))
    magic_negative = bool(d.get("magic_negative_depth_valid", True))

    mean, std = compute_channel_stats(train_pairs, band_indices=[0, 1, 2], reflectance_scale=reflectance_scale) if bool(t.get("normalize", True)) else (None, None)
    depth_mean, depth_std = compute_depth_stats(train_pairs, magic_negative_valid=magic_negative) if bool(t.get("normalize_depth", True)) else (None, None)
    depth_std_val = float(depth_std) if depth_std is not None else 1.0

    train_ds = _build_dataset_from_pairs(
        train_pairs,
        img_dir=str(d["image_dir"]),
        depth_dir=str(d["depth_dir"]),
        img_glob=str(d.get("image_suffix", "img_*.tif")),
        depth_suffixes_to_try=d.get("depth_suffixes_to_try"),
        reflectance_scale=reflectance_scale,
        magic_negative_depth=magic_negative,
        normalize=bool(t.get("normalize", True)),
        mean=(mean.tolist() if mean is not None else None),
        std=(std.tolist() if std is not None else None),
        normalize_depth=bool(t.get("normalize_depth", True)),
        depth_mean=depth_mean,
        depth_std=depth_std,
    )
    val_ds = _build_dataset_from_pairs(
        val_pairs,
        img_dir=str(d["image_dir"]),
        depth_dir=str(d["depth_dir"]),
        img_glob=str(d.get("image_suffix", "img_*.tif")),
        depth_suffixes_to_try=d.get("depth_suffixes_to_try"),
        reflectance_scale=reflectance_scale,
        magic_negative_depth=magic_negative,
        normalize=bool(t.get("normalize", True)),
        mean=(mean.tolist() if mean is not None else None),
        std=(std.tolist() if std is not None else None),
        normalize_depth=bool(t.get("normalize_depth", True)),
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    pin = device.type == "cuda"
    nw = int(t.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=int(t.get("batch_size", 1)), shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=int(t.get("batch_size", 1)), shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))

    model = DensePredictionTransformer(
        in_channels=3,
        width=int(m.get("width", 128)),
        patch_size=int(m.get("patch_size", 8)),
        layers=int(m.get("layers", 2)),
        heads=int(m.get("heads", 4)),
        dropout=float(m.get("dropout", 0.1)),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(t.get("learning_rate", 1e-4)), weight_decay=float(t.get("weight_decay", 1e-4)))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_val_rmse = float("inf")
    best_epoch = -1
    epochs_wo = 0
    patience = int(t.get("patience", 25))

    best_path = str(save_dir / str(l.get("best_model_name", "best_model.pt")))
    last_path = str(save_dir / str(l.get("last_model_name", "last_model.pt")))

    config_for_ckpt = {"data": d, "train": t, "model": m, "stats": {"mean": mean.tolist() if mean is not None else None, "std": std.tolist() if std is not None else None, "depth_mean": depth_mean, "depth_std": depth_std}}

    for epoch in range(1, int(t.get("epochs", 50)) + 1):
        trm = train_one_epoch(model, train_loader, optimizer, device, depth_std=depth_std_val)
        vm = evaluate(model=model, dataloader=val_loader, device=device, loss_fn=masked_combined_loss)

        val_rmse_m = float(vm["rmse"]) * depth_std_val
        val_mae_m = float(vm["mae"]) * depth_std_val
        scheduler.step(float(vm["rmse"]))
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{int(t.get('epochs', 50))}] | LR: {lr:.6f} | "
            f"Train Loss: {trm['loss']:.6f} | Train RMSE: {trm['rmse']:.6f} m | Train MAE: {trm['mae']:.6f} m | "
            f"Val RMSE: {val_rmse_m:.6f} m | Val MAE: {val_mae_m:.6f} m"
        )

        save_checkpoint(last_path, model, optimizer, epoch, best_val_rmse, config_for_ckpt)
        if val_rmse_m < best_val_rmse:
            best_val_rmse = val_rmse_m
            best_epoch = epoch
            epochs_wo = 0
            save_checkpoint(best_path, model, optimizer, epoch, best_val_rmse, config_for_ckpt)
            print(f"  -> New best model @ epoch {epoch} | Val RMSE={best_val_rmse:.6f} m")
        else:
            epochs_wo += 1

        if epochs_wo >= patience:
            print(f"Early stopping. Best epoch={best_epoch} Val RMSE={best_val_rmse:.6f} m")
            break

    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()

