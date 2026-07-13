import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import (
    BathymetryDataset,
    build_pairs,
    build_pairs_magic,
    compute_channel_stats,
    compute_depth_stats,
    S2_BAND_TO_INDEX,
)
from model import SimpleBathymetryCNN
from eval import evaluate, masked_mse


PairType = Tuple[Path, Path, str]

_CNN_ROOT = Path(__file__).resolve().parent
_NEW_TEST = _CNN_ROOT.parent / "new_test"
if _NEW_TEST.is_dir():
    sys.path.insert(0, str(_NEW_TEST))
try:
    from common import load_yaml_config, pick_torch_device
except ImportError:

    def load_yaml_config(path: str) -> Dict[str, Any]:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def pick_torch_device(device_pref: str, gpu_id: int = 0):
        pref = (device_pref or "auto").strip().lower()
        if pref == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        if pref in ("cuda", "gpu"):
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cpu") if not torch.cuda.is_available() else torch.device(f"cuda:{gpu_id}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def debug_tensor(name, x):
    print(
        f"{name}: shape={tuple(x.shape)}, "
        f"finite={torch.isfinite(x).all().item()}, "
        f"min={torch.nan_to_num(x).min().item():.6f}, "
        f"max={torch.nan_to_num(x).max().item():.6f}, "
        f"mean={torch.nan_to_num(x).mean().item():.6f}"
    )


def masked_l1(pred, target, valid_mask, eps: float = 1e-8):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = torch.nan_to_num(valid_mask, nan=0.0, posinf=0.0, neginf=0.0)

    abs_error = torch.abs(pred - target) * valid_mask
    return abs_error.sum() / (valid_mask.sum() + eps)


def masked_combined_loss(pred, target, valid_mask, alpha: float = 0.3):
    """
    alpha: weight for MSE
    (1 - alpha): weight for L1

    alpha=0.3 => 0.3*MSE + 0.7*L1
    More robust than 0.5/0.5 for noisy depth targets.
    """
    return alpha * masked_mse(pred, target, valid_mask) + (1.0 - alpha) * masked_l1(
        pred, target, valid_mask
    )


def train_one_epoch(model, dataloader, optimizer, device, depth_std=1.0, epoch=1):
    model.train()

    total_loss = 0.0
    total_valid = 0.0
    total_sq_error = 0.0
    total_abs_error = 0.0

    for step, batch in enumerate(dataloader):
        nb = device.type == "cuda"
        image = batch["image"].to(device, non_blocking=nb)
        depth = batch["depth"].to(device, non_blocking=nb)
        valid_mask = batch["valid_mask"].to(device, non_blocking=nb)

        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = torch.nan_to_num(valid_mask, nan=0.0, posinf=0.0, neginf=0.0)

        if epoch == 1 and step == 0:
            debug_tensor("image", image)
            debug_tensor("depth_norm", depth)
            debug_tensor("valid_mask", valid_mask)

        optimizer.zero_grad(set_to_none=True)

        pred = model(image)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        if epoch == 1 and step == 0:
            debug_tensor("pred_norm_before_loss", pred)

        loss = masked_combined_loss(pred, depth, valid_mask, alpha=0.3)

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss detected at epoch={epoch}, step={step}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            residual = pred - depth
            sq_error = ((residual ** 2) * valid_mask).sum().item()
            abs_error = (residual.abs() * valid_mask).sum().item()
            num_valid = valid_mask.sum().item()

        total_loss += loss.item()
        total_sq_error += sq_error
        total_abs_error += abs_error
        total_valid += num_valid

    if total_valid == 0:
        raise ValueError("No valid pixels found in training.")

    mse_norm = total_sq_error / total_valid
    rmse_norm = mse_norm ** 0.5
    mae_norm = total_abs_error / total_valid

    rmse_m = rmse_norm * depth_std
    mae_m = mae_norm * depth_std

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "rmse_norm": rmse_norm,
        "mae_norm": mae_norm,
        "rmse": rmse_m,
        "mae": mae_m,
        "num_valid_pixels": total_valid,
    }


def split_pairs(
    pairs: List[PairType],
    val_ratio: float,
    seed: int,
) -> Tuple[List[PairType], List[PairType]]:
    total_size = len(pairs)
    if total_size < 2:
        raise ValueError(f"Need at least 2 matched pairs, got {total_size}.")

    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = max(1, int(round(total_size * val_ratio)))
    train_size = total_size - val_size

    if train_size <= 0:
        raise ValueError(
            f"Not enough matched samples to split dataset. "
            f"total_size={total_size}, val_size={val_size}"
        )

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]

    if len(train_pairs) == 0 or len(val_pairs) == 0:
        raise ValueError(
            f"Invalid split: train={len(train_pairs)}, val={len(val_pairs)}"
        )

    return train_pairs, val_pairs


def compute_train_stats(
    train_pairs: List[PairType],
    selected_bands: Optional[Sequence[str]],
    normalize: bool,
    normalize_depth: bool,
    depth_min: Optional[float],
    depth_max: Optional[float],
    invalid_depth_values: Optional[Sequence[float]],
    image_mode: str = "s2",
    reflectance_scale: float = 1.0,
    magic_negative_depth: bool = False,
):
    im = str(image_mode).lower().strip()
    if im == "rgb":
        band_indices = [0, 1, 2]
    elif selected_bands is None:
        band_indices = list(S2_BAND_TO_INDEX.values())
    else:
        band_indices = [S2_BAND_TO_INDEX[b] for b in selected_bands]

    mean = None
    std = None
    depth_mean = None
    depth_std = None

    if normalize:
        mean, std = compute_channel_stats(
            train_pairs,
            band_indices=band_indices,
            reflectance_scale=reflectance_scale,
        )
        mean = mean.astype(np.float32)
        std = np.maximum(std.astype(np.float32), 1e-6)

    if normalize_depth:
        depth_mean, depth_std = compute_depth_stats(
            train_pairs,
            depth_min=depth_min,
            depth_max=depth_max,
            invalid_depth_values=invalid_depth_values,
            magic_negative_valid=magic_negative_depth,
        )
        depth_mean = float(depth_mean)
        depth_std = max(float(depth_std), 1e-6)

    return mean, std, depth_mean, depth_std


def make_dataset_with_fixed_pairs(
    img_dir: str,
    depth_dir: str,
    pairs_subset: List[PairType],
    selected_bands: Optional[Sequence[str]],
    normalize: bool,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
    normalize_depth: bool,
    depth_mean: Optional[float],
    depth_std: Optional[float],
    depth_min: Optional[float],
    depth_max: Optional[float],
    invalid_depth_values: Optional[Sequence[float]],
    return_metadata: bool = False,
    pairing_mode: str = "prefix",
    img_glob: str = "img_*.tif",
    depth_glob: str = "depth_*.tif",
    depth_suffixes_to_try: Optional[Sequence[str]] = None,
    image_mode: str = "s2",
    reflectance_scale: float = 1.0,
    magic_negative_depth: bool = False,
):
    ds = BathymetryDataset(
        img_dir=img_dir,
        depth_dir=depth_dir,
        img_glob=img_glob,
        depth_glob=depth_glob,
        selected_bands=selected_bands,
        normalize=normalize,
        mean=mean,
        std=std,
        normalize_depth=normalize_depth,
        depth_mean=depth_mean,
        depth_std=depth_std,
        depth_min=depth_min,
        depth_max=depth_max,
        invalid_depth_values=invalid_depth_values,
        return_metadata=return_metadata,
        pairing_mode=pairing_mode,
        depth_suffixes_to_try=depth_suffixes_to_try,
        image_mode=image_mode,
        reflectance_scale=reflectance_scale,
        magic_negative_depth=magic_negative_depth,
    )

    # BathymetryDataset hiện tại tự rebuild toàn bộ pairs trong __init__.
    # Override lại để tránh stats leakage nhưng vẫn tương thích code cũ.
    ds.pairs = list(pairs_subset)

    if len(ds.pairs) == 0:
        raise ValueError("Dataset subset is empty after overriding pairs.")

    return ds


def build_datasets(config: Dict):
    img_dir = Path(config["img_dir"])
    depth_dir = Path(config["depth_dir"])
    pairing = str(config.get("pairing_mode", "prefix")).lower()

    if pairing == "magic":
        all_pairs = build_pairs_magic(
            img_dir=img_dir,
            depth_dir=depth_dir,
            image_suffix=config["img_glob"],
            depth_suffixes_to_try=config.get("depth_suffixes_to_try"),
        )
    else:
        all_pairs = build_pairs(
            img_dir=img_dir,
            depth_dir=depth_dir,
            img_glob=config["img_glob"],
            depth_glob=config["depth_glob"],
        )

    if len(all_pairs) == 0:
        raise ValueError("No matched image-depth pairs found.")

    train_pairs, val_pairs = split_pairs(
        pairs=all_pairs,
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )

    mean, std, depth_mean, depth_std = compute_train_stats(
        train_pairs=train_pairs,
        selected_bands=config["selected_bands"],
        normalize=config["normalize"],
        normalize_depth=config["normalize_depth"],
        depth_min=config["depth_min"],
        depth_max=config["depth_max"],
        invalid_depth_values=config["invalid_depth_values"],
        image_mode=config.get("image_mode", "s2"),
        reflectance_scale=float(config.get("reflectance_scale", 1.0)),
        magic_negative_depth=bool(config.get("magic_negative_depth", False)),
    )

    ds_kw = dict(
        selected_bands=config["selected_bands"],
        normalize=config["normalize"],
        mean=mean,
        std=std,
        normalize_depth=config["normalize_depth"],
        depth_mean=depth_mean,
        depth_std=depth_std,
        depth_min=config["depth_min"],
        depth_max=config["depth_max"],
        invalid_depth_values=config["invalid_depth_values"],
        return_metadata=False,
        pairing_mode=pairing,
        img_glob=config["img_glob"],
        depth_glob=config["depth_glob"],
        depth_suffixes_to_try=config.get("depth_suffixes_to_try"),
        image_mode=config.get("image_mode", "s2"),
        reflectance_scale=float(config.get("reflectance_scale", 1.0)),
        magic_negative_depth=bool(config.get("magic_negative_depth", False)),
    )

    train_dataset = make_dataset_with_fixed_pairs(
        str(img_dir),
        str(depth_dir),
        train_pairs,
        **ds_kw,
    )
    val_dataset = make_dataset_with_fixed_pairs(
        str(img_dir),
        str(depth_dir),
        val_pairs,
        **ds_kw,
    )

    stats = {
        "mean": mean.tolist() if mean is not None else None,
        "std": std.tolist() if std is not None else None,
        "depth_mean": float(depth_mean) if depth_mean is not None else None,
        "depth_std": float(depth_std) if depth_std is not None else None,
        "num_all_pairs": len(all_pairs),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
    }

    return train_dataset, val_dataset, stats


def save_checkpoint(path, model, optimizer, epoch, best_val_rmse, config):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_rmse": best_val_rmse,
        "config": config,
    }
    torch.save(ckpt, path)


def cnn_train_config_from_yaml(full: Dict[str, Any]) -> Dict[str, Any]:
    d = full.get("data", {}) or {}
    t = full.get("train", {}) or {}
    m = full.get("model", {}) or {}
    l = full.get("logging", {}) or {}
    pairing = str(d.get("pairing", "magic")).lower().strip()
    imode = str(d.get("image_mode", "rgb")).lower().strip()
    hc = m.get("hidden_channels", [32, 64, 64, 32])
    if isinstance(hc, tuple):
        hc = list(hc)
    return {
        "img_dir": d["image_dir"],
        "depth_dir": d["depth_dir"],
        "pairing_mode": pairing,
        "img_glob": d.get("image_suffix") or d.get("img_glob") or "img_*.tif",
        "depth_glob": d.get("depth_glob", "depth_*.tif"),
        "depth_suffixes_to_try": d.get("depth_suffixes_to_try"),
        "image_mode": imode,
        "reflectance_scale": float(d.get("reflectance_scale", 255.0)),
        "magic_negative_depth": bool(d.get("magic_negative_depth_valid", True)),
        "selected_bands": (None if imode == "rgb" else d.get("selected_bands")),
        "normalize": bool(t.get("normalize", True)),
        "normalize_depth": bool(t.get("normalize_depth", True)),
        "depth_min": t.get("depth_min"),
        "depth_max": t.get("depth_max"),
        "invalid_depth_values": t.get("invalid_depth_values") or [],
        "batch_size": int(t.get("batch_size", 8)),
        "num_workers": int(t.get("num_workers", 0)),
        "epochs": int(t.get("epochs", 50)),
        "learning_rate": float(t.get("learning_rate", 1e-4)),
        "weight_decay": float(t.get("weight_decay", 1e-4)),
        "val_ratio": float(t.get("val_ratio", 0.2)),
        "seed": int(t.get("seed", 42)),
        "patience": int(t.get("patience", 25)),
        "hidden_channels": tuple(int(x) for x in hc),
        "use_batchnorm": bool(m.get("use_batchnorm", False)),
        "dropout": float(m.get("dropout", 0.05)),
        "use_coordconv": bool(m.get("use_coordconv", True)),
        "norm_type": str(m.get("norm_type", "group")),
        "num_groups": int(m.get("num_groups", 8)),
        "save_dir": str(l.get("save_dir", "checkpoints_cnn_rgb")),
        "best_model_name": str(l.get("best_model_name", "best_model.pt")),
        "last_model_name": str(l.get("last_model_name", "last_model.pt")),
        "device_pref": str(t.get("device", "cpu")),
        "gpu_id": int(t.get("gpu_id", 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(_CNN_ROOT / "config.yaml"),
        help="YAML config (same env vars as new_test: IMAGE_DIR, DEPTH_DIR, …).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override train.device from YAML (cpu | cuda | auto).",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_CNN_ROOT / ".env")
    except ImportError:
        pass

    raw = load_yaml_config(args.config)
    config = cnn_train_config_from_yaml(raw)

    os.makedirs(config["save_dir"], exist_ok=True)
    set_seed(config["seed"])

    dev_pref = args.device if args.device is not None else config["device_pref"]
    device = pick_torch_device(str(dev_pref), int(config["gpu_id"]))
    print(f"Using device: {device}")

    train_dataset, val_dataset, stats = build_datasets(config)

    config["mean"] = stats["mean"]
    config["std"] = stats["std"]
    config["depth_mean"] = stats["depth_mean"]
    config["depth_std"] = stats["depth_std"]

    depth_std = float(config["depth_std"]) if config["depth_std"] is not None else 1.0

    print(f"Pairing                   : {config['pairing_mode']}")
    print(f"Image mode                : {config['image_mode']}")
    print(f"Magic negative depth GT   : {config['magic_negative_depth']}")
    print(f"Reflectance scale         : {config['reflectance_scale']}")
    print(f"Matched samples (all pairs): {stats['num_all_pairs']}")
    print(f"Train samples             : {stats['num_train_pairs']}")
    print(f"Val samples               : {stats['num_val_pairs']}")
    print(f"Image mean (train only)   : {config['mean']}")
    print(f"Image std (train only)    : {config['std']}")
    print(f"Depth mean (train only)   : {config['depth_mean']}")
    print(f"Depth std (train only)    : {config['depth_std']}")
    print(f"Depth min/max filter      : {config['depth_min']}, {config['depth_max']}")
    print(f"Invalid depth values      : {config['invalid_depth_values']}")
    print(f"Model hidden channels     : {config['hidden_channels']}")
    print(f"Use CoordConv             : {config['use_coordconv']}")
    print(f"Norm type                 : {config['norm_type']}")
    print(f"Num groups                : {config['num_groups']}")

    if str(config["image_mode"]).lower() == "rgb":
        in_channels = 3
        print("Using RGB (3 channels).")
    elif config["selected_bands"] is None:
        in_channels = 13
        print("Using all 13 Sentinel-2 bands.")
    else:
        in_channels = len(config["selected_bands"])
        print(f"Using selected bands: {config['selected_bands']}")

    pin = device.type == "cuda"
    nw = int(config["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
    )

    model = SimpleBathymetryCNN(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        use_batchnorm=config["use_batchnorm"],
        dropout=config["dropout"],
        use_coordconv=config.get("use_coordconv", True),
        norm_type=config.get("norm_type", "group"),
        num_groups=config.get("num_groups", 8),
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7,
    )

    best_val_rmse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    best_model_path = str(Path(config["save_dir"]) / config["best_model_name"])
    last_model_path = str(Path(config["save_dir"]) / config["last_model_name"])

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            depth_std=depth_std,
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=masked_combined_loss,
        )

        # evaluate() returns metrics in normalized space if normalize_depth=True
        val_rmse_norm = float(val_metrics["rmse"])
        val_mae_norm = float(val_metrics["mae"])
        val_std_norm = float(val_metrics["std"])

        val_rmse_m = val_rmse_norm * depth_std
        val_mae_m = val_mae_norm * depth_std
        val_std_m = val_std_norm * depth_std

        scheduler.step(val_rmse_norm)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{config['epochs']}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Train RMSE: {train_metrics['rmse']:.6f} m | "
            f"Train MAE: {train_metrics['mae']:.6f} m | "
            f"Val Loss: {val_metrics.get('loss', float('nan')):.6f} | "
            f"Val RMSE: {val_rmse_m:.6f} m | "
            f"Val MAE: {val_mae_m:.6f} m | "
            f"Val STD: {val_std_m:.6f} m | "
            f"Valid px train: {int(train_metrics['num_valid_pixels'])}"
        )

        save_checkpoint(
            path=last_model_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_rmse=best_val_rmse,
            config=config,
        )

        if val_rmse_m < best_val_rmse:
            best_val_rmse = val_rmse_m
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_rmse=best_val_rmse,
                config=config,
            )

            print(
                f"  -> New best model saved at epoch {epoch} "
                f"with Val RMSE = {best_val_rmse:.6f} m"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["patience"]:
            print(
                f"Early stopping triggered after {epoch} epochs. "
                f"Best epoch: {best_epoch}, best Val RMSE: {best_val_rmse:.6f} m"
            )
            break

    print("=" * 80)
    print("Training finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation RMSE: {best_val_rmse:.6f} m")
    print(f"Best model saved to: {best_model_path}")
    print(f"Last model saved to: {last_model_path}")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except MemoryError as e:
        print(f"MemoryError: {e}")
        print(
            "Lower train.batch_size in cnn_src/config.yaml (try 1 for 720x720 RGB on CPU). "
            "If the process vanishes with no message, Windows may have killed it for RAM (OOM)."
        )
        raise
    except Exception:
        traceback.print_exc()
        raise