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

from dataset import BathymetryDataset, build_pairs_magic, compute_channel_stats, compute_depth_stats
from eval import evaluate, masked_mse
from model import DASDB


PairType = Tuple[Path, Path, str]

_ROOT = Path(__file__).resolve().parent


def _resolve_data_section(raw_d: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(raw_d)
    if not d.get("source_image_dir"):
        sid = d.get("train_image_dir") or d.get("image_dir")
        if sid:
            d["source_image_dir"] = sid
    if not d.get("source_depth_dir"):
        sdd = d.get("depth_dir")
        if sdd:
            d["source_depth_dir"] = sdd
    if not d.get("source_image_dir"):
        raise KeyError("data.source_image_dir (or data.train_image_dir / data.image_dir) is required.")
    if not d.get("source_depth_dir"):
        raise KeyError("data.source_depth_dir (or data.depth_dir) is required.")
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
    depth_glob: str,
    depth_suffixes_to_try: Optional[Sequence[str]],
    image_mode: str,
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
        depth_glob=depth_glob,
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
        image_mode=image_mode,
        reflectance_scale=reflectance_scale,
        magic_negative_depth=magic_negative_depth,
    )
    ds.pairs = list(pairs_subset)
    return ds


def train_one_epoch(
    model: DASDB,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    depth_std: float,
    lambda_domain: float,
    grl_alpha: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_depth_loss = 0.0
    total_domain_loss = 0.0
    total_valid = 0.0
    total_sq_error = 0.0
    total_abs_error = 0.0

    bce = torch.nn.BCEWithLogitsLoss()
    target_iter = iter(target_loader)

    nb = device.type == "cuda"
    for batch_s in source_loader:
        try:
            batch_t = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            batch_t = next(target_iter)

        xs = batch_s["image"].to(device, non_blocking=nb)
        ys = batch_s["depth"].to(device, non_blocking=nb)
        ms = batch_s["valid_mask"].to(device, non_blocking=nb)
        xt = batch_t["image"].to(device, non_blocking=nb)

        xs = torch.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
        ys = torch.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0)
        ms = torch.nan_to_num(ms, nan=0.0, posinf=0.0, neginf=0.0)
        xt = torch.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0)

        optimizer.zero_grad(set_to_none=True)

        pred_s = model.forward_depth(xs)
        pred_s = torch.nan_to_num(pred_s, nan=0.0, posinf=0.0, neginf=0.0)
        depth_loss = masked_combined_loss(pred_s, ys, ms, alpha=0.3)

        logit_s = model.forward_domain(xs, alpha=grl_alpha)
        logit_t = model.forward_domain(xt, alpha=grl_alpha)
        dom_y_s = torch.zeros_like(logit_s)
        dom_y_t = torch.ones_like(logit_t)
        domain_loss = 0.5 * (bce(logit_s, dom_y_s) + bce(logit_t, dom_y_t))

        loss = depth_loss + float(lambda_domain) * domain_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            residual = pred_s - ys
            sq_error = ((residual ** 2) * ms).sum().item()
            abs_error = (residual.abs() * ms).sum().item()
            num_valid = ms.sum().item()

        total_loss += float(loss.detach().cpu())
        total_depth_loss += float(depth_loss.detach().cpu())
        total_domain_loss += float(domain_loss.detach().cpu())
        total_sq_error += sq_error
        total_abs_error += abs_error
        total_valid += num_valid

    mse_norm = total_sq_error / max(total_valid, 1.0)
    rmse_norm = mse_norm ** 0.5
    mae_norm = total_abs_error / max(total_valid, 1.0)
    return {
        "loss": total_loss / max(len(source_loader), 1),
        "depth_loss": total_depth_loss / max(len(source_loader), 1),
        "domain_loss": total_domain_loss / max(len(source_loader), 1),
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
        help="Labeled RGB folder (source); pairs depth only when a file exists in source depth dir (image-first, same as CNN magic pairing).",
    )
    parser.add_argument(
        "--depth_dir",
        type=str,
        default=None,
        help="Ground-truth depth folder for the source domain (maps to source_depth_dir).",
    )
    parser.add_argument(
        "--target_image_dir",
        type=str,
        default=None,
        help="Unlabeled (or extra) RGB folder for the target domain.",
    )
    parser.add_argument(
        "--target_depth_dir",
        type=str,
        default=None,
        help="Depth dir used to resolve target pairs via magic pairing (defaults to source depth dir).",
    )
    parser.add_argument(
        "--image_suffix",
        type=str,
        default=None,
        help="Glob for RGB files under each image dir (e.g. img_*.tif or *.tif).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except Exception:
        pass

    raw = load_yaml_config(args.config)
    d = _resolve_data_section(raw.get("data", {}) or {})
    if args.train_image_dir is not None:
        d["source_image_dir"] = str(args.train_image_dir)
    if args.depth_dir is not None:
        d["source_depth_dir"] = str(args.depth_dir)
    if args.target_image_dir is not None:
        d["target_image_dir"] = str(args.target_image_dir)
    if args.target_depth_dir is not None:
        d["target_depth_dir"] = str(args.target_depth_dir)
    if args.image_suffix is not None:
        d["image_suffix"] = str(args.image_suffix)
    raw = dict(raw)
    raw["data"] = d
    t = dict(raw.get("train", {}) or {})
    if args.epochs is not None:
        t["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        t["batch_size"] = int(args.batch_size)
    raw["train"] = t
    m = raw.get("model", {}) or {}
    l = raw.get("logging", {}) or {}

    if not d.get("target_image_dir"):
        raise KeyError("data.target_image_dir is required for domain-adaptation training.")

    os.makedirs(str(l.get("save_dir", "checkpoints_da_sdb")), exist_ok=True)
    set_seed(int(t.get("seed", 42)))

    dev_pref = args.device if args.device is not None else str(t.get("device", "cpu"))
    device = pick_torch_device(dev_pref, int(t.get("gpu_id", 0)))

    source_pairs = build_pairs_magic(
        img_dir=Path(d["source_image_dir"]),
        depth_dir=Path(d["source_depth_dir"]),
        image_suffix=str(d.get("image_suffix", "img_*.tif")),
        depth_suffixes_to_try=d.get("depth_suffixes_to_try"),
    )
    if len(source_pairs) == 0:
        raise ValueError("No matched image-depth pairs found for source domain.")

    target_pairs = build_pairs_magic(
        img_dir=Path(d["target_image_dir"]),
        depth_dir=Path(d.get("target_depth_dir") or d["source_depth_dir"]),
        image_suffix=str(d.get("image_suffix", "img_*.tif")),
        depth_suffixes_to_try=d.get("depth_suffixes_to_try"),
    )
    if len(target_pairs) == 0:
        raise ValueError("No target images found (target_image_dir) for domain adaptation.")

    train_pairs, val_pairs = split_pairs(source_pairs, float(t.get("val_ratio", 0.2)), int(t.get("seed", 42)))

    reflectance_scale = float(d.get("reflectance_scale", 255.0))
    magic_negative = bool(d.get("magic_negative_depth_valid", True))

    mean, std = compute_channel_stats(train_pairs, band_indices=[0, 1, 2], reflectance_scale=reflectance_scale) if bool(t.get("normalize", True)) else (None, None)

    depth_mean, depth_std = compute_depth_stats(train_pairs, magic_negative_valid=magic_negative) if bool(t.get("normalize_depth", True)) else (None, None)
    depth_std_val = float(depth_std) if depth_std is not None else 1.0

    train_ds = _build_dataset_from_pairs(
        train_pairs,
        img_dir=str(d["source_image_dir"]),
        depth_dir=str(d["source_depth_dir"]),
        img_glob=str(d.get("image_suffix", "img_*.tif")),
        depth_glob="depth_*.tif",
        depth_suffixes_to_try=d.get("depth_suffixes_to_try"),
        image_mode=str(d.get("image_mode", "rgb")),
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
        img_dir=str(d["source_image_dir"]),
        depth_dir=str(d["source_depth_dir"]),
        img_glob=str(d.get("image_suffix", "img_*.tif")),
        depth_glob="depth_*.tif",
        depth_suffixes_to_try=d.get("depth_suffixes_to_try"),
        image_mode=str(d.get("image_mode", "rgb")),
        reflectance_scale=reflectance_scale,
        magic_negative_depth=magic_negative,
        normalize=bool(t.get("normalize", True)),
        mean=(mean.tolist() if mean is not None else None),
        std=(std.tolist() if std is not None else None),
        normalize_depth=bool(t.get("normalize_depth", True)),
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    target_ds = BathymetryDataset(
        img_dir=str(d["target_image_dir"]),
        depth_dir=str(d.get("target_depth_dir") or d["source_depth_dir"]),
        img_glob=str(d.get("image_suffix", "img_*.tif")),
        depth_glob="depth_*.tif",
        selected_bands=None,
        normalize=bool(t.get("normalize", True)),
        mean=(mean.tolist() if mean is not None else None),
        std=(std.tolist() if std is not None else None),
        normalize_depth=bool(t.get("normalize_depth", True)),
        depth_mean=depth_mean,
        depth_std=depth_std,
        depth_min=None,
        depth_max=None,
        invalid_depth_values=[],
        return_metadata=False,
        pairing_mode="magic",
        depth_suffixes_to_try=d.get("depth_suffixes_to_try"),
        image_mode=str(d.get("image_mode", "rgb")),
        reflectance_scale=reflectance_scale,
        magic_negative_depth=magic_negative,
    )
    target_ds.pairs = [(p[0], p[1], p[2]) for p in target_pairs]

    pin = device.type == "cuda"
    nw = int(t.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=int(t.get("batch_size", 2)), shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=int(t.get("batch_size", 2)), shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    target_loader = DataLoader(target_ds, batch_size=int(t.get("batch_size", 2)), shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))

    hidden_channels = tuple(int(x) for x in (m.get("hidden_channels") or [32, 64, 64, 32]))
    model = DASDB(
        in_channels=3,
        hidden_channels=hidden_channels,
        dropout=float(m.get("dropout", 0.05)),
        use_coordconv=bool(m.get("use_coordconv", True)),
        norm_type=str(m.get("norm_type", "group")),
        num_groups=int(m.get("num_groups", 8)),
        domain_hidden=int(m.get("domain_hidden", 128)),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(t.get("learning_rate", 1e-4)), weight_decay=float(t.get("weight_decay", 1e-4)))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_val_rmse = float("inf")
    best_epoch = -1
    epochs_wo = 0

    save_dir = Path(str(l.get("save_dir", "checkpoints_da_sdb")))
    best_path = str(save_dir / str(l.get("best_model_name", "best_model.pt")))
    last_path = str(save_dir / str(l.get("last_model_name", "last_model.pt")))

    config_for_ckpt = {
        "data": d,
        "train": t,
        "model": m,
        "stats": {
            "mean": mean.tolist() if mean is not None else None,
            "std": std.tolist() if std is not None else None,
            "depth_mean": float(depth_mean) if depth_mean is not None else None,
            "depth_std": float(depth_std) if depth_std is not None else None,
        },
    }

    lambda_domain = float(t.get("lambda_domain", 0.1))
    grl_alpha = float(t.get("grl_alpha", 1.0))
    patience = int(t.get("patience", 25))

    for epoch in range(1, int(t.get("epochs", 50)) + 1):
        train_metrics = train_one_epoch(
            model=model,
            source_loader=train_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            device=device,
            depth_std=depth_std_val,
            lambda_domain=lambda_domain,
            grl_alpha=grl_alpha,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=masked_combined_loss,
        )

        val_rmse_norm = float(val_metrics["rmse"])
        val_mae_norm = float(val_metrics["mae"])
        val_rmse_m = val_rmse_norm * depth_std_val
        val_mae_m = val_mae_norm * depth_std_val

        scheduler.step(val_rmse_norm)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{int(t.get('epochs', 50))}] | LR: {lr:.6f} | "
            f"Train Loss: {train_metrics['loss']:.6f} | DepthLoss: {train_metrics['depth_loss']:.6f} | DomLoss: {train_metrics['domain_loss']:.6f} | "
            f"Train RMSE: {train_metrics['rmse']:.6f} m | Train MAE: {train_metrics['mae']:.6f} m | "
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

