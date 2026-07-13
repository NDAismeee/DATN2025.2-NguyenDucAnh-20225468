import argparse
import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from common import load_yaml_config, pick_torch_device
from dataset import BathymetryDataset, build_pairs_magic
from eval import evaluate, masked_mse
from model import build_model_from_config, load_pretrained_weights, save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_l1(pred, target, valid_mask, eps: float = 1e-8):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = torch.nan_to_num(valid_mask, nan=0.0, posinf=0.0, neginf=0.0)
    abs_error = torch.abs(pred - target) * valid_mask
    return abs_error.sum() / (valid_mask.sum() + eps)


def masked_combined_loss(pred, target, valid_mask, alpha: float = 0.3):
    return alpha * masked_mse(pred, target, valid_mask) + (1.0 - alpha) * masked_l1(pred, target, valid_mask)


def train_one_epoch(model, dataloader, optimizer, device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

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
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at epoch={epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=root / ".env", override=True)
    cfg = load_yaml_config(args.config)

    data_cfg = cfg.get("data", {}) or {}
    tr_cfg = cfg.get("train", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}

    set_seed(int(tr_cfg.get("seed", 42)))
    device = pick_torch_device(str(tr_cfg.get("device", "auto")), gpu_id=int(tr_cfg.get("gpu_id", 0)))

    img_dir = Path(str(data_cfg.get("image_dir", ""))).expanduser()
    depth_dir = Path(str(data_cfg.get("depth_dir", ""))).expanduser()
    image_suffix = str(data_cfg.get("image_suffix", "img_*.tif"))
    depth_suffixes_to_try = data_cfg.get("depth_suffixes_to_try", ["_depth", "_bathy", "_gt", "_label"])
    reflectance_scale = float(data_cfg.get("reflectance_scale", 1.0))
    magic_negative_depth = bool(data_cfg.get("magic_negative_depth_valid", True))

    pairs = build_pairs_magic(
        img_dir=img_dir,
        depth_dir=depth_dir,
        image_suffix=image_suffix,
        depth_suffixes_to_try=depth_suffixes_to_try,
    )
    if not pairs:
        raise ValueError("No matched image-depth pairs found. Check IMAGE_DIR/DEPTH_DIR and config.yaml.")

    val_ratio = float(tr_cfg.get("val_ratio", 0.2))
    rng = np.random.default_rng(int(tr_cfg.get("seed", 42)))
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = int(round(len(pairs) * val_ratio))
    n_val = max(1, n_val) if len(pairs) >= 2 else 0
    val_idx = set(idx[:n_val].tolist())
    train_pairs = [p for k, p in enumerate(pairs) if k not in val_idx]
    val_pairs = [p for k, p in enumerate(pairs) if k in val_idx]
    if len(val_pairs) == 0:
        val_pairs = list(train_pairs)

    train_ds = BathymetryDataset(
        img_dir=str(img_dir),
        depth_dir=str(depth_dir),
        img_glob=image_suffix,
        depth_glob="depth_*.tif",
        selected_bands=None,
        normalize=False,
        mean=None,
        std=None,
        normalize_depth=False,
        pairing_mode="magic",
        depth_suffixes_to_try=depth_suffixes_to_try,
        image_mode=str(data_cfg.get("image_mode", "rgb")),
        reflectance_scale=reflectance_scale,
        magic_negative_depth=magic_negative_depth,
    )
    train_ds.pairs = train_pairs

    val_ds = BathymetryDataset(
        img_dir=str(img_dir),
        depth_dir=str(depth_dir),
        img_glob=image_suffix,
        depth_glob="depth_*.tif",
        selected_bands=None,
        normalize=False,
        mean=None,
        std=None,
        normalize_depth=False,
        pairing_mode="magic",
        depth_suffixes_to_try=depth_suffixes_to_try,
        image_mode=str(data_cfg.get("image_mode", "rgb")),
        reflectance_scale=reflectance_scale,
        magic_negative_depth=magic_negative_depth,
    )
    val_ds.pairs = val_pairs

    bs = int(tr_cfg.get("batch_size", 2))
    num_workers = int(tr_cfg.get("num_workers", 2))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model, weights_path = build_model_from_config(cfg)
    model.to(device)
    if weights_path:
        load_pretrained_weights(model, weights_path, device=device)

    lr = float(tr_cfg.get("learning_rate", 3e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    save_dir = Path(str(log_cfg.get("save_dir", "checkpoints_unet_pretrained"))).expanduser()
    best_name = str(log_cfg.get("best_name", "best_model.pt"))
    last_name = str(log_cfg.get("last_name", "last_model.pt"))

    best_rmse = float("inf")
    rows: List[Dict[str, Any]] = []

    epochs = int(tr_cfg.get("epochs", 10))
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch)
        val_metrics = evaluate(model=model, dataloader=val_loader, device=device, loss_fn=None)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_mae": float(val_metrics.get("mae", float("nan"))),
            "val_rmse": float(val_metrics.get("rmse", float("nan"))),
            "val_std": float(val_metrics.get("std", float("nan"))),
            "val_num_valid_pixels": float(val_metrics.get("num_valid_pixels", 0.0)),
        }
        rows.append(row)
        print(f"[unet] epoch={epoch} train_loss={tr_loss:.6f} val_mae={row['val_mae']:.6f} val_rmse={row['val_rmse']:.6f}", flush=True)

        save_checkpoint(save_dir / last_name, model=model, config=cfg, epoch=epoch, val_metrics={"rmse": row["val_rmse"], "mae": row["val_mae"]})
        if np.isfinite(row["val_rmse"]) and float(row["val_rmse"]) < best_rmse:
            best_rmse = float(row["val_rmse"])
            save_checkpoint(save_dir / best_name, model=model, config=cfg, epoch=epoch, val_metrics={"rmse": row["val_rmse"], "mae": row["val_mae"]})

    log_path = save_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {log_path}")
    print(f"Saved: {save_dir / best_name}")


if __name__ == "__main__":
    main()

