from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import create_experiment_folder, load_yaml_config, pick_torch_device, save_config, set_seed
from dataset import create_dataloaders
from model import LLMGuidedBathymetryModel


def build_model(config: Dict[str, Any]) -> nn.Module:
    return LLMGuidedBathymetryModel(config)


def make_run_name(config: Dict[str, Any], seed: int | None = None) -> str:
    suffix = f"_seed{seed}" if seed is not None else ""
    return f"vlm_bathymetry_autoencoder{suffix}"


def forward_model(model: nn.Module, batch: Dict[str, Any], device: torch.device):
    nb = device.type == "cuda"
    return model(
        batch["image"].to(device, non_blocking=nb),
        reliability_mask=batch["reliability_mask"].to(device, non_blocking=nb),
        disturbance_masks=batch["disturbance_masks"].to(device, non_blocking=nb),
        text_embeddings=batch["text_embeddings"].to(device, non_blocking=nb),
        region_valid_mask=batch["region_valid_mask"].to(device, non_blocking=nb),
        prior_depth_map=batch["prior_depth_map"].to(device, non_blocking=nb),
        prior_valid_mask=batch["prior_valid_mask"].to(device, non_blocking=nb),
        prior_confidence=batch["prior_confidence"].to(device, non_blocking=nb),
        water_mask=batch["water_mask"].to(device, non_blocking=nb),
        depth_gt=batch["depth"].to(device, non_blocking=nb),
        valid_mask=batch["valid_mask"].to(device, non_blocking=nb),
    )


def _valid_errors(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask > 0
    if valid.sum() == 0:
        return np.zeros((0,), dtype=np.float32)
    return (y_pred[valid] - y_true[valid]).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    err = _valid_errors(y_true, y_pred, mask)
    if err.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "error_std": 0.0}
    abs_err = np.abs(err)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "error_std": float(abs_err.std()),
    }


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(1, int(total_epochs))

    def step(self, epoch_index: int) -> float:
        epoch = int(epoch_index) + 1
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            lr = self.base_lr * epoch / float(self.warmup_epochs)
        else:
            denom = max(1, self.total_epochs - self.warmup_epochs)
            progress = min(1.0, max(0.0, (epoch - self.warmup_epochs) / float(denom)))
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def tensor_value(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    if not torch.is_tensor(value):
        return float(value)
    return float(value.detach().mean().item()) if value.numel() > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {
        "loss": 0.0,
        "nll_loss": 0.0,
        "align_loss": 0.0,
        "recon_loss": 0.0,
        "alpha_mean": 0.0,
        "var_mean": 0.0,
    }
    pred_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    mask_all: List[np.ndarray] = []
    num_batches = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        pred, info = forward_model(model, batch, device)
        loss = info["total"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        totals["loss"] += tensor_value(loss)
        totals["nll_loss"] += tensor_value(info.get("nll_loss"))
        totals["align_loss"] += tensor_value(info.get("align_loss"))
        totals["recon_loss"] += tensor_value(info.get("recon_loss"))
        totals["alpha_mean"] += tensor_value(info.get("alpha"))
        totals["var_mean"] += tensor_value(info.get("var"))
        pred_all.append(pred.detach().cpu().numpy())
        gt_all.append(batch["depth"].cpu().numpy())
        mask_all.append(batch["valid_mask"].cpu().numpy())
        num_batches += 1

    n = max(num_batches, 1)
    metrics = compute_metrics(np.concatenate(gt_all), np.concatenate(pred_all), np.concatenate(mask_all))
    return {**{k: v / n for k, v in totals.items()}, **metrics}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: Iterable[Dict[str, Any]], device: torch.device) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {
        "loss": 0.0,
        "nll_loss": 0.0,
        "align_loss": 0.0,
        "recon_loss": 0.0,
        "alpha_mean": 0.0,
        "var_mean": 0.0,
    }
    pred_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    mask_all: List[np.ndarray] = []
    num_batches = 0

    for batch in loader:
        pred, info = forward_model(model, batch, device)
        totals["loss"] += tensor_value(info.get("total"))
        totals["nll_loss"] += tensor_value(info.get("nll_loss"))
        totals["align_loss"] += tensor_value(info.get("align_loss"))
        totals["recon_loss"] += tensor_value(info.get("recon_loss"))
        totals["alpha_mean"] += tensor_value(info.get("alpha"))
        totals["var_mean"] += tensor_value(info.get("var"))
        pred_all.append(pred.detach().cpu().numpy())
        gt_all.append(batch["depth"].cpu().numpy())
        mask_all.append(batch["valid_mask"].cpu().numpy())
        num_batches += 1

    n = max(num_batches, 1)
    metrics = compute_metrics(np.concatenate(gt_all), np.concatenate(pred_all), np.concatenate(mask_all))
    return {**{k: v / n for k, v in totals.items()}, **metrics}


@torch.inference_mode()
def save_predictions(model: nn.Module, loader: Iterable[Dict[str, Any]], device: torch.device, out_dir: Path) -> None:
    rows = []
    best = {"mae": float("inf")}
    for batch in loader:
        pred, info = forward_model(model, batch, device)
        gt = batch["depth"].cpu().numpy()
        mask = batch["valid_mask"].cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        sample_ids = batch["sample_id"]
        for idx, sample_id in enumerate(sample_ids):
            metrics = compute_metrics(gt[idx : idx + 1], pred_np[idx : idx + 1], mask[idx : idx + 1])
            rows.append({"sample_id": sample_id, **metrics})
            if metrics["mae"] < best["mae"]:
                best = {
                    "mae": metrics["mae"],
                    "sample_id": sample_id,
                    "pred": pred_np[idx],
                    "gt": gt[idx],
                    "mask": mask[idx],
                    "mu": info["mu"][idx].detach().cpu().numpy(),
                    "var": info["var"][idx].detach().cpu().numpy(),
                    "alpha": info["alpha"][idx].detach().cpu().numpy(),
                    "d_phys": info["d_phys"][idx].detach().cpu().numpy(),
                    "reconstruction": info["reconstruction"][idx].detach().cpu().numpy(),
                    "M": batch["reliability_mask"][idx].cpu().numpy(),
                    "water_mask": batch["water_mask"][idx].cpu().numpy(),
                }
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    if best.get("sample_id") is not None:
        with open(out_dir / "best_sample_id.txt", "w", encoding="utf-8") as f:
            f.write(str(best["sample_id"]))
        for key, value in best.items():
            if isinstance(value, np.ndarray):
                np.save(out_dir / f"best_{key}.npy", value)


def dataloaders_from_config(config: Dict[str, Any], seed: int):
    data_cfg = config.get("data", {})
    semantic_cfg = config.get("semantic", {})
    split_cfg = config.get("split", {})
    train_cfg = config.get("train", {})
    text_cfg = config.get("text_encoder", {})
    batch_size = int(train_cfg.get("batch_size", data_cfg.get("batch_size", 8)))
    return create_dataloaders(
        image_dir=data_cfg["image_dir"],
        depth_dir=data_cfg["depth_dir"],
        modality=data_cfg.get("modality", "rgb"),
        image_mode=str(data_cfg.get("image_mode", "rgb")),
        image_size=data_cfg.get("image_size"),
        batch_size=batch_size,
        num_workers=int(data_cfg.get("num_workers", 0)),
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.10)),
        test_ratio=float(split_cfg.get("test_ratio", 0.20)),
        seed=seed,
        selected_bands=data_cfg.get("selected_bands"),
        reflectance_scale=float(data_cfg.get("reflectance_scale", 255.0)),
        image_suffix=data_cfg.get("image_suffix", "img_*.tif"),
        depth_suffixes_to_try=data_cfg.get("depth_suffixes_to_try"),
        semantic_dir=semantic_cfg.get("semantic_dir"),
        require_semantic_if_enabled=bool(semantic_cfg.get("require_semantic_if_enabled", True)),
        reliability_suffix=semantic_cfg.get("reliability_suffix", "_M.npy"),
        disturbance_masks_suffix=semantic_cfg.get("disturbance_masks_suffix", "_R.npy"),
        depth_prior_suffix=semantic_cfg.get("depth_prior_suffix", "_prior.npy"),
        depth_prior_valid_suffix=semantic_cfg.get("depth_prior_valid_suffix", "_prior_valid.npy"),
        depth_prior_conf_suffix=semantic_cfg.get("depth_prior_conf_suffix", "_prior_conf.npy"),
        text_embeddings_suffix=semantic_cfg.get("text_embeddings_suffix", "_text_embeddings.npy"),
        region_texts_suffix=semantic_cfg.get("region_texts_suffix", "_region_texts.json"),
        water_suffix=semantic_cfg.get("water_suffix", "_water.npy"),
        text_dim=int(text_cfg.get("output_dim", config.get("model", {}).get("text_dim", 384))),
    )


def run_seed(config: Dict[str, Any], seed: int, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    set_seed(seed)
    exp_dir_s, run_id = create_experiment_folder(
        config.get("logging", {}).get("base_dir", "logs"),
        make_run_name(config, seed),
    )
    exp_dir = Path(exp_dir_s)
    save_config(config, str(exp_dir / "config_used.yaml"))

    train_loader, val_loader, test_loader = dataloaders_from_config(config, seed)
    model = build_model(config).to(device)
    train_cfg = config.get("train", {})
    lr = float(train_cfg.get("learning_rate", 1.0e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    num_epochs = int(train_cfg.get("num_epochs", 100))
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=lr,
        warmup_epochs=int(train_cfg.get("warmup_epochs", 5)),
        total_epochs=num_epochs,
    )
    grad_clip = float(train_cfg.get("gradient_clip_norm", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 15))
    best_path = exp_dir / "best_model.pt"
    train_log_path = exp_dir / "train_log.csv"
    best_val_mae = float("inf")
    patience_counter = 0

    if args.train:
        for epoch in range(num_epochs):
            start = time.time()
            current_lr = scheduler.step(epoch)
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
            val_metrics = evaluate(model, val_loader, device)
            row = {
                "seed": seed,
                "epoch": epoch + 1,
                "lr": current_lr,
                "time_sec": round(time.time() - start, 3),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            pd.DataFrame([row]).to_csv(
                train_log_path,
                mode="a",
                header=not train_log_path.exists(),
                index=False,
            )
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "seed": seed,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )
            else:
                patience_counter += 1
            print(
                f"seed={seed} epoch={epoch + 1} train_mae={train_metrics['mae']:.4f} "
                f"val_mae={val_metrics['mae']:.4f} "
                f"nll={train_metrics['nll_loss']:.4f} "
                f"align={train_metrics['align_loss']:.4f} "
                f"recon={train_metrics['recon_loss']:.4f} "
                f"loss={train_metrics['loss']:.4f}"
            )
            if patience_counter >= patience:
                break

    if args.test and best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    result = {"seed": seed, "run_id": run_id, **test_metrics}
    pd.DataFrame([result]).to_csv(exp_dir / "metrics.csv", index=False)
    save_predictions(model, test_loader, device, exp_dir)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, exp_dir / "last_model.pt")
    print(f"seed={seed} test_mae={test_metrics['mae']:.4f} test_rmse={test_metrics['rmse']:.4f} dir={exp_dir}")
    return result


def aggregate_results(results: List[Dict[str, float]], output_dir: Path) -> None:
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    metric_cols = [
        col
        for col in [
            "mae",
            "rmse",
            "error_std",
            "loss",
            "nll_loss",
            "align_loss",
            "recon_loss",
        ]
        if col in df
    ]
    rows = []
    for col in metric_cols:
        rows.append({"metric": col, "mean": float(df[col].mean()), "std": float(df[col].std(ddof=0))})
    pd.DataFrame(rows).to_csv(output_dir / "aggregate_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    train_cfg = config.get("train", {})
    device_pref = args.device if args.device is not None else train_cfg.get("device", "auto")
    device = pick_torch_device(str(device_pref), int(args.gpu_id))
    seeds = [int(s) for s in train_cfg.get("seeds", [train_cfg.get("seed", 42)])]
    print(f"device={device} seeds={seeds}")

    results = [run_seed(config, seed, device, args) for seed in seeds]
    aggregate_dir = Path(config.get("logging", {}).get("base_dir", "logs")) / "llm_guided_bathymetry_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_results(results, aggregate_dir)
    print(f"Aggregate metrics saved to: {aggregate_dir}")


if __name__ == "__main__":
    main()
