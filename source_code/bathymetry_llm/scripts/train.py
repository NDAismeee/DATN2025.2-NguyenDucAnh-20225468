from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from bathymetry_llm.data.dataset import bathymetry_collate_fn, build_bathymetry_dataset
from bathymetry_llm.models.llm_guided_bathymetry import LLMGuidedBathymetryModel
from bathymetry_llm.models.losses import compute_total_loss
from bathymetry_llm.utils.io import append_csv, load_config, package_root, pick_device, resolve_path, save_json, set_seed
from bathymetry_llm.utils.metrics import compute_metrics


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _forward(model: LLMGuidedBathymetryModel, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return model(
        image=batch["image"],
        unreliable_mask=batch["unreliable_mask"],
        d_phys=batch["d_phys"],
        region_masks=batch["region_masks"],
        text_embeddings=batch["text_embeddings"],
        gamma_map=batch["gamma_map"],
        w_phys=batch["w_phys"],
        region_valid_mask=batch["region_valid_mask"],
    )


def _metrics_from_batches(targets, preds, masks) -> Dict[str, float]:
    return compute_metrics(np.concatenate(targets, axis=0), np.concatenate(preds, axis=0), np.concatenate(masks, axis=0))


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(0, int(warmup_epochs))
    total_epochs = max(1, int(total_epochs))

    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, scale)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    lambda_align: float,
    lambda_int: float,
    align_tau: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    totals = {"total": 0.0, "nll": 0.0, "align": 0.0, "int": 0.0, "alpha": 0.0, "var": 0.0}
    targets, preds, masks = [], [], []
    n = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch)
        losses = compute_total_loss(outputs, batch, lambda_align=lambda_align, lambda_int=lambda_int, align_tau=align_tau)
        losses["total"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()
        for key in ("total", "nll", "align", "int"):
            totals[key] += float(losses[key].detach().cpu())
        totals["alpha"] += float(outputs["alpha"].detach().mean().cpu())
        totals["var"] += float(outputs["var"].detach().mean().cpu())
        targets.append(batch["depth"].detach().cpu().numpy())
        preds.append(outputs["depth"].detach().cpu().numpy())
        masks.append(batch["valid_mask"].detach().cpu().numpy())
        n += 1
    metrics = _metrics_from_batches(targets, preds, masks)
    for key in totals:
        metrics[key] = totals[key] / max(n, 1)
    return metrics


def evaluate_epoch(model, loader, device, lambda_align: float, lambda_int: float, align_tau: float) -> Dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "nll": 0.0, "align": 0.0, "int": 0.0, "alpha": 0.0, "var": 0.0}
    targets, preds, masks = [], [], []
    n = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            outputs = _forward(model, batch)
            losses = compute_total_loss(outputs, batch, lambda_align=lambda_align, lambda_int=lambda_int, align_tau=align_tau)
            for key in ("total", "nll", "align", "int"):
                totals[key] += float(losses[key].detach().cpu())
            totals["alpha"] += float(outputs["alpha"].detach().mean().cpu())
            totals["var"] += float(outputs["var"].detach().mean().cpu())
            targets.append(batch["depth"].detach().cpu().numpy())
            preds.append(outputs["depth"].detach().cpu().numpy())
            masks.append(batch["valid_mask"].detach().cpu().numpy())
            n += 1
    metrics = _metrics_from_batches(targets, preds, masks)
    for key in totals:
        metrics[key] = totals[key] / max(n, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    root = package_root()
    cfg = load_config(resolve_path(args.config, root))
    set_seed(int(cfg.get("training", {}).get("seed", 42)))
    device = pick_device(cfg.get("training", {}).get("device", "auto"))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    out_cfg = cfg.get("output", {})
    image_size = data_cfg.get("image_size")
    k_max = int(data_cfg.get("k_max", 8))
    text_dim = int(data_cfg.get("text_dim", model_cfg.get("text_dim", 384)))
    train_set = build_bathymetry_dataset(data_cfg, "train", image_size, k_max, text_dim, config_root=root)
    val_set = build_bathymetry_dataset(data_cfg, "val", image_size, k_max, text_dim, config_root=root)
    train_loader = DataLoader(
        train_set,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=bathymetry_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=bathymetry_collate_fn,
    )
    model = LLMGuidedBathymetryModel(
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        text_dim=text_dim,
        gate_hidden=int(model_cfg.get("gate_hidden", 128)),
        gate_dropout=float(model_cfg.get("gate_dropout", 0.1)),
    ).to(device)
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 100))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs, warmup_epochs, base_lr)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 15))
    checkpoint_dir = resolve_path(out_cfg.get("checkpoint_dir", "outputs/checkpoints"), root)
    metric_dir = resolve_path(out_cfg.get("metric_dir", "outputs/metrics"), root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    log_path = metric_dir / "train_log.csv"
    best_path = checkpoint_dir / "best.pt"
    best_mae = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    lambda_align = float(train_cfg.get("lambda_align", 1e-2))
    lambda_int = float(train_cfg.get("lambda_int", train_cfg.get("lambda_range", 1e-2)))
    if not bool(model_cfg.get("use_range_loss", True)):
        lambda_int = 0.0
    align_tau = float(model_cfg.get("align_tau", 0.07))
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer,
            lambda_align, lambda_int, align_tau, grad_clip,
        )
        val_metrics = evaluate_epoch(model, val_loader, device, lambda_align, lambda_int, align_tau)
        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": current_lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_csv(row, log_path)
        improved = val_metrics["mae"] < best_mae
        if improved:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": val_metrics},
                best_path,
            )
        else:
            epochs_since_improve += 1
        print(
            f"epoch={epoch} lr={current_lr:.2e} train_rmse={train_metrics['rmse']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} val_rmse={val_metrics['rmse']:.4f} "
            f"best_mae={best_mae:.4f}@{best_epoch} no_improve={epochs_since_improve}"
        )
        if early_stop_patience > 0 and epochs_since_improve >= early_stop_patience:
            print(f"early_stop epoch={epoch} no_improve={epochs_since_improve} best_mae={best_mae:.4f}@{best_epoch}")
            break
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, checkpoint_dir / "last.pt")
    save_json(
        {"best_mae": best_mae, "best_epoch": best_epoch, "best_checkpoint": str(best_path)},
        metric_dir / "training_summary.json",
    )


if __name__ == "__main__":
    main()
