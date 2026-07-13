#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LOSS_COLUMNS = [
    ("train_loss", "val_loss", "Total loss"),
    ("train_nll_loss", "val_nll_loss", "NLL loss"),
    ("train_align_loss", "val_align_loss", "Align loss"),
]

METRIC_COLUMNS = [
    ("train_mae", "val_mae", "MAE"),
    ("train_rmse", "val_rmse", "RMSE"),
]

AUX_COLUMNS = [
    ("train_alpha_mean", "val_alpha_mean", "Alpha mean"),
    ("train_var_mean", "val_var_mean", "Variance mean"),
    ("lr", None, "Learning rate"),
]


def find_train_logs(logs_dir: Path, run_dir: Optional[Path] = None) -> List[Path]:
    if run_dir is not None:
        path = run_dir / "train_log.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing train_log.csv in {run_dir}")
        return [path]

    paths = sorted(logs_dir.glob("llm_guided_bathymetry_seed*/**/train_log.csv"))
    if not paths:
        paths = sorted(logs_dir.glob("**/train_log.csv"))
    if not paths:
        raise FileNotFoundError(f"No train_log.csv found under {logs_dir}")
    return paths


def _plot_pair(
    ax: plt.Axes,
    df: pd.DataFrame,
    train_col: str,
    val_col: Optional[str],
    title: str,
    seed_label: str,
) -> None:
    epochs = df["epoch"]
    if train_col in df.columns:
        ax.plot(epochs, df[train_col], label=f"train ({seed_label})", linewidth=1.8)
    if val_col and val_col in df.columns:
        ax.plot(epochs, df[val_col], label=f"val ({seed_label})", linewidth=1.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_single_run(df: pd.DataFrame, out_path: Path, title_prefix: str = "") -> None:
    seed = int(df["seed"].iloc[0]) if "seed" in df.columns else 0
    seed_label = f"seed {seed}"
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()

    for ax, (train_col, val_col, title) in zip(axes[:3], LOSS_COLUMNS):
        _plot_pair(ax, df, train_col, val_col, title, seed_label)

    for ax, (train_col, val_col, title) in zip(axes[3:5], METRIC_COLUMNS):
        _plot_pair(ax, df, train_col, val_col, title, seed_label)

    ax = axes[5]
    if "lr" in df.columns:
        ax.plot(df["epoch"], df["lr"], color="tab:purple", linewidth=1.8)
        ax.set_title("Learning rate")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")

    prefix = f"{title_prefix} " if title_prefix else ""
    fig.suptitle(f"{prefix}{seed_label}", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_seed(log_paths: Sequence[Path], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        (axes[0, 0], "train_loss", "val_loss", "Total loss"),
        (axes[0, 1], "train_nll_loss", "val_nll_loss", "NLL loss"),
        (axes[1, 0], "train_mae", "val_mae", "MAE"),
        (axes[1, 1], "train_rmse", "val_rmse", "RMSE"),
    ]

    for log_path in log_paths:
        df = pd.read_csv(log_path)
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else 0
        label = f"seed {seed}"
        for ax, train_col, val_col, title in panels:
            if train_col in df.columns:
                ax.plot(df["epoch"], df[train_col], label=f"train {label}", linewidth=1.5)
            if val_col in df.columns:
                ax.plot(df["epoch"], df[val_col], label=f"val {label}", linewidth=1.5, linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)

    for ax in axes.ravel():
        ax.legend(fontsize=7)

    fig.suptitle("Training curves (all seeds)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_test_summary(aggregate_dir: Path, out_path: Path) -> None:
    per_seed = aggregate_dir / "per_seed_metrics.csv"
    agg = aggregate_dir / "aggregate_metrics.csv"
    if not per_seed.exists():
        return

    df = pd.read_csv(per_seed)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [c for c in ["mae", "rmse", "error_std"] if c in df.columns]
    for ax, metric in zip(axes, metrics):
        seeds = df["seed"].astype(str) if "seed" in df.columns else df.index.astype(str)
        ax.bar(seeds, df[metric], color="steelblue", alpha=0.85)
        ax.set_title(f"Test {metric.upper()}")
        ax.set_xlabel("Seed")
        ax.grid(True, axis="y", alpha=0.3)
        if agg.exists() and metric in pd.read_csv(agg)["metric"].values:
            row = pd.read_csv(agg).set_index("metric").loc[metric]
            ax.axhline(row["mean"], color="crimson", linestyle="--", linewidth=1.2, label=f"mean={row['mean']:.3f}")
            ax.legend(fontsize=8)

    fig.suptitle("Test metrics per seed", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from main.py train_log.csv")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--run_dir", type=str, default=None, help="Specific experiment folder")
    parser.add_argument("--out_dir", type=str, default="train_viz")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    run_dir = Path(args.run_dir) if args.run_dir else None

    log_paths = find_train_logs(logs_dir, run_dir)
    for log_path in log_paths:
        df = pd.read_csv(log_path)
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else 0
        run_name = log_path.parent.name
        plot_single_run(df, out_dir / f"train_curves_{run_name}_seed{seed}.png")

    if len(log_paths) > 1:
        plot_multi_seed(log_paths, out_dir / "train_curves_all_seeds.png")

    aggregate_dir = logs_dir / "llm_guided_bathymetry_aggregate"
    plot_test_summary(aggregate_dir, out_dir / "test_metrics_per_seed.png")

    print(f"Saved plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
