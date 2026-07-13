from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import numpy as np

from bathymetry_llm.data.dataset import BathymetryDataset
from bathymetry_llm.utils.io import load_config, package_root, resolve_path
from bathymetry_llm.utils.visualization import save_prediction_figure


def _load_prediction(scene_id: str, pred_dir: Path, key: str) -> np.ndarray | None:
    path = pred_dir / f"{scene_id}_{key}.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prediction_dir", default=None)
    args = parser.parse_args()
    root = package_root()
    cfg = load_config(resolve_path(args.config, root))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    scene = Path(args.scene)
    if not scene.is_absolute():
        scene = root / scene
    dataset = BathymetryDataset(scene.parent, "all", data_cfg.get("image_size"), int(data_cfg.get("k_max", 8)), text_dim=int(data_cfg.get("text_dim", model_cfg.get("text_dim", 384))))
    matches = [i for i, p in enumerate(dataset.scenes) if p.resolve() == scene.resolve()]
    if not matches:
        raise FileNotFoundError(f"Scene not found: {scene}")
    sample = dataset[matches[0]]
    pred_dir = Path(args.prediction_dir) if args.prediction_dir else resolve_path(cfg.get("output", {}).get("prediction_dir", "outputs/predictions"), root)
    fig_dir = resolve_path(cfg.get("output", {}).get("figure_dir", "outputs/figures"), root) / sample["scene_id"]
    depth_pred = _load_prediction(sample["scene_id"], pred_dir, "depth")
    var = _load_prediction(sample["scene_id"], pred_dir, "var")
    alpha = _load_prediction(sample["scene_id"], pred_dir, "alpha")
    if depth_pred is None:
        depth_pred = sample["d_phys"].numpy()
    if var is None:
        var = np.zeros_like(depth_pred)
    if alpha is None:
        alpha = np.zeros_like(depth_pred)
    save_prediction_figure(sample["image"].numpy(), sample["depth"].numpy(), depth_pred, sample["valid_mask"].numpy(), var, alpha, sample["unreliable_mask"].numpy(), sample["d_phys"].numpy(), fig_dir / "summary.png")
    print(fig_dir / "summary.png")


if __name__ == "__main__":
    main()
