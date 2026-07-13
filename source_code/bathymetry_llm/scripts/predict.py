from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import torch

from bathymetry_llm.data.dataset import BathymetryDataset
from bathymetry_llm.models.llm_guided_bathymetry import LLMGuidedBathymetryModel
from bathymetry_llm.utils.io import load_config, package_root, pick_device, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    root = package_root()
    scene = Path(args.scene)
    if not scene.is_absolute():
        scene = root / scene
    device = pick_device("auto")
    ckpt = torch.load(resolve_path(args.checkpoint, root), map_location=device)
    cfg = ckpt.get("config", load_config(root / "configs/default.yaml"))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    dataset = BathymetryDataset(scene.parent, "all", data_cfg.get("image_size"), int(data_cfg.get("k_max", 8)), text_dim=int(data_cfg.get("text_dim", model_cfg.get("text_dim", 384))))
    matches = [i for i, p in enumerate(dataset.scenes) if p.resolve() == scene.resolve()]
    if not matches:
        raise FileNotFoundError(f"Scene not found in dataset parent: {scene}")
    sample = dataset[matches[0]]
    model = LLMGuidedBathymetryModel(
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        text_dim=int(data_cfg.get("text_dim", model_cfg.get("text_dim", 384))),
        gate_hidden=int(model_cfg.get("gate_hidden", 128)),
        gate_dropout=float(model_cfg.get("gate_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    with torch.inference_mode():
        outputs = model(
            batch["image"],
            batch["unreliable_mask"],
            batch["d_phys"],
            batch["region_masks"],
            batch["text_embeddings"],
            batch["gamma_map"],
            batch["w_phys"],
            batch["region_valid_mask"],
        )
    out_dir = Path(args.output_dir) if args.output_dir else resolve_path(cfg.get("output", {}).get("prediction_dir", "outputs/predictions"), root)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_id = sample["scene_id"]
    for key in ("depth", "mu", "var", "alpha", "d_phys"):
        np.save(out_dir / f"{scene_id}_{key}.npy", outputs[key][0].detach().cpu().numpy())
    print(out_dir / f"{scene_id}_depth.npy")


if __name__ == "__main__":
    main()
