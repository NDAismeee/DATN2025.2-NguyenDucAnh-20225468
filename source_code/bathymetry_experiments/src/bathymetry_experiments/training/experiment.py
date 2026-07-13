from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bathymetry_experiments.common.config import load_config, save_config
from bathymetry_experiments.common.metrics import mean_metrics, regression_metrics
from bathymetry_experiments.common.utils import make_run_dir, set_seed, write_csv
from bathymetry_experiments.data.io import Pair, find_pairs, load_sample, pixel_features, sample_pixels, split_pairs

TABULAR_MODELS = {"knn", "random_forest", "mlp"}
TORCH_MODELS = {"proposed", "cnn", "unet", "depth_anything_v2", "da_sdb", "dpt"}
ALL_MODELS = sorted(TABULAR_MODELS | TORCH_MODELS)


def default_config() -> dict[str, Any]:
    return {
        "data": {
            "image_dir": "/mnt/disk3/anhnd2468/MagicBathyNet/agia_napa/img/aerial",
            "depth_dir": "/mnt/disk3/anhnd2468/MagicBathyNet/agia_napa/depth/aerial",
            "image_glob": "img_*.tif",
            "reflectance_scale": 255.0,
            "val_ratio": 0.2,
        },
        "train": {
            "seed": 42,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "pixels_per_image": 2500,
            "image_size": 128,
            "limit_images": 0,
        },
        "logging": {"base_dir": "runs"},
    }


def load_experiment_config(path: str | None) -> dict[str, Any]:
    config = default_config()
    user = load_config(path)
    return _deep_update(config, user)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def prepare_pairs(config: dict[str, Any]) -> tuple[list[Pair], list[Pair]]:
    data = config["data"]
    train = config["train"]
    pairs = find_pairs(data["image_dir"], data["depth_dir"], data.get("image_glob", "img_*.tif"))
    limit = int(train.get("limit_images") or 0)
    if limit > 0:
        pairs = pairs[:limit]
    return split_pairs(pairs, float(data.get("val_ratio", 0.2)), int(train.get("seed", 42)))


def train_model(model_key: str, config_path: str | None = None, output_dir: str | None = None) -> Path:
    if model_key not in ALL_MODELS:
        raise ValueError(f"Unknown model {model_key!r}. Choices: {', '.join(ALL_MODELS)}")
    config = load_experiment_config(config_path)
    config["model_key"] = model_key
    set_seed(int(config["train"].get("seed", 42)))
    run_dir = make_run_dir(output_dir or config["logging"].get("base_dir", "runs"), model_key)
    save_config(config, run_dir / "config_used.yaml")
    train_pairs, val_pairs = prepare_pairs(config)
    if model_key in TABULAR_MODELS:
        _train_tabular(model_key, config, train_pairs, val_pairs, run_dir)
    else:
        _train_torch(model_key, config, train_pairs, val_pairs, run_dir)
    return run_dir


def _build_tabular(model_key: str, seed: int):
    if model_key == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=15, weights="distance", metric="minkowski")
    if model_key == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=40, max_depth=18, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if model_key == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", batch_size=2048, max_iter=80, random_state=seed, early_stopping=True)
    raise ValueError(model_key)


def _train_tabular(model_key: str, config: dict[str, Any], train_pairs: list[Pair], val_pairs: list[Pair], run_dir: Path) -> None:
    import joblib
    seed = int(config["train"].get("seed", 42))
    data = config["data"]
    train = config["train"]
    x_train, y_train = sample_pixels(train_pairs, int(train.get("pixels_per_image", 2500)), seed, float(data.get("reflectance_scale", 255.0)))
    model = _build_tabular(model_key, seed)
    model.fit(x_train, y_train)
    joblib.dump(model, run_dir / "best_model.joblib")
    rows = _evaluate_predictor(model.predict, val_pairs, config, run_dir / "infer_outputs")
    summary = mean_metrics(rows)
    write_csv(run_dir / "train_log.csv", [{"epoch": 1, "train_l1": float("nan"), **summary}])
    write_csv(run_dir / "metrics.csv", [{"model": model_key, **summary}])
    write_csv(run_dir / "predictions.csv", rows)
    (run_dir / "summary.json").write_text(json.dumps({"model": model_key, "n_train_pixels": int(y_train.size), **summary}, indent=2), encoding="utf-8")


def _evaluate_predictor(predict_fn, pairs: list[Pair], config: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    scale = float(config["data"].get("reflectance_scale", 255.0))
    for pair in pairs:
        image, target, valid, sample_id = load_sample(pair, scale)
        features = pixel_features(image)
        pred_flat = np.full(target.size, np.nan, dtype=np.float32)
        valid_flat = valid.reshape(-1) > 0
        valid_idx = np.flatnonzero(valid_flat)
        for start in range(0, valid_idx.size, 200000):
            idx = valid_idx[start:start + 200000]
            pred_flat[idx] = predict_fn(features[idx]).astype(np.float32)
        pred = pred_flat.reshape(target.shape)
        metrics = regression_metrics(pred, target, valid)
        np.save(out_dir / f"{sample_id}_pred.npy", pred)
        np.save(out_dir / f"{sample_id}_gt.npy", target)
        np.save(out_dir / f"{sample_id}_valid_mask.npy", valid.astype(np.uint8))
        rows.append({"sample_id": sample_id, **metrics})
    write_csv(out_dir / "infer_summary.csv", rows)
    return rows


def _train_torch(model_key: str, config: dict[str, Any], train_pairs: list[Pair], val_pairs: list[Pair], run_dir: Path) -> None:
    import cv2
    import json as _json
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from bathymetry_experiments.models.torch_models import ProposedPaperRegressor, build_torch_model
    from bathymetry_experiments.expert.generate import ensure_expert_artifacts

    class RasterDataset(Dataset):
        def __init__(self, pairs: list[Pair], cfg: dict[str, Any]):
            self.pairs = pairs
            self.scale = float(cfg["data"].get("reflectance_scale", 255.0))
            self.size = int(cfg["train"].get("image_size", 128))
            self.expert = cfg.get("expert") or {}
            self.expert_enabled = bool(self.expert.get("enabled")) and model_key == "proposed"
            if self.expert_enabled:
                self.semantic_dir = Path(str(self.expert.get("semantic_dir", "semantic")))
                self.annotation_dir = Path(str(self.expert.get("annotation_dir", "annotation")))
        def __len__(self) -> int:
            return len(self.pairs)
        def __getitem__(self, idx: int):
            image, target, valid, sample_id = load_sample(self.pairs[idx], self.scale)
            image_hw = np.moveaxis(image, 0, -1)
            image_rs = cv2.resize(image_hw, (self.size, self.size), interpolation=cv2.INTER_AREA)
            target_rs = cv2.resize(np.nan_to_num(target, nan=0.0), (self.size, self.size), interpolation=cv2.INTER_AREA)
            valid_rs = cv2.resize(valid.astype(np.float32), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            image_chw = np.moveaxis(image_rs, -1, 0).astype(np.float32)
            if not self.expert_enabled:
                return torch.from_numpy(image_chw), torch.from_numpy(target_rs[None].astype(np.float32)), torch.from_numpy(valid_rs[None].astype(np.float32)), sample_id
            rel = np.load(self.semantic_dir / f"{sample_id}_reliability.npy").astype(np.float32)
            rel_rs = cv2.resize(rel, (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            emb = np.load(self.semantic_dir / f"{sample_id}_expert_embedding.npy").astype(np.float32)
            ann = _json.loads((self.annotation_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
            zones = ann.get("depth_zones") or []
            return (
                torch.from_numpy(image_chw),
                torch.from_numpy(target_rs[None].astype(np.float32)),
                torch.from_numpy(valid_rs[None].astype(np.float32)),
                sample_id,
                torch.from_numpy(rel_rs[None]),
                torch.from_numpy(emb),
                zones,
            )

    device_pref = str(config["train"].get("device", "cpu"))
    device = torch.device("cuda" if device_pref.startswith("cuda") and torch.cuda.is_available() else "cpu")
    expert = config.get("expert") or {}
    expert_enabled = bool(expert.get("enabled")) and model_key == "proposed"
    if expert_enabled:
        ensure_expert_artifacts(train_pairs + val_pairs, config, overwrite=False)
        model = ProposedPaperRegressor(in_channels=4).to(device)
    else:
        model = build_torch_model(model_key).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["train"].get("learning_rate", 0.001)), weight_decay=1e-4)
    loader = DataLoader(RasterDataset(train_pairs, config), batch_size=int(config["train"].get("batch_size", 2)), shuffle=True)
    logs: list[dict[str, Any]] = []
    epochs = int(config["train"].get("epochs", 1))

    def _zones_to_prior(intensity: torch.Tensor, zones: list[list[dict[str, Any]]]) -> torch.Tensor:
        batch, _, h, w = intensity.shape
        centers = torch.tensor([0.15, 0.45, 0.75], device=intensity.device, dtype=intensity.dtype).view(1, 3, 1, 1)
        dist2 = (intensity - centers) ** 2
        weights = torch.softmax(-dist2 / 0.02, dim=1)
        mids = []
        for b in range(batch):
            z = zones[b] if b < len(zones) else []
            mids_b = {"nearshore": 1.0, "mid": 4.0, "offshore": 8.0}
            for item in z:
                if not isinstance(item, dict):
                    continue
                zone = str(item.get("zone") or "").strip()
                if zone in mids_b and item.get("d_min_m") is not None and item.get("d_max_m") is not None:
                    dmin = float(item.get("d_min_m"))
                    dmax = float(item.get("d_max_m"))
                    mids_b[zone] = 0.5 * (dmin + dmax)
            mids.append([mids_b["nearshore"], mids_b["mid"], mids_b["offshore"]])
        mids_t = torch.tensor(mids, device=intensity.device, dtype=intensity.dtype).view(batch, 3, 1, 1)
        return (weights * mids_t).sum(dim=1, keepdim=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            if not expert_enabled:
                image, target, valid, _ = batch
                image, target, valid = image.to(device), target.to(device), valid.to(device)
                pred = model(image)
                loss = (torch.abs(pred - target) * valid).sum() / (valid.sum() + 1e-6)
            else:
                image, target, valid, _, rel, emb, zones = batch
                image, target, valid = image.to(device), target.to(device), valid.to(device)
                rel, emb = rel.to(device), emb.to(device)
                intensity = image.mean(dim=1, keepdim=True)
                d_phys = _zones_to_prior(intensity, zones)
                x = torch.cat([image, rel], dim=1)
                d_hat, sigma2, _, align = model(x, reliability=rel, d_phys=d_phys, expert_embedding=emb)
                err2 = (d_hat - target) ** 2
                nll = (err2 / (2.0 * sigma2) + 0.5 * torch.log(sigma2)) * valid
                loss = nll.sum() / (valid.sum() + 1e-6)
                lam = float(expert.get("contrastive_lambda") or 0.0)
                if lam > 0 and emb.shape[0] > 1:
                    loss = loss + lam * (1.0 - align.mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_rows = _evaluate_torch(model, val_pairs, config, out_dir=None, device=device, save_outputs=False)
        val_summary = mean_metrics(val_rows)
        logs.append({"epoch": epoch, "train_l1": float(np.mean(losses)) if losses else float("nan"), **val_summary})
    torch.save({"model_key": model_key, "state_dict": model.state_dict(), "config": config}, run_dir / "best_model.pt")
    write_csv(run_dir / "train_log.csv", logs)
    rows = _evaluate_torch(model, val_pairs, config, out_dir=run_dir / "infer_outputs", device=device, save_outputs=True)
    summary = mean_metrics(rows)
    write_csv(run_dir / "metrics.csv", [{"model": model_key, **summary}])
    write_csv(run_dir / "predictions.csv", rows)
    (run_dir / "summary.json").write_text(json.dumps({"model": model_key, **summary}, indent=2), encoding="utf-8")


def _evaluate_torch(
    model,
    pairs: list[Pair],
    config: dict[str, Any],
    out_dir: Path | None,
    device,
    save_outputs: bool,
) -> list[dict[str, Any]]:
    import cv2
    import json as _json
    import torch
    expert = config.get("expert") or {}
    expert_enabled = bool(expert.get("enabled")) and str(config.get("model_key")) == "proposed"
    if expert_enabled:
        semantic_dir = Path(str(expert.get("semantic_dir", "semantic")))
        annotation_dir = Path(str(expert.get("annotation_dir", "annotation")))
    if save_outputs:
        if out_dir is None:
            raise ValueError("out_dir is required when save_outputs=True")
        out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    scale = float(config["data"].get("reflectance_scale", 255.0))
    size = int(config["train"].get("image_size", 128))
    model.eval()

    def _prior_for_sample(image_chw01: np.ndarray, sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rel = np.load(semantic_dir / f"{sample_id}_reliability.npy").astype(np.float32)
        rel_rs = cv2.resize(rel, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        emb = np.load(semantic_dir / f"{sample_id}_expert_embedding.npy").astype(np.float32)
        ann = _json.loads((annotation_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
        zones = ann.get("depth_zones") or []
        img_hw = np.moveaxis(image_chw01, 0, -1)
        img_rs = cv2.resize(img_hw, (size, size), interpolation=cv2.INTER_AREA)
        intensity = img_rs.mean(axis=2, keepdims=True).astype(np.float32)
        centers = np.array([0.15, 0.45, 0.75], dtype=np.float32).reshape(1, 1, 3)
        d2 = (intensity - centers) ** 2
        w = np.exp(-d2 / 0.02)
        w = w / (w.sum(axis=2, keepdims=True) + 1e-6)
        mids = {"nearshore": 1.0, "mid": 4.0, "offshore": 8.0}
        for item in zones:
            if not isinstance(item, dict):
                continue
            zone = str(item.get("zone") or "").strip()
            if zone in mids and item.get("d_min_m") is not None and item.get("d_max_m") is not None:
                mids[zone] = 0.5 * (float(item.get("d_min_m")) + float(item.get("d_max_m")))
        midv = np.array([mids["nearshore"], mids["mid"], mids["offshore"]], dtype=np.float32).reshape(1, 1, 3)
        d_phys = (w * midv).sum(axis=2).astype(np.float32)
        return rel_rs, emb, d_phys

    with torch.no_grad():
        for pair in pairs:
            image, target, valid, sample_id = load_sample(pair, scale)
            image_hw = np.moveaxis(image, 0, -1)
            image_rs = cv2.resize(image_hw, (size, size), interpolation=cv2.INTER_AREA)
            image_chw = np.moveaxis(image_rs, -1, 0).astype(np.float32)
            if not expert_enabled:
                pred_rs = model(torch.from_numpy(image_chw[None]).to(device)).cpu().numpy()[0, 0]
            else:
                rel_rs, emb, d_phys = _prior_for_sample(image, sample_id)
                rel_t = torch.from_numpy(rel_rs[None, None]).to(device)
                dphys_t = torch.from_numpy(d_phys[None, None]).to(device)
                emb_t = torch.from_numpy(emb[None]).to(device)
                x = torch.from_numpy(np.concatenate([image_chw[None], rel_rs[None, None]], axis=1).astype(np.float32)).to(device)
                d_hat, _, _, _ = model(x, reliability=rel_t, d_phys=dphys_t, expert_embedding=emb_t)
                pred_rs = d_hat.cpu().numpy()[0, 0]
            pred = cv2.resize(pred_rs, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            metrics = regression_metrics(pred, target, valid)
            if save_outputs:
                assert out_dir is not None
                np.save(out_dir / f"{sample_id}_pred.npy", pred)
                np.save(out_dir / f"{sample_id}_gt.npy", target)
                np.save(out_dir / f"{sample_id}_valid_mask.npy", valid.astype(np.uint8))
            rows.append({"sample_id": sample_id, **metrics})
    if save_outputs:
        assert out_dir is not None
        write_csv(out_dir / "infer_summary.csv", rows)
    return rows


def run_all(config_path: str | None, models: list[str], output_dir: str | None = None) -> list[Path]:
    selected = models or ALL_MODELS
    return [train_model(model, config_path, output_dir) for model in selected]
