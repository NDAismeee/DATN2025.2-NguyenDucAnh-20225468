import argparse
import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from joblib import dump
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset import build_pairs_magic, read_raster
from pixel_mlp_core import pick_torch_device, predict_from_bundle, train_torch_pixel_mlp

PairType = Tuple[Path, Path, str]


def _resolve_data_section(raw_d: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(raw_d)
    img_dir = d.get("train_image_dir") or d.get("image_dir")
    if not img_dir:
        raise KeyError("data.image_dir (or data.train_image_dir) is required in config YAML.")
    d["image_dir"] = img_dir
    if not d.get("depth_dir"):
        raise KeyError("data.depth_dir is required in config YAML.")
    return d


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        key = obj[2:-1]
        return os.environ.get(key, obj)
    return obj


def load_yaml_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _expand_env(cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _valid_mask_from_depth(depth_1hw: np.ndarray, magic_negative_depth_valid: bool) -> np.ndarray:
    d = depth_1hw.astype(np.float32, copy=False)
    if magic_negative_depth_valid:
        valid = np.isfinite(d) & (d < 0)
    else:
        valid = np.isfinite(d)
    return valid[0].astype(np.uint8, copy=False)


def _to_positive_depth(depth_1hw: np.ndarray, magic_negative_depth_valid: bool) -> np.ndarray:
    d = depth_1hw.astype(np.float32, copy=False)[0]
    if magic_negative_depth_valid:
        return -d
    return d


def _rgb_hw3(img_chw: np.ndarray, reflectance_scale: float) -> np.ndarray:
    x = img_chw[:3].astype(np.float32, copy=False)
    if reflectance_scale and reflectance_scale != 1.0:
        x = x / float(reflectance_scale)
    return np.transpose(x, (1, 2, 0))


def _pixel_features(rgb: np.ndarray, include_xy: bool) -> np.ndarray:
    h, w, _ = rgb.shape
    feats = [rgb.reshape(-1, 3)]
    if include_xy:
        yy, xx = np.mgrid[0:h, 0:w]
        fx = (xx.astype(np.float32) / max(w - 1, 1)).reshape(-1, 1)
        fy = (yy.astype(np.float32) / max(h - 1, 1)).reshape(-1, 1)
        feats.extend([fx, fy])
    return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


def _sample_training_pixels(
    rgb_hw3: np.ndarray,
    depth_pos_hw: np.ndarray,
    valid_hw: np.ndarray,
    rng: np.random.Generator,
    pixels_per_image: int,
    include_xy: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_idx = np.flatnonzero(valid_hw.reshape(-1) > 0)
    if valid_idx.size == 0:
        return np.zeros((0, 3 + (2 if include_xy else 0)), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    n = min(int(pixels_per_image), int(valid_idx.size))
    choose = rng.choice(valid_idx, size=n, replace=False)
    feats = _pixel_features(rgb_hw3, include_xy=include_xy)[choose]
    y = depth_pos_hw.reshape(-1)[choose].astype(np.float32, copy=False)
    return feats, y


def _train_val_split(pairs: Sequence[PairType], val_ratio: float, seed: int) -> Tuple[List[PairType], List[PairType]]:
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = int(round(len(pairs) * float(val_ratio)))
    n_val = max(1, n_val) if len(pairs) >= 2 else 0
    return pairs[n_val:], pairs[:n_val]


def _masked_mae(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    m = valid > 0
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(pred[m] - gt[m])))


def _masked_rmse(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    m = valid > 0
    if not np.any(m):
        return float("nan")
    r = pred[m] - gt[m]
    return float(np.sqrt(np.mean(r * r)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--train_image_dir", type=str, default=None)
    ap.add_argument("--image_dir", type=str, default=None)
    ap.add_argument("--depth_dir", type=str, default=None)
    ap.add_argument("--image_suffix", type=str, default=None)
    args = ap.parse_args()

    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
    cfg = load_yaml_config(args.config)
    cfg = dict(cfg)

    data_cfg = _resolve_data_section(dict(cfg.get("data", {}) or {}))
    if args.train_image_dir is not None:
        data_cfg["image_dir"] = str(args.train_image_dir)
    elif args.image_dir is not None:
        data_cfg["image_dir"] = str(args.image_dir)
    if args.depth_dir is not None:
        data_cfg["depth_dir"] = str(args.depth_dir)
    if args.image_suffix is not None:
        data_cfg["image_suffix"] = str(args.image_suffix)
    cfg["data"] = data_cfg

    train_cfg = cfg.get("train", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}

    img_dir = Path(str(data_cfg.get("image_dir", ""))).expanduser()
    depth_dir = Path(str(data_cfg.get("depth_dir", ""))).expanduser()
    image_suffix = str(data_cfg.get("image_suffix", "img_*.tif"))
    depth_suffixes_to_try = data_cfg.get("depth_suffixes_to_try", ["_depth", "_bathy", "_gt", "_label"])
    reflectance_scale = float(data_cfg.get("reflectance_scale", 1.0))
    magic_negative_depth_valid = bool(data_cfg.get("magic_negative_depth_valid", True))

    seed = int(train_cfg.get("seed", 42))
    val_ratio = float(train_cfg.get("val_ratio", 0.2))
    pixels_per_image = int(train_cfg.get("pixels_per_image", 5000))
    include_xy = bool(train_cfg.get("include_xy", True))
    max_train_pixels = int(train_cfg.get("max_train_pixels", 200000))

    set_seed(seed)
    rng = np.random.default_rng(seed)

    pairs = build_pairs_magic(
        img_dir=img_dir,
        depth_dir=depth_dir,
        image_suffix=image_suffix,
        depth_suffixes_to_try=depth_suffixes_to_try,
    )
    if not pairs:
        raise ValueError("No matched image-depth pairs found. Check IMAGE_DIR/DEPTH_DIR and config.yaml.")
    print(f"[mlp train] magic pairs (image-first): {len(pairs)} | image_dir={img_dir} | depth_dir={depth_dir}")

    train_pairs, val_pairs = _train_val_split(pairs, val_ratio=val_ratio, seed=seed)
    if not train_pairs:
        raise ValueError("Train split is empty. Reduce val_ratio or add more data.")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for img_path, depth_path, _pid in train_pairs:
        img = read_raster(img_path)
        depth = read_raster(depth_path)
        rgb = _rgb_hw3(img, reflectance_scale=reflectance_scale)
        valid = _valid_mask_from_depth(depth, magic_negative_depth_valid=magic_negative_depth_valid)
        depth_pos = _to_positive_depth(depth, magic_negative_depth_valid=magic_negative_depth_valid)

        x_i, y_i = _sample_training_pixels(
            rgb_hw3=rgb,
            depth_pos_hw=depth_pos,
            valid_hw=valid,
            rng=rng,
            pixels_per_image=pixels_per_image,
            include_xy=include_xy,
        )
        if x_i.shape[0] > 0:
            xs.append(x_i)
            ys.append(y_i)

        if sum(a.shape[0] for a in xs) >= max_train_pixels:
            break

    if not xs:
        raise ValueError("No valid training pixels sampled. Check depth validity rules.")

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if X.shape[0] > max_train_pixels:
        sel = rng.choice(np.arange(X.shape[0]), size=max_train_pixels, replace=False)
        X = X[sel]
        y = y[sel]

    hls = model_cfg.get("hidden_layer_sizes", [256, 128, 64])
    if isinstance(hls, list):
        hidden_layer_sizes = tuple(int(x) for x in hls)
    else:
        hidden_layer_sizes = tuple(int(x) for x in str(hls).strip("()[]").split(",") if str(x).strip())

    dev_pref = str(train_cfg.get("device", "cpu")).lower().strip()
    train_device = pick_torch_device(dev_pref, int(train_cfg.get("gpu_id", 0)))
    use_torch_gpu = dev_pref in ("cuda", "gpu") and train_device.type == "cuda"

    if use_torch_gpu:
        epochs = int(train_cfg.get("epochs", model_cfg.get("max_iter", 80)))
        batch_size = int(train_cfg.get("batch_size", 8192))
        lr = float(model_cfg.get("learning_rate_init", 1e-3))
        wd = float(train_cfg.get("weight_decay", 1e-4))
        net, scaler = train_torch_pixel_mlp(
            X,
            y,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=str(model_cfg.get("activation", "relu")),
            learning_rate=lr,
            weight_decay=wd,
            epochs=epochs,
            batch_size=batch_size,
            device=train_device,
            seed=seed,
        )
        net.eval()
        net_cpu = net.cpu()
        bundle: Dict[str, Any] = {
            "backend": "torch",
            "model": None,
            "torch_state_dict": net_cpu.state_dict(),
            "torch_arch": {
                "in_dim": int(X.shape[1]),
                "hidden_layer_sizes": list(hidden_layer_sizes),
                "activation": str(model_cfg.get("activation", "relu")),
            },
            "scaler": scaler,
            "config": cfg,
            "feature_dim": int(X.shape[1]),
        }
        print(f"[mlp train] PyTorch MLP on {train_device} | epochs={epochs} batch_size={batch_size}")
    else:
        if dev_pref in ("cuda", "gpu"):
            print("[mlp train] CUDA requested but not available; using sklearn MLPRegressor on CPU.")
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=str(model_cfg.get("activation", "relu")),
            alpha=float(model_cfg.get("alpha", 1e-4)),
            learning_rate_init=float(model_cfg.get("learning_rate_init", 1e-3)),
            max_iter=int(model_cfg.get("max_iter", 60)),
            early_stopping=bool(model_cfg.get("early_stopping", True)),
            random_state=seed,
            verbose=True,
        )
        model = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("mlp", mlp)])
        model.fit(X, y)
        bundle = {
            "backend": "sklearn",
            "model": model,
            "config": cfg,
            "feature_dim": int(X.shape[1]),
        }
        print("[mlp train] sklearn MLPRegressor on CPU (sklearn has no GPU backend).")

    val_infer_device = pick_torch_device(
        str(train_cfg.get("val_infer_device", train_cfg.get("device", "cpu"))),
        int(train_cfg.get("gpu_id", 0)),
    )
    val_mae = float("nan")
    val_rmse = float("nan")
    if val_pairs:
        maes: List[float] = []
        rmses: List[float] = []
        for img_path, depth_path, _pid in val_pairs:
            img = read_raster(img_path)
            depth = read_raster(depth_path)
            rgb = _rgb_hw3(img, reflectance_scale=reflectance_scale)
            valid = _valid_mask_from_depth(depth, magic_negative_depth_valid=magic_negative_depth_valid)
            gt = _to_positive_depth(depth, magic_negative_depth_valid=magic_negative_depth_valid)

            feats = _pixel_features(rgb, include_xy=include_xy)
            pred = predict_from_bundle(bundle, feats, chunk=200000, device=val_infer_device).reshape(gt.shape)
            maes.append(_masked_mae(pred, gt, valid))
            rmses.append(_masked_rmse(pred, gt, valid))
        val_mae = float(np.nanmean(np.array(maes, dtype=np.float64))) if maes else float("nan")
        val_rmse = float(np.nanmean(np.array(rmses, dtype=np.float64))) if rmses else float("nan")

    save_dir = Path(str(log_cfg.get("save_dir", "checkpoints_mlp"))).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(log_cfg.get("model_name", "mlp.joblib"))
    out_path = save_dir / model_name
    dump(bundle, out_path)

    train_log_path = save_dir / "train_log.csv"
    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_pixels", "val_mae", "val_rmse"])
        w.writeheader()
        w.writerow({"epoch": 1, "train_pixels": int(X.shape[0]), "val_mae": val_mae, "val_rmse": val_rmse})

    print(f"Saved model: {out_path}")
    print(f"Saved log  : {train_log_path}")


if __name__ == "__main__":
    main()

