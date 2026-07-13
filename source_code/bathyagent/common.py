import os
import random
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml


def _expand_env_strings(obj: Any) -> Any:
    pat = re.compile(r"^\$\{([^}]+)\}$")

    def subst(s: str) -> str:
        m = pat.match(s.strip())
        if not m:
            return s
        key = m.group(1)
        v = os.environ.get(key)
        if v is None or str(v).strip() == "":
            return s
        return str(v)

    if isinstance(obj, dict):
        return {k: _expand_env_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_strings(v) for v in obj]
    if isinstance(obj, str):
        return subst(obj)
    return obj


def sync_model_encoder_layout(config: Dict[str, Any]) -> None:
    from dataset import DEFAULT_SEMANTIC_CHANNEL_ORDER, resolve_selected_bands

    data_cfg = config.get("data", {})
    model_cfg = config.setdefault("model", {})
    sem_cfg = config.get("semantic", {})

    selected_bands = data_cfg.get("selected_bands", None)
    image_mode = str(data_cfg.get("image_mode", "rgb"))
    resolved = resolve_selected_bands(selected_bands, image_mode=image_mode)
    image_c = 13 if resolved is None else len(resolved)

    fuse = bool(model_cfg.get("fuse_semantic_channels", False)) and bool(
        sem_cfg.get("use_semantic_channels", False)
    )
    if fuse:
        co = sem_cfg.get("channel_order", None)
        if co is None:
            co = list(DEFAULT_SEMANTIC_CHANNEL_ORDER)
            sem_cfg["channel_order"] = co
        sem_c = len(list(co))
    else:
        sem_c = 0

    model_cfg["image_channels"] = image_c
    model_cfg["semantic_extra_channels"] = sem_c
    model_cfg["encoder_in_channels"] = image_c + sem_c
    model_cfg["in_channels"] = image_c


def validate_llm_training_pipeline(config: Dict[str, Any]) -> None:
    model_cfg = config.get("model", {})
    sem_cfg = config.get("semantic", {})
    vlm_cfg = config.get("vlm", {}) or {}

    fuse = bool(model_cfg.get("fuse_semantic_channels", False)) and bool(
        sem_cfg.get("use_semantic_channels", False)
    )
    use_sem = bool(sem_cfg.get("use_semantic_channels", False))
    ur = bool(model_cfg.get("use_raster_prior_depth", False))
    up = bool(sem_cfg.get("use_prior_depth_map", False))
    ullm = bool(model_cfg.get("use_llm_prior", False))
    urlm = bool(model_cfg.get("use_real_llm", False))

    use_dual_unc = bool(model_cfg.get("use_dual_uncertainty", False))
    fusion_mode = str(model_cfg.get("fusion_mode", "precision_weighted")).strip().lower()
    llm_depth_source = str(model_cfg.get("llm_depth_source", "geometric")).strip().lower()
    llm_uncertainty_mode = str(
        model_cfg.get("llm_uncertainty_mode", "risk_score")
    ).strip().lower()

    llm_unc_min = float(model_cfg.get("llm_uncertainty_min", 0.05))
    llm_unc_max = float(model_cfg.get("llm_uncertainty_max", 1.0))
    risk_score_channel_name = str(model_cfg.get("risk_score_channel_name", "risk_score"))
    risk_high_means_trust_llm = bool(model_cfg.get("risk_high_means_trust_llm", True))

    llm_name = model_cfg.get("llm_model_name", None)
    if llm_name is None or str(llm_name).strip() == "":
        llm_name = vlm_cfg.get("openai_model") or vlm_cfg.get("model_name") or "gpt-4o"
    llm_dev = str(model_cfg.get("llm_device", "auto"))

    channel_order = sem_cfg.get("channel_order", None)
    if channel_order is None:
        channel_order = []

    issue_only = list(channel_order) == ["issue_map"]

    lines = [
        "[pipeline] DEFAULT: RGB (3) + issue_map (1) => encoder_in_channels=4 when fuse_semantic_channels is on",
        f"           fuse active={fuse} encoder_in={model_cfg.get('encoder_in_channels')}",
        f"[pipeline] semantic.channel_order={list(channel_order)}",
        "[pipeline] offline annotation: OpenAI vision API (OPENAI_API_KEY); no local Qwen in the default workflow",
        f"[pipeline] use_raster_prior_depth={ur} semantic.use_prior_depth_map={up} use_llm_prior={ullm} use_real_llm={urlm}",
        f"[pipeline] OpenAI annotate model={vlm_cfg.get('openai_model', 'gpt-4o')!r}",
    ]

    if ullm:
        lines.extend(
            [
                "[pipeline] OPTIONAL prior branch: geometric/raster prior + fusion (expert / legacy mode)",
                f"[pipeline]   use_dual_uncertainty={use_dual_unc} fusion_mode={fusion_mode!r}",
                f"[pipeline]   llm_depth_source={llm_depth_source!r} llm_uncertainty_mode={llm_uncertainty_mode!r}",
                f"[pipeline]   llm_uncertainty_range=[{llm_unc_min:.4f}, {llm_unc_max:.4f}] "
                f"legacy risk_score_channel_name={risk_score_channel_name!r} "
                f"risk_high_means_trust_llm={risk_high_means_trust_llm}",
            ]
        )
    if urlm:
        lines.append(
            "[pipeline] use_real_llm=True: local transformers Qwen (legacy experiments only)."
        )

    print("\n".join(lines))

    # -----------------------------------------------------
    # Hard validation
    # -----------------------------------------------------
    if fuse and not sem_cfg.get("semantic_dir"):
        raise ValueError("fuse_semantic_channels requires semantic.semantic_dir")

    if ullm:
        if use_dual_unc and fusion_mode not in {"precision_weighted", "gated_blend"}:
            raise ValueError(
                f"Unsupported model.fusion_mode={fusion_mode!r}. "
                "Use 'precision_weighted' or 'gated_blend'."
            )

        if llm_depth_source not in {"raster_prior", "geometric", "blend"}:
            raise ValueError(
                f"Unsupported model.llm_depth_source={llm_depth_source!r}. "
                "Use 'raster_prior', 'geometric', or 'blend'."
            )

        if llm_uncertainty_mode not in {"risk_score", "fixed"}:
            raise ValueError(
                f"Unsupported model.llm_uncertainty_mode={llm_uncertainty_mode!r}. "
                "Use 'risk_score' or 'fixed'."
            )

        if llm_unc_min <= 0 or llm_unc_max <= 0:
            raise ValueError(
                "model.llm_uncertainty_min and model.llm_uncertainty_max must be > 0."
            )

        if llm_unc_min > llm_unc_max:
            raise ValueError(
                f"model.llm_uncertainty_min ({llm_unc_min}) must be <= "
                f"model.llm_uncertainty_max ({llm_unc_max})."
            )

    # -----------------------------------------------------
    # Warnings / notes
    # -----------------------------------------------------
    if ur and not up:
        print(
            "[pipeline] WARNING: use_raster_prior_depth is True but semantic.use_prior_depth_map "
            "is False; prior_depth_map batches will be zeros unless you enable loading."
        )

    if ur and not ullm:
        print(
            "[pipeline] WARNING: use_raster_prior_depth is True but use_llm_prior is False; "
            "the forward path may still build prior maps, but final depth will ignore the LLM/prior branch."
        )

    if ullm and not ur and llm_depth_source == "raster_prior":
        print(
            "[pipeline] WARNING: llm_depth_source='raster_prior' but use_raster_prior_depth=False; "
            "depth_llm will effectively fall back to geometric prior."
        )

    if ullm and not ur:
        print(
            "[pipeline] NOTE: use_llm_prior without raster prior uses distance-to-shore geometric prior "
            "from config prior.* ranges, unless you enable *_prior.npy loading."
        )

    if use_dual_unc and not ullm:
        print(
            "[pipeline] NOTE: use_dual_uncertainty is True but use_llm_prior is False; "
            "no model-vs-prior fusion (neural head only unless you enable use_llm_prior)."
        )

    if llm_uncertainty_mode == "risk_score" and ullm:
        if not use_sem:
            print(
                "[pipeline] NOTE: llm_uncertainty_mode='risk_score' but semantic.use_semantic_channels=False; "
                "prior-branch uncertainty falls back to zeros unless fixed mode."
            )
        elif issue_only:
            print(
                "[pipeline] NOTE: issue_map-only semantics: prior-branch uncertainty scales with issue_map "
                "(higher issue_map -> higher LLM/prior uncertainty)."
            )
        elif risk_score_channel_name not in channel_order:
            print(
                f"[pipeline] WARNING: risk_score_channel_name={risk_score_channel_name!r} not in "
                "semantic.channel_order; legacy risk-score uncertainty unavailable."
            )

    if urlm and llm_depth_source == "raster_prior" and ur:
        print(
            "[pipeline] NOTE: use_real_llm=True with raster prior enabled (legacy / expert setups)."
        )

    if llm_depth_source == "blend" and not ur:
        print(
            "[pipeline] WARNING: llm_depth_source='blend' but raster prior disabled; blend uses geometric only."
        )


# =========================================================
# Config / reproducibility
# =========================================================
def load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand_env_strings(cfg or {})


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def set_seed(seed: int) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch.
    This version is stricter than the old one for better reproducibility.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_torch_device(device_pref: str, gpu_id: int = 0):
    import torch

    pref = (device_pref or "auto").strip().lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("device is cuda/gpu but torch.cuda.is_available() is False.")
        n = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= n:
            raise ValueError(f"gpu_id={gpu_id} is out of range (found {n} GPU(s)).")
        dev = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(dev)
        return dev
    if pref != "auto":
        raise ValueError(f"Unknown device: {device_pref!r} (use cpu, cuda, or auto).")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= n:
        print(f"[device] auto: gpu_id={gpu_id} invalid for {n} GPU(s); using CPU.")
        return torch.device("cpu")
    dev = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(dev)
    return dev


# =========================================================
# Logging / experiment folders
# =========================================================
def create_experiment_folder(base_dir: str, model_name: str) -> Tuple[str, str]:
    """
    Create an experiment folder:
        logs/<model_name>/<run_id>/
    where run_id is timestamp + short uuid.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    run_id = f"{ts}_{short_id}"
    exp_dir = os.path.join(base_dir, model_name, run_id)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir, run_id


def append_row_to_csv(path: str, row: Dict[str, Any]) -> None:
    """
    Append one row to CSV. Create file with header if it does not exist.
    """
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


# =========================================================
# Standard metrics
# =========================================================
def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# =========================================================
# Masked metrics for bathymetry
# =========================================================
def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert mask to boolean mask.
    """
    return mask.astype(bool)


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean Absolute Error computed only on valid pixels.
    """
    mask = _prepare_mask(mask)
    if mask.sum() == 0:
        return 0.0
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """
    RMSE computed only on valid pixels.
    """
    mask = _prepare_mask(mask)
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(((y_true[mask] - y_pred[mask]) ** 2).mean()))


def masked_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    MAPE computed only on valid pixels.
    """
    mask = _prepare_mask(mask)
    if mask.sum() == 0:
        return 0.0
    denom = np.maximum(np.abs(y_true[mask]), eps)
    return float(np.abs((y_true[mask] - y_pred[mask]) / denom).mean())


def masked_r2(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """
    R² computed only on valid pixels.
    """
    mask = _prepare_mask(mask)
    if mask.sum() == 0:
        return 0.0

    yt = y_true[mask]
    yp = y_pred[mask]

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_bias(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean Error / Bias on valid pixels.
    Positive means overestimation if y_pred > y_true on average.
    """
    mask = _prepare_mask(mask)
    if mask.sum() == 0:
        return 0.0
    return float((y_pred[mask] - y_true[mask]).mean())


def compute_p95ae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """
    95th percentile absolute error on valid pixels.
    Useful for tail-error evaluation.
    """
    mask = _prepare_mask(mask)
    if mask.sum() == 0:
        return 0.0
    abs_err = np.abs(y_true[mask] - y_pred[mask])
    return float(np.percentile(abs_err, 95))


# =========================================================
# Optional helper for masked loss debugging
# =========================================================
def masked_mean(values: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> float:
    """
    Mean of values over valid pixels only.
    """
    mask = mask.astype(np.float32)
    denom = float(mask.sum())
    if denom < eps:
        return 0.0
    return float((values * mask).sum() / denom)