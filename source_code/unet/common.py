import os
from typing import Any, Dict

import torch


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


def pick_torch_device(device_pref: str, gpu_id: int = 0) -> torch.device:
    pref = (device_pref or "auto").strip().lower()
    if pref == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if pref in ("cuda", "gpu", "auto"):
        return torch.device(f"cuda:{int(gpu_id)}")
    return torch.device("cpu")

