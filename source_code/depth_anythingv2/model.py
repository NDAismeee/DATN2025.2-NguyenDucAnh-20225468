import importlib
from typing import Any, Dict, Tuple

import torch


def pick_torch_device(device_pref: str, gpu_id: int = 0) -> torch.device:
    pref = (device_pref or "auto").strip().lower()
    if pref == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if pref in ("cuda", "gpu", "auto"):
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _try_import_depth_anything_v2():
    try:
        return importlib.import_module("depth_anything_v2.dpt")
    except Exception:
        return None


def load_depth_anything_v2(checkpoint_path: str, encoder: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Loads Depth Anything V2 model definition if available, then loads weights.

    This runner supports two setups:
    - You have the Depth-Anything-V2 code installed (module `depth_anything_v2` importable)
    - Otherwise, you must install/clone it so the import works.
    """
    mod = _try_import_depth_anything_v2()
    if mod is None:
        raise RuntimeError(
            "Depth Anything V2 code not found (cannot import `depth_anything_v2`). "
            "Please install/clone Depth-Anything-V2 so `depth_anything_v2.dpt` is importable."
        )

    DepthAnythingV2 = getattr(mod, "DepthAnythingV2", None)
    if DepthAnythingV2 is None:
        raise RuntimeError("`depth_anything_v2.dpt.DepthAnythingV2` not found in installed package.")

    enc = (encoder or "vitl").lower().strip()
    if enc == "vits":
        cfg = dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    elif enc == "vitb":
        cfg = dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
    else:
        cfg = dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])

    model = DepthAnythingV2(**cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, cfg

