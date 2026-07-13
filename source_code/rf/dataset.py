import importlib.util
import sys
from pathlib import Path

_RF_ROOT = Path(__file__).resolve().parent
_CNN_SRC = _RF_ROOT.parent / "cnn_src"
_CNN_DATASET = _CNN_SRC / "dataset.py"
if not _CNN_DATASET.is_file():
    raise FileNotFoundError(f"cnn_src dataset.py not found at: {_CNN_DATASET}")

_spec = importlib.util.spec_from_file_location("_cnn_src_dataset", str(_CNN_DATASET))
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to create module spec for cnn_src.dataset")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

__all__ = []
for _name in dir(_mod):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_mod, _name)
    __all__.append(_name)

