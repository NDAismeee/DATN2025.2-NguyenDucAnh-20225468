import importlib.util
from pathlib import Path

_UNET_ROOT = Path(__file__).resolve().parent
_CNN_EVAL = (_UNET_ROOT.parent / "cnn_src" / "eval.py").resolve()
if not _CNN_EVAL.is_file():
    raise FileNotFoundError(f"cnn_src eval.py not found at: {_CNN_EVAL}")

_spec = importlib.util.spec_from_file_location("_cnn_src_eval", str(_CNN_EVAL))
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to create module spec for cnn_src.eval")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

__all__ = []
for _name in dir(_mod):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_mod, _name)
    __all__.append(_name)

