import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC_ROOT = _HERE.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from cnn_src.dataset import *  # noqa: F401,F403

