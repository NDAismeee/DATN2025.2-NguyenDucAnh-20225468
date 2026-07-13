from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LegacyModelAdapter:
    key: str
    train_script: Path
    infer_script: Path
    default_config: Path
    supports_device: bool = True
    supports_gpu_id: bool = True
    supports_sample_id: bool = True
