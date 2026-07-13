from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PretrainedAerialUNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.encoder = nn.ModuleList(
            [
                DoubleConv(n_channels, 32),
                Down(32, 64),
                Down(64, 128),
                Down(128, 256),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                Up(256, 128, bilinear),
                Up(128, 64, bilinear),
                Up(64, 32, bilinear),
                nn.Conv2d(32, n_classes, kernel_size=1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x = self.decoder[0](x4, x3)
        x = self.decoder[1](x, x2)
        x = self.decoder[2](x, x1)
        return self.decoder[3](x)


def load_pretrained_weights(model: nn.Module, weights_path: str, device: torch.device) -> None:
    p = Path(weights_path)
    ckpt = torch.load(str(p), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: Dict[str, Any],
    epoch: int,
    val_metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": int(epoch),
            "val_metrics": dict(val_metrics),
        },
        str(path),
    )


def build_model_from_config(cfg: Dict[str, Any]) -> Tuple[PretrainedAerialUNet, Optional[str]]:
    mc = cfg.get("model", {}) or {}
    bilinear = bool(mc.get("bilinear", False))
    weights = mc.get("pretrained_weights", None)
    weights_path = str(weights) if weights else None
    model = PretrainedAerialUNet(n_channels=3, n_classes=1, bilinear=bilinear)
    return model, weights_path

