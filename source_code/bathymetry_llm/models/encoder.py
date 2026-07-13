from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoiseAwareUNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        c1 = max(hidden_dim // 4, 16)
        c2 = max(hidden_dim // 2, 32)
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.bottleneck = ConvBlock(c2, hidden_dim)
        self.up2 = nn.Conv2d(hidden_dim + c2, hidden_dim, 3, padding=1)
        self.up1 = nn.Conv2d(hidden_dim + c1, hidden_dim, 3, padding=1)
        self.out = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor, unreliable_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, unreliable_mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))
        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = F.relu(self.up2(torch.cat([u2, e2], dim=1)), inplace=True)
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = F.relu(self.up1(torch.cat([u1, e1], dim=1)), inplace=True)
        return self.out(u1)
