from __future__ import annotations

import torch
import torch.nn as nn


class SoftGate(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        in_ch = hidden_dim + 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mlp_hidden, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=float(dropout)),
            nn.Conv2d(mlp_hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        variance: torch.Tensor,
        unreliable_mask: torch.Tensor,
        gamma_map: torch.Tensor,
        w_phys: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([fused_features, variance, unreliable_mask, gamma_map, w_phys], dim=1)
        return self.net(x)
