from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from bathymetry_llm.models.encoder import NoiseAwareUNetEncoder
from bathymetry_llm.models.fusion import RegionTextFusion
from bathymetry_llm.models.gate import SoftGate


VAR_MIN = 1e-4


class DepthUncertaintyHead(nn.Module):
    def __init__(self, hidden_dim: int, var_min: float = VAR_MIN) -> None:
        super().__init__()
        self.var_min = float(var_min)
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 2, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(features)
        mu = F.softplus(raw[:, 0:1])
        var = F.softplus(raw[:, 1:2]) + self.var_min
        log_var = torch.log(var)
        return mu, log_var, var


class LLMGuidedBathymetryModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        text_dim: int = 384,
        gate_hidden: int = 128,
        gate_dropout: float = 0.1,
        var_min: float = VAR_MIN,
    ) -> None:
        super().__init__()
        self.encoder = NoiseAwareUNetEncoder(in_channels=4, hidden_dim=hidden_dim)
        self.fusion = RegionTextFusion(hidden_dim=hidden_dim, text_dim=text_dim)
        self.depth_head = DepthUncertaintyHead(hidden_dim=hidden_dim, var_min=var_min)
        self.gate = SoftGate(hidden_dim=hidden_dim, mlp_hidden=gate_hidden, dropout=gate_dropout)

    def forward(
        self,
        image: torch.Tensor,
        unreliable_mask: torch.Tensor,
        d_phys: torch.Tensor,
        region_masks: torch.Tensor,
        text_embeddings: torch.Tensor,
        gamma_map: torch.Tensor,
        w_phys: torch.Tensor,
        region_valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if image.shape[1] != 3:
            raise ValueError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}")
        if unreliable_mask.shape[1] != 1 or d_phys.shape[1] != 1:
            raise ValueError("unreliable_mask and d_phys must have one channel")
        m_bin = (unreliable_mask > 0.5).to(dtype=image.dtype)
        f_vis = self.encoder(image, m_bin)
        fused, z_v, z_t = self.fusion(f_vis, region_masks, text_embeddings, region_valid_mask)
        mu, log_var, var = self.depth_head(fused)
        alpha = self.gate(fused, var, m_bin, gamma_map, w_phys)
        depth = (1.0 - alpha) * mu + alpha * d_phys
        return {
            "depth": depth,
            "mu": mu,
            "log_var": log_var,
            "var": var,
            "alpha": alpha,
            "z_v": z_v,
            "z_t": z_t,
            "f_vis": f_vis,
            "fused": fused,
            "d_phys": d_phys,
        }
