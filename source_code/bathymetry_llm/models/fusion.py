from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_region_pool(features: torch.Tensor, region_masks: torch.Tensor) -> torch.Tensor:
    if region_masks.shape[-2:] != features.shape[-2:]:
        region_masks = F.interpolate(region_masks, size=features.shape[-2:], mode="nearest")
    masks = region_masks.float().clamp(0.0, 1.0)
    denom = masks.flatten(2).sum(dim=-1).clamp_min(1e-6)
    pooled = torch.einsum("bchw,bkhw->bkc", features, masks) / denom.unsqueeze(-1)
    return pooled


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionExpertPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, z_t: torch.Tensor, region_valid_mask: torch.Tensor | None) -> torch.Tensor:
        b, k, c = z_t.shape
        q = self.query.expand(b, 1, c)
        scores = torch.bmm(q, z_t.transpose(1, 2)) / (c**0.5)
        scores = scores.squeeze(1)
        if region_valid_mask is not None:
            valid = region_valid_mask.to(dtype=z_t.dtype, device=z_t.device).clamp(0.0, 1.0)
            scores = scores.masked_fill(valid < 0.5, -1e4)
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        v = torch.bmm(attn.unsqueeze(1), z_t).squeeze(1)
        return v


class ExpertFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_pool = AttentionExpertPool(hidden_dim)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, visual_features: torch.Tensor, z_t: torch.Tensor, region_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        v_expert = self.attn_pool(z_t, region_valid_mask)
        e_proj = self.adapter(v_expert)
        expert_map = e_proj[:, :, None, None].expand(-1, -1, visual_features.shape[-2], visual_features.shape[-1])
        return self.net(torch.cat([visual_features, expert_map], dim=1))


class RegionTextFusion(nn.Module):
    def __init__(self, hidden_dim: int, text_dim: int) -> None:
        super().__init__()
        self.text_projection = ProjectionHead(text_dim, hidden_dim)
        self.visual_projection = ProjectionHead(hidden_dim, hidden_dim)
        self.fusion = ExpertFusion(hidden_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        region_masks: torch.Tensor,
        text_embeddings: torch.Tensor,
        region_valid_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled_visual = masked_region_pool(visual_features, region_masks)
        z_v = self.visual_projection(pooled_visual)
        z_t = self.text_projection(text_embeddings)
        fused = self.fusion(visual_features, z_t, region_valid_mask=region_valid_mask)
        return fused, z_v, z_t
