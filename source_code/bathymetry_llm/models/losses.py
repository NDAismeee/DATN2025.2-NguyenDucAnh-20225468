from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype, device=values.device)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


VAR_MIN = 1e-4


def heteroscedastic_nll(pred: torch.Tensor, target: torch.Tensor, variance: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    variance = variance.clamp_min(VAR_MIN)
    loss = ((target - pred) ** 2) / (2.0 * variance) + 0.5 * torch.log(variance)
    return masked_mean(loss, valid_mask)


def contrastive_alignment_loss(z_v: torch.Tensor, z_t: torch.Tensor, region_valid_mask: torch.Tensor | None, tau: float) -> torch.Tensor:
    b, k, c = z_v.shape
    z_vn = F.normalize(z_v, dim=-1, eps=1e-8)
    z_tn = F.normalize(z_t, dim=-1, eps=1e-8)
    logits = torch.bmm(z_vn, z_tn.transpose(1, 2)) / float(tau)
    if region_valid_mask is None:
        valid = torch.ones((b, k), device=z_v.device, dtype=z_v.dtype)
    else:
        valid = region_valid_mask.to(device=z_v.device, dtype=z_v.dtype).clamp(0.0, 1.0)
    col_mask = (1.0 - valid).unsqueeze(1) * (-1e4)
    logits = logits + col_mask
    log_probs = F.log_softmax(logits, dim=-1)
    pos = torch.diagonal(log_probs, dim1=-2, dim2=-1)
    per_k = -pos
    if valid.sum() <= 0:
        return per_k.sum() * 0.0
    return (per_k * valid).sum() / valid.sum().clamp_min(1.0)


def interval_consistency_loss(
    depth: torch.Tensor,
    d_min: torch.Tensor | None,
    d_max: torch.Tensor | None,
    gamma_map: torch.Tensor | None,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if d_min is None or d_max is None:
        return depth.sum() * 0.0
    hinge = F.relu(d_min - depth).pow(2) + F.relu(depth - d_max).pow(2)
    if gamma_map is None:
        g = torch.ones_like(hinge)
    else:
        g = gamma_map.to(dtype=hinge.dtype, device=hinge.device).clamp(0.0, 1.0)
    return masked_mean(hinge * g, valid_mask)


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    lambda_align: float = 0.01,
    lambda_int: float = 0.01,
    align_tau: float = 0.07,
) -> Dict[str, torch.Tensor]:
    nll = heteroscedastic_nll(outputs["depth"], batch["depth"], outputs["var"], batch["valid_mask"])
    align = contrastive_alignment_loss(outputs["z_v"], outputs["z_t"], batch.get("region_valid_mask"), tau=float(align_tau))
    int_loss = interval_consistency_loss(
        outputs["depth"],
        batch.get("d_min"),
        batch.get("d_max"),
        batch.get("gamma_map"),
        batch["valid_mask"],
    )
    total = nll + float(lambda_align) * align + float(lambda_int) * int_loss
    return {"total": total, "nll": nll, "align": align, "int": int_loss}
