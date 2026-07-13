from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_gaussian_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    mask: Optional[torch.Tensor],
    variance_floor: float = 1.0e-4,
) -> torch.Tensor:
    """Masked Gaussian negative log-likelihood."""
    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.float()
    var = var.clamp_min(float(variance_floor))
    per_pixel = ((target - pred) ** 2) / (2.0 * var) + 0.5 * torch.log(var)
    return (per_pixel * mask).sum() / mask.sum().clamp_min(1.0)


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Masked L1 reconstruction loss.

    pred, target: (B, C, H, W)
    mask: (B, 1, H, W) or None
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Reconstruction shape mismatch: pred={tuple(pred.shape)}, "
            f"target={tuple(target.shape)}"
        )

    if mask is None:
        mask = torch.ones_like(target[:, :1])

    mask = mask.float()
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")

    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)

    error = torch.abs(pred - target)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm helper for batch_size=1 training."""
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    """Basic conv block with GroupNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """Downsampling block with stride=2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.block = ConvBlock(out_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    """Upsampling block with bilinear interpolation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels, dropout)

    def forward(
        self,
        x: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.block(x)


class VisualAutoencoderEncoder(nn.Module):
    """4-channel encoder with H/4 × W/4 spatial bottleneck."""

    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, 64, dropout=0.0)
        self.down1 = DownBlock(64, 128, dropout=dropout * 0.5)
        self.down2 = DownBlock(128, latent_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feature_h = self.stem(x)  # B×64×H×W
        feature_h2 = self.down1(feature_h)  # B×128×H/2×W/2
        latent = self.down2(feature_h2)  # B×D×H/4×W/4

        return latent, {
            "feature_h": feature_h,
            "feature_h2": feature_h2,
        }


class RGBReconstructionDecoder(nn.Module):
    """Decoder for RGB reconstruction from latent."""

    def __init__(
        self,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.up1 = UpBlock(latent_dim, 128, dropout=dropout * 0.5)
        self.up2 = UpBlock(128, 64, dropout=dropout * 0.25)
        self.output = nn.Conv2d(64, 3, kernel_size=1)

    def forward(
        self,
        latent: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        height, width = output_size
        half_size = ((height + 1) // 2, (width + 1) // 2)

        x = self.up1(latent, half_size)
        x = self.up2(x, (height, width))
        reconstruction = self.output(x)
        return reconstruction


class DepthDecoder(nn.Module):
    """Decoder for depth features from fused latent."""

    def __init__(
        self,
        latent_dim: int = 256,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.up1 = UpBlock(latent_dim, 128, dropout=dropout * 0.5)
        self.up2 = UpBlock(128, output_dim, dropout=dropout * 0.25)

    def forward(
        self,
        fused_latent: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        height, width = output_size
        half_size = ((height + 1) // 2, (width + 1) // 2)

        x = self.up1(fused_latent, half_size)
        decoded_features = self.up2(x, (height, width))
        return decoded_features


class RegionSemanticGrounding(nn.Module):
    """Region-text semantic grounding with expert extraction."""

    def __init__(self, feature_dim: int = 256, text_dim: int = 384, align_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.visual_proj = nn.Sequential(
            nn.Linear(feature_dim, align_dim),
            nn.ReLU(inplace=True),
            nn.Linear(align_dim, align_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, align_dim),
            nn.ReLU(inplace=True),
            nn.Linear(align_dim, align_dim),
        )
        self.attn_score = nn.Linear(align_dim, 1)
        self.expert_adapter = nn.Sequential(
            nn.Linear(align_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    @staticmethod
    def masked_average_pool(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Pool features by region masks."""
        bsz, channels, _, _ = features.shape
        if masks.ndim != 4:
            raise ValueError(f"disturbance masks must be (B,K,H,W), got {tuple(masks.shape)}")
        if masks.shape[1] == 0:
            return features.new_zeros((bsz, 0, channels))
        mask = masks.unsqueeze(2).float()
        num = (features.unsqueeze(1) * mask).sum(dim=(-1, -2))
        den = mask.sum(dim=(-1, -2)).clamp_min(1.0e-6)
        return num / den

    def forward(
        self,
        features: torch.Tensor,
        disturbance_masks: torch.Tensor,
        text_embeddings: torch.Tensor,
        region_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        bsz = features.shape[0]
        if disturbance_masks.shape[1] == 0 or text_embeddings.shape[1] == 0:
            zero = features.new_zeros((bsz, self.feature_dim))
            align_loss = features.new_zeros(())
            return zero, align_loss, {
                "z_v": features.new_zeros((bsz, 0, self.feature_dim)),
                "z_t": features.new_zeros((bsz, 0, self.feature_dim)),
                "attn": features.new_zeros((bsz, 0)),
            }

        valid = region_valid_mask.float()
        pooled_visual = self.masked_average_pool(features, disturbance_masks)
        z_v = self.visual_proj(pooled_visual)
        z_t = self.text_proj(text_embeddings.float())

        cosine = F.cosine_similarity(z_v, z_t, dim=-1)
        align_loss = ((1.0 - cosine) * valid).sum() / valid.sum().clamp_min(1.0)

        attn_logits = self.attn_score(z_t).squeeze(-1)
        attn_logits = attn_logits.masked_fill(valid <= 0, -1.0e4)
        attn = torch.softmax(attn_logits, dim=1) * valid
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

        expert = (attn.unsqueeze(-1) * z_t).sum(dim=1)
        expert = self.expert_adapter(expert)
        
        # Zero out expert if no valid regions
        has_valid_region = (valid.sum(dim=1, keepdim=True) > 0).float()
        expert = expert * has_valid_region
        
        return expert, align_loss, {"z_v": z_v, "z_t": z_t, "attn": attn}


class MultimodalFusion(nn.Module):
    """Fuse latent features with expert embedding."""

    def __init__(self, feature_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = features.shape
        expert_map = expert[:, :, None, None].expand(bsz, channels, height, width)
        return self.net(torch.cat([features, expert_map], dim=1))


class DepthHead(nn.Module):
    """Predict depth mean and variance from decoded features."""

    def __init__(self, feature_dim: int = 64, variance_floor: float = 1.0e-4):
        super().__init__()
        self.variance_floor = float(variance_floor)
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(
        self,
        decoded_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(decoded_features)
        mu = F.softplus(out[:, 0:1])
        var = F.softplus(out[:, 1:2]) + self.variance_floor
        return mu, var


class UncertaintyAwareGate(nn.Module):
    """Predict gating weights based on uncertainty and reliability."""

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim + 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(
        self,
        decoded_features: torch.Tensor,
        var: torch.Tensor,
        reliability_mask: torch.Tensor,
    ) -> torch.Tensor:
        log_var = torch.log(var.clamp_min(1.0e-8))
        gate_in = torch.cat([decoded_features, log_var, reliability_mask.float()], dim=1)
        return torch.sigmoid(self.net(gate_in))


class LLMGuidedBathymetryModel(nn.Module):
    """VLM-guided mask-conditioned bathymetry autoencoder."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        mcfg = config.get("model", {})
        tcfg = config.get("train", {})
        
        # Model dimensions
        feature_dim = int(mcfg.get("latent_dim", mcfg.get("feature_dim", 256)))
        text_dim = int(mcfg.get("text_dim", config.get("text_encoder", {}).get("output_dim", 384)))
        align_dim = int(mcfg.get("align_dim", 256))
        in_channels = int(mcfg.get("encoder_in_channels", 4))
        decoder_feature_dim = int(mcfg.get("decoder_feature_dim", 64))
        gate_hidden = int(mcfg.get("gating_hidden_dim", 128))
        dropout = float(mcfg.get("dropout", 0.1))

        # Losses and settings
        self.variance_floor = float(mcfg.get("variance_floor", 1.0e-4))
        self.lambda_align = float(tcfg.get("lambda_align", mcfg.get("lambda_align", 1.0e-2)))
        self.use_reconstruction_loss = bool(
            mcfg.get("use_reconstruction_loss", True)
        )
        self.lambda_recon = float(
            tcfg.get("lambda_recon", mcfg.get("lambda_recon", 5.0e-2))
        )
        self.reconstruction_mask_mode = str(
            mcfg.get("reconstruction_mask", "water")
        ).lower()
        
        self.assert_local_prior = bool(config.get("debug", {}).get("assert_local_prior", False))
        self.multiply_alpha_by_reliability = bool(
            mcfg.get("multiply_alpha_by_reliability_mask", True)
        )

        # Encoder with spatial bottleneck
        self.encoder = VisualAutoencoderEncoder(
            in_channels=in_channels,
            latent_dim=feature_dim,
            dropout=dropout,
        )

        # Semantic grounding at bottleneck resolution
        self.semantic_grounding = RegionSemanticGrounding(
            feature_dim=feature_dim,
            text_dim=text_dim,
            align_dim=align_dim,
        )

        # Fusion at bottleneck resolution
        self.fusion = MultimodalFusion(feature_dim=feature_dim, dropout=dropout)

        # Reconstruction decoder: latent → RGB
        self.reconstruction_decoder = RGBReconstructionDecoder(
            latent_dim=feature_dim,
            dropout=dropout,
        )

        # Depth decoder: fused latent → full-resolution features
        self.depth_decoder = DepthDecoder(
            latent_dim=feature_dim,
            output_dim=decoder_feature_dim,
            dropout=dropout,
        )

        # Depth and variance prediction heads
        self.depth_head = DepthHead(
            feature_dim=decoder_feature_dim,
            variance_floor=self.variance_floor,
        )

        # Gating network
        self.gate = UncertaintyAwareGate(
            feature_dim=decoder_feature_dim,
            hidden_dim=gate_hidden,
            dropout=dropout,
        )

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str) -> torch.Tensor:
        """Resize tensor to match reference shape."""
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        if mode in {"bilinear", "bicubic"}:
            return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)
        return F.interpolate(x, size=ref.shape[-2:], mode=mode)

    def forward(
        self,
        image: torch.Tensor,
        reliability_mask: Optional[torch.Tensor] = None,
        disturbance_masks: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        region_valid_mask: Optional[torch.Tensor] = None,
        prior_depth_map: Optional[torch.Tensor] = None,
        prior_valid_mask: Optional[torch.Tensor] = None,
        prior_confidence: Optional[torch.Tensor] = None,
        water_mask: Optional[torch.Tensor] = None,
        depth_gt: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, _, height, width = image.shape
        device = image.device
        dtype = image.dtype

        # Initialize missing inputs
        if reliability_mask is None:
            reliability_mask = torch.zeros((bsz, 1, height, width), device=device, dtype=dtype)
        if prior_depth_map is None:
            prior_depth_map = torch.zeros((bsz, 1, height, width), device=device, dtype=dtype)
        if prior_valid_mask is None:
            prior_valid_mask = (prior_depth_map > 0).float()
        if prior_confidence is None:
            prior_confidence = torch.ones((bsz, 1, height, width), device=device, dtype=dtype)
        if water_mask is None:
            water_mask = torch.ones((bsz, 1, height, width), device=device, dtype=dtype)
        if valid_mask is None:
            valid_mask = water_mask
        if disturbance_masks is None:
            disturbance_masks = torch.zeros((bsz, 0, height, width), device=device, dtype=dtype)
        if text_embeddings is None:
            text_embeddings = torch.zeros((bsz, disturbance_masks.shape[1], 384), device=device, dtype=dtype)
        if region_valid_mask is None:
            region_area = disturbance_masks.flatten(2).sum(-1)
            region_valid_mask = (region_area > 0).float()

        # Resize all inputs to image resolution
        reliability_mask = self._resize_like(reliability_mask.float(), image, "nearest")
        water_mask = self._resize_like(water_mask.float(), image, "nearest")
        valid_mask = self._resize_like(valid_mask.float(), image, "nearest")
        prior_depth_map = self._resize_like(prior_depth_map.float(), image, "nearest")
        prior_valid_mask = self._resize_like(prior_valid_mask.float(), image, "nearest")
        prior_confidence = self._resize_like(prior_confidence.float(), image, "nearest")
        prior_valid_mask = (prior_valid_mask > 0.5).float() * water_mask
        prior_confidence = prior_confidence.clamp(0.0, 1.0) * prior_valid_mask
        if disturbance_masks.shape[1] > 0:
            disturbance_masks = self._resize_like(disturbance_masks.float(), image, "nearest")

        # 1. Spatial bottleneck encoding
        encoder_input = torch.cat([image, reliability_mask], dim=1)
        latent, encoder_features = self.encoder(encoder_input)

        # 2. Downsample problem-region masks to latent resolution
        if disturbance_masks.shape[1] > 0:
            latent_disturbance_masks = F.adaptive_max_pool2d(
                disturbance_masks.float(),
                output_size=latent.shape[-2:],
            )
            latent_disturbance_masks = (
                latent_disturbance_masks > 0.0
            ).float()
        else:
            latent_disturbance_masks = disturbance_masks

        latent_region_area = latent_disturbance_masks.flatten(2).sum(-1)
        latent_region_valid_mask = (
            region_valid_mask.float()
            * (latent_region_area > 0).float()
        )

        # 3. Region-text grounding at bottleneck resolution
        expert, align_loss, align_info = self.semantic_grounding(
            latent,
            latent_disturbance_masks,
            text_embeddings,
            latent_region_valid_mask,
        )

        # 4. Fuse latent visual representation and expert embedding
        fused_latent = self.fusion(latent, expert)

        # 5. Decode RGB from pure visual latent
        reconstruction = self.reconstruction_decoder(
            latent,
            output_size=(height, width),
        )

        # 6. Decode depth features from fused latent
        decoded_features = self.depth_decoder(
            fused_latent,
            output_size=(height, width),
        )

        # 7. Predict full-resolution depth and variance
        mu, var = self.depth_head(decoded_features)

        # 8. Predict full-resolution raw gate
        alpha_raw = self.gate(
            decoded_features,
            var,
            reliability_mask,
        )

        # Local prior correction logic
        d_phys = prior_depth_map.float() * prior_valid_mask
        alpha_eff = alpha_raw * prior_valid_mask * prior_confidence
        if self.multiply_alpha_by_reliability:
            alpha_eff = alpha_eff * reliability_mask
        depth = (1.0 - alpha_eff) * mu + alpha_eff * d_phys

        if self.assert_local_prior:
            if torch.max(prior_valid_mask * (1.0 - reliability_mask)) > 1e-5:
                raise ValueError("prior_valid_mask must be a subset of reliability_mask.")
            if torch.max(alpha_eff * (1.0 - reliability_mask)) > 1e-5:
                raise ValueError("alpha_eff must be zero outside reliability_mask.")

        depth = depth * water_mask
        mu = mu * water_mask
        d_phys = d_phys * water_mask
        alpha_raw = alpha_raw * water_mask
        alpha_eff = alpha_eff * water_mask
        var = var * water_mask + (1.0 - water_mask) * self.variance_floor

        # Compute losses
        if self.reconstruction_mask_mode == "water":
            reconstruction_mask = water_mask
        elif self.reconstruction_mask_mode == "valid":
            reconstruction_mask = valid_mask
        elif self.reconstruction_mask_mode == "all":
            reconstruction_mask = None
        else:
            raise ValueError(
                "model.reconstruction_mask must be one of: "
                "water, valid, all"
            )

        if self.use_reconstruction_loss:
            recon_loss = masked_l1_loss(
                reconstruction,
                image,
                reconstruction_mask,
            )
        else:
            recon_loss = depth.new_zeros(())

        # NLL loss using intersection of valid depth and water mask
        loss_mask = valid_mask * water_mask

        if depth_gt is not None:
            nll_loss = masked_gaussian_nll(
                depth,
                depth_gt,
                var,
                loss_mask,
                self.variance_floor,
            )
            total = (
                nll_loss
                + self.lambda_align * align_loss
                + self.lambda_recon * recon_loss
            )
        else:
            nll_loss = depth.new_zeros(())
            total = depth.new_zeros(())

        info: Dict[str, torch.Tensor] = {
            "depth": depth,
            "mu": mu,
            "depth_model": mu,
            "d_phys": d_phys,
            "depth_llm": d_phys,
            "var": var,
            "sigma2": var,
            "prior_valid_mask": prior_valid_mask,
            "prior_confidence": prior_confidence,
            "reliability_mask": reliability_mask,
            "alpha_raw": alpha_raw,
            "alpha": alpha_eff,
            "alpha_eff": alpha_eff,
            "w_model": (1.0 - alpha_eff) * water_mask,
            "w_llm": alpha_eff,
            "A": fused_latent,
            "F_vis": latent,
            "latent": latent,
            "fused_latent": fused_latent,
            "decoded_features": decoded_features,
            "reconstruction": reconstruction,
            "recon_rgb": reconstruction,
            "recon_loss": recon_loss,
            "latent_disturbance_masks": latent_disturbance_masks,
            "latent_region_valid_mask": latent_region_valid_mask,
            "E_proj": expert,
            "align_loss": align_loss,
            "cl_loss": align_loss,
            "nll_loss": nll_loss,
            "depth_loss": nll_loss,
            "total": total,
        }
        info.update(align_info)
        return depth, info


DepthAutoencoderModel = LLMGuidedBathymetryModel
