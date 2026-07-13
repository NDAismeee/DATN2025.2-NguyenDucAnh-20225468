from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNRegressor(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, width),
            ConvBlock(width, width),
            nn.Conv2d(width, 1, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetRegressor(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 24):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.head = nn.Sequential(nn.Conv2d(width, 1, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


class ProposedRegressor(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 32):
        super().__init__()
        self.encoder = UNetRegressor(in_channels, width)
        self.prior = nn.Sequential(ConvBlock(in_channels + 1, width), nn.Conv2d(width, 1, 1), nn.Softplus())
        self.gate = nn.Sequential(ConvBlock(in_channels + 2, width), nn.Conv2d(width, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neural = self.encoder(x)
        intensity = x.mean(dim=1, keepdim=True)
        reliability = torch.clamp(1.0 - torch.abs(intensity - intensity.mean(dim=(-2, -1), keepdim=True)) * 2.0, 0.0, 1.0)
        prior = self.prior(torch.cat([x, reliability], dim=1))
        gate = self.gate(torch.cat([x, reliability, neural], dim=1))
        return (1.0 - gate) * neural + gate * prior


class ProposedPaperRegressor(nn.Module):
    def __init__(self, in_channels: int = 4, width: int = 32, embed_dim: int = 1536):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.mu_head = nn.Sequential(nn.Conv2d(width, 1, 1), nn.Softplus())
        self.logvar_head = nn.Sequential(nn.Conv2d(width, 1, 1), nn.Tanh())
        self.embed_proj = nn.Sequential(nn.Linear(embed_dim, width), nn.ReLU(inplace=True))
        self.gate = nn.Sequential(ConvBlock(width + 2, width), nn.Conv2d(width, 1, 1), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        *,
        reliability: torch.Tensor,
        d_phys: torch.Tensor,
        expert_embedding: torch.Tensor,
        return_mu: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        mu = self.mu_head(d1)
        logvar = self.logvar_head(d1)
        sigma2 = F.softplus(logvar) + 1e-6

        pooled = d1.mean(dim=(-2, -1))
        exp_proj = self.embed_proj(expert_embedding)
        exp_proj = exp_proj / (exp_proj.norm(dim=1, keepdim=True) + 1e-6)
        pooled_n = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-6)
        align = (pooled_n * exp_proj).sum(dim=1)

        gate_in = torch.cat([d1, sigma2, reliability], dim=1)
        alpha = self.gate(gate_in)
        d_hat = (1.0 - alpha) * mu + alpha * d_phys
        if return_mu:
            return d_hat, sigma2, alpha, align, mu
        return d_hat, sigma2, alpha, align


class DASDBRegressor(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 32):
        super().__init__()
        self.shared = UNetRegressor(in_channels, width)
        self.residual = nn.Sequential(ConvBlock(in_channels + 1, width), nn.Conv2d(width, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.shared(x)
        shallow_prior = torch.relu(1.0 - x[:, 1:2]) * 4.0
        return F.softplus(base + self.residual(torch.cat([x, shallow_prior], dim=1)))


class DensePredictionTransformer(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 64, patch_size: int = 8, layers: int = 2, heads: int = 4):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, width, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * 4, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.refine = nn.Sequential(ConvBlock(width, width // 2), nn.Conv2d(width // 2, 1, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        z = self.patch(x)
        batch, channels, ph, pw = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        z = tokens.transpose(1, 2).reshape(batch, channels, ph, pw)
        z = F.interpolate(z, size=(height, width), mode="bilinear", align_corners=False)
        return self.refine(z)


def build_torch_model(model_key: str, in_channels: int = 3) -> nn.Module:
    if model_key == "cnn":
        return CNNRegressor(in_channels)
    if model_key == "unet":
        return UNetRegressor(in_channels)
    if model_key == "proposed":
        return ProposedRegressor(in_channels)
    if model_key == "da_sdb":
        return DASDBRegressor(in_channels)
    if model_key in {"dpt", "depth_anything_v2"}:
        return DensePredictionTransformer(in_channels)
    raise ValueError(f"Unsupported torch model: {model_key}")
