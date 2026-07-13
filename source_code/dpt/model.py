import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DensePredictionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        width: int = 128,
        patch_size: int = 8,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch = nn.Conv2d(in_channels, width, kernel_size=self.patch_size, stride=self.patch_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            batch_first=True,
            dropout=float(dropout),
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(layers))
        self.refine = nn.Sequential(
            ConvNormAct(width, width // 2, 3),
            nn.Conv2d(width // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        z = self.patch(x)
        b, c, ph, pw = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        z = tokens.transpose(1, 2).reshape(b, c, ph, pw)
        z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
        return self.refine(z)

