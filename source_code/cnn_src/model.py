import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        norm_type: str = "group",
        num_groups: int = 8,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        padding = dilation * (kernel_size // 2)
        bias = norm_type in [None, "none"]

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
        ]

        norm = self._make_norm(out_channels, norm_type, num_groups)
        if norm is not None:
            layers.append(norm)

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation in [None, "none"]:
            pass
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    @staticmethod
    def _make_norm(channels: int, norm_type: str, num_groups: int):
        if norm_type is None or norm_type == "none":
            return None
        if norm_type == "batch":
            return nn.BatchNorm2d(channels)
        if norm_type == "instance":
            return nn.InstanceNorm2d(channels, affine=True)
        if norm_type == "group":
            groups = min(num_groups, channels)
            while channels % groups != 0 and groups > 1:
                groups -= 1
            return nn.GroupNorm(groups, channels)

        raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        norm_type: str = "group",
        num_groups: int = 8,
        dropout: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()

        self.conv1 = ConvNormAct(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            dilation=dilation,
            norm_type=norm_type,
            num_groups=num_groups,
            activation="relu",
            dropout=dropout,
        )
        self.conv2 = ConvNormAct(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            dilation=dilation,
            norm_type=norm_type,
            num_groups=num_groups,
            activation="none",
            dropout=0.0,
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.act(out)
        return out


class CoordConvInput(nn.Module):
    """
    Append normalized x/y coordinate channels to the input.
    Input : [B, C, H, W]
    Output: [B, C+2, H, W]
    """

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype).view(1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w)

        yy = yy.expand(b, 1, h, w)
        xx = xx.expand(b, 1, h, w)

        return torch.cat([x, xx, yy], dim=1)


class SimpleBathymetryCNN(nn.Module):
    """
    Improved lightweight CNN for dense bathymetry regression.

    Main improvements over the old version:
    - residual learning
    - more stable normalization for small patch training
    - dilated context block for a larger receptive field
    - optional CoordConv positional channels
    - skip fusion from shallow and deep features

    Input:
        x: [B, in_channels, H, W]

    Output:
        y: [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int = 13,
        hidden_channels: tuple = (32, 64, 64, 32),
        use_batchnorm: bool = False,
        dropout: float = 0.0,
        use_coordconv: bool = True,
        norm_type: str = "group",
        num_groups: int = 8,
    ):
        super().__init__()

        if len(hidden_channels) != 4:
            raise ValueError(
                f"hidden_channels must have length 4, got {len(hidden_channels)}"
            )

        c1, c2, c3, c4 = hidden_channels

        if use_batchnorm:
            norm_type = "batch"

        self.use_coordconv = use_coordconv
        self.coord = CoordConvInput(enabled=use_coordconv)

        input_channels = in_channels + 2 if use_coordconv else in_channels

        # stem
        self.stem = ConvNormAct(
            in_channels=input_channels,
            out_channels=c1,
            kernel_size=3,
            norm_type=norm_type,
            num_groups=num_groups,
            activation="relu",
            dropout=0.0,
        )

        # encoder-like feature extraction without spatial downsampling
        self.stage1_proj = ConvNormAct(
            in_channels=c1,
            out_channels=c2,
            kernel_size=3,
            norm_type=norm_type,
            num_groups=num_groups,
            activation="relu",
            dropout=dropout,
        )
        self.stage1_res = ResidualBlock(
            channels=c2,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
            dilation=1,
        )

        self.stage2_proj = ConvNormAct(
            in_channels=c2,
            out_channels=c3,
            kernel_size=3,
            norm_type=norm_type,
            num_groups=num_groups,
            activation="relu",
            dropout=dropout,
        )
        self.stage2_res = ResidualBlock(
            channels=c3,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
            dilation=1,
        )

        # larger receptive field for bathymetry context
        self.context = ResidualBlock(
            channels=c3,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
            dilation=2,
        )

        # decoder/fusion
        self.fuse = ConvNormAct(
            in_channels=c1 + c2 + c3,
            out_channels=c4,
            kernel_size=3,
            norm_type=norm_type,
            num_groups=num_groups,
            activation="relu",
            dropout=dropout,
        )

        self.refine = ResidualBlock(
            channels=c4,
            norm_type=norm_type,
            num_groups=num_groups,
            dropout=dropout,
            dilation=1,
        )

        self.head = nn.Sequential(
            ConvNormAct(
                in_channels=c4,
                out_channels=max(c4 // 2, 16),
                kernel_size=3,
                norm_type=norm_type,
                num_groups=num_groups,
                activation="relu",
                dropout=0.0,
            ),
            nn.Conv2d(max(c4 // 2, 16), 1, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W]
        x = self.coord(x)

        s0 = self.stem(x)            # [B, c1, H, W]
        s1 = self.stage1_proj(s0)    # [B, c2, H, W]
        s1 = self.stage1_res(s1)     # [B, c2, H, W]

        s2 = self.stage2_proj(s1)    # [B, c3, H, W]
        s2 = self.stage2_res(s2)     # [B, c3, H, W]
        s2 = self.context(s2)        # [B, c3, H, W]

        fused = torch.cat([s0, s1, s2], dim=1)
        fused = self.fuse(fused)     # [B, c4, H, W]
        fused = self.refine(fused)   # [B, c4, H, W]

        out = self.head(fused)       # [B, 1, H, W]
        return out


if __name__ == "__main__":
    model = SimpleBathymetryCNN(
        in_channels=13,
        hidden_channels=(32, 64, 64, 32),
        use_batchnorm=False,
        dropout=0.05,
        use_coordconv=True,
        norm_type="group",
        num_groups=8,
    )

    x = torch.randn(4, 13, 18, 18)
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Num params  :", sum(p.numel() for p in model.parameters()))