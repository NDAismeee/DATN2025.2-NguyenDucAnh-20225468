import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class CoordConvInput(nn.Module):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype
        yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xx, yy], dim=1)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        norm_type: str = "group",
        num_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        bias = norm_type in (None, "none")
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)]
        if norm_type == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm_type == "instance":
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif norm_type == "group":
            groups = min(num_groups, out_channels)
            while out_channels % groups != 0 and groups > 1:
                groups -= 1
            layers.append(nn.GroupNorm(groups, out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, norm_type: str, num_groups: int, dropout: float, dilation: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, dilation=dilation, norm_type=norm_type, num_groups=num_groups, dropout=dropout)
        self.conv2 = ConvNormAct(channels, channels, dilation=dilation, norm_type=norm_type, num_groups=num_groups, dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels=(32, 64, 64, 32),
        dropout: float = 0.05,
        use_coordconv: bool = True,
        norm_type: str = "group",
        num_groups: int = 8,
    ):
        super().__init__()
        c1, c2, c3, c4 = hidden_channels
        self.coord = CoordConvInput(enabled=use_coordconv)
        input_channels = in_channels + (2 if use_coordconv else 0)
        self.stem = ConvNormAct(input_channels, c1, norm_type=norm_type, num_groups=num_groups, dropout=0.0)
        self.stage1 = nn.Sequential(
            ConvNormAct(c1, c2, norm_type=norm_type, num_groups=num_groups, dropout=dropout),
            ResidualBlock(c2, norm_type, num_groups, dropout),
        )
        self.stage2 = nn.Sequential(
            ConvNormAct(c2, c3, norm_type=norm_type, num_groups=num_groups, dropout=dropout),
            ResidualBlock(c3, norm_type, num_groups, dropout),
            ResidualBlock(c3, norm_type, num_groups, dropout, dilation=2),
        )
        self.fuse = ConvNormAct(c1 + c2 + c3, c4, norm_type=norm_type, num_groups=num_groups, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coord(x)
        s0 = self.stem(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        z = torch.cat([s0, s1, s2], dim=1)
        return self.fuse(z)


class DepthHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class DomainClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class DASDB(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels=(32, 64, 64, 32),
        dropout: float = 0.05,
        use_coordconv: bool = True,
        norm_type: str = "group",
        num_groups: int = 8,
        domain_hidden: int = 128,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            use_coordconv=use_coordconv,
            norm_type=norm_type,
            num_groups=num_groups,
        )
        feat_ch = hidden_channels[-1]
        self.depth_head = DepthHead(feat_ch)
        self.domain_head = DomainClassifier(feat_ch, hidden=domain_hidden)

    def forward_depth(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.depth_head(feat)

    def forward_domain(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        feat = self.encoder(x)
        feat = GradReverse.apply(feat, float(alpha))
        return self.domain_head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_depth(x)

