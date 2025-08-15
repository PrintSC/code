import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models.sam.modules.blocks import DropPath


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        b, _, h, w = x.size()
        xx = torch.linspace(-1, 1, w, device=x.device).repeat(h, 1).unsqueeze(0)
        yy = torch.linspace(-1, 1, h, device=x.device).unsqueeze(1).repeat(1, w).unsqueeze(0)
        xx = xx.expand(b, -1, -1, -1)
        yy = yy.expand(b, -1, -1, -1)
        x = torch.cat([x, xx, yy], dim=1)
        return self.conv(x)


class GlobalContextFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_scales = [1, 3, 5]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(inplace=True)
            ) for scale in self.pool_scales
        ])
        self.project = nn.Conv2d(out_channels * len(self.pool_scales), out_channels, 1)

    def forward(self, x):
        features = [F.interpolate(conv(x), x.shape[2:], mode='nearest') for conv in self.convs]
        out = torch.cat(features, dim=1)
        return self.project(out)


class GatedFusionAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        attn = self.gate(x)
        spatial = self.spatial(x)
        out = x * attn + spatial
        return self.out_proj(out)


class MLPEnhancer(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.mlp(x)


class FocusBlock(nn.Module):
    def __init__(self, dim, ctx_dim=None, mlp_ratio=4, drop_path=0.1):
        super().__init__()
        ctx_dim = ctx_dim or dim // 4
        self.coord_embed = CoordConv(dim, dim)
        self.ctx_extractor = GlobalContextFusion(dim, ctx_dim)
        self.attn = GatedFusionAttention(dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLPEnhancer(dim, dim * mlp_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, ctx=None):
        z = self.coord_embed(x)
        p = self.ctx_extractor(x) if ctx is None else ctx
        p_up = F.interpolate(p, size=z.shape[2:], mode='nearest')
        fused = z + self.drop_path(self.attn(z + p_up))
        fused = self.norm1(fused)
        out = self.norm2(fused + self.mlp(fused))
        return out
