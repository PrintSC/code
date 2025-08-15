import torch
import torch.nn as nn
from timm.models.layers import DropPath
from models.nn.Add_modules.BatchNorm import LayerNorm2d, DilatedReparamBlock, SEModule, GRN, LayerScale


class ResDWConv(nn.Conv2d):
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + super().forward(x)


class MultiScaleConv(nn.Module):
    def __init__(self, dim, scales=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim) for k in scales
        ])
        self.project = nn.Conv2d(dim * len(scales), dim, kernel_size=1)

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        x = torch.cat(feats, dim=1)
        return self.project(x)


class VANBlock(nn.Module):
    def __init__(self, dim, ffn_expansion=2):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * ffn_expansion, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * ffn_expansion, dim, kernel_size=1)
        )

    def forward(self, x):
        x = x + self.conv(self.norm(x))
        x = x + self.mlp(x)
        return x


class EnhancedBasicBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=7,
                 mlp_ratio=4,
                 drop_path=0.0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 ls_init_value=None,
                 res_scale=False):
        super().__init__()
        self.res_scale = res_scale
        mlp_dim = int(dim * mlp_ratio)

        self.dwconv = ResDWConv(dim, kernel_size=3)

        self.use_dilated = kernel_size >= 5
        if self.use_dilated:
            conv_block = DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy,
                                             attempt_use_lk_impl=use_gemm)
        else:
            conv_block = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
                                   groups=dim, bias=False)

        self.block = nn.Sequential(
            norm_layer(dim),
            conv_block,
            nn.BatchNorm2d(dim),
            SEModule(dim),
            MultiScaleConv(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            VANBlock(dim),
        )

        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)
        residual = x
        out = self.block(x)

        if self.res_scale:
            return self.ls(residual) + self.drop_path(out)
        else:
            return residual + self.drop_path(self.ls(out))


class ConvStemBlock(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 kernel_size=7,
                 mlp_ratio=4,
                 drop_path=0.0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 ls_init_value=None,
                 res_scale=False):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.block = EnhancedBasicBlock(
            dim=out_channels,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_gemm=use_gemm,
            deploy=deploy,
            ls_init_value=ls_init_value,
            res_scale=res_scale
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        return x
