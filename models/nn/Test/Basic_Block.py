import torch
import torch.nn as nn
from timm.models.layers import DropPath

from models.nn.Add_modules.BatchNorm import LayerNorm2d, DilatedReparamBlock, SEModule, GRN, LayerScale


class ResDWConv(nn.Conv2d):
    """Depthwise convolution with residual connection."""

    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + super().forward(x)


class BasicBlock(nn.Module):
    """
    Basic Block as shown in the diagram:
    [DWConv] -> [Norm] -> [Dilated RepConv] -> [SE Layer] -> [ConvFFN] with residuals
    """

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

        self.block = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
        )

        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)  # First DWConv + residual
        residual = x
        out = self.block(x)

        if self.res_scale:
            return self.ls(residual) + self.drop_path(out)
        else:
            return residual + self.drop_path(self.ls(out))


block = BasicBlock(dim=3, kernel_size=7, mlp_ratio=4, drop_path=0.1)
x = torch.randn(1, 3, 640, 640)
out = block(x)
print(out.shape)
