import torch
from torch import nn

from models.nn.Add_modules import autopad


class Conv(nn.Module):
    """Standard convolution with optional activation."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


if __name__ == '__main__':
    x = (1, 3, 640, 640)
    image = torch.rand(*x)
    model1 = Conv(3, 64, s=3, p=2)
    model2 = Conv(64, 128, s=3, p=2)
    out1 = model1(image)
    print(out1.shape)
    out2 = model2(out1)
    print(out2.shape)