import torch
import torch.nn as nn


class GlobalContextExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1, 1)

    def forward(self, x):
        return self.proj(self.pool(x))


