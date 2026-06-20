import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNReLU(nn.Module):
    """
    Standardized Conv3D block:
    Conv3D → GroupNorm → ReLU
    Safe for batch size = 1
    """
    def __init__(self, in_ch, out_ch, k=3, p=1, groups=8):
        super().__init__()
        g = groups if out_ch % groups == 0 else 1
        self.conv = nn.Conv3d(in_ch, out_ch, k, padding=p, bias=False)
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class DoubleConv3D(nn.Module):
    """
    Two consecutive ConvGNReLU blocks
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNReLU(in_ch, out_ch),
            ConvGNReLU(out_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)

