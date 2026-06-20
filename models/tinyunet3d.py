import torch
import torch.nn as nn
from models.blocks import DoubleConv3D


class TinyUNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base=16):
        super().__init__()
        self.pool = nn.MaxPool3d(2)

        self.enc1 = DoubleConv3D(in_channels, base)
        self.enc2 = DoubleConv3D(base, base * 2)
        self.enc3 = DoubleConv3D(base * 2, base * 4)

        self.bottleneck = DoubleConv3D(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, 2)

        self.dec3 = DoubleConv3D(base * 8, base * 4)
        self.dec2 = DoubleConv3D(base * 4, base * 2)
        self.dec1 = DoubleConv3D(base * 2, base)

        self.outc = nn.Conv3d(base, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))

        b = self.bottleneck(self.pool(s3))

        x = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return self.outc(x)

