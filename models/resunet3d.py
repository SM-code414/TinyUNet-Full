import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import ConvGNReLU


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvGNReLU(in_ch, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn(self.conv2(out))
        return F.relu(out + self.skip(x), inplace=True)


class ResUNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base=32):
        super().__init__()
        self.pool = nn.MaxPool3d(2)

        self.enc1 = ResidualBlock3D(in_channels, base)
        self.enc2 = ResidualBlock3D(base, base * 2)
        self.enc3 = ResidualBlock3D(base * 2, base * 4)
        self.enc4 = ResidualBlock3D(base * 4, base * 8)

        self.bottleneck = ResidualBlock3D(base * 8, base * 8)

        self.up4 = nn.ConvTranspose3d(base * 8, base * 8, 2, 2)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, 2)

        self.dec4 = ResidualBlock3D(base * 16, base * 8)
        self.dec3 = ResidualBlock3D(base * 8, base * 4)
        self.dec2 = ResidualBlock3D(base * 4, base * 2)
        self.dec1 = ResidualBlock3D(base * 2, base)

        self.outc = nn.Conv3d(base, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        x = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return self.outc(x)

