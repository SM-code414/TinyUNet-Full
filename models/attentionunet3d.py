import torch
import torch.nn as nn
from models.blocks import DoubleConv3D


class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv3d(F_g, F_int, 1, bias=False)
        self.W_x = nn.Conv3d(F_l, F_int, 1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(F_int, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        return x * self.psi(self.W_g(g) + self.W_x(x))


class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base=32):
        super().__init__()
        self.pool = nn.MaxPool3d(2)

        self.enc1 = DoubleConv3D(in_channels, base)
        self.enc2 = DoubleConv3D(base, base * 2)
        self.enc3 = DoubleConv3D(base * 2, base * 4)
        self.enc4 = DoubleConv3D(base * 4, base * 8)

        self.bottleneck = DoubleConv3D(base * 8, base * 16)

        self.att4 = AttentionGate3D(base * 8, base * 8, base * 4)
        self.att3 = AttentionGate3D(base * 4, base * 4, base * 2)
        self.att2 = AttentionGate3D(base * 2, base * 2, base)
        self.att1 = AttentionGate3D(base, base, base // 2)

        self.up4 = nn.ConvTranspose3d(base * 16, base * 8, 2, 2)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, 2)

        self.dec4 = DoubleConv3D(base * 16, base * 8)
        self.dec3 = DoubleConv3D(base * 8, base * 4)
        self.dec2 = DoubleConv3D(base * 4, base * 2)
        self.dec1 = DoubleConv3D(base * 2, base)

        self.outc = nn.Conv3d(base, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        g4 = self.up4(b)
        s4 = self.att4(g4, s4)
        x = self.dec4(torch.cat([g4, s4], dim=1))

        g3 = self.up3(x)
        s3 = self.att3(g3, s3)
        x = self.dec3(torch.cat([g3, s3], dim=1))

        g2 = self.up2(x)
        s2 = self.att2(g2, s2)
        x = self.dec2(torch.cat([g2, s2], dim=1))

        g1 = self.up1(x)
        s1 = self.att1(g1, s1)
        x = self.dec1(torch.cat([g1, s1], dim=1))

        return self.outc(x)

