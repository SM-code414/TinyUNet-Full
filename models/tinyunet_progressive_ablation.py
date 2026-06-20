"""
tinyunet_progressive_ablation.py
────────────────────────────────
Progressive TinyUNet ablation pipeline for BraTS Meningioma.

Designed to work DIRECTLY with your existing:
    - train.py
    - evaluate.py
    - visualization.py
    - metrics.py
    - dataloaders
    - inference pipeline

NO changes required in:
    - training loop
    - optimizer
    - scheduler
    - evaluation
    - checkpointing

────────────────────────────────────────────────────────────────────
PROGRESSIVE MANUSCRIPT ORDER
────────────────────────────────────────────────────────────────────

1. TinyUNet_Baseline
    Plain TinyUNet

2. TinyUNet_Edge
    + Sobel edge fusion

3. TinyUNet_Edge_DS
    + Deep supervision

4. TinyUNet_Full
    + Boundary-aware training support

Boundary loss is NOT embedded inside model forward.
It should be added in train.py loss computation ONLY.

This keeps evaluate.py fully compatible.

────────────────────────────────────────────────────────────────────
IMPORTANT
────────────────────────────────────────────────────────────────────

Use:
    base = 32

for ALL experiments for fair reviewer comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import DoubleConv3D


# ============================================================
# GroupNorm helper
# ============================================================

def _gn(channels, groups=8):
    g = groups if channels % groups == 0 else 1
    return nn.GroupNorm(g, channels)


# ============================================================
# EDGE MODULE
# ============================================================

class SobelEdge3D(nn.Module):
    """
    3D Sobel edge extraction.

    Input:
        (B,C,D,H,W)

    Output:
        (B,C,D,H,W)

    Kx, Ky, Kz are derived from a single base kernel via index permutation,
    one for each axis (width, height, depth respectively). Verified against
    synthetic single-axis ramp volumes to confirm no cross-axis response.
    """

    def __init__(self):
        super().__init__()

        Kx = torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],

            [[-2, 0, 2],
             [-4, 0, 4],
             [-2, 0, 2]],

            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
        ], dtype=torch.float32)

        Ky = Kx.permute(0, 2, 1)
        Kz = Kx.permute(2, 1, 0)

        self.register_buffer("Kx", Kx.unsqueeze(0).unsqueeze(0))
        self.register_buffer("Ky", Ky.unsqueeze(0).unsqueeze(0))
        self.register_buffer("Kz", Kz.unsqueeze(0).unsqueeze(0))

    def forward(self, x):

        B, C, D, H, W = x.shape

        edges = []

        for c in range(C):

            ch = x[:, c:c+1]

            gx = F.conv3d(ch, self.Kx, padding=1)
            gy = F.conv3d(ch, self.Ky, padding=1)
            gz = F.conv3d(ch, self.Kz, padding=1)

            mag = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-6)

            edges.append(mag)

        return torch.cat(edges, dim=1)


# ============================================================
# BASE TINYUNET
# ============================================================

class TinyUNet_Baseline(nn.Module):

    def __init__(self,
                 in_channels=4,
                 num_classes=4,
                 base=32):

        super().__init__()

        B = base

        self.pool = nn.MaxPool3d(2)

        self.enc1 = DoubleConv3D(in_channels, B)
        self.enc2 = DoubleConv3D(B, B * 2)
        self.enc3 = DoubleConv3D(B * 2, B * 4)

        self.bottleneck = DoubleConv3D(B * 4, B * 8)

        self.up3 = nn.ConvTranspose3d(B * 8, B * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(B * 4, B * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(B * 2, B, 2, 2)

        self.dec3 = DoubleConv3D(B * 8, B * 4)
        self.dec2 = DoubleConv3D(B * 4, B * 2)
        self.dec1 = DoubleConv3D(B * 2, B)

        self.outc = nn.Conv3d(B, num_classes, 1)

    def forward(self, x):

        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))

        b = self.bottleneck(self.pool(s3))

        x = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return self.outc(x)


# ============================================================
# EDGE AWARE TINYUNET
# ============================================================

class TinyUNet_Edge(nn.Module):

    def __init__(self,
                 in_channels=4,
                 num_classes=4,
                 base=32):

        super().__init__()

        B = base

        self.pool = nn.MaxPool3d(2)

        self.sobel = SobelEdge3D()

        self.enc1 = DoubleConv3D(in_channels * 2, B)
        self.enc2 = DoubleConv3D(B, B * 2)
        self.enc3 = DoubleConv3D(B * 2, B * 4)

        self.bottleneck = DoubleConv3D(B * 4, B * 8)

        self.up3 = nn.ConvTranspose3d(B * 8, B * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(B * 4, B * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(B * 2, B, 2, 2)

        self.dec3 = DoubleConv3D(B * 8, B * 4)
        self.dec2 = DoubleConv3D(B * 4, B * 2)
        self.dec1 = DoubleConv3D(B * 2, B)

        self.outc = nn.Conv3d(B, num_classes, 1)

    def forward(self, x):

        edges = self.sobel(x)

        x_in = torch.cat([x, edges], dim=1)

        s1 = self.enc1(x_in)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))

        b = self.bottleneck(self.pool(s3))

        x = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return self.outc(x)


# ============================================================
# EDGE + DEEP SUPERVISION
# ============================================================

class TinyUNet_Edge_DS(nn.Module):

    """
    Deep supervision version.

    TRAINING:
        returns:
            main_out, aux2, aux3

    EVAL:
        returns:
            main_out

    This preserves evaluate.py compatibility.
    """

    def __init__(self,
                 in_channels=4,
                 num_classes=4,
                 base=32):

        super().__init__()

        B = base

        self.pool = nn.MaxPool3d(2)

        self.sobel = SobelEdge3D()

        self.enc1 = DoubleConv3D(in_channels * 2, B)
        self.enc2 = DoubleConv3D(B, B * 2)
        self.enc3 = DoubleConv3D(B * 2, B * 4)

        self.bottleneck = DoubleConv3D(B * 4, B * 8)

        self.up3 = nn.ConvTranspose3d(B * 8, B * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(B * 4, B * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(B * 2, B, 2, 2)

        self.dec3 = DoubleConv3D(B * 8, B * 4)
        self.dec2 = DoubleConv3D(B * 4, B * 2)
        self.dec1 = DoubleConv3D(B * 2, B)

        self.outc = nn.Conv3d(B, num_classes, 1)

        # deep supervision heads
        self.ds2 = nn.Conv3d(B * 2, num_classes, 1)
        self.ds3 = nn.Conv3d(B * 4, num_classes, 1)

    def forward(self, x):

        input_size = x.shape[2:]

        edges = self.sobel(x)

        x_in = torch.cat([x, edges], dim=1)

        s1 = self.enc1(x_in)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))

        b = self.bottleneck(self.pool(s3))

        d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        out = self.outc(d1)

        # inference compatibility
        if not self.training:
            return out

        aux3 = self.ds3(d3)
        aux2 = self.ds2(d2)

        aux3 = F.interpolate(
            aux3,
            size=input_size,
            mode="trilinear",
            align_corners=False
        )

        aux2 = F.interpolate(
            aux2,
            size=input_size,
            mode="trilinear",
            align_corners=False
        )

        return out, aux2, aux3


# ============================================================
# FULL MODEL
# EDGE + DEEP SUPERVISION + BOUNDARY SUPPORT
# ============================================================

class TinyUNet_Full(TinyUNet_Edge_DS):
    """
    Same architecture as Edge_DS.

    Boundary loss is added externally in train.py.

    This keeps:
        evaluate.py
        inference
        visualization

    fully compatible.
    """
    pass


# ============================================================
# REGISTRY
# ============================================================

ABLATION_REGISTRY = {

    "tinyunet_baseline": TinyUNet_Baseline,

    "tinyunet_edge": TinyUNet_Edge,

    "tinyunet_edge_ds": TinyUNet_Edge_DS,

    "tinyunet_full": TinyUNet_Full,
}


def get_ablation_model(name, **kwargs):

    name = name.lower()

    if name not in ABLATION_REGISTRY:

        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(ABLATION_REGISTRY.keys())}"
        )

    return ABLATION_REGISTRY[name](**kwargs)
