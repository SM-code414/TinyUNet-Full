"""
tinyunet_ablations.py
─────────────────────
Four TinyUNet ablation variants for BraTS Meningioma.
Drop this file into your project directory alongside tinyunet3d.py.

Variants
────────
  TinyUNet_DW    — Depthwise Separable Conv  (reduce params, maintain receptive field)
  TinyUNet_Ghost — GhostNet-style conv        (cheap ops generate redundant features)
  TinyUNet_SE    — Squeeze-and-Excitation     (channel attention after each block)
  TinyUNet_Edge  — Edge-enhanced              (Sobel edge maps fused at encoder input)

All variants keep the same macro architecture as TinyUNet3D:
  3-level encoder → bottleneck → 3-level decoder → 1×1 output
  GroupNorm throughout (safe for batch_size=1 on free Colab T4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared primitive: GroupNorm-safe normalisation
# ============================================================

def _gn(channels, groups=8):
    """GroupNorm with fallback to groups=1 if channels < groups."""
    g = groups if channels % groups == 0 else 1
    return nn.GroupNorm(g, channels)


# ============================================================
# 1. DEPTHWISE SEPARABLE CONV variant
#    Standard Conv3d → DepthwiseConv3d + PointwiseConv3d
#    Reduces params by ~8-9x per conv, similar receptive field.
# ============================================================

class DWConvGNReLU(nn.Module):
    """Depthwise separable Conv3D → GroupNorm → ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, 3, padding=1,
                            groups=in_ch, bias=False)   # depthwise
        self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=False)  # pointwise
        self.gn  = _gn(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.pw(self.dw(x))))


class DWDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            DWConvGNReLU(in_ch, out_ch),
            DWConvGNReLU(out_ch, out_ch),
        )
    def forward(self, x):
        return self.block(x)


class TinyUNet_DW(nn.Module):
    """TinyUNet with Depthwise Separable Convolutions."""
    def __init__(self, in_channels=4, num_classes=4, base=16):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        B = base

        self.enc1      = DWDoubleConv(in_channels, B)
        self.enc2      = DWDoubleConv(B,     B * 2)
        self.enc3      = DWDoubleConv(B * 2, B * 4)
        self.bottleneck= DWDoubleConv(B * 4, B * 8)

        self.up3 = nn.ConvTranspose3d(B * 8, B * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(B * 4, B * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(B * 2, B,     2, 2)

        self.dec3 = DWDoubleConv(B * 8, B * 4)
        self.dec2 = DWDoubleConv(B * 4, B * 2)
        self.dec1 = DWDoubleConv(B * 2, B)

        self.outc = nn.Conv3d(B, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b  = self.bottleneck(self.pool(s3))
        x  = self.dec3(torch.cat([self.up3(b),  s3], dim=1))
        x  = self.dec2(torch.cat([self.up2(x),  s2], dim=1))
        x  = self.dec1(torch.cat([self.up1(x),  s1], dim=1))
        return self.outc(x)


# ============================================================
# 2. GHOST CONV variant
#    GhostNet (Han et al. 2020): generate n/2 features with
#    standard conv, then apply cheap 3×3 depthwise conv to
#    each to produce another n/2 — concatenated to give n.
#    Reduces FLOPs ~2x vs standard conv.
# ============================================================

class GhostConv3D(nn.Module):
    """
    Ghost Module for 3D:
      half features from 1×1 conv (primary)
      half from cheap 3×3 depthwise (ghost)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        half = out_ch // 2
        self.primary = nn.Sequential(
            nn.Conv3d(in_ch, half, 1, bias=False),
            _gn(half),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv3d(half, half, 3, padding=1,
                      groups=half, bias=False),   # depthwise
            _gn(half),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        p = self.primary(x)
        g = self.cheap(p)
        return torch.cat([p, g], dim=1)


class GhostDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = GhostConv3D(in_ch,  out_ch)
        self.c2 = GhostConv3D(out_ch, out_ch)

    def forward(self, x):
        return self.c2(self.c1(x))


class TinyUNet_Ghost(nn.Module):
    """TinyUNet with Ghost Convolutions."""
    def __init__(self, in_channels=4, num_classes=4, base=16):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        B = base

        self.enc1      = GhostDoubleConv(in_channels, B)
        self.enc2      = GhostDoubleConv(B,     B * 2)
        self.enc3      = GhostDoubleConv(B * 2, B * 4)
        self.bottleneck= GhostDoubleConv(B * 4, B * 8)

        self.up3 = nn.ConvTranspose3d(B * 8, B * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(B * 4, B * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(B * 2, B,     2, 2)

        self.dec3 = GhostDoubleConv(B * 8, B * 4)
        self.dec2 = GhostDoubleConv(B * 4, B * 2)
        self.dec1 = GhostDoubleConv(B * 2, B)

        self.outc = nn.Conv3d(B, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b  = self.bottleneck(self.pool(s3))
        x  = self.dec3(torch.cat([self.up3(b),  s3], dim=1))
        x  = self.dec2(torch.cat([self.up2(x),  s2], dim=1))
        x  = self.dec1(torch.cat([self.up1(x),  s1], dim=1))
        return self.outc(x)


# ============================================================
# 3. SQUEEZE-AND-EXCITATION variant
#    SE block (Hu et al. 2018) added after each DoubleConv.
#    Global average pool → FC → ReLU → FC → Sigmoid → rescale.
#    Adds channel-wise attention with minimal extra params.
# ============================================================

class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1, 1)
        return x * w


class SEDoubleConv(nn.Module):
    """DoubleConv3D + SE channel attention."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        from models.blocks import DoubleConv3D
        self.conv = DoubleConv3D(in_ch, out_ch)
        self.se   = SEBlock3D(out_ch)

    def forward(self, x):
        return self.se(self.conv(x))


class TinyUNet_SE(nn.Module):
    """TinyUNet with Squeeze-and-Excitation after every block."""
    def __init__(self, in_channels=4, num_classes=4, base=16):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        B = base

        self.enc1      = SEDoubleConv(in_channels, B)
        self.enc2      = SEDoubleConv(B,     B * 2)
        self.enc3      = SEDoubleConv(B * 2, B * 4)
        self.bottleneck= SEDoubleConv(B * 4, B * 8)

        self.up3 = nn.ConvTranspose3d(B * 8, B * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(B * 4, B * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(B * 2, B,     2, 2)

        self.dec3 = SEDoubleConv(B * 8, B * 4)
        self.dec2 = SEDoubleConv(B * 4, B * 2)
        self.dec1 = SEDoubleConv(B * 2, B)

        self.outc = nn.Conv3d(B, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b  = self.bottleneck(self.pool(s3))
        x  = self.dec3(torch.cat([self.up3(b),  s3], dim=1))
        x  = self.dec2(torch.cat([self.up2(x),  s2], dim=1))
        x  = self.dec1(torch.cat([self.up1(x),  s1], dim=1))
        return self.outc(x)


# ============================================================
# 4. EDGE-ENHANCED variant
#    Sobel edge maps computed from each MRI channel are
#    concatenated to the input before enc1.
#    Gives the network explicit boundary information —
#    useful for meningioma where tumour margin delineation
#    (ET vs NETC, NETC vs brain) is the hard part.
#
#    Input:  (B, 4, D, H, W)  — 4 MRI channels
#    Edges:  (B, 4, D, H, W)  — Sobel magnitude per channel
#    Cat  :  (B, 8, D, H, W)  → enc1
#    enc1 input channels = in_channels * 2
# ============================================================

class SobelEdge3D(nn.Module):
    """
    Compute 3D Sobel edge magnitude for each input channel independently.
    Uses 3×3×3 Sobel kernels along X, Y, Z axes.
    Output same shape as input — magnitude = sqrt(Gx²+Gy²+Gz²).
    """
    def __init__(self):
        super().__init__()
        # 3D Sobel kernels
        Kx = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        ], dtype=torch.float32)
        Ky = Kx.permute(1, 0, 2)
        Kz = Kx.permute(2, 1, 0)
        # Register as non-trainable buffers
        self.register_buffer("Kx", Kx.unsqueeze(0).unsqueeze(0))
        self.register_buffer("Ky", Ky.unsqueeze(0).unsqueeze(0))
        self.register_buffer("Kz", Kz.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, C, D, H, W = x.shape
        out = []
        for c in range(C):
            ch = x[:, c:c+1]                        # (B,1,D,H,W)
            gx = F.conv3d(ch, self.Kx, padding=1)
            gy = F.conv3d(ch, self.Ky, padding=1)
            gz = F.conv3d(ch, self.Kz, padding=1)
            mag = (gx**2 + gy**2 + gz**2).sqrt()
            out.append(mag)
        return torch.cat(out, dim=1)                 # (B,C,D,H,W)


class TinyUNet_Edge(nn.Module):
    """
    TinyUNet with Sobel edge maps fused at input.
    enc1 receives [original 4 channels | edge 4 channels] = 8 channels.
    All other layers identical to TinyUNet3D.
    """
    def __init__(self, in_channels=4, num_classes=4, base=16):
        super().__init__()
        from models.blocks import DoubleConv3D
        self.pool  = nn.MaxPool3d(2)
        self.sobel = SobelEdge3D()
        B = base

        # enc1 takes doubled input channels
        self.enc1      = DoubleConv3D(in_channels * 2, B)
        self.enc2      = DoubleConv3D(B,     B * 2)
        self.enc3      = DoubleConv3D(B * 2, B * 4)
        self.bottleneck= DoubleConv3D(B * 4, B * 8)

        self.up3 = nn.ConvTranspose3d(B * 8, B * 4, 2, 2)
        self.up2 = nn.ConvTranspose3d(B * 4, B * 2, 2, 2)
        self.up1 = nn.ConvTranspose3d(B * 2, B,     2, 2)

        self.dec3 = DoubleConv3D(B * 8, B * 4)
        self.dec2 = DoubleConv3D(B * 4, B * 2)
        self.dec1 = DoubleConv3D(B * 2, B)

        self.outc = nn.Conv3d(B, num_classes, 1)

    def forward(self, x):
        edges = self.sobel(x)                        # (B, 4, D, H, W)
        x_in  = torch.cat([x, edges], dim=1)         # (B, 8, D, H, W)
        s1 = self.enc1(x_in)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b  = self.bottleneck(self.pool(s3))
        x  = self.dec3(torch.cat([self.up3(b),  s3], dim=1))
        x  = self.dec2(torch.cat([self.up2(x),  s2], dim=1))
        x  = self.dec1(torch.cat([self.up1(x),  s1], dim=1))
        return self.outc(x)


# ============================================================
# Registry
# ============================================================

ABLATION_REGISTRY = {
    "tinyunet_dw":    TinyUNet_DW,
    "tinyunet_ghost": TinyUNet_Ghost,
    "tinyunet_se":    TinyUNet_SE,
    "tinyunet_edge":  TinyUNet_Edge,
}

def get_ablation_model(name, **kwargs):
    name = name.lower()
    if name not in ABLATION_REGISTRY:
        raise ValueError(
            f"Unknown ablation model '{name}'. "
            f"Choose from: {list(ABLATION_REGISTRY)}"
        )
    return ABLATION_REGISTRY[name](**kwargs)
