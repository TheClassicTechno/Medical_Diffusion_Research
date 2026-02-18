"""
FNO 3D with full Fourier/spectral layers (copy of fno_3d_cvr with SpectralFNO3D).
SpectralConv3d: FFT -> multiply in Fourier space (modes) -> IFFT.
Use for Week7 or standard data; same interface as SimpleFNO3D (in_ch, out_ch, modes, width).
"""
import os
import numpy as np
import torch
import torch.nn as nn

# Reuse data loading and Week7 dataset from original
from fno_3d_cvr import (
    load_volume, pre_to_post, add_position_channels,
    VolumePairs, VolumePairsFromPairs,
    Week7VolumePairsFNO, WEEK7_PAD_SHAPE, WEEK7_ORIGINAL_SHAPE,
    TARGET_SIZE, DATA_DIR, _pad_vol_to_96_112_96,
)

# Re-export for trainers
__all__ = [
    "SpectralConv3d", "SpectralFNO3D",
    "load_volume", "pre_to_post", "add_position_channels",
    "VolumePairs", "VolumePairsFromPairs", "Week7VolumePairsFNO",
    "WEEK7_PAD_SHAPE", "WEEK7_ORIGINAL_SHAPE", "TARGET_SIZE", "DATA_DIR",
]


class SpectralConv3d(nn.Module):
    """3D Fourier layer: FFT -> multiply by learnable weights (truncated to modes) -> IFFT."""

    def __init__(self, in_ch, out_ch, modes1, modes2, modes3):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1.0 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (B, C, H, W, D). rfftn last dim = D//2+1. Per-mode matmul: out[b,o,k] = sum_i x_ft[b,i,k] * W[i,o,k]
        B, C, H, W, D = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros_like(x_ft)
        m1 = min(self.modes1, x_ft.shape[-3])
        m2 = min(self.modes2, x_ft.shape[-2])
        m3 = min(self.modes3, x_ft.shape[-1])
        # (B, in_ch, m1, m2, m3) @ (in_ch, out_ch, m1, m2, m3) -> (B, out_ch, m1, m2, m3) via einsum
        w1 = self.weights1[:, :, :m1, :m2, :m3]
        w2 = self.weights2[:, :, :m1, :m2, :m3]
        w3 = self.weights3[:, :, :m1, :m2, :m3]
        out_ft[:, :, :m1, :m2, :m3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :m1, :m2, :m3], w1)
        out_ft[:, :, -m1:, :m2, :m3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -m1:, :m2, :m3], w2)
        out_ft[:, :, :m1, -m2:, :m3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :m1, -m2:, :m3], w3)
        x_out = torch.fft.irfftn(out_ft, s=(H, W, D), dim=(-3, -2, -1))
        return x_out


class SpectralFNO3D(nn.Module):
    """FNO 3D with full spectral layers: lift -> 2 x (SpectralConv3d + skip) -> project. Same interface as SimpleFNO3D."""

    def __init__(self, in_ch=4, out_ch=1, modes=12, width=64):
        super().__init__()
        self.modes = modes
        self.width = width
        # modes3 for rfftn: last dim is D//2+1; use same as modes for simplicity (clamped in forward)
        m3 = max(modes // 2 + 1, 1)
        self.fc0 = nn.Linear(in_ch, width)
        self.spectral0 = SpectralConv3d(width, width, modes, modes, m3)
        self.spectral1 = SpectralConv3d(width, width, modes, modes, m3)
        self.w0 = nn.Conv3d(width, width, 1)
        self.w1 = nn.Conv3d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_ch)

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = torch.relu(x)

        x1 = self.spectral0(x)
        x2 = self.w0(x)
        x = torch.relu(x1 + x2)

        x1 = self.spectral1(x)
        x2 = self.w1(x)
        x = torch.relu(x1 + x2)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x
