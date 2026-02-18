#!/usr/bin/env python3
"""
Patch-Volume Autoencoder for 3D Diffusion (3D MedDiffusion Style)
==================================================================
Based on research: "3D MedDiffusion: A 3D Medical Diffusion Model for 
Controllable and High-quality Medical Image Generation" (2024)

Key Innovation:
- Patch-wise encoding: Compresses volumes into overlapping patches
- Volume-wise decoding: Reconstructs full volumes with smooth blending
- Ensures 3D consistency while maintaining memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from monai.networks.blocks import ResidualUnit
from monai.networks.layers import Norm, get_norm_layer


class PatchEncoder3D(nn.Module):
    """
    Encodes 3D volumes into patch-based latent representations.
    
    Process:
    1. Split volume into overlapping patches
    2. Encode each patch to latent space
    3. Maintain patch spatial relationships
    """
    def __init__(
        self,
        in_channels=1,
        latent_channels=4,
        channels=(32, 64, 128),  # Reduced to 3 levels to avoid too small latents
        num_res_blocks=2,
        patch_size=(32, 32, 16),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        
        # Encoder: Process each patch
        encoder_layers = []
        in_ch = in_channels
        
        for out_ch in channels:
            # Downsample block
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm3d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            # Residual blocks
            for _ in range(num_res_blocks):
                encoder_layers.append(
                    ResidualUnit(
                        spatial_dims=3,
                        in_channels=out_ch,
                        out_channels=out_ch,
                        act=("LeakyReLU", {"inplace": True}),
                        norm="INSTANCE",
                    )
                )
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent projection
        self.latent_proj = nn.Conv3d(
            channels[-1], latent_channels, kernel_size=1
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W, D) - Full volume or patch
        Returns:
            latent: (B, latent_channels, H', W', D') - Encoded representation
        """
        features = self.encoder(x)
        latent = self.latent_proj(features)
        return latent


class PatchDecoder3D(nn.Module):
    """
    Decodes patch-based latent representations back to full volumes.
    
    Process:
    1. Decode each patch from latent space
    2. Reconstruct full volume with overlap blending
    """
    def __init__(
        self,
        latent_channels=4,
        out_channels=1,
        channels=(128, 64, 32),  # Match reduced encoder
        num_res_blocks=2,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Latent projection
        self.latent_proj = nn.Conv3d(
            latent_channels, channels[0], kernel_size=1
        )
        
        # Decoder: Upsample from latent to full resolution
        decoder_layers = []
        in_ch = channels[0]
        
        # First upsample (no residual blocks before first upsample)
        for i, out_ch in enumerate(channels[1:]):
            # Upsample block
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm3d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            # Residual blocks after upsample
            for _ in range(num_res_blocks):
                decoder_layers.append(
                    ResidualUnit(
                        spatial_dims=3,
                        in_channels=out_ch,
                        out_channels=out_ch,
                        act=("LeakyReLU", {"inplace": True}),
                        norm="INSTANCE",
                    )
                )
            in_ch = out_ch
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output projection
        self.output_proj = nn.Conv3d(channels[-1], out_channels, kernel_size=1)
        self.output_act = nn.Tanh()  # Output in [-1, 1], will normalize to [0, 1]
    
    def forward(self, latent):
        """
        Args:
            latent: (B, latent_channels, H', W', D') - Encoded representation
        Returns:
            x: (B, out_channels, H, W, D) - Reconstructed volume or patch
        """
        features = self.latent_proj(latent)
        decoded = self.decoder(features)
        output = self.output_proj(decoded)
        output = self.output_act(output)
        return output


class PatchVolumeVAE(nn.Module):
    """
    Complete Patch-Volume Autoencoder.
    
    Encodes volumes into patch-based latent space and decodes back.
    Designed for 3D medical image diffusion.
    """
    def __init__(
        self,
        in_channels=1,
        latent_channels=4,
        encoder_channels=(32, 64, 128),  # Reduced to 3 levels
        decoder_channels=(128, 64, 32, 1),  # Match reduced encoder channels
        num_res_blocks=2,
        patch_size=(32, 32, 16),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        
        self.encoder = PatchEncoder3D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            channels=encoder_channels,
            num_res_blocks=num_res_blocks,
            patch_size=patch_size,
        )
        
        self.decoder = PatchDecoder3D(
            latent_channels=latent_channels,
            out_channels=in_channels,
            channels=decoder_channels,
            num_res_blocks=num_res_blocks,
        )
    
    def encode_to_latent(self, x):
        """Encode volume to latent space."""
        return self.encoder(x)
    
    def decode_from_latent(self, latent):
        """Decode latent to volume space."""
        return self.decoder(latent)
    
    def forward(self, x):
        """Full forward pass: encode then decode."""
        latent = self.encode_to_latent(x)
        recon = self.decode_from_latent(latent)
        return recon, latent


def extract_patches(volume, patch_size, stride):
    """
    Extract overlapping patches from a volume.
    
    Args:
        volume: (H, W, D) numpy array
        patch_size: (ph, pw, pd) patch dimensions
        stride: Stride for patch extraction (overlap = patch_size - stride)
    
    Returns:
        patches: List of patches
        patch_coords: List of (h, w, d) coordinates for each patch
    """
    H, W, D = volume.shape
    ph, pw, pd = patch_size
    
    patches = []
    patch_coords = []
    
    for h in range(0, H - ph + 1, stride):
        for w in range(0, W - pw + 1, stride):
            for d in range(0, D - pd + 1, stride):
                patch = volume[h:h+ph, w:w+pw, d:d+pd]
                if patch.shape == patch_size:
                    patches.append(patch)
                    patch_coords.append((h, w, d))
    
    return patches, patch_coords


def reconstruct_volume_from_patches(patches, patch_coords, volume_shape, patch_size, stride):
    """
    Reconstruct full volume from patches with overlap blending.
    
    Uses weighted averaging for overlapping regions.
    """
    H, W, D = volume_shape
    ph, pw, pd = patch_size
    
    # Initialize output and weight arrays
    output = np.zeros((H, W, D), dtype=np.float32)
    weights = np.zeros((H, W, D), dtype=np.float32)
    
    # Create blending weights (Gaussian or linear falloff)
    patch_weights = np.ones(patch_size, dtype=np.float32)
    overlap = patch_size[0] - stride
    
    if overlap > 0:
        # Linear falloff at edges for smooth blending
        for dim in range(3):
            if patch_size[dim] > 1:
                edge_size = min(overlap, patch_size[dim] // 4)
                if edge_size > 0:
                    # Create linear falloff
                    falloff = np.linspace(0.5, 1.0, edge_size)
                    # Apply to all edges
                    for i in range(edge_size):
                        if dim == 0:
                            patch_weights[i, :, :] *= falloff[i]
                            patch_weights[patch_size[0]-1-i, :, :] *= falloff[i]
                        elif dim == 1:
                            patch_weights[:, i, :] *= falloff[i]
                            patch_weights[:, patch_size[1]-1-i, :] *= falloff[i]
                        else:
                            patch_weights[:, :, i] *= falloff[i]
                            patch_weights[:, :, patch_size[2]-1-i] *= falloff[i]
    
    # Accumulate patches with weights
    for patch, (h, w, d) in zip(patches, patch_coords):
        output[h:h+ph, w:w+pw, d:d+pd] += patch * patch_weights
        weights[h:h+ph, w:w+pw, d:d+pd] += patch_weights
    
    # Normalize by weights (avoid division by zero)
    weights = np.maximum(weights, 1e-6)
    output = output / weights
    
    return output
