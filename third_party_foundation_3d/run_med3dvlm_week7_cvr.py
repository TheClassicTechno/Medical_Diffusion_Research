#!/usr/bin/env python3
"""
Train/finetune Med3DVLM DCFormer encoder + 3D decoder for Week7 CVR (pre->post).
- Encoder: DCFormer (decomp_small) with input (128,128,128); encoder kept or finetuned.
- Decoder: ConvTranspose3d from encoder features to 1 channel, then interpolate to 91x109x91.
- Data: Week7 get_week7_splits + load_volume(91,109,91); resize to 128^3 for encoder; loss on 91x109x91.
Run from repo root with PYTHONPATH including scripts and Med3DVLM:
  cd /data1/julih && PYTHONPATH=/data1/julih/scripts:/data1/julih/third_party_foundation_3d/Med3DVLM python3 third_party_foundation_3d/run_med3dvlm_week7_cvr.py
"""
import os
import sys
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Week7 data (run from /data1/julih)
ROOT = "/data1/julih"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "third_party_foundation_3d", "Med3DVLM"))

from week7_data import get_week7_splits, Week7VolumePairs3D
from week7_preprocess import TARGET_SHAPE, metrics_in_brain, get_brain_mask_for_shape, get_region_weight_mask_for_shape

# DCFormer from Med3DVLM
from src.model.encoder.dcformer import decomp_small

OUT_DIR = os.path.join(ROOT, "third_party_foundation_3d", "med3dvlm_week7_cvr")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# DCFormer expects (B,C,H,W,D) after internal permute; we use 128^3 to get 4^3 feature map (32x downsample)
ENC_SIZE = (128, 128, 128)
TARGET_SHAPE_3D = TARGET_SHAPE  # (91, 109, 91)
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4
FREEZE_ENCODER = True  # set False to finetune encoder


class CVRDecoder3D(nn.Module):
    """Upsample encoder feature map (B, 768, 2, 2, 2) to (B, 1, 128, 128, 128) via 6x stride-2."""
    def __init__(self, in_ch=768):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, 384, 4, stride=2, padding=1),
            nn.BatchNorm3d(384),
            nn.GELU(),
            nn.ConvTranspose3d(384, 192, 4, stride=2, padding=1),
            nn.BatchNorm3d(192),
            nn.GELU(),
            nn.ConvTranspose3d(192, 96, 4, stride=2, padding=1),
            nn.BatchNorm3d(96),
            nn.GELU(),
            nn.ConvTranspose3d(96, 48, 4, stride=2, padding=1),
            nn.BatchNorm3d(48),
            nn.GELU(),
            nn.ConvTranspose3d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.GELU(),
            nn.ConvTranspose3d(24, 1, 4, stride=2, padding=1),
        )
        self.out_size = (128, 128, 128)

    def forward(self, x):
        return self.up(x)


class DCFormerCVR(nn.Module):
    def __init__(self, input_size=(128, 128, 128), freeze_encoder=True):
        super().__init__()
        # DCFormer input is (B, C, D, H, W); internally permuted to (B,C,H,W,D). input_size = (H,W,D)
        self.encoder = decomp_small(input_size=input_size)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        enc_ch = self.encoder.channels[-1]
        self.decoder = CVRDecoder3D(in_ch=enc_ch)
        self.enc_size = input_size

    def forward(self, x):
        # x: (B, 1, 128, 128, 128) in (B,C,D,H,W). DecompModel returns list of (B, N, C); last is (B, 8, 768)
        feats = self.encoder(x)
        last = feats[-1]
        B, N, C = last.shape
        s = 2
        last = last.permute(0, 2, 1).view(B, C, s, s, s)
        out = self.decoder(last)
        return out


def resize_vol(vol, size):
    """vol: (B,1,D,H,W) or (D,H,W). Resize to size (d,h,w)."""
    if vol.dim() == 3:
        return F.interpolate(vol.unsqueeze(0).unsqueeze(0), size=size, mode="trilinear", align_corners=False).squeeze(0).squeeze(0)
    return F.interpolate(vol, size=size, mode="trilinear", align_corners=False)


def train_epoch(model, loader, criterion, optimizer, mask_t=None):
    model.train()
    total, n = 0.0, 0
    for pre, post in loader:
        pre = pre.to(DEVICE)
        post = post.to(DEVICE)
        pre_128 = resize_vol(pre, ENC_SIZE)
        optimizer.zero_grad()
        pred_128 = model(pre_128)
        pred = resize_vol(pred_128, TARGET_SHAPE_3D)
        if mask_t is not None:
            l1_masked = (torch.abs(pred - post) * mask_t).sum() / (mask_t.sum() + 1e-8)
            loss = l1_masked
        else:
            loss = criterion(pred, post)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def evaluate(model, loader):
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    with torch.no_grad():
        for pre, post in loader:
            pre = pre.to(DEVICE)
            post = post.to(DEVICE)
            pre_128 = resize_vol(pre, ENC_SIZE)
            pred_128 = model(pre_128)
            pred = resize_vol(pred_128, TARGET_SHAPE_3D)
            for i in range(pred.shape[0]):
                p = pred[i, 0].cpu().numpy()
                t = post[i, 0].cpu().numpy()
                m = metrics_in_brain(p, t, data_range=1.0)
                mae_list.append(m["mae_mean"])
                ssim_list.append(m["ssim_mean"])
                psnr_list.append(m["psnr_mean"])
    return {
        "mae_mean": float(np.mean(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)),
    }


def main():
    print("Med3DVLM DCFormer + decoder for Week7 CVR (pre->post)")
    train_pairs, val_pairs, test_pairs = get_week7_splits()
    train_ds = Week7VolumePairs3D(train_pairs, augment=True)
    val_ds = Week7VolumePairs3D(val_pairs, augment=False)
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = DCFormerCVR(input_size=ENC_SIZE, freeze_encoder=FREEZE_ENCODER).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LR)
    criterion = nn.L1Loss()
    use_region_weight = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    mask_np = get_region_weight_mask_for_shape(TARGET_SHAPE_3D, vascular_weight=1.5) if use_region_weight else get_brain_mask_for_shape(TARGET_SHAPE_3D)
    mask_t = torch.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

    best_val_psnr = -1.0
    for ep in range(EPOCHS):
        loss = train_epoch(model, loader=train_loader, criterion=criterion, optimizer=optimizer, mask_t=mask_t)
        metrics = evaluate(model, val_loader)
        if metrics["psnr_mean"] > best_val_psnr:
            best_val_psnr = metrics["psnr_mean"]
            torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(OUT_DIR, "med3dvlm_cvr_best.pt"))
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1} loss={loss:.4f} val MAE={metrics['mae_mean']:.4f} SSIM={metrics['ssim_mean']:.4f} PSNR={metrics['psnr_mean']:.2f}")

    ckpt = torch.load(os.path.join(OUT_DIR, "med3dvlm_cvr_best.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader)
    print("Test:", test_metrics)
    use_region = os.environ.get("WEEK7_REGION_WEIGHT", "").lower() in ("1", "true", "yes")
    out_name = "med3dvlm_week7_phase2_results.json" if use_region else "med3dvlm_week7_results.json"
    out_json = os.path.join(OUT_DIR, out_name)
    with open(out_json, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Saved", out_json)


if __name__ == "__main__":
    main()
