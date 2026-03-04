#!/usr/bin/env python3
# Export per-subject metrics for Patch_3D (Week7). Run from /data1/julih.
# Writes week8_per_subject_metrics/Patch_3D_<subject_id>.json
import os
import sys
import json
ROOT = "/data1/julih"
OUT_DIR = os.path.join(ROOT, "week8_per_subject_metrics")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from week7_data import get_week7_splits, _subject_id_from_path
from week7_preprocess import metrics_in_brain, get_brain_mask
import numpy as np
import torch
WEEK7_ORIGINAL = (91, 109, 91)

def _brain_mean(pred_vol, target_vol, mask=None):
    if mask is None:
        mask = get_brain_mask()
    if mask.shape != pred_vol.shape:
        from scipy.ndimage import zoom as _z
        factors = [pred_vol.shape[i] / mask.shape[i] for i in range(3)]
        mask = _z(mask.astype(np.float32), factors, order=0)
    mask = (mask > 0.5).astype(np.float32)
    n = mask.sum() + 1e-8
    return float((pred_vol * mask).sum() / n), float((target_vol * mask).sum() / n)

def load_volume_week7(nii_path, pad_shape=(96, 96, 96)):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from week7_preprocess import load_volume, TARGET_SHAPE
    vol = load_volume(nii_path, target_shape=TARGET_SHAPE, apply_mask=True, minmax=True)
    if vol.shape != pad_shape:
        out = np.zeros(pad_shape, dtype=vol.dtype)
        sh = [min(vol.shape[i], pad_shape[i]) for i in range(3)]
        out[:sh[0], :sh[1], :sh[2]] = vol[:sh[0], :sh[1], :sh[2]]
        return out.astype(np.float32)
    return vol.astype(np.float32)

def main():
    patch_dir = os.path.join(ROOT, "Diffusion_3D_PatchVolume")
    sys.path.insert(0, patch_dir)
    sys.path.insert(0, os.path.join(ROOT, "Diffusion_3D_Latent"))
    from utils import strict_normalize_volume
    from patch_volume_vae import PatchVolumeVAE, extract_patches, reconstruct_volume_from_patches
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_size = (96, 96, 96)
    patch_size = (24, 24, 24)
    stride = 12
    vae_ckpt = os.path.join(patch_dir, "patch_vae_week7_best.pt")
    if not os.path.isfile(vae_ckpt):
        print("Patch VAE checkpoint not found:", vae_ckpt)
        return
    vae_model = PatchVolumeVAE(
        in_channels=1, latent_channels=4,
        encoder_channels=(32, 64, 128), decoder_channels=(128, 64, 32, 1),
        num_res_blocks=2, patch_size=patch_size,
    ).to(device)
    vae_model.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae_model.eval()
    _, _, test_pairs = get_week7_splits()
    n = 0
    with torch.no_grad():
        for pre_p, post_p in test_pairs:
            if not os.path.isfile(pre_p) or not os.path.isfile(post_p):
                continue
            sid = _subject_id_from_path(pre_p)
            try:
                pre_vol = strict_normalize_volume(load_volume_week7(pre_p, pad_shape=target_size))
                post_vol = strict_normalize_volume(load_volume_week7(post_p, pad_shape=target_size))
            except Exception as e:
                print("Skip", sid, e)
                continue
            try:
                pre_patches, patch_coords = extract_patches(pre_vol, patch_size, stride)
                predicted_patches = []
                for pre_patch in pre_patches:
                    pre_t = torch.from_numpy(pre_patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    pre_latent = vae_model.encode_to_latent(pre_t)
                    pred_patch = vae_model.decode_from_latent(pre_latent)
                    predicted_patches.append(pred_patch[0, 0].cpu().numpy())
                pred_vol = reconstruct_volume_from_patches(
                    predicted_patches, patch_coords, target_size, patch_size, stride
                )
            except Exception as e:
                print("Skip", sid, "inference:", e)
                continue
            if np.isnan(pred_vol).any() or np.isinf(pred_vol).any():
                continue
            if pred_vol.shape != WEEK7_ORIGINAL:
                pred_vol = pred_vol[:91, :109, :91].copy()
            if post_vol.shape != WEEK7_ORIGINAL:
                post_vol = post_vol[:91, :109, :91].copy()
            met = metrics_in_brain(pred_vol, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_vol, post_vol)
            out = {
                "model": "Patch_3D",
                "subject_id": sid,
                "mae": float(met["mae_mean"]),
                "ssim": float(met["ssim_mean"]),
                "psnr": float(met["psnr_mean"]),
                "pred_mean": pred_mean,
                "target_mean": target_mean,
            }
            with open(os.path.join(OUT_DIR, "Patch_3D_%s.json" % sid), "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for Patch_3D to", OUT_DIR)

if __name__ == "__main__":
    main()
