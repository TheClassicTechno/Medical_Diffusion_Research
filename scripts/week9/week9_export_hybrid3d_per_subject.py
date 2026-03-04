#!/usr/bin/env python3
# Export per-subject metrics for Hybrid_3D. Run from /data1/julih.
# Writes week8_per_subject_metrics/Hybrid_3D_<subject_id>.json
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
    if mask is None: mask = get_brain_mask()
    if mask.shape != pred_vol.shape:
        from scipy.ndimage import zoom as _z
        factors = [pred_vol.shape[i] / mask.shape[i] for i in range(3)]
        mask = _z(mask.astype(np.float32), factors, order=0)
    mask = (mask > 0.5).astype(np.float32)
    n = mask.sum() + 1e-8
    return float((pred_vol * mask).sum() / n), float((target_vol * mask).sum() / n)
def main():
    hybrid_dir = os.path.join(ROOT, "Hybrid_UNet_Diffusion")
    sys.path.insert(0, hybrid_dir)
    sys.path.insert(0, os.path.join(ROOT, "Diffusion_3D_Latent"))
    from unet_diffusion_hybrid import load_volume_week7, strict_normalize_volume, p_sample_ddim_hybrid, make_beta_schedule, HybridDiffusionNet
    from vae_3d import VAE3D
    from monai.networks.nets import UNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_size = (96, 112, 96)
    n_timesteps_train, n_steps_ddim, residual_scale = 1000, 25, 0.5
    vae_path = "/data1/julih/Diffusion_3D_Latent/vae_3d_week7_best.pt"
    if not os.path.isfile(vae_path): vae_path = "/data1/julih/Diffusion_3D_Latent/vae_3d_best.pt"
    unet_path = "/data1/julih/UNet_3D/unet_3d_week7_best.pt"
    ckpt_path = os.path.join(hybrid_dir, "hybrid_unet_diffusion_week7_best.pt")
    if not all(os.path.isfile(p) for p in (ckpt_path, unet_path, vae_path)):
        print("Missing checkpoint:", ckpt_path, unet_path, vae_path)
        return
    vae_model = VAE3D(in_channels=1, latent_channels=4).to(device)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.eval()
    unet_model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2, act=("LeakyReLU", {"inplace": True}), norm="INSTANCE", dropout=0.0).to(device)
    unet_model.load_state_dict(torch.load(unet_path, map_location=device))
    unet_model.eval()
    model = HybridDiffusionNet(latent_channels=4, channels=(32, 64, 128), use_v_prediction=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    betas, alphas, alpha_bar = make_beta_schedule(n_timesteps=n_timesteps_train, schedule="cosine")
    alphas_bar_sqrt = torch.from_numpy(np.sqrt(alpha_bar)).float().to(device)
    one_minus_alphas_bar_sqrt = torch.from_numpy(np.sqrt(1.0 - alpha_bar)).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)
    alphas_t = torch.from_numpy(alphas).float().to(device)
    _, _, test_pairs = get_week7_splits()
    n = 0
    with torch.no_grad():
        for pre_p, post_p in test_pairs:
            if not os.path.isfile(pre_p) or not os.path.isfile(post_p): continue
            sid = _subject_id_from_path(pre_p)
            try:
                pre_vol = strict_normalize_volume(load_volume_week7(pre_p, pad_shape=target_size))
                post_vol = strict_normalize_volume(load_volume_week7(post_p, pad_shape=target_size))
            except Exception as e: print("Skip", sid, e); continue
            try:
                pred_vol = p_sample_ddim_hybrid(model, vae_model, unet_model, pre_vol, n_timesteps_train, n_steps_ddim, betas_t, alphas_t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, residual_scale, target_size, device)
                pred_np = pred_vol[0, 0].cpu().numpy()
            except Exception as e: print("Skip", sid, "inference:", e); continue
            if pred_np.shape[-3:] != WEEK7_ORIGINAL: pred_np = pred_np[:WEEK7_ORIGINAL[0], :WEEK7_ORIGINAL[1], :WEEK7_ORIGINAL[2]]
            if post_vol.shape != WEEK7_ORIGINAL: post_vol = post_vol[:WEEK7_ORIGINAL[0], :WEEK7_ORIGINAL[1], :WEEK7_ORIGINAL[2]]
            met = metrics_in_brain(pred_np, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_np, post_vol)
            out = {"model": "Hybrid_3D", "subject_id": sid, "mae": float(met["mae_mean"]), "ssim": float(met["ssim_mean"]), "psnr": float(met["psnr_mean"]), "pred_mean": pred_mean, "target_mean": target_mean}
            with open(os.path.join(OUT_DIR, "Hybrid_3D_%s.json" % sid), "w") as f: json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for Hybrid_3D to", OUT_DIR)
if __name__ == "__main__": main()
