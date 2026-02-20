#!/usr/bin/env python3
"""
Export per-subject metrics for UNet_3D, Cold_3D, Residual_3D, DDPM_3D (test set) for Bland-Altman and Wilcoxon.
Run from /data1/julih with PyTorch and project dependencies. Writes to week8_per_subject_metrics/<model>_<subject_id>.json.
Each model requires its Week 7 checkpoint (e.g. unet_3d_week7_best.pt, cold_diffusion_3d_week7_best.pt, etc.).
"""
from __future__ import annotations

import os
import sys
import json
import traceback

ROOT = "/data1/julih"
OUT_DIR = os.path.join(ROOT, "week8_per_subject_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

# Ensure scripts and week7 data/preprocess are importable
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from week7_data import get_week7_splits, Week7VolumePairs3D, _subject_id_from_path
from week7_preprocess import load_volume, TARGET_SHAPE, metrics_in_brain, get_brain_mask

import numpy as np
import torch

WEEK7_ORIGINAL = (91, 109, 91)
PAD_3D = (96, 112, 96)


def _pad_3d(pre_t, post_t, target_shape):
    if (pre_t.shape[2], pre_t.shape[3], pre_t.shape[4]) == target_shape:
        return pre_t, post_t
    th, tw, td = target_shape
    import torch.nn.functional as F
    _, _, h, w, d = pre_t.shape
    ph, pw, pd = max(0, th - h), max(0, tw - w), max(0, td - d)
    if ph or pw or pd:
        pre_t = F.pad(pre_t, (0, pd, 0, pw, 0, ph), mode="constant", value=0)
        post_t = F.pad(post_t, (0, pd, 0, pw, 0, ph), mode="constant", value=0)
    return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]


def _crop_to_91(pred_np, post_np):
    if pred_np.shape[-3:] != WEEK7_ORIGINAL:
        pred_np = pred_np[:, :, :WEEK7_ORIGINAL[0], :WEEK7_ORIGINAL[1], :WEEK7_ORIGINAL[2]]
        post_np = post_np[:, :, :WEEK7_ORIGINAL[0], :WEEK7_ORIGINAL[1], :WEEK7_ORIGINAL[2]]
    return pred_np, post_np


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


def export_unet_3d():
    """UNet_3D (tips): single forward pass."""
    sys.path.insert(0, os.path.join(ROOT, "UNet_3D"))
    try:
        from model_3d import make_unet_3d, _pad_3d_if_needed
    except Exception as e:
        print("Skip UNet_3D:", e)
        return 0
    ckpt_path = os.path.join(ROOT, "UNet_3D", "unet_3d_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip UNet_3D: checkpoint not found", ckpt_path)
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_unet_3d().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i in range(len(test_pairs)):
            pre_path, _ = test_pairs[i]
            sid = _subject_id_from_path(pre_path)
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = _pad_3d_if_needed(pre_t, post_t, PAD_3D)
            pred_t = model(pre_t)
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np, post_np = _crop_to_91(pred_np, post_np)
            pred_vol = pred_np[0, 0]
            post_vol = post_np[0, 0]
            met = metrics_in_brain(pred_vol, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_vol, post_vol)
            out = {"model": "UNet_3D", "subject_id": sid, "mae": float(met["mae_mean"]), "ssim": float(met["ssim_mean"]), "psnr": float(met["psnr_mean"]), "pred_mean": pred_mean, "target_mean": target_mean}
            with open(os.path.join(OUT_DIR, f"UNet_3D_{sid}.json"), "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for UNet_3D")
    return n


def export_cold_3d():
    """Cold Diffusion 3D: iterative sampling."""
    cold_dir = os.path.join(ROOT, "Diffusion_ColdDiffusion_3D")
    # Ensure Cold's model_3d is used (remove UNet_3D from path if present)
    while cold_dir in sys.path:
        sys.path.remove(cold_dir)
    sys.path.insert(0, cold_dir)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cold_model_3d", os.path.join(cold_dir, "model_3d.py"))
        cold_model_3d = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cold_model_3d)
        ColdDiffusionNet3D = cold_model_3d.ColdDiffusionNet3D
        cold_sample = cold_model_3d.cold_sample
        make_alpha_schedule = cold_model_3d.make_alpha_schedule
        _pad_3d = cold_model_3d._pad_3d
    except Exception as e:
        print("Skip Cold_3D:", e)
        return 0
    ckpt_path = os.path.join(ROOT, "Diffusion_ColdDiffusion_3D", "cold_diffusion_3d_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip Cold_3D: checkpoint not found", ckpt_path)
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ColdDiffusionNet3D().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    alpha = make_alpha_schedule(100)
    alpha_t = torch.from_numpy(alpha).float().to(device)
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i in range(len(test_pairs)):
            pre_path, _ = test_pairs[i]
            sid = _subject_id_from_path(pre_path)
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = _pad_3d(pre_t, post_t, PAD_3D)
            pred_t = cold_sample(model, pre_t, 100, alpha_t, device)
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np = pred_np[:, :, :91, :109, :91]
            post_np = post_np[:, :, :91, :109, :91]
            pred_vol = pred_np[0, 0]
            post_vol = post_np[0, 0]
            met = metrics_in_brain(pred_vol, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_vol, post_vol)
            out = {"model": "Cold_3D", "subject_id": sid, "mae": float(met["mae_mean"]), "ssim": float(met["ssim_mean"]), "psnr": float(met["psnr_mean"]), "pred_mean": pred_mean, "target_mean": target_mean}
            with open(os.path.join(OUT_DIR, f"Cold_3D_{sid}.json"), "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for Cold_3D")
    return n


def export_residual_3d():
    """Residual Diffusion 3D: full sampling loop over test set."""
    res_dir = os.path.join(ROOT, "Diffusion_ResidualDiffusion_3D")
    ckpt_path = os.path.join(res_dir, "residual_diffusion_3d_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip Residual_3D: checkpoint not found", ckpt_path)
        return 0
    import importlib.util
    spec = importlib.util.spec_from_file_location("residual_model_3d", os.path.join(res_dir, "model_3d.py"))
    res_model_3d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(res_model_3d)
    SimpleResidualDiffusion3D = res_model_3d.SimpleResidualDiffusion3D
    make_beta_schedule = res_model_3d.make_beta_schedule
    p_sample_loop_residual = res_model_3d.p_sample_loop_residual
    res_pad_3d = res_model_3d._pad_3d
    n_timesteps = 1000
    residual_scale = 0.2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    betas = make_beta_schedule(n_timesteps)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_sqrt = torch.from_numpy(np.sqrt(alphas_bar)).float().to(device)
    one_minus_alphas_bar_sqrt = torch.from_numpy(np.sqrt(1.0 - alphas_bar)).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)
    alphas_t = torch.from_numpy(alphas).float().to(device)
    model = SimpleResidualDiffusion3D(ch=16).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i in range(len(test_pairs)):
            pre_path, _ = test_pairs[i]
            sid = _subject_id_from_path(pre_path)
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = res_pad_3d(pre_t, post_t, PAD_3D)
            residual_pred = p_sample_loop_residual(
                model, pre_t, n_timesteps,
                betas_t, alphas_t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device,
            )
            residual_pred = residual_pred * residual_scale
            pred_t = torch.clamp(pre_t + residual_pred, 0.0, 1.0)
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np, post_np = _crop_to_91(pred_np, post_np)
            pred_vol = pred_np[0, 0]
            post_vol = post_np[0, 0]
            met = metrics_in_brain(pred_vol, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_vol, post_vol)
            out = {"model": "Residual_3D", "subject_id": sid, "mae": float(met["mae_mean"]), "ssim": float(met["ssim_mean"]), "psnr": float(met["psnr_mean"]), "pred_mean": pred_mean, "target_mean": target_mean}
            with open(os.path.join(OUT_DIR, f"Residual_3D_{sid}.json"), "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for Residual_3D")
    return n


def export_ddpm_3d():
    """DDPM 3D: full sampling loop over test set."""
    ddpm_dir = os.path.join(ROOT, "Diffusion_baseline_3D")
    ckpt_path = os.path.join(ddpm_dir, "ddpm_3d_week7_best.pt")
    if not os.path.isfile(ckpt_path):
        print("Skip DDPM_3D: checkpoint not found", ckpt_path)
        return 0
    import importlib.util
    spec = importlib.util.spec_from_file_location("ddpm_model_3d", os.path.join(ddpm_dir, "diffusion_model_3d.py"))
    ddpm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ddpm_module)
    SimpleCondDiffusion3D = ddpm_module.SimpleCondDiffusion3D
    make_beta_schedule = ddpm_module.make_beta_schedule
    p_sample_loop = ddpm_module.p_sample_loop
    ddpm_pad_3d = ddpm_module._pad_3d
    n_timesteps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    betas = make_beta_schedule(n_timesteps)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    alphas_bar_sqrt = torch.from_numpy(np.sqrt(alphas_bar)).float().to(device)
    one_minus_alphas_bar_sqrt = torch.from_numpy(np.sqrt(1.0 - alphas_bar)).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)
    alphas_t = torch.from_numpy(alphas).float().to(device)
    model = SimpleCondDiffusion3D(ch=16).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    n = 0
    with torch.no_grad():
        for i in range(len(test_pairs)):
            pre_path, _ = test_pairs[i]
            sid = _subject_id_from_path(pre_path)
            pre_t, post_t = test_ds[i]
            pre_t = pre_t.unsqueeze(0).to(device)
            post_t = post_t.unsqueeze(0).to(device)
            pre_t, post_t = ddpm_pad_3d(pre_t, post_t, PAD_3D)
            pred_t = p_sample_loop(
                model, pre_t, n_timesteps,
                betas_t, alphas_t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device,
            )
            pred_np = pred_t.cpu().numpy()
            post_np = post_t.cpu().numpy()
            pred_np, post_np = _crop_to_91(pred_np, post_np)
            pred_vol = pred_np[0, 0]
            post_vol = post_np[0, 0]
            met = metrics_in_brain(pred_vol, post_vol, data_range=1.0)
            pred_mean, target_mean = _brain_mean(pred_vol, post_vol)
            out = {"model": "DDPM_3D", "subject_id": sid, "mae": float(met["mae_mean"]), "ssim": float(met["ssim_mean"]), "psnr": float(met["psnr_mean"]), "pred_mean": pred_mean, "target_mean": target_mean}
            with open(os.path.join(OUT_DIR, f"DDPM_3D_{sid}.json"), "w") as f:
                json.dump(out, f, indent=0)
            n += 1
    print("Wrote", n, "per-subject JSONs for DDPM_3D")
    return n


def main():
    print("Exporting per-subject metrics for external models (Week 7 test set)...")
    export_unet_3d()
    export_cold_3d()
    export_residual_3d()
    export_ddpm_3d()
    print("Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
