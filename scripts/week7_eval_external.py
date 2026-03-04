#!/usr/bin/env python3
"""Evaluate Week 7 model on external pairs JSON. See EXTERNAL_DATASET_CONTRACT.md.
Usage: week7_eval_external.py --pairs_json path --model unet3d --checkpoint path --out external_eval.json
Validate: export test pairs to JSON, run with same checkpoint; compare mae_mean/ssim_mean/psnr_mean to existing test metrics."""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from week7_preprocess import TARGET_SHAPE, metrics_in_brain

TARGET_3D_PAD = (96, 112, 96)

def _crop_to_target(vol):
    if vol.shape == TARGET_SHAPE:
        return vol
    h, w, d = TARGET_SHAPE
    return vol[:h, :w, :d].copy()

def _pad_3d(pre_t, post_t, target_shape):
    import torch.nn.functional as F
    _, _, h, w, d = pre_t.shape
    th, tw, td = target_shape
    if h < th or w < tw or d < td:
        pd = (0, max(0, td - d), 0, max(0, tw - w), 0, max(0, th - h))
        pre_t = F.pad(pre_t, pd, mode="constant", value=0)
        post_t = F.pad(post_t, pd, mode="constant", value=0)
    return pre_t[:, :, :th, :tw, :td], post_t[:, :, :th, :tw, :td]

def run_eval(pairs_json, model_name, checkpoint_path, out_path, device=None):
    import torch
    from torch.utils.data import DataLoader
    from external_dataset import ExternalWeek7Pairs

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "unet3d":
        from monai.networks.nets import UNet
        model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
            channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
            act=("LeakyReLU", {"inplace": True}), norm="INSTANCE", dropout=0.0)
    elif model_name == "resnet3d":
        from week7_train_resnet3d import ResNet3DCVR
        model = ResNet3DCVR(pretrained=False)
    else:
        raise ValueError("Use --model unet3d or resnet3d")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.to(device).eval()

    ds = ExternalWeek7Pairs(pairs_json)
    if len(ds) == 0:
        print("No pairs loaded from", pairs_json)
        return
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    mae_list, ssim_list, psnr_list = [], [], []
    per_subject = []
    with torch.no_grad():
        for idx, (pre_t, post_t) in enumerate(loader):
            pre_t, post_t = pre_t.to(device), post_t.to(device)
            pre_t, post_t = _pad_3d(pre_t, post_t, TARGET_3D_PAD)
            pred_vol = model(pre_t)[0, 0].cpu().numpy()
            gt_vol = post_t[0, 0].cpu().numpy()
            # Pass same shape as internal trainer (96,112,96) so metrics_in_brain matches week7_train_unet3d evaluate()
            m = metrics_in_brain(pred_vol, gt_vol, data_range=1.0)
            mae_list.append(m["mae_mean"])
            ssim_list.append(m["ssim_mean"])
            psnr_list.append(m["psnr_mean"])
            per_subject.append({"id": "ext_%d" % idx, "mae": m["mae_mean"], "ssim": m["ssim_mean"], "psnr": m["psnr_mean"]})

    out = {"source": "week7_eval_external", "pairs_json": os.path.abspath(pairs_json), "model": model_name,
           "checkpoint": checkpoint_path, "n_volumes": len(mae_list),
           "mae_mean": float(np.mean(mae_list)), "ssim_mean": float(np.mean(ssim_list)), "psnr_mean": float(np.mean(psnr_list)),
           "per_subject": per_subject}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", out_path, "MAE=%.4f SSIM=%.4f PSNR=%.2f n=%d" % (out["mae_mean"], out["ssim_mean"], out["psnr_mean"], out["n_volumes"]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_json", required=True)
    ap.add_argument("--model", required=True, choices=("unet3d", "resnet3d"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="external_eval.json")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    if not os.path.isfile(args.pairs_json):
        print("Pairs JSON not found:", args.pairs_json)
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print("Checkpoint not found:", args.checkpoint)
        sys.exit(1)
    run_eval(args.pairs_json, args.model, args.checkpoint, args.out, args.device or None)

if __name__ == "__main__":
    main()
