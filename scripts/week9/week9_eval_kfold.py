#!/usr/bin/env python3
"""
Evaluate each K-fold checkpoint on the hold-out test set. Saves one JSON per fold.

Usage (run from scripts/):
  python week9/week9_eval_kfold.py --model unet3d --checkpoints week7_results/unet3d_fold0_best.pt week7_results/unet3d_fold1_best.pt ...
  python week9/week9_eval_kfold.py --model unet3d --kfold 5  # auto glob week7_results/unet3d_fold*_best.pt

Output: week9/unet3d_fold{N}_test_results.json per checkpoint (mae_mean, ssim_mean, psnr_mean).
"""
import os
import sys
import json
import argparse
import glob

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
from torch.utils.data import DataLoader

from week7_train_unet3d import make_unet_3d, evaluate, OUT_DIR, DEVICE, BATCH_SIZE
from week7_data import get_week7_splits, Week7VolumePairs3D

WEEK9_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet3d", help="Model name (for output filenames)")
    ap.add_argument("--checkpoints", nargs="+", default=[], help="Paths to fold checkpoints")
    ap.add_argument("--kfold", type=int, default=0, help="If >0, glob checkpoints: week7_results/{model}_fold*_best.pt")
    ap.add_argument("--out_dir", default="", help="Directory for result JSONs (default week9/)")
    args = ap.parse_args()

    if args.kfold > 0:
        pattern = os.path.join(OUT_DIR, "%s_fold*_best.pt" % args.model)
        paths = sorted(glob.glob(pattern))
        # Sort by fold number
        def fold_num(p):
            base = os.path.basename(p)
            import re
            m = re.search(r"fold(\d+)", base)
            return int(m.group(1)) if m else -1
        paths = sorted(paths, key=fold_num)
        if len(paths) != args.kfold:
            print("Warning: expected %d checkpoints, found %d: %s" % (args.kfold, len(paths), paths))
    else:
        paths = [os.path.abspath(p) for p in args.checkpoints]
    if not paths:
        print("No checkpoints found. Use --checkpoints or --kfold K")
        return

    _, _, test_pairs = get_week7_splits()
    test_ds = Week7VolumePairs3D(test_pairs, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)
    out_dir = args.out_dir or WEEK9_DIR
    os.makedirs(out_dir, exist_ok=True)

    for ckpt_path in paths:
        base = os.path.basename(ckpt_path)
        import re
        m = re.search(r"fold(\d+)", base)
        fold_idx = int(m.group(1)) if m else -1
        if not os.path.isfile(ckpt_path):
            print("Skip missing: %s" % ckpt_path)
            continue
        model = make_unet_3d().to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, test_loader)
        result = {"fold": fold_idx, "checkpoint": ckpt_path, "mae_mean": test_metrics["mae_mean"], "ssim_mean": test_metrics["ssim_mean"], "psnr_mean": test_metrics["psnr_mean"]}
        out_path = os.path.join(out_dir, "%s_fold%d_test_results.json" % (args.model, fold_idx))
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print("Fold %d -> %s  MAE=%.4f SSIM=%.4f PSNR=%.2f" % (fold_idx, out_path, result["mae_mean"], result["ssim_mean"], result["psnr_mean"]))


if __name__ == "__main__":
    main()
