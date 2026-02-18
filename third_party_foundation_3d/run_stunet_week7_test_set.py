#!/usr/bin/env python3
"""
Run STU-Net on full Week7 test set; compute MAE, SSIM, PSNR, Dice (pred mask vs brain mask).
Saves third_party_foundation_3d/stunet_week7_results.json.
STU-Net outputs 105-class segmentation; we binarize (pred > 0) vs brain mask for metrics.
"""
import os
import sys
import json
import shutil
import subprocess
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

ROOT = os.path.dirname(os.path.abspath(__file__))
WEEK7_SPLIT = "/data1/julih/combined_subject_split.json"
BRAIN_MASK = "/data1/julih/MNI152_T1_2mm_brain_mask_dil.nii.gz"
INPUT_DIR = os.path.join(ROOT, "stunet_week7_test_input")
OUTPUT_DIR = os.path.join(ROOT, "stunet_week7_test_output")
RESULTS_JSON = os.path.join(ROOT, "stunet_week7_results.json")
NNUNET_PYTHON = os.environ.get("NNUNET_PYTHON", "/data1/julih/miniconda3/envs/julih_monai/bin/python3")


def load_nii(path):
    return np.asarray(nib.load(path).get_fdata()).squeeze().astype(np.float32)


def dice(a, b, thresh=0.5):
    a = (a > thresh).flatten()
    b = (b > thresh).flatten()
    if a.sum() == 0 and b.sum() == 0:
        return 1.0
    inter = (a & b).sum()
    return 2 * inter / (a.sum() + b.sum() + 1e-8)


def main():
    with open(WEEK7_SPLIT) as f:
        data = json.load(f)
    pairs = data.get("test", [])
    if not pairs:
        print("No Week7 test pairs.")
        sys.exit(1)

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Prepare input: nnUNet expects CASENAME_0000.nii.gz
    case_ids = []
    for i, item in enumerate(pairs):
        pre_path = item.get("pre_path")
        if not pre_path or not os.path.isfile(pre_path):
            continue
        cid = "week7_test_%03d" % i
        case_ids.append((cid, pre_path))
        shutil.copy2(pre_path, os.path.join(INPUT_DIR, cid + "_0000.nii.gz"))

    if not case_ids:
        print("No valid test paths.")
        sys.exit(1)

    # Run nnUNet predict (single call on folder)
    env = os.environ.copy()
    env["RESULTS_FOLDER"] = env.get("RESULTS_FOLDER", os.path.join(ROOT, "stunet_results"))
    env["nnUNet_raw_data_base"] = env.get("nnUNet_raw_data_base", os.path.join(ROOT, "nnunet_data", "raw"))
    env["nnUNet_preprocessed"] = env.get("nnUNet_preprocessed", os.path.join(ROOT, "nnunet_data", "preprocessed"))
    print("Running STU-Net on %d test volumes..." % len(case_ids))
    cmd = [
        NNUNET_PYTHON, "-m", "nnunet.inference.predict_simple",
        "-i", INPUT_DIR, "-o", OUTPUT_DIR,
        "-t", "101", "-m", "3d_fullres", "-f", "0",
        "-tr", "STUNetTrainer_base", "-chk", "base_ep4k",
        "--mode", "fast", "--disable_tta",
    ]
    subprocess.run(cmd, env=env, cwd=ROOT, check=False, capture_output=False)

    # Metrics: pred (binarized) vs brain mask
    mask_ref = load_nii(BRAIN_MASK)
    mae_list, ssim_list, psnr_list, dice_list = [], [], [], []
    for cid, _ in case_ids:
        pred_path = os.path.join(OUTPUT_DIR, cid + ".nii.gz")
        if not os.path.isfile(pred_path):
            continue
        pred = load_nii(pred_path)
        # Binarize: any structure = 1 (or use (pred > 0))
        pred_bin = (pred > 0.5).astype(np.float32)
        gt = load_nii(BRAIN_MASK)
        if pred_bin.shape != gt.shape:
            factors = [gt.shape[k] / pred_bin.shape[k] for k in range(3)]
            pred_bin = zoom(pred_bin, factors, order=0)
        pred_bin = np.clip(pred_bin, 0, 1).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)
        mae_list.append(np.abs(pred_bin - gt).mean())
        try:
            ssim_list.append(ssim(gt, pred_bin, data_range=1.0))
        except Exception:
            ssim_list.append(0.0)
        try:
            psnr_list.append(psnr(gt, pred_bin, data_range=1.0))
        except Exception:
            psnr_list.append(0.0)
        dice_list.append(dice(pred_bin, gt))

    if not mae_list:
        print("No predictions found.")
        sys.exit(1)

    results = {
        "model": "STU-Net",
        "task": "mask_vs_brain_mask",
        "note": "Segmentation (105-class); binarized pred vs brain mask. Not CVR.",
        "n_test": len(mae_list),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list)),
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "dice_mean": float(np.mean(dice_list)),
        "dice_std": float(np.std(dice_list)),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("Results:", results)
    print("Saved", RESULTS_JSON)


if __name__ == "__main__":
    main()
