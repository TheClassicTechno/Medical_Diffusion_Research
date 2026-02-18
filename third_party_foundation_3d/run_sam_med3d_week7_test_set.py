#!/usr/bin/env python3
"""
Run SAM-Med3D on full Week7 test set; compute MAE, SSIM, PSNR, Dice (pred mask vs brain mask).
Saves third_party_foundation_3d/sam_med3d_week7_results.json for comparison with other 3D models.
Note: SAM-Med3D outputs segmentation masks, so MAE/SSIM/PSNR here are mask-vs-mask (not CVR post).
"""
import os
import sys
import json
import shutil
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

SAM_ROOT = os.path.dirname(os.path.abspath(__file__))
WEEK7_SPLIT = "/data1/julih/combined_subject_split.json"
BRAIN_MASK = "/data1/julih/MNI152_T1_2mm_brain_mask_dil.nii.gz"
OUT_DIR = os.path.join(SAM_ROOT, "test_data", "week7_sam_test_set")
IMAGES_DIR = os.path.join(OUT_DIR, "imagesVa")
LABELS_DIR = os.path.join(OUT_DIR, "labelsVa")
PRED_DIR = os.path.join(OUT_DIR, "pred")
RESULTS_JSON = os.path.join(SAM_ROOT, "sam_med3d_week7_results.json")


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

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    sam_dir = os.path.join(SAM_ROOT, "SAM-Med3D")
    if os.path.isdir(sam_dir):
        sys.path.insert(0, sam_dir)
    else:
        sys.path.insert(0, SAM_ROOT)
    try:
        import medim
        from utils.infer_utils import validate_paired_img_gt
    except ImportError as e:
        print("Import error:", e)
        sys.exit(1)

    ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    for loc in [os.path.join(SAM_ROOT, "SAM-Med3D", "ckpt", "sam_med3d_turbo.pth"),
                os.path.join(SAM_ROOT, "ckpt", "sam_med3d_turbo.pth")]:
        if os.path.isfile(loc):
            ckpt_path = loc
            break
    print("Loading SAM-Med3D...")
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)

    n = len(pairs)
    print("Running inference on", n, "test volumes...")
    for i, item in enumerate(pairs):
        pre_path = item.get("pre_path")
        if not pre_path or not os.path.isfile(pre_path):
            continue
        base = os.path.splitext(os.path.splitext(os.path.basename(pre_path))[0])[0]
        sid = f"test_{i:03d}_{base}"[:60]
        img_dest = os.path.join(IMAGES_DIR, sid + ".nii.gz")
        gt_dest = os.path.join(LABELS_DIR, sid + ".nii.gz")
        out_path = os.path.join(PRED_DIR, sid + ".nii.gz")
        shutil.copy2(pre_path, img_dest)
        shutil.copy2(BRAIN_MASK, gt_dest)
        validate_paired_img_gt(model, img_dest, gt_dest, out_path, num_clicks=1)
        if (i + 1) % 8 == 0:
            print("  ", i + 1, "/", n)

    # Compute metrics: pred mask vs brain mask (resize to match if needed)
    mask_ref = load_nii(BRAIN_MASK)
    mae_list, ssim_list, psnr_list, dice_list = [], [], [], []
    for i, item in enumerate(pairs):
        pre_path = item.get("pre_path")
        if not pre_path or not os.path.isfile(pre_path):
            continue
        base = os.path.splitext(os.path.splitext(os.path.basename(pre_path))[0])[0]
        sid = f"test_{i:03d}_{base}"[:60]
        pred_path = os.path.join(PRED_DIR, sid + ".nii.gz")
        gt_path = os.path.join(LABELS_DIR, sid + ".nii.gz")
        if not os.path.isfile(pred_path):
            continue
        pred = load_nii(pred_path)
        gt = load_nii(gt_path)
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            factors = [gt.shape[k] / pred.shape[k] for k in range(3)]
            pred = zoom(pred, factors, order=0)
        pred = np.clip(pred, 0, 1).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)
        mae_list.append(np.abs(pred - gt).mean())
        try:
            ssim_list.append(ssim(gt, pred, data_range=1.0))
        except Exception:
            ssim_list.append(0.0)
        try:
            psnr_list.append(psnr(gt, pred, data_range=1.0))
        except Exception:
            psnr_list.append(0.0)
        dice_list.append(dice(pred, gt))

    if not mae_list:
        print("No predictions found.")
        sys.exit(1)

    results = {
        "model": "SAM-Med3D",
        "task": "mask_vs_brain_mask",
        "note": "Segmentation (mask); MAE/SSIM/PSNR are pred mask vs brain mask, not CVR.",
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
