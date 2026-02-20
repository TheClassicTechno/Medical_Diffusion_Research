#!/usr/bin/env python3
"""
Week 7 unified preprocessing for fair 2D vs 3D comparison.
- Apply MNI brain mask to all volumes (outside mask = 0).
- Same dimensions: 91 x 109 x 91 (match MNI152_T1_2mm_brain_mask_dil).
- Pad with 0s when needed; resize to target; min-max norm.
- Same augmentations for 2D and 3D: flip LR, flip UD, intensity scale (optional, at train time).
Use with combined 2020-2023 split so train/val/test are identical across all models.
"""
import os
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from typing import Tuple, Optional, List

# Same for everyone (from week7tasks.txt)
TARGET_SHAPE = (91, 109, 91)  # MNI 2mm brain mask dimensions
BRAIN_MASK_PATH = "/data1/julih/MNI152_T1_2mm_brain_mask_dil.nii.gz"
COMBINED_SPLIT_PATH = "/data1/julih/combined_subject_split.json"

_mask_cache = None


def get_brain_mask() -> np.ndarray:
    """Load MNI brain mask (91, 109, 91), 1 = brain, 0 = non-brain. Cached."""
    global _mask_cache
    if _mask_cache is None:
        m = nib.load(BRAIN_MASK_PATH)
        _mask_cache = np.asarray(m.dataobj).squeeze().astype(np.float32)
        if _mask_cache.max() > 1:
            _mask_cache = (_mask_cache > 0).astype(np.float32)
    return _mask_cache


def get_brain_bounding_box(mask: Optional[np.ndarray] = None) -> Tuple[slice, slice, slice]:
    """Axis-aligned bounding box of brain (where mask > 0). Returns (sl_d, sl_h, sl_w) to crop volume to brain only."""
    if mask is None:
        mask = get_brain_mask()
    mask = (mask > 0.5) if mask.dtype != bool else mask
    where = np.argwhere(mask)
    if where.size == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2])
    d_min, d_max = where[:, 0].min(), where[:, 0].max() + 1
    h_min, h_max = where[:, 1].min(), where[:, 1].max() + 1
    w_min, w_max = where[:, 2].min(), where[:, 2].max() + 1
    return slice(d_min, d_max), slice(h_min, h_max), slice(w_min, w_max)


def get_brain_crop_shape(mask: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    """Shape of the brain-only crop (same for all subjects when using fixed MNI mask). Returns (D, H, W)."""
    sl_d, sl_h, sl_w = get_brain_bounding_box(mask)
    return (sl_d.stop - sl_d.start, sl_h.stop - sl_h.start, sl_w.stop - sl_w.start)


def load_volume_cropped(
    nii_path: str,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    apply_mask: bool = True,
    pad_to_shape: Optional[Tuple[int, int, int]] = None,
    minmax: bool = True,
) -> np.ndarray:
    """
    Load NIfTI, resize to target_shape, apply brain mask, crop to brain bbox, optionally pad to pad_to_shape.
    Returns float32 array of shape pad_to_shape if given, else get_brain_crop_shape().
    Use for brain-only crop experiment: smaller volume, same preprocessing otherwise.
    """
    vol = load_volume(nii_path, target_shape=target_shape, apply_mask=apply_mask, pad_zeros=True, minmax=minmax)
    # Bbox from MNI mask (same shape as TARGET_SHAPE)
    sl_d, sl_h, sl_w = get_brain_bounding_box(get_brain_mask())
    cropped = vol[sl_d, sl_h, sl_w].copy()
    if pad_to_shape is not None:
        out = np.zeros(pad_to_shape, dtype=np.float32)
        cd, ch, cw = cropped.shape
        pd, ph, pw = pad_to_shape
        out[:min(cd, pd), :min(ch, ph), :min(cw, pw)] = cropped[:min(cd, pd), :min(ch, ph), :min(cw, pw)]
        return out
    return cropped


def load_volume(
    nii_path: str,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    apply_mask: bool = True,
    pad_zeros: bool = True,
    minmax: bool = True,
) -> np.ndarray:
    """
    Load NIfTI, optionally resize to target_shape, apply brain mask, pad, min-max norm.
    Returns float32 array of shape target_shape.
    """
    img = nib.load(nii_path)
    data = np.asarray(img.get_fdata()).astype(np.float32).squeeze()
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape} from {nii_path}")

    # Resize to target (trilinear)
    if data.shape != target_shape:
        factors = [target_shape[i] / data.shape[i] for i in range(3)]
        data = zoom(data, factors, order=1)

    # Pad with 0s if smaller (shouldn't happen after zoom, but if target was larger)
    if pad_zeros and (data.shape[0] < target_shape[0] or data.shape[1] < target_shape[1] or data.shape[2] < target_shape[2]):
        out = np.zeros(target_shape, dtype=np.float32)
        s0 = min(data.shape[0], target_shape[0])
        s1 = min(data.shape[1], target_shape[1])
        s2 = min(data.shape[2], target_shape[2])
        out[:s0, :s1, :s2] = data[:s0, :s1, :s2]
        data = out
    elif data.shape != target_shape:
        # Crop to target if larger
        data = data[: target_shape[0], : target_shape[1], : target_shape[2]]

    # Apply brain mask: outside mask = 0
    if apply_mask:
        mask = get_brain_mask()
        if mask.shape != target_shape:
            factors = [target_shape[i] / mask.shape[i] for i in range(3)]
            mask = zoom(mask, factors, order=0)
        mask = (mask > 0.5).astype(np.float32)
        data = data * mask

    if minmax:
        mn, mx = data.min(), data.max()
        if (mx - mn) > 1e-8:
            data = (data - mn) / (mx - mn)
        else:
            data = np.zeros_like(data)

    return data.astype(np.float32)


def augment_volume(
    vol: np.ndarray,
    flip_lr: bool = False,
    flip_ud: bool = False,
    flip_fb: bool = False,
    intensity_scale: Optional[float] = None,
) -> np.ndarray:
    """In-place style augmentations; returns augmented copy. Same for 2D and 3D."""
    out = vol.copy()
    if flip_lr:
        out = np.flip(out, axis=1).copy()
    if flip_ud:
        out = np.flip(out, axis=0).copy()
    if flip_fb:
        out = np.flip(out, axis=2).copy()
    if intensity_scale is not None and abs(intensity_scale - 1.0) > 1e-6:
        out = (out * intensity_scale).clip(0.0, 1.0)
    return out


def load_combined_split() -> dict:
    """Load combined 2020-2023 subject-level split."""
    with open(COMBINED_SPLIT_PATH) as f:
        return json.load(f)


def get_pre_post_pairs(split_key: str = "train") -> List[Tuple[str, str]]:
    """Return list of (pre_path, post_path) for split_key in combined split."""
    data = load_combined_split()
    pairs = []
    for item in data.get(split_key, []):
        pre = item.get("pre_path")
        post = item.get("post_path")
        if pre and post and os.path.isfile(pre) and os.path.isfile(post):
            pairs.append((pre, post))
    return pairs


def metrics_in_brain(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    data_range: float = 1.0,
) -> dict:
    """
    MAE, SSIM, PSNR computed only inside the brain (best for reporting CVR quality).
    pred, target: 3D arrays same shape as mask (e.g. 91,109,91).
    mask: same shape, 1 = brain, 0 = outside. If None, uses get_brain_mask().
    Returns dict with mae_mean, ssim_mean, psnr_mean (brain-only).
    """
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

    if mask is None:
        mask = get_brain_mask()
    if mask.shape != pred.shape:
        from scipy.ndimage import zoom as _zoom
        factors = [pred.shape[i] / mask.shape[i] for i in range(3)]
        mask = _zoom(mask.astype(np.float32), factors, order=0) > 0.5
    mask_bool = (mask > 0.5).astype(bool)
    n = mask_bool.sum()
    if n == 0:
        return {"mae_mean": float("nan"), "ssim_mean": float("nan"), "psnr_mean": float("nan")}
    mae = np.abs(pred.astype(np.float64) - target.astype(np.float64))[mask_bool].mean()
    sl_d, sl_h, sl_w = get_brain_bounding_box(mask)
    p_crop = pred[sl_d, sl_h, sl_w]
    t_crop = target[sl_d, sl_h, sl_w]
    ssim_val = ssim(t_crop, p_crop, data_range=data_range)
    psnr_val = psnr(t_crop, p_crop, data_range=data_range)
    return {"mae_mean": float(mae), "ssim_mean": float(ssim_val), "psnr_mean": float(psnr_val)}


def get_brain_mask_2d_slice(mask_3d: Optional[np.ndarray] = None) -> np.ndarray:
    """Middle axial slice of brain mask (D//2), shape (H, W). For 2D models."""
    if mask_3d is None:
        mask_3d = get_brain_mask()
    d = mask_3d.shape[0] // 2
    return (mask_3d[d] > 0.5).astype(np.float32)


def metrics_in_brain_2d(
    pred: np.ndarray,
    target: np.ndarray,
    mask_2d: Optional[np.ndarray] = None,
    data_range: float = 1.0,
) -> dict:
    """
    MAE, SSIM, PSNR for 2D (single slice) computed only inside brain.
    pred, target: (H, W). mask_2d: (H, W), 1 = brain. If None, uses middle slice of get_brain_mask().
    """
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

    if mask_2d is None:
        mask_2d = get_brain_mask_2d_slice()
    if mask_2d.shape != pred.shape:
        from scipy.ndimage import zoom as _zoom
        factors = [pred.shape[i] / mask_2d.shape[i] for i in range(2)]
        mask_2d = (_zoom(mask_2d.astype(np.float32), factors, order=0) > 0.5).astype(np.float32)
    mask_bool = (mask_2d > 0.5).astype(bool)
    n = mask_bool.sum()
    if n == 0:
        return {"mae_mean": float("nan"), "ssim_mean": float("nan"), "psnr_mean": float("nan")}
    mae = np.abs(pred.astype(np.float64) - target.astype(np.float64))[mask_bool].mean()
    # SSIM/PSNR on full slice (skimage doesn't support mask); for consistency crop to bbox
    where = np.argwhere(mask_bool)
    if where.size == 0:
        return {"mae_mean": float(mae), "ssim_mean": float("nan"), "psnr_mean": float("nan")}
    rmin, rmax = where[:, 0].min(), where[:, 0].max() + 1
    cmin, cmax = where[:, 1].min(), where[:, 1].max() + 1
    p_crop = pred[rmin:rmax, cmin:cmax]
    t_crop = target[rmin:rmax, cmin:cmax]
    ssim_val = ssim(t_crop, p_crop, data_range=data_range)
    psnr_val = psnr(t_crop, p_crop, data_range=data_range)
    return {"mae_mean": float(mae), "ssim_mean": float(ssim_val), "psnr_mean": float(psnr_val)}


def masked_loss_3d(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """L1 loss only over voxels where mask > 0. pred, target, mask: (D,H,W) or (1,D,H,W)."""
    if pred.ndim == 4:
        pred, target = pred.squeeze(0), target.squeeze(0)
    if mask is None:
        mask = get_brain_mask()
    if mask.shape != pred.shape:
        from scipy.ndimage import zoom as _zoom
        factors = [pred.shape[i] / mask.shape[i] for i in range(3)]
        mask = _zoom(mask.astype(np.float32), factors, order=0) > 0.5
    m = (mask > 0.5).astype(bool)
    if m.sum() == 0:
        return float(np.abs(pred - target).mean())
    return float(np.abs(pred.astype(np.float64) - target.astype(np.float64))[m].mean())


def masked_loss_2d(pred: np.ndarray, target: np.ndarray, mask_2d: Optional[np.ndarray] = None) -> float:
    """L1 loss only over pixels where mask > 0. pred, target: (H,W)."""
    if mask_2d is None:
        mask_2d = get_brain_mask_2d_slice()
    if mask_2d.shape != pred.shape:
        from scipy.ndimage import zoom as _zoom
        factors = [pred.shape[i] / mask_2d.shape[i] for i in range(2)]
        mask_2d = _zoom(mask_2d.astype(np.float32), factors, order=0) > 0.5
    m = (mask_2d > 0.5).astype(bool)
    if m.sum() == 0:
        return float(np.abs(pred - target).mean())
    return float(np.abs(pred.astype(np.float64) - target.astype(np.float64))[m].mean())


def get_brain_mask_for_shape(shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """Brain mask resized to shape (e.g. (96,112,96) or (96,112)). Returns 0/1 float array."""
    mask = get_brain_mask()
    if len(shape) == 3:
        if mask.shape != shape:
            factors = [shape[i] / mask.shape[i] for i in range(3)]
            mask = zoom(mask.astype(np.float32), factors, order=0)
    else:
        mask = get_brain_mask_2d_slice(mask)
        if mask.shape != shape:
            factors = [shape[i] / mask.shape[i] for i in range(2)]
            mask = zoom(mask.astype(np.float32), factors, order=0)
    return (mask > 0.5).astype(dtype)


# ---------------------------------------------------------------------------
# Phase 2: Vascular / MNI territory region-weighted loss
# ---------------------------------------------------------------------------
MASKS_DIR_DEFAULT = "/data1/julih/Masks"
MASKS_DIR_RYDHAM = "/data/rydham/Masks"


def _get_masks_dir() -> Optional[str]:
    """Return first existing of MASKS_DIR_DEFAULT, MASKS_DIR_RYDHAM, or None."""
    if os.path.isdir(MASKS_DIR_DEFAULT):
        return MASKS_DIR_DEFAULT
    if os.path.isdir(MASKS_DIR_RYDHAM):
        return MASKS_DIR_RYDHAM
    return None


def load_territory_masks(
    masks_dir: str,
    target_shape: Tuple[int, int, int],
) -> List[Tuple[str, np.ndarray]]:
    """
    Load MNI territory masks (*2mm*.nii.gz or MNI_*.nii.gz) and resize to target_shape.
    Returns [(name, mask_3d), ...] with mask_3d float 0/1.
    """
    import glob
    out = []
    pattern = os.path.join(masks_dir, "*.nii.gz")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "2mm" in f]
    if not files:
        files = sorted(glob.glob(os.path.join(masks_dir, "MNI_*.nii.gz")))
    for path in files:
        try:
            img = nib.load(path)
            data = np.asarray(img.dataobj).squeeze().astype(np.float32)
            if data.ndim == 4:
                data = data[..., 0]
            mask = (data > 0).astype(np.float32)
            if mask.shape != target_shape:
                factors = [target_shape[i] / mask.shape[i] for i in range(3)]
                mask = zoom(mask, factors, order=0)
                mask = (mask > 0.5).astype(np.float32)
            name = os.path.basename(path).replace(".nii.gz", "")
            out.append((name, mask))
        except Exception:
            continue
    return out


def get_region_weight_mask_for_shape(
    shape: Tuple[int, ...],
    masks_dir: Optional[str] = None,
    vascular_weight: float = 1.5,
    dtype=np.float32,
) -> np.ndarray:
    """
    Weight map for region-weighted loss: brain = 1.0, vascular territories = vascular_weight.
    If masks_dir is None or no territory masks found, returns brain mask (1.0 in brain, 0 outside).
    shape: (D,H,W) for 3D or (H,W) for 2D.
    """
    base = get_brain_mask_for_shape(shape, dtype=np.float32)  # 0/1
    mdir = masks_dir or _get_masks_dir()
    if not mdir or not os.path.isdir(mdir):
        return base.astype(dtype)
    if len(shape) == 2:
        target_3d = TARGET_SHAPE
        territory_list = load_territory_masks(mdir, target_3d)
        if not territory_list:
            return base.astype(dtype)
        mid = target_3d[0] // 2
        weight = base.copy()
        for _name, m3 in territory_list:
            m2 = (m3[mid] > 0.5).astype(np.float32)
            if m2.shape != shape:
                factors = [shape[i] / m2.shape[i] for i in range(2)]
                m2 = zoom(m2, factors, order=0)
            weight[m2 > 0.5] = vascular_weight
        return weight.astype(dtype)
    else:
        territory_list = load_territory_masks(mdir, shape)
        if not territory_list:
            return base.astype(dtype)
        weight = base.astype(np.float32)
        for _name, m in territory_list:
            weight[m > 0.5] = vascular_weight
        return weight.astype(dtype)
