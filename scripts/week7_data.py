#!/usr/bin/env python3
"""
Week 7 unified dataset for 2D and 3D models.
- Uses combined 2020-2023 split only.
- Same preprocessing: brain mask, 91x109x91, pad, min-max (via week7_preprocess).
- Same augmentations at train time: flip_lr, flip_ud, intensity_scale (optional flip_fb for 3D).
- 2D: returns middle axial slice (91, 109) from the preprocessed volume.
- 3D: returns full volume (1, 91, 109, 91).
Metrics should be computed in same dimensionality (e.g. 2D middle slice vs 2D middle slice; 3D full vs 3D full).
Phase 3: Week7VolumePairs3DWithMasks returns (pre, post, mask) for mask-weighted loss.
"""
import os
import re
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import nibabel as nib

from week7_preprocess import (
    load_volume,
    load_volume_cropped,
    augment_volume,
    get_pre_post_pairs,
    get_brain_mask_for_shape,
    get_brain_bounding_box,
    get_brain_mask,
    TARGET_SHAPE,
)


# Same augmentation range for 2D and 3D (from week7tasks: left/right, top/bottom flips; future: intensity)
def _random_augment() -> Tuple[bool, bool, bool, Optional[float]]:
    flip_lr = random.random() < 0.5
    flip_ud = random.random() < 0.5
    flip_fb = random.random() < 0.5
    # intensity scale in [0.9, 1.1]
    intensity_scale = 0.9 + 0.2 * random.random()
    return flip_lr, flip_ud, flip_fb, intensity_scale


class Week7VolumePairs3D(Dataset):
    """3D dataset: returns (pre_vol, post_vol) as (1, H, W, D) tensors, same preprocessing and aug."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        augment: bool = False,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    ):
        self.pairs = pairs
        self.augment = augment
        self.target_shape = target_shape

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_path, post_path = self.pairs[idx]
        pre = load_volume(pre_path, target_shape=self.target_shape)
        post = load_volume(post_path, target_shape=self.target_shape)
        if self.augment:
            fl, fu, ff, scale = _random_augment()
            pre = augment_volume(pre, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
            post = augment_volume(post, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
        pre_t = torch.from_numpy(pre).unsqueeze(0).float()   # (1, H, W, D)
        post_t = torch.from_numpy(post).unsqueeze(0).float()
        return pre_t, post_t


class Week7VolumePairs3DCropped(Dataset):
    """3D dataset with brain-only crop: load 91x109x91, apply mask, crop to bbox, pad to crop_pad_shape.
    Same split and augmentations as Week7VolumePairs3D. For controlled crop experiment vs full-volume."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        augment: bool = False,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        crop_pad_shape: Tuple[int, int, int] = (72, 88, 72),
    ):
        self.pairs = list(pairs)
        self.augment = augment
        self.target_shape = target_shape
        self.crop_pad_shape = crop_pad_shape

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_path, post_path = self.pairs[idx]
        pre = load_volume_cropped(pre_path, target_shape=self.target_shape, pad_to_shape=self.crop_pad_shape)
        post = load_volume_cropped(post_path, target_shape=self.target_shape, pad_to_shape=self.crop_pad_shape)
        if self.augment:
            fl, fu, ff, scale = _random_augment()
            pre = augment_volume(pre, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
            post = augment_volume(post, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
        pre_t = torch.from_numpy(pre).unsqueeze(0).float()
        post_t = torch.from_numpy(post).unsqueeze(0).float()
        return pre_t, post_t


class Week7SlicePairs2D(Dataset):
    """2D dataset: same preprocessing as 3D, then take middle axial slice. Same augment (flip_lr, flip_ud, scale)."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        augment: bool = False,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    ):
        self.pairs = pairs
        self.augment = augment
        self.target_shape = target_shape

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_path, post_path = self.pairs[idx]
        pre = load_volume(pre_path, target_shape=self.target_shape)
        post = load_volume(post_path, target_shape=self.target_shape)
        # Middle axial slice: index D//2
        mid = self.target_shape[2] // 2
        pre_2d = pre[:, :, mid]   # (H, W)
        post_2d = post[:, :, mid]
        if self.augment:
            fl, fu, _, scale = _random_augment()
            # (H,W) -> (H,W,1) so augment_volume works; flip_fb on dim 2 is no-op for 1 slice
            pre_2d = augment_volume(pre_2d[:, :, np.newaxis], flip_lr=fl, flip_ud=fu, flip_fb=False, intensity_scale=scale).squeeze(-1)
            post_2d = augment_volume(post_2d[:, :, np.newaxis], flip_lr=fl, flip_ud=fu, flip_fb=False, intensity_scale=scale).squeeze(-1)
        pre_t = torch.from_numpy(pre_2d).unsqueeze(0).float()   # (1, H, W)
        post_t = torch.from_numpy(post_2d).unsqueeze(0).float()
        return pre_t, post_t


def _subject_id_from_path(pre_path: str) -> str:
    """Derive subject id from pre path (e.g. pre_2021_001.nii.gz -> 2021_001)."""
    base = os.path.basename(pre_path).replace(".nii.gz", "").replace(".nii", "")
    if base.startswith("pre_"):
        return base.replace("pre_", "", 1)
    m = re.match(r"(.+)_pre$", base)
    if m:
        return m.group(1)
    return base.replace("pre", "").strip("_") or "unknown"


# Phase 3: subject-specific masks dir (run generate_week7_subject_masks.py first)
SUBJECT_MASKS_DIR_DEFAULT = "/data1/julih/week7_subject_masks"


class Week7VolumePairs3DWithMasks(Dataset):
    """3D dataset returning (pre_vol, post_vol, mask_vol). Mask is per-subject from cache or MNI fallback."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        augment: bool = False,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        masks_dir: Optional[str] = None,
        pad_shape: Optional[Tuple[int, int, int]] = None,
    ):
        self.pairs = pairs
        self.augment = augment
        self.target_shape = target_shape
        self.masks_dir = masks_dir or os.environ.get("WEEK7_SUBJECT_MASKS_DIR", SUBJECT_MASKS_DIR_DEFAULT)
        self.pad_shape = pad_shape  # if set, mask is resized to pad_shape for loss (e.g. 96,112,96)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_path, post_path = self.pairs[idx]
        pre = load_volume(pre_path, target_shape=self.target_shape)
        post = load_volume(post_path, target_shape=self.target_shape)
        sid = _subject_id_from_path(pre_path)
        mask_path = os.path.join(self.masks_dir, f"{sid}.nii.gz")
        if os.path.isfile(mask_path):
            img = nib.load(mask_path)
            mask = np.asarray(img.dataobj).squeeze().astype(np.float32)
            if mask.shape != self.target_shape:
                factors = [self.target_shape[i] / mask.shape[i] for i in range(3)]
                mask = zoom(mask, factors, order=0)
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = get_brain_mask_for_shape(self.target_shape)
        if self.augment:
            fl, fu, ff, scale = _random_augment()
            pre = augment_volume(pre, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
            post = augment_volume(post, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=scale)
            mask = augment_volume(mask, flip_lr=fl, flip_ud=fu, flip_fb=ff, intensity_scale=1.0)
            mask = (mask > 0.5).astype(np.float32)
        out_shape = self.pad_shape if self.pad_shape else self.target_shape
        if mask.shape != out_shape:
            factors = [out_shape[i] / mask.shape[i] for i in range(3)]
            mask = zoom(mask, factors, order=0)
            mask = (mask > 0.5).astype(np.float32)
        pre_t = torch.from_numpy(pre).unsqueeze(0).float()
        post_t = torch.from_numpy(post).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return pre_t, post_t, mask_t


def get_week7_splits():
    """Return train, val, test pairs (pre_path, post_path) from combined 2020-2023."""
    train = get_pre_post_pairs("train")
    val = get_pre_post_pairs("val")
    test = get_pre_post_pairs("test")
    return train, val, test
