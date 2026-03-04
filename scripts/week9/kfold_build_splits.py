#!/usr/bin/env python3
"""
Build K-fold splits from train+val only; test set is fixed and untouched.

Reads combined_subject_split.json. Merges train and val into 220 subjects, splits by
subject into K folds (each fold: one part as val, rest as train). Writes kfold_splits.json
and verifies: no test subject in any fold; each subject in exactly one val fold; disjoint
train/val per fold.

Usage:
  python scripts/week9/kfold_build_splits.py --K 5 --out /data1/julih/scripts/week9/kfold_splits.json
  python scripts/week9/kfold_build_splits.py --K 5  # writes to week9/kfold_splits.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from week7_preprocess import load_combined_split, get_pre_post_pairs, COMBINED_SPLIT_PATH
from week7_data import _subject_id_from_path


def _subject_to_pairs(pairs):
    """Build dict subject_id -> (pre_path, post_path). One pair per subject."""
    out = {}
    for pre_path, post_path in pairs:
        sid = _subject_id_from_path(pre_path)
        if sid in out:
            continue
        out[sid] = (pre_path, post_path)
    return out


def _subject_to_pairs_from_data(data, split_key):
    """Build dict subject_id -> (pre_path, post_path) from data[split_key] using subject_id when present."""
    out = {}
    for item in data.get(split_key, []):
        pre = item.get("pre_path")
        post = item.get("post_path")
        if not pre or not post:
            continue
        sid = item.get("subject_id") or _subject_id_from_path(pre)
        out[sid] = (pre, post)
    return out


def build_kfold_splits(K: int, seed: int, split_path: str, out_path: str) -> dict:
    """
    Build K-fold splits and write to out_path. Return the written dict.
    Raises ValueError if verification fails.
    """
    if K < 2:
        raise ValueError("K must be >= 2")
    data = load_combined_split() if split_path == COMBINED_SPLIT_PATH else _load_json(split_path)
    # Use subject_id from split items when present (consistent IDs across path styles)
    train_s2p = _subject_to_pairs_from_data(data, "train")
    val_s2p = _subject_to_pairs_from_data(data, "val")
    subject_to_pairs = {**train_s2p, **val_s2p}
    all_subjects = list(subject_to_pairs.keys())
    test_subjects = list(data.get("test_subjects", [])) or list(_subject_to_pairs_from_data(data, "test").keys())

    # Verification: no test subject in train+val
    test_set = set(test_subjects)
    trainval_set = set(all_subjects)
    overlap = test_set & trainval_set
    if overlap:
        raise ValueError("Test set must be disjoint from train+val. Overlap: %s" % (list(overlap)[:5],))

    n_trainval = len(all_subjects)
    if n_trainval < K:
        raise ValueError("train+val has %d subjects; K=%d would leave empty folds" % (n_trainval, K))

    # Shuffle with fixed seed for reproducibility
    rng = __import__("random").Random(seed)
    shuffled = list(all_subjects)
    rng.shuffle(shuffled)

    # Split into K folds (fold i = chunk i as val)
    fold_size = n_trainval // K
    remainder = n_trainval % K
    folds = []
    start = 0
    for i in range(K):
        size = fold_size + (1 if i < remainder else 0)
        val_subjects = shuffled[start : start + size]
        train_subjects = [s for s in shuffled if s not in val_subjects]
        start += size
        folds.append({"fold": i, "train_subjects": train_subjects, "val_subjects": val_subjects})

    out = {
        "K": K,
        "seed": seed,
        "n_trainval": n_trainval,
        "n_test": len(test_subjects),
        "test_subjects": test_subjects,
        "folds": folds,
        "subject_to_pairs_source": "train+val from combined split; pairs resolved by subject_id",
    }

    # Verification
    all_val = set()
    for f in folds:
        t_set = set(f["train_subjects"])
        v_set = set(f["val_subjects"])
        if t_set & v_set:
            raise ValueError("Fold %d: train and val must be disjoint" % f["fold"])
        if t_set | v_set != trainval_set:
            raise ValueError("Fold %d: train U val must equal full train+val" % f["fold"])
        all_val |= v_set
    if all_val != trainval_set:
        raise ValueError("Union of val_subjects across folds must equal train+val")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _pairs_from_data(data, split_key):
    pairs = []
    for item in data.get(split_key, []):
        pre = item.get("pre_path")
        post = item.get("post_path")
        if pre and post and os.path.isfile(pre) and os.path.isfile(post):
            pairs.append((pre, post))
    return pairs


def get_train_val_pairs_for_fold(kfold_path: str, fold_index: int, split_path: str = None):
    """
    Load kfold_splits.json and return (train_pairs, val_pairs) for the given fold.
    Pairs are (pre_path, post_path) lists. split_path: combined split JSON (default COMBINED_SPLIT_PATH).
    """
    with open(kfold_path) as f:
        kfold = json.load(f)
    split_path = split_path or COMBINED_SPLIT_PATH
    data = load_combined_split() if split_path == COMBINED_SPLIT_PATH else _load_json(split_path)
    train_s2p = _subject_to_pairs_from_data(data, "train")
    val_s2p = _subject_to_pairs_from_data(data, "val")
    subject_to_pairs = {**train_s2p, **val_s2p}

    folds = kfold["folds"]
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError("fold_index %d out of range [0, %d)" % (fold_index, len(folds)))
    f = folds[fold_index]
    train_pairs = [subject_to_pairs[s] for s in f["train_subjects"] if s in subject_to_pairs]
    val_pairs = [subject_to_pairs[s] for s in f["val_subjects"] if s in subject_to_pairs]
    return train_pairs, val_pairs


def main():
    ap = argparse.ArgumentParser(description="Build K-fold splits (train+val only; test fixed)")
    ap.add_argument("--K", type=int, default=5, help="Number of folds")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for fold assignment")
    ap.add_argument("--split", default=COMBINED_SPLIT_PATH, help="Path to combined_subject_split.json")
    ap.add_argument("--out", default="", help="Output path for kfold_splits.json")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "kfold_splits.json")
    result = build_kfold_splits(args.K, args.seed, args.split, out)
    print("Wrote %s (K=%d, n_trainval=%d, n_test=%d)" % (out, result["K"], result["n_trainval"], result["n_test"]))


if __name__ == "__main__":
    main()
