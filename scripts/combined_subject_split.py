#!/usr/bin/env python3
"""
Build combined dataset: existing pre/post (/data1/julih/pre, post) + 2020 pairs.
Subject-level split so no subject appears in both train and test.
Output: /data1/julih/combined_subject_split.json (same format as 2020_single_delay_split.json).
"""
import os
import json
import random
import argparse

DATA_DIR = "/data1/julih"
PRE_DIR = os.path.join(DATA_DIR, "pre")
POST_DIR = os.path.join(DATA_DIR, "post")
SPLIT_2020 = "/data1/julih/2020_single_delay_split.json"
SEED = 42
TRAIN_FRAC = 0.75
VAL_FRAC = 0.125


def pre_to_post_path(pre_path):
    base = os.path.basename(pre_path).replace("pre_", "post_")
    return os.path.join(POST_DIR, base)


def subject_id_from_pre_path(pre_path):
    """pre/pre_2021_008.nii.gz -> 2021_008. pre_2022_032.nii.gz -> 2022_032."""
    base = os.path.basename(pre_path).replace(".nii.gz", "").replace("pre_", "")
    return base  # e.g. 2021_008, 2022_032


def subject_id_from_2020(sid):
    """moyamoya_stanford_2020_001 -> 2020_001."""
    if "moyamoya_stanford_2020_" in sid:
        return "2020_" + sid.split("_")[-1]
    return sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data1/julih/combined_subject_split.json")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    args = ap.parse_args()

    random.seed(args.seed)
    pairs_by_subject = {}  # subject_id -> list of (pre_path, post_path)

    # Existing pre/post
    if os.path.isdir(PRE_DIR):
        for f in sorted(os.listdir(PRE_DIR)):
            if not f.startswith("pre_") or not f.endswith(".nii.gz"):
                continue
            pre_path = os.path.join(PRE_DIR, f)
            post_path = pre_to_post_path(pre_path)
            if not os.path.isfile(post_path):
                continue
            sid = subject_id_from_pre_path(pre_path)
            if sid not in pairs_by_subject:
                pairs_by_subject[sid] = []
            pairs_by_subject[sid].append((pre_path, post_path))

    # 2020
    if os.path.isfile(SPLIT_2020):
        with open(SPLIT_2020) as f:
            data = json.load(f)
        for part in ("train", "val", "test"):
            for item in data.get(part, []):
                sid = subject_id_from_2020(item["subject_id"])
                pre_path = item["pre_path"]
                post_path = item["post_path"]
                if os.path.isfile(pre_path) and os.path.isfile(post_path):
                    if sid not in pairs_by_subject:
                        pairs_by_subject[sid] = []
                    pairs_by_subject[sid].append((pre_path, post_path))

    # One pair per subject (take first if multiple)
    subject_ids = sorted(pairs_by_subject.keys())
    pairs = []
    for sid in subject_ids:
        pre_path, post_path = pairs_by_subject[sid][0]
        pairs.append((sid, pre_path, post_path))

    n = len(pairs)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train

    random.shuffle(pairs)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val :]

    out = {
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_test": len(test_pairs),
        "train_subjects": [p[0] for p in train_pairs],
        "val_subjects": [p[0] for p in val_pairs],
        "test_subjects": [p[0] for p in test_pairs],
        "train": [{"subject_id": p[0], "pre_path": p[1], "post_path": p[2]} for p in train_pairs],
        "val": [{"subject_id": p[0], "pre_path": p[1], "post_path": p[2]} for p in val_pairs],
        "test": [{"subject_id": p[0], "pre_path": p[1], "post_path": p[2]} for p in test_pairs],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Combined: %d subjects, %d train / %d val / %d test. Written %s" % (n, len(train_pairs), len(val_pairs), len(test_pairs), args.out))


if __name__ == "__main__":
    main()
