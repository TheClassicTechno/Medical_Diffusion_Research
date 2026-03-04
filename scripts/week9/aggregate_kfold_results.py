#!/usr/bin/env python3
"""
Aggregate K-fold test results into mean ± std (and optional 95% CI) across folds.

Input: K JSON files with mae_mean, ssim_mean, psnr_mean (e.g. unet3d_fold0_test_results.json, ...).
Output: One summary with mean ± std; optional 95% CI = mean ± 1.96*std/sqrt(K).

Usage:
  python scripts/week9/aggregate_kfold_results.py --results_dir scripts/week9 --pattern "unet3d_fold*_test_results.json"
  python scripts/week9/aggregate_kfold_results.py --results_dir scripts/week9 --model unet3d --K 5
"""
import argparse
import json
import os
import re
import glob
import numpy as np


def load_fold_results(results_dir: str, pattern: str = None, model: str = None, K: int = 0) -> list:
    """Load per-fold result JSONs. Returns list of dicts with fold, mae_mean, ssim_mean, psnr_mean."""
    results_dir = os.path.abspath(results_dir)
    if pattern:
        paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    elif model:
        paths = sorted(glob.glob(os.path.join(results_dir, "%s_fold*_test_results.json" % model)))
    else:
        paths = sorted(glob.glob(os.path.join(results_dir, "*_fold*_test_results.json")))
    out = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
            out.append({
                "fold": d.get("fold", -1),
                "mae_mean": float(d.get("mae_mean", d.get("MAE", float("nan")))),
                "ssim_mean": float(d.get("ssim_mean", d.get("SSIM", float("nan")))),
                "psnr_mean": float(d.get("psnr_mean", d.get("PSNR", float("nan")))),
                "path": p,
            })
        except Exception as e:
            print("Skip %s: %s" % (p, e))
    # Sort by fold index
    out.sort(key=lambda x: x["fold"])
    if K and len(out) != K:
        print("Warning: expected %d folds, found %d" % (K, len(out)))
    return out


def mean_std_ci(vals: list, confidence: float = 0.95) -> tuple:
    """Return (mean, std, ci_half). 95%% CI half-width = 1.96*std/sqrt(n) (normal approx)."""
    a = np.array([float(x) for x in vals if not (x != x or x == float("inf"))])
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = len(a)
    mean = float(np.mean(a))
    std = float(np.std(a))
    if n < 2:
        ci_half = 0.0
    else:
        # 1.96 for 95% CI (normal); for small n could use t-distribution
        z = 1.96 if confidence >= 0.95 else 2.576  # 99%
        ci_half = float(z * std / np.sqrt(n))
    return mean, std, ci_half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="", help="Directory containing fold result JSONs")
    ap.add_argument("--pattern", default="", help="Glob pattern (e.g. unet3d_fold*_test_results.json)")
    ap.add_argument("--model", default="unet3d", help="Model name for glob")
    ap.add_argument("--K", type=int, default=0, help="Expected number of folds (for validation)")
    ap.add_argument("--ci", action="store_true", help="Include 95%% CI in output")
    ap.add_argument("--output", default="", help="Write summary to this path (JSON or .md)")
    args = ap.parse_args()
    results_dir = args.results_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week9")
    results_dir = os.path.abspath(results_dir)
    pattern = args.pattern or None
    items = load_fold_results(results_dir, pattern=pattern, model=args.model, K=args.K)
    if not items:
        print("No fold results found in %s" % results_dir)
        return
    K = len(items)
    mae_vals = [x["mae_mean"] for x in items]
    ssim_vals = [x["ssim_mean"] for x in items]
    psnr_vals = [x["psnr_mean"] for x in items]
    mae_mean, mae_std, mae_ci = mean_std_ci(mae_vals) if args.ci else (float(np.mean(mae_vals)), float(np.std(mae_vals)), None)
    if not args.ci:
        ssim_mean, ssim_std, ssim_ci = float(np.mean(ssim_vals)), float(np.std(ssim_vals)), None
        psnr_mean, psnr_std, psnr_ci = float(np.mean(psnr_vals)), float(np.std(psnr_vals)), None
    else:
        ssim_mean, ssim_std, ssim_ci = mean_std_ci(ssim_vals)
        psnr_mean, psnr_std, psnr_ci = mean_std_ci(psnr_vals)
    print("K-fold (K=%d) test set: MAE %.4f +/- %.4f  SSIM %.4f +/- %.4f  PSNR %.2f +/- %.2f" % (
        K, mae_mean, mae_std, ssim_mean, ssim_std, psnr_mean, psnr_std))
    if args.ci:
        print("95%% CI: MAE [%.4f, %.4f]  SSIM [%.4f, %.4f]  PSNR [%.2f, %.2f]" % (
            mae_mean - mae_ci, mae_mean + mae_ci,
            ssim_mean - ssim_ci, ssim_mean + ssim_ci,
            psnr_mean - psnr_ci, psnr_mean + psnr_ci))
    summary = {
        "n_folds": K,
        "mae_mean": mae_mean, "mae_std": mae_std,
        "ssim_mean": ssim_mean, "ssim_std": ssim_std,
        "psnr_mean": psnr_mean, "psnr_std": psnr_std,
    }
    if args.ci:
        summary["mae_ci95_half"] = mae_ci
        summary["ssim_ci95_half"] = ssim_ci
        summary["psnr_ci95_half"] = psnr_ci
    if args.output:
        with open(args.output, "w") as f:
            if args.output.endswith(".md"):
                f.write("| Metric | Mean ± Std |\n|--------|------------|\n")
                f.write("| MAE | %.4f ± %.4f |\n" % (mae_mean, mae_std))
                f.write("| SSIM | %.4f ± %.4f |\n" % (ssim_mean, ssim_std))
                f.write("| PSNR | %.2f ± %.2f |\n" % (psnr_mean, psnr_std))
                if args.ci:
                    f.write("\n95%% CI: MAE [%.4f, %.4f], SSIM [%.4f, %.4f], PSNR [%.2f, %.2f]\n" % (
                        mae_mean - mae_ci, mae_mean + mae_ci, ssim_mean - ssim_ci, ssim_mean + ssim_ci, psnr_mean - psnr_ci, psnr_mean + psnr_ci))
            else:
                json.dump(summary, f, indent=2)
        print("Wrote %s" % args.output)


if __name__ == "__main__":
    main()
