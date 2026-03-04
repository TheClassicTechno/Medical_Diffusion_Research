#!/bin/bash
# Run after: (1) K-fold UNet 3D scripts with 50 epochs, (2) 5 models × 3 seeds with full epochs.
# Aggregates results and updates summary files.

set -e
ROOT="/data1/julih"
cd "$ROOT"

echo "=== Aggregating 3-seed results (week8_results) ==="
python3 scripts/aggregate_week8_seeds.py --results_dir week8_results --output scripts/week9/week8_best5_mean_std_full_epochs.md --ci

echo ""
echo "=== Aggregating K-fold results (UNet 3D scripts, 50 ep) ==="
if [[ -f scripts/week9/unet3d_fold4_test_results.json ]]; then
  python3 scripts/week9/aggregate_kfold_results.py --results_dir scripts/week9 --model unet3d --K 5 --ci --output scripts/week9/unet3d_kfold_50ep_summary.json
  echo "K-fold (50 ep) summary written to scripts/week9/unet3d_kfold_50ep_summary.json"
else
  echo "K-fold test result files not all present yet; run after week9_train_unet3d_kfold.py --epochs 50 completes for folds 0..4."
fi

echo ""
echo "Done. Update WEEK10_MENTOR_PROGRESS_SUMMARY.txt and BEST_MODELS_MEAN_STD_REPORTING.md with the new paths/numbers if needed."
