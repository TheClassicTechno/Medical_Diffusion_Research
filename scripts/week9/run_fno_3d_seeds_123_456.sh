#!/bin/bash
# Run only FNO_3D seeds 123 and 456 with full epochs (no WEEK9_QUICK_EPOCHS).
# Use after the main 3-seed run completed FNO seed 42 but stopped before 123/456.

set -e
ROOT="/data1/julih"
cd "$ROOT"
mkdir -p week8_results

PYTHON="python3"
[[ -x /data1/julih/miniconda3/envs/julih_monai/bin/python3 ]] && PYTHON="/data1/julih/miniconda3/envs/julih_monai/bin/python3"
export PYTHON
export WEEK7=1
unset WEEK7_REGION_WEIGHT
unset WEEK7_SUBJECT_MASKS
unset WEEK9_QUICK_EPOCHS

for seed in 123 456; do
  echo "===== FNO_3D seed=$seed ====="
  cd "$ROOT/NeuralOperators"
  export SEED="$seed"
  $PYTHON fno_3d_finetune_slice_brain.py --week7
  cp fno_3d_week7_best_results.json "$ROOT/week8_results/FNO_3D_seed${seed}.json"
  echo "  -> week8_results/FNO_3D_seed${seed}.json"
  cd "$ROOT"
done

echo "FNO_3D seeds 123 and 456 done."
echo "=== Running finalize_full_epochs_results.sh ==="
"$ROOT/scripts/week9/finalize_full_epochs_results.sh"
