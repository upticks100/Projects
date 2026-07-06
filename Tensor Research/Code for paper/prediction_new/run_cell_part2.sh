#!/usr/bin/env bash
# Part 2 (extended, fixed-origin) pipeline for ONE locked cell.
# Usage: run_cell_part2.sh <objective> <L>
#   1. rebuild aggregate_summary.csv from all finished per-cell CSVs
#   2. dump extended (21Q) predictions for the cell (rank1)
#   3. build the multi-target event-study panel (FIXED builder + EXTENDED data)
#   4. run the outlier-robust multi-target analyzer -> report txt
set -euo pipefail
OBJ="$1"; L="$2"
PRED="/student/mcnama53/Projects/Tensor Research/Code for paper/prediction_new"
PY=/student/mcnama53/.local/share/mamba/envs/research/bin/python
EXT="$PRED/results/v3_holdout_ext_20260629_230144"
EDX="$PRED/../pre_prediction_cache/event_study_extended"
FUND="$PRED/../90-26_Q_Fundamentals_v2_extended.csv"
TAG="${OBJ}_L${L}"
cd "$PRED"

echo "=== [$TAG] (1) rebuild aggregate_summary ==="
$PY - <<PY
import glob, pandas as pd, os
ext = "$EXT"
cells = [f for f in glob.glob(os.path.join(ext, "*_L*.csv"))
         if all(s not in f for s in ("_per_window", "event_study",
                                     "aggregate", "multitarget"))]
pd.concat([pd.read_csv(c) for c in cells], ignore_index=True).to_csv(
    os.path.join(ext, "aggregate_summary.csv"), index=False)
print("  aggregate cells:", sorted(os.path.basename(c) for c in cells))
PY

echo "=== [$TAG] (2) dump extended predictions ==="
PRED_CACHE_DIR="$PRED/tensor_cache_ext" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 \
  $PY -u dump_test_predictions.py "$EXT" --objective "$OBJ" --L "$L" --rank-order 1

echo "=== [$TAG] (3) build multi-target panel ==="
$PY build_event_study_dataset.py \
  --preds "$EXT/predictions_${TAG}_rank1.pkl" \
  --features ALL \
  --fundamentals "$FUND" \
  --link-table "$EDX/link_table.csv" \
  --daily-returns "$EDX/daily_returns.csv" \
  --daily-market "$EDX/daily_market.csv" \
  --ff3 "$EDX/ff3_daily.csv" \
  --iv "$EDX/optionmetrics_iv.csv" \
  --out "$EXT/event_study_multitarget_ext_${TAG}.csv"

echo "=== [$TAG] (4) analyze (outlier-robust) ==="
$PY analyze_event_study_multi.py \
  "$EXT/event_study_multitarget_ext_${TAG}.csv" \
  --features ALL --fdr 0.10 > "$EXT/multitarget_report_ext_${TAG}.txt" 2>&1

echo "DONE $TAG -> $EXT/multitarget_report_ext_${TAG}.txt"
