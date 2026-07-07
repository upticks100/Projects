#!/usr/bin/env bash
# Part 2 battery on the HY dumps (pre-registered H-HY-IV, log 2026-07-07).
set -uo pipefail
PRED="/student/mcnama53/Projects/Tensor Research/Code for paper/prediction_new"
PY=/student/mcnama53/.local/share/mamba/envs/research/bin/python
HY="$PRED/results/v3_holdout_hy_20260707"
EDX="$PRED/../pre_prediction_cache/event_study_hy"
EXT="$PRED/../pre_prediction_cache/event_study_extended"
FUND="$PRED/../90-26_Q_Fundamentals_v2_HY.csv"
cd "$PRED"
for cell in "ridge_delta_v3 2" "ridge_delta_v3 4" "residual_delta_v3 2" "residual_delta_v3 4"; do
  read -r OBJ L <<< "$cell"; TAG="${OBJ}_L${L}"
  [ -f "$HY/multitarget_report_hy_${TAG}.txt" ] && { echo "[skip] $TAG"; continue; }
  echo "=== [$TAG] build panel ==="
  "$PY" build_event_study_dataset.py \
    --preds "$HY/predictions_${TAG}_rank1.pkl" \
    --features ALL \
    --fundamentals "$FUND" \
    --link-table "$EDX/link_table.csv" \
    --daily-returns "$EDX/daily_returns.csv.gz" \
    --daily-market "$EDX/daily_market.csv" \
    --ff3 "$EXT/ff3_daily.csv" \
    --iv "$EDX/optionmetrics_iv.csv.gz" \
    --out "$HY/event_study_multitarget_hy_${TAG}.csv" || exit 1
  echo "=== [$TAG] analyze ==="
  "$PY" analyze_event_study_multi.py \
    "$HY/event_study_multitarget_hy_${TAG}.csv" \
    --features ALL --fdr 0.10 > "$HY/multitarget_report_hy_${TAG}.txt" 2>&1 || exit 1
  echo "DONE $TAG"
done
echo "PART2_HY_ALL_DONE"
