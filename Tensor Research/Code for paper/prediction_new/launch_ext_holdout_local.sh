#!/usr/bin/env bash
# Extended (T=21) holdout re-score, LOCAL, on tensor_cache_ext.
#
# Reuses the LOCKED v3 hyperparameters from the unchanged optuna_journal and
# re-scores each cell's top-K trials on the calendar-fixed test block
# (2021Q1..2026Q1). Part 1's first 16 test windows must reproduce; +5 new
# windows (2025Q1..2026Q1) append. Writes one CSV pair per cell:
#   results/v3_holdout_ext_<ts>/<objective>_L<L>.csv
#   results/v3_holdout_ext_<ts>/<objective>_L<L>_per_window.csv
set -euo pipefail

TOP_K="${TOP_K:-5}"
N_JOBS_CP="${N_JOBS_CP:-3}"
PRED="/student/mcnama53/Projects/Tensor Research/Code for paper/prediction_new"
PY=/student/mcnama53/.local/share/mamba/envs/research/bin/python
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$PRED/results/v3_holdout_ext_${TS}"
LOG_DIR="$PRED/logs/v3_holdout_ext_${TS}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Point the whole pipeline at the EXTENDED cache (calendar split anchors 2021Q1)
export PRED_CACHE_DIR="$PRED/tensor_cache_ext"

echo "=== extended holdout (local) ==="
echo "out_dir = $OUT_DIR"
echo "cache   = $PRED_CACHE_DIR"
echo "top_k=$TOP_K  n_jobs_cp=$N_JOBS_CP"
echo

cd "$PRED"
for cell in "residual_delta_v3 2" "residual_delta_v3 4" "ridge_delta_v3 2" "ridge_delta_v3 4"; do
  read -r obj L <<< "$cell"
  tag="${obj}_L${L}"
  out_csv="$OUT_DIR/${tag}.csv"
  pw_csv="$OUT_DIR/${tag}_per_window.csv"
  log="$LOG_DIR/${tag}.log"
  echo "[launch] $tag -> $log"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    PRED_CACHE_DIR="$PRED_CACHE_DIR" \
    nohup setsid "$PY" -u evaluate_top_trials_test.py \
      --objective "$obj" \
      --top-k "$TOP_K" \
      --min-completed 100 \
      --n-jobs-cp "$N_JOBS_CP" \
      --modes LEVELS \
      --lookbacks "$L" \
      --output "$out_csv" \
      --per-window-output "$pw_csv" \
      > "$log" 2>&1 < /dev/null &
  disown
  sleep 0.5
done

echo
echo "OUT_DIR=$OUT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "launched 4 cells; monitor logs in $LOG_DIR"
