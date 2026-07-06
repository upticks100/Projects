#!/usr/bin/env bash
# Distributed per-fold audit launcher.
# Fans out 45 single-fit tasks (3 cells × 5 trials × 3 folds, excluding the
# already-completed L=2 booster) to ~29 reachable lab hosts. Each task writes
# one CSV row to a unique file on NFS; the aggregator concatenates them.
set -euo pipefail

PROJECT="/student/mcnama53/Projects/Tensor Research/Code for paper"
PRED="$PROJECT/prediction_new"
PY=/student/mcnama53/.local/share/mamba/envs/research/bin/python
TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$PRED/results/distributed_audit_${TS}"
LOGS_DIR="$PRED/logs/distributed_audit_${TS}"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Reachable host list (utmlab26-05 excluded — publickey rejected).
HOSTS=(
  utmlab10-02 utmlab10-03 utmlab10-05 utmlab10-07 utmlab10-11
  utmlab10-13 utmlab10-17 utmlab10-19 utmlab10-23 utmlab10-29
  utmlab20-02 utmlab20-03 utmlab20-05 utmlab20-07 utmlab20-11
  utmlab20-13 utmlab20-17 utmlab20-19 utmlab20-23 utmlab20-29
  utmlab26-02 utmlab26-03 utmlab26-07 utmlab26-11 utmlab26-13
  utmlab26-17 utmlab26-19 utmlab26-23 utmlab26-29
)
NHOSTS=${#HOSTS[@]}

# Tasks: 3 cells we still need × 5 trials × 3 folds = 45.
TASKS=()
for cell in "residual_delta_v3 2" "residual_delta_v3 4" "ridge_delta_v3 4"; do
  obj=$(echo "$cell" | awk '{print $1}')
  L=$(echo "$cell" | awk '{print $2}')
  for rank in 1 2 3 4 5; do
    for fold in 1 2 3; do
      TASKS+=("$obj $L $rank $fold")
    done
  done
done
NTASKS=${#TASKS[@]}

echo "=== Distributed audit launch ==="
echo "results_dir = $RESULTS_DIR"
echo "logs_dir    = $LOGS_DIR"
echo "hosts       = $NHOSTS"
echo "tasks       = $NTASKS"
echo

# Fan out: round-robin host assignment, all SSHs in parallel.
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  host="${HOSTS[$((i % NHOSTS))]}"
  read -r obj L rank fold <<< "$task"
  tag="${obj}_L${L}_rank${rank}_fold${fold}"
  out_csv="$RESULTS_DIR/${tag}.csv"
  log="$LOGS_DIR/${tag}_${host}.log"

  # Use a single nohup'd backgrounded SSH per task so the parent can wait.
  (
    timeout 1800 ssh -o BatchMode=yes -o ConnectTimeout=10 \
      -o ControlMaster=no -o ControlPath=none "$host" \
      "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       NUMEXPR_NUM_THREADS=4 \
       $PY -u '$PRED/audit_one_fit.py' \
         --objective '$obj' --L '$L' \
         --rank-order '$rank' --outer-fold '$fold' \
         --output '$out_csv'" \
      > "$log" 2>&1
    rc=$?
    echo "$tag  host=$host  rc=$rc  $(date +%H:%M:%S)" \
      >> "$LOGS_DIR/_dispatch.log"
  ) &
done

echo "Launched $NTASKS tasks across $NHOSTS hosts."
echo "Waiting for all SSH calls to return..."
wait
echo
echo "=== All SSH calls returned at $(date) ==="
echo "Result rows on disk:"
ls "$RESULTS_DIR" | wc -l
echo
echo "Aggregating..."
agg="$RESULTS_DIR/_aggregated.csv"
header_written=false
for f in "$RESULTS_DIR"/*.csv; do
  [[ "$(basename "$f")" == _*.csv ]] && continue
  if ! $header_written; then
    head -1 "$f" > "$agg"
    header_written=true
  fi
  tail -n +2 "$f" >> "$agg"
done
echo "Aggregated CSV: $agg"
wc -l "$agg"
