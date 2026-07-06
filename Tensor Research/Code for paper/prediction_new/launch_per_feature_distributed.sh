#!/usr/bin/env bash
# Distributed per-feature R² launcher.
# Fans out N trials (default: top-5 per cell across 4 cells = 20 tasks) to lab
# hosts. Each task writes one CSV to <results_dir>/cell_<obj>_L<L>_rank<r>.csv
# on NFS; the aggregator concatenates them.
set -euo pipefail

HOLDOUT_DIR="${HOLDOUT_DIR:-/student/mcnama53/Projects/Tensor Research/Code for paper/prediction_new/results/v3_holdout_20260620_084220}"
TOP_K="${TOP_K:-5}"

PROJECT="/student/mcnama53/Projects/Tensor Research/Code for paper"
PRED="$PROJECT/prediction_new"
PY=/student/mcnama53/.local/share/mamba/envs/research/bin/python
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$HOLDOUT_DIR/per_feature_${TS}"
LOG_DIR="$PRED/logs/per_feature_${TS}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Reachable host list (from launch_distributed_audit.sh).
HOSTS=(
  utmlab10-02 utmlab10-03 utmlab10-05 utmlab10-07 utmlab10-08
  utmlab10-11 utmlab10-13 utmlab10-17 utmlab10-19 utmlab10-23
  utmlab10-29
  utmlab20-02 utmlab20-03 utmlab20-05 utmlab20-07 utmlab20-11
  utmlab20-13 utmlab20-17 utmlab20-19 utmlab20-23 utmlab20-29
  utmlab26-02 utmlab26-03 utmlab26-07 utmlab26-11 utmlab26-13
  utmlab26-17 utmlab26-19 utmlab26-23 utmlab26-29
)
NHOSTS=${#HOSTS[@]}

# Build task list: 4 cells × TOP_K ranks.
TASKS=()
for cell in "residual_delta_v3 2" "residual_delta_v3 4" "ridge_delta_v3 2" "ridge_delta_v3 4"; do
  read -r obj L <<< "$cell"
  for r in $(seq 1 "$TOP_K"); do
    TASKS+=("$obj $L $r")
  done
done
NTASKS=${#TASKS[@]}

# Remote runner script (written to NFS, invoked by each ssh).
REMOTE_RUNNER="$OUT_DIR/run_one_trial.sh"
cat > "$REMOTE_RUNNER" <<'RUNNER_EOF'
#!/usr/bin/env bash
# Usage: run_one_trial.sh <holdout_dir> <objective> <L> <rank> <out_csv> <log>
set -euo pipefail
HOLDOUT="$1"; OBJ="$2"; L="$3"; RANK="$4"; OUT_CSV="$5"; LOG="$6"
PROJECT="/student/mcnama53/Projects/Tensor Research/Code for paper"
PY=/student/mcnama53/.local/share/mamba/envs/research/bin/python
cd "$PROJECT/prediction_new"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  nohup setsid "$PY" -u evaluate_per_feature.py \
    "$HOLDOUT" \
    --ranks "$RANK" \
    --objective "$OBJ" \
    --L "$L" \
    --out-csv "$OUT_CSV" \
    > "$LOG" 2>&1 < /dev/null &
disown
sleep 0.3
echo "launched pid=$!"
RUNNER_EOF
chmod +x "$REMOTE_RUNNER"

echo "=== per-feature distributed launch ==="
echo "holdout_dir = $HOLDOUT_DIR"
echo "out_dir     = $OUT_DIR"
echo "log_dir     = $LOG_DIR"
echo "tasks       = $NTASKS  (TOP_K=$TOP_K × 4 cells)"
echo "hosts       = $NHOSTS"
echo

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  host="${HOSTS[$((i % NHOSTS))]}"
  read -r obj L rank <<< "$task"
  tag="${obj}_L${L}_rank${rank}"
  out_csv="$OUT_DIR/cell_${tag}.csv"
  log="$LOG_DIR/${tag}_${host}.log"

  echo "[$host] $tag"
  remote_cmd=$(printf 'bash %q %q %q %q %q %q %q' \
    "$REMOTE_RUNNER" "$HOLDOUT_DIR" "$obj" "$L" "$rank" "$out_csv" "$log")
  ssh -o BatchMode=yes -o ConnectTimeout=10 \
      -o ControlMaster=no -o ControlPath=none "$host" \
      "$remote_cmd" 2>&1 | head -2 || \
    echo "  WARN: ssh launch on $host returned non-zero"
done

echo
echo "=== launch summary ==="
echo "out_dir = $OUT_DIR"
echo "Monitor with:"
echo "  ls $OUT_DIR"
echo "  for h in $(echo "${HOSTS[@]}" | tr ' ' '\\n' | sort -u | head -10 | tr '\\n' ' '); do echo == \$h ==; ssh \$h 'ps -eo pid,etime,args --no-headers | grep evaluate_per_feature | grep -v grep | head -2'; done"
