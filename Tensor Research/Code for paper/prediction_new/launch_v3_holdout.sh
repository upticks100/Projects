#!/usr/bin/env bash
# v3 holdout evaluation launcher (post 2026-06-19 OOF-fallback fix).
#
# Fans out 4 evaluator processes (one per cell), each scoring its cell's
# top-K trials on the held-out test set with the rewritten
# evaluate_top_trials_test.py. Each cell writes its own pair of CSVs:
#   results/v3_holdout_<ts>/<objective>_L<L>.csv               (pooled)
#   results/v3_holdout_<ts>/<objective>_L<L>_per_window.csv    (per-window)
#
# Combine with aggregate_v3_holdout.py after.
set -euo pipefail

TOP_K="${TOP_K:-5}"
N_JOBS_CP="${N_JOBS_CP:-2}"

PROJECT="/student/mcnama53/Projects/Tensor Research/Code for paper"
PRED="$PROJECT/prediction_new"
PY=/student/mcnama53/.local/share/mamba/envs/research/bin/python
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$PRED/results/v3_holdout_${TS}"
LOG_DIR="$PRED/logs/v3_holdout_${TS}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Cell -> host mapping. Slowest cells (L=4 FE) go to the freshest hosts.
# Pick from launch_distributed_audit.sh's reachable list, exclude hosts
# already busy (utmlab10-08 is running the smokes; check before re-running).
declare -A HOST_FOR
HOST_FOR["residual_delta_v3 4"]=utmlab10-02
HOST_FOR["residual_delta_v3 2"]=utmlab10-03
HOST_FOR["ridge_delta_v3 4"]=utmlab10-05
HOST_FOR["ridge_delta_v3 2"]=utmlab10-07

echo "=== v3 holdout launch ==="
echo "out_dir = $OUT_DIR"
echo "log_dir = $LOG_DIR"
echo "top_k   = $TOP_K"
echo

REMOTE_RUNNER="$OUT_DIR/run_on_host.sh"
cat > "$REMOTE_RUNNER" <<'RUNNER_EOF'
#!/usr/bin/env bash
# Invoked on a remote host as: bash run_on_host.sh <obj> <L> <topk> <njobs> <out_csv> <pw_csv> <log>
set -euo pipefail
OBJ="$1"; L="$2"; TOPK="$3"; NJOBS="$4"; OUT_CSV="$5"; PW_CSV="$6"; LOG="$7"
PROJECT="/student/mcnama53/Projects/Tensor Research/Code for paper"
PY=/student/mcnama53/.local/share/mamba/envs/research/bin/python
cd "$PROJECT/prediction_new"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  nohup setsid "$PY" -u evaluate_top_trials_test.py \
    --objective "$OBJ" \
    --top-k "$TOPK" \
    --min-completed 100 \
    --n-jobs-cp "$NJOBS" \
    --modes LEVELS \
    --lookbacks "$L" \
    --output "$OUT_CSV" \
    --per-window-output "$PW_CSV" \
    > "$LOG" 2>&1 < /dev/null &
disown
sleep 0.5
echo "launched pid=$!"
RUNNER_EOF
chmod +x "$REMOTE_RUNNER"

PIDS=()
for cell in "residual_delta_v3 4" "residual_delta_v3 2" "ridge_delta_v3 4" "ridge_delta_v3 2"; do
  read -r obj L <<< "$cell"
  host="${HOST_FOR[$cell]}"
  tag="${obj}_L${L}"
  out_csv="$OUT_DIR/${tag}.csv"
  pw_csv="$OUT_DIR/${tag}_per_window.csv"
  log="$LOG_DIR/${tag}_${host}.log"

  echo "[$host] $tag -> $log"
  remote_cmd=$(printf 'bash %q %q %q %q %q %q %q %q' \
    "$REMOTE_RUNNER" "$obj" "$L" "$TOP_K" "$N_JOBS_CP" \
    "$out_csv" "$pw_csv" "$log")
  ssh -o BatchMode=yes -o ConnectTimeout=10 \
      -o ControlMaster=no -o ControlPath=none "$host" \
      "$remote_cmd" || \
    echo "  WARN: ssh launch on $host returned non-zero"
done

echo
echo "=== launch summary ==="
echo "out_dir = $OUT_DIR"
echo "log_dir = $LOG_DIR"
echo
echo "Monitor with:"
echo "  for h in utmlab10-02 utmlab10-03 utmlab10-05 utmlab10-07; do"
echo "    echo \"== \$h ==\""
echo "    ssh \$h pgrep -af evaluate_top_trials_test"
echo "  done"
echo
echo "Or tail any log: tail -f $LOG_DIR/<tag>_<host>.log"
