#!/usr/bin/env bash
# Launch one residual_delta_v2 watchdog + 8 CP workers on this host.
# Each worker = (mode, L, batch_label) tuple. Each host gets 2 of each cell.
# Designed to be invoked via:
#   ssh <host> "bash '/student/mcnama53/Projects/Tensor Research/Code for paper/prediction_new/launch_rdv2_on_host.sh'"
# The script returns immediately after spawning; workers + watchdog detach via nohup.
set -euo pipefail

PROJECT="/student/mcnama53/Projects/Tensor Research/Code for paper"
PRED="$PROJECT/prediction_new"
PY=/student/mcnama53/.local/share/mamba/envs/research/bin/python
WATCHDOG="$PROJECT/cp_sweep_watchdog.py"
LOGS="$PRED/logs"
HOSTLABEL=$(hostname)
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOGS"
cd "$PRED"

# Skip if a v2 worker already running on this host.
if pgrep -af 'worker.py.*residual_delta_v2' >/dev/null 2>&1; then
  echo "$HOSTLABEL: residual_delta_v2 workers already present; skipping launch"
  exit 0
fi

# Start the per-host watchdog covering only residual_delta_v2 workers.
WD_LOG="$LOGS/rdv2_48h_${HOSTLABEL}_watchdog.log"
WD_STDOUT="$LOGS/rdv2_48h_${HOSTLABEL}_watchdog.stdout.log"
MEM_GB=$(awk '/MemTotal:/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
LIMIT_GB=$(awk -v m="$MEM_GB" 'BEGIN{printf "%.1f", m*0.8}')

nohup "$PY" -u "$WATCHDOG" \
  --match residual_delta_v2 \
  --interval 10 \
  --system-limit-gb "$LIMIT_GB" \
  --process-pct 85 \
  --cascade \
  --cascade-match residual_delta_v2 \
  --cleanup-margin-gb 6 \
  --start-timeout 900 \
  --log "$WD_LOG" \
  > "$WD_STDOUT" 2>&1 < /dev/null &
echo "$HOSTLABEL: watchdog pid=$! limit_gb=$LIMIT_GB"

# Launch 8 workers (2 per cell).
N_TRIALS=5000
BUDGET=172800  # 48h
launch_worker () {
  local mode="$1" L="$2" tag="$3"
  local log="$LOGS/rdv2_48h_${mode}_L${L}_${HOSTLABEL}_${TS}_${tag}.log"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    nohup "$PY" -u worker.py \
      --mode "$mode" --L "$L" \
      --n-trials "$N_TRIALS" \
      --time-budget-s "$BUDGET" \
      --objective residual_delta_v2 \
      --worker-name "${HOSTLABEL}_rdv2_${tag}" \
    > "$log" 2>&1 < /dev/null &
  echo "$HOSTLABEL: worker pid=$! mode=$mode L=$L tag=$tag log=$log"
}

launch_worker LEVELS   2 l2a
launch_worker LEVELS   4 l4a
launch_worker SURPRISE 2 s2a
launch_worker SURPRISE 4 s4a
launch_worker LEVELS   2 l2b
launch_worker LEVELS   4 l4b
launch_worker SURPRISE 2 s2b
launch_worker SURPRISE 4 s4b

# Give nohup a moment to disassociate before SSH exits.
sleep 1
echo "$HOSTLABEL: launch_rdv2_on_host.sh complete"
