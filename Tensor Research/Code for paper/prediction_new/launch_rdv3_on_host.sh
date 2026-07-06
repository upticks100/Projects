#!/usr/bin/env bash
# Launch one watchdog + 8 CP workers covering both v3 objectives on this host.
#   4 workers on residual_delta_v3  (2 × LEVELS L=2, 2 × LEVELS L=4)
#   4 workers on ridge_delta_v3     (2 × LEVELS L=2, 2 × LEVELS L=4)
# Designed to be invoked via:
#   ssh <host> "bash '/student/mcnama53/Projects/Tensor Research/Code for paper/prediction_new/launch_rdv3_on_host.sh'"
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

# Skip if any v3 worker is already running on this host.
if pgrep -af 'worker.py.*_delta_v3' >/dev/null 2>&1; then
  echo "$HOSTLABEL: _delta_v3 workers already present; skipping launch"
  exit 0
fi

# One watchdog covers both v3 studies via the shared "_v3" substring match.
WD_LOG="$LOGS/rdv3_72h_${HOSTLABEL}_watchdog.log"
WD_STDOUT="$LOGS/rdv3_72h_${HOSTLABEL}_watchdog.stdout.log"
MEM_GB=$(awk '/MemTotal:/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
LIMIT_GB=$(awk -v m="$MEM_GB" 'BEGIN{printf "%.1f", m*0.8}')

nohup "$PY" -u "$WATCHDOG" \
  --match _delta_v3 \
  --interval 10 \
  --system-limit-gb "$LIMIT_GB" \
  --process-pct 85 \
  --cascade \
  --cascade-match _delta_v3 \
  --cleanup-margin-gb 6 \
  --start-timeout 1200 \
  --log "$WD_LOG" \
  > "$WD_STDOUT" 2>&1 < /dev/null &
echo "$HOSTLABEL: watchdog pid=$! limit_gb=$LIMIT_GB"

N_TRIALS=10000
BUDGET=259200  # 72h
launch_worker () {
  local objective="$1" mode="$2" L="$3" tag="$4"
  local log="$LOGS/${objective}_72h_${mode}_L${L}_${HOSTLABEL}_${TS}_${tag}.log"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    nohup "$PY" -u worker.py \
      --mode "$mode" --L "$L" \
      --n-trials "$N_TRIALS" \
      --time-budget-s "$BUDGET" \
      --objective "$objective" \
      --worker-name "${HOSTLABEL}_${objective}_${tag}" \
    > "$log" 2>&1 < /dev/null &
  echo "$HOSTLABEL: worker pid=$! obj=$objective mode=$mode L=$L tag=$tag log=$log"
}

# residual_delta_v3 cluster (FE-residual + per-feature X scale toggle)
launch_worker residual_delta_v3 LEVELS 2 fe_l2a
launch_worker residual_delta_v3 LEVELS 2 fe_l2b
launch_worker residual_delta_v3 LEVELS 4 fe_l4a
launch_worker residual_delta_v3 LEVELS 4 fe_l4b

# ridge_delta_v3 cluster (Ridge-residual booster + per-feature X scale toggle)
launch_worker ridge_delta_v3 LEVELS 2 rd_l2a
launch_worker ridge_delta_v3 LEVELS 2 rd_l2b
launch_worker ridge_delta_v3 LEVELS 4 rd_l4a
launch_worker ridge_delta_v3 LEVELS 4 rd_l4b

# Let nohup detach before SSH exits.
sleep 1
echo "$HOSTLABEL: launch_rdv3_on_host.sh complete"
