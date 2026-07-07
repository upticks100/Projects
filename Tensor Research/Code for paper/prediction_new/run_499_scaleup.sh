#!/usr/bin/env bash
# 499-firm scale-up orchestrator (2026-07-06). IDEMPOTENT: every stage checks
# its outputs and is skipped if they already exist, so the script can be
# re-run after any failure or partial completion.
#
# Stages
#   1. caches   build tensor_cache_499 (local)
#   2. refits   dump 4 locked cells, one per lab host (ssh + setsid nohup)
#   3. gate     transfer check (locked hyperparams; FAIL blocks stage 4)
#   4. veer499  pre-registered veer test (H1 drift_cashflow->dDD, H2 veers->dIV)
#
# The MFI v2 rebuild is a parallel track launched separately
# (rebuild_mfi_tensor_v2.py); it shares no state with these stages.
#
# Launch DETACHED so it survives disconnects:
#   cd prediction_new && setsid nohup bash run_499_scaleup.sh \
#       > logs/run_499_scaleup.log 2>&1 < /dev/null &
set -uo pipefail

PROJECT="/student/mcnama53/Projects/Tensor Research/Code for paper"
PRED="$PROJECT/prediction_new"
PY=/student/mcnama53/.local/share/mamba/envs/research/bin/python
LOGS="$PRED/logs/scaleup_499"
OUT_DIR="$PRED/results/v3_holdout_499_20260706"
REF_SUMMARY="$PRED/results/v3_holdout_ext_20260629_230144/aggregate_summary.csv"
EVENT_DIR="$PROJECT/pre_prediction_cache/event_study_499"

# Frozen 499 environment (everything else identical to the extended run)
export PRED_FUNDAMENTALS_FILE="$PROJECT/90-26_Q_Fundamentals_v2_extended.csv"
export PRED_END_DATE="2026-06-30"
export PRED_CACHE_DIR="$PRED/tensor_cache_499"
export PRED_UNIVERSE_TOP_N="499"
# PRED_CP_LOWMEM: stock CPRegressor OOMs at 498 firms x rank 12-13 (~65 GB
# design matrix on 62 GB hosts). LowMemCPRegressor = identical math, blocked
# accumulation (test_cp_lowmem_equiv.py: exact agreement).
ENV_STR="PRED_FUNDAMENTALS_FILE='$PRED_FUNDAMENTALS_FILE' PRED_END_DATE='$PRED_END_DATE' PRED_CACHE_DIR='$PRED_CACHE_DIR' PRED_UNIVERSE_TOP_N=499 PRED_CP_LOWMEM=1 PRED_CP_LOWMEM_BLOCK=4"

# utmlab10-07 replaced by -04 (2026-07-06 21:05): 07 went unresponsive after
# the stock-CPRegressor OOM thrashing; residual L4 relaunched on idle -04.
HOSTS=(utmlab10-02 utmlab10-03 utmlab10-05 utmlab10-04)
CELLS=("ridge_delta_v3 2" "ridge_delta_v3 4" "residual_delta_v3 2" "residual_delta_v3 4")

mkdir -p "$LOGS" "$OUT_DIR"
cd "$PRED"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(stamp)] $*"; }

# ---------------- stage 1: caches -------------------------------------------
if [ -f "$PRED_CACHE_DIR/meta.pkl" ] && [ -f "$PRED_CACHE_DIR/tensor_levels_L4.pkl" ]; then
  say "stage1 caches: already built ($PRED_CACHE_DIR) — skip"
else
  say "stage1 caches: building into $PRED_CACHE_DIR"
  "$PY" -u build_prediction_caches.py > "$LOGS/build_caches_499.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$PRED_CACHE_DIR/meta.pkl" ]; then
    say "stage1 caches: FAILED (rc=$rc) — see $LOGS/build_caches_499.log"; exit 1
  fi
  say "stage1 caches: done"
fi

# locked trials into the 499 results dir
if [ ! -f "$OUT_DIR/aggregate_summary.csv" ]; then
  cp "$REF_SUMMARY" "$OUT_DIR/aggregate_summary.csv"
  say "copied locked aggregate_summary.csv -> $OUT_DIR"
fi

# ---------------- stage 2: refits on lab hosts ------------------------------
say "stage2 refits: checking/launching 4 cells"
for i in 0 1 2 3; do
  read -r obj L <<< "${CELLS[$i]}"
  host="${HOSTS[$i]}"
  pkl="$OUT_DIR/predictions_${obj}_L${L}_rank1.pkl"
  log="$LOGS/refit_${obj}_L${L}.log"
  if [ -f "$pkl" ]; then say "  ${obj} L${L}: dump exists — skip"; continue; fi
  # skip relaunch if the remote job is still alive
  if timeout 20 ssh -o BatchMode=yes "$host" "pgrep -af 'dump_test_predictions.*${obj}.*--L ${L}\b'" >/dev/null 2>&1; then
    say "  ${obj} L${L}: already running on $host — skip launch"; continue
  fi
  say "  ${obj} L${L}: launching on $host (log: $(basename "$log"))"
  timeout 30 ssh -o BatchMode=yes "$host" \
    "cd '$PRED' && setsid nohup env $ENV_STR OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 \
       '$PY' -u dump_test_predictions.py '$OUT_DIR' --objective $obj --L $L \
       > '$log' 2>&1 < /dev/null & echo launched"
done

# monitor loop: wait for all 4 dumps (or dead jobs), max 48h
say "stage2 refits: monitoring (poll 120s, max 48h)"
deadline=$(( $(date +%s) + 48*3600 ))
while :; do
  n_done=0; alive_report=""
  for i in 0 1 2 3; do
    read -r obj L <<< "${CELLS[$i]}"
    [ -f "$OUT_DIR/predictions_${obj}_L${L}_rank1.pkl" ] && n_done=$((n_done+1))
  done
  [ "$n_done" -eq 4 ] && { say "stage2 refits: all 4 dumps present"; break; }
  if [ "$(date +%s)" -gt "$deadline" ]; then
    say "stage2 refits: TIMEOUT with $n_done/4 dumps — aborting"; exit 1
  fi
  # dead-job detection: no pkl AND no remote process -> relaunch once per loop
  for i in 0 1 2 3; do
    read -r obj L <<< "${CELLS[$i]}"
    host="${HOSTS[$i]}"
    pkl="$OUT_DIR/predictions_${obj}_L${L}_rank1.pkl"
    log="$LOGS/refit_${obj}_L${L}.log"
    [ -f "$pkl" ] && continue
    if ! timeout 20 ssh -o BatchMode=yes "$host" "pgrep -f 'dump_test_predictions.*${obj}.*--L ${L}\b'" >/dev/null 2>&1; then
      say "  ${obj} L${L}: NOT running on $host and no dump — relaunching"
      timeout 30 ssh -o BatchMode=yes "$host" \
        "cd '$PRED' && setsid nohup env $ENV_STR OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 \
           '$PY' -u dump_test_predictions.py '$OUT_DIR' --objective $obj --L $L \
           > '$log' 2>&1 < /dev/null & echo relaunched"
    fi
  done
  sleep 120
done

# ---------------- stage 3: transfer-check gate ------------------------------
say "stage3 gate: transfer check vs 50-firm locked values"
"$PY" -u transfer_check_499.py "$OUT_DIR" --reference-summary "$REF_SUMMARY" \
  > "$LOGS/transfer_check.log" 2>&1
gate_rc=$?
cat "$LOGS/transfer_check.log"
if [ $gate_rc -eq 2 ]; then
  say "stage3 gate: FAIL — CP delta collapsed in all 4 cells. Pre-registered"
  say "veer test BLOCKED per protocol; this is a finding, report it."
  exit 2
elif [ $gate_rc -ne 0 ]; then
  say "stage3 gate: ERROR (rc=$gate_rc)"; exit 1
fi
say "stage3 gate: PASS"

# ---------------- stage 4: pre-registered veer test --------------------------
last_report="$OUT_DIR/veer_report_residual_delta_v3_L4_499.txt"
if [ -f "$last_report" ]; then
  say "stage4 veer499: reports already present — skip"
else
  say "stage4 veer499: running veer_anomaly_experiment on 499 dumps"
  "$PY" -u veer_anomaly_experiment.py "$OUT_DIR" \
    --event-dir "$EVENT_DIR" --tag 499 \
    > "$LOGS/veer_499.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    say "stage4 veer499: FAILED (rc=$rc) — see $LOGS/veer_499.log"; exit 1
  fi
fi
say "ALL STAGES DONE — results in $OUT_DIR"
