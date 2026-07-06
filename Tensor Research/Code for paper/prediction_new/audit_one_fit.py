#!/usr/bin/env python3
"""Run a SINGLE (objective, L, rank_order, outer_fold) CP fit and emit one CSV row.

Designed to be invoked over SSH from a distributed launcher. The fit logic is
identical to the per_fold_signal_audit_parallel.py evaluate_task function; the
trial selection logic is identical to load_top_trials. Output is one CSV row
written atomically to a unique filename so a downstream aggregator can simply
concatenate all rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorly as tl  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from tensorly.regression.cp_regression import CPRegressor  # noqa: E402


def decode_param(value, distribution: str):
    try:
        dist = json.loads(distribution)
        if dist.get("name") == "CategoricalDistribution":
            return dist["attributes"]["choices"][int(value)]
    except Exception:
        pass
    return value


def load_top_trial(journal_path: Path, rank_order: int) -> dict:
    params: dict[int, dict] = {}
    values: dict[int, float] = {}
    with journal_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            op = ev.get("op_code")
            if op == 5:
                tid = ev["trial_id"]
                params.setdefault(tid, {})[ev["param_name"]] = decode_param(
                    ev["param_value_internal"],
                    ev.get("distribution", ""),
                )
            elif op == 6 and ev.get("state") == 1 and ev.get("values"):
                values[ev["trial_id"]] = float(ev["values"][0])
    ranked = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    if rank_order < 1 or rank_order > len(ranked):
        raise SystemExit(f"rank_order {rank_order} out of range (have {len(ranked)} completed trials)")
    tid, value = ranked[rank_order - 1]
    p = params[tid].copy()
    return {
        "rank_order": rank_order,
        "trial_number": int(tid),
        "journal_value": float(value),
        "params": {
            "RANK_REGRESS": int(p["RANK_REGRESS"]),
            "REG_W": float(p["REG_W"]),
            "GAMMA": float(p.get("GAMMA", 1.0)),
            "USE_RMS_SCALING": bool(p["USE_RMS_SCALING"]),
            "FEATURE_TARGET_SCALE": bool(p.get("FEATURE_TARGET_SCALE", False)),
            "FEATURE_X_SCALE": bool(p.get("FEATURE_X_SCALE", False)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path,
                        default=Path("/student/mcnama53/Projects/Tensor Research"))
    parser.add_argument("--objective", required=True,
                        choices=["residual_delta_v3", "ridge_delta_v3"])
    parser.add_argument("--mode", default="LEVELS")
    parser.add_argument("--L", type=int, required=True, choices=[2, 4])
    parser.add_argument("--rank-order", type=int, required=True,
                        help="1-indexed rank of the trial within the study")
    parser.add_argument("--outer-fold", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--output", type=Path, required=True,
                        help="Per-fit CSV row will be written here (single line + header)")
    args = parser.parse_args()

    pred_dir = args.project_root / "Code for paper/prediction_new"
    sys.path.insert(0, str(pred_dir))
    sys.path.insert(0, str(pred_dir.parent))

    from prediction_config import N_ITER_MAX, SEED, cache_path  # noqa: PLC0415
    from worker import (  # noqa: PLC0415
        _compute_ridge_predictions_for_fold,
        _per_feature_x_scale,
        evaluate_model,
        firm_feature_means,
    )

    tl.set_backend("numpy")
    np.random.seed(SEED)

    journal_path = (
        pred_dir
        / f"optuna_journal/study_{args.mode.lower()}_L{args.L}_{args.objective}.log"
    )
    if not journal_path.exists():
        raise SystemExit(f"journal not found: {journal_path}")

    trial = load_top_trial(journal_path, args.rank_order)
    p = trial["params"]
    is_booster = args.objective == "ridge_delta_v3"

    t_load = time.time()
    cache = joblib.load(cache_path(args.mode, args.L))
    X_all, Y_all, M_all = cache["X"], cache["Y"], cache["Mask"]
    split_idx = int(0.8 * len(X_all))
    X_dev, Y_dev, M_dev = X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx]

    folds = list(TimeSeriesSplit(n_splits=3).split(X_dev))
    if args.outer_fold < 1 or args.outer_fold > len(folds):
        raise SystemExit(f"outer_fold {args.outer_fold} out of range")
    tr_idx, va_idx = folds[args.outer_fold - 1]
    X_tr, Y_tr, M_tr = X_dev[tr_idx], Y_dev[tr_idx], M_dev[tr_idx]
    X_va, Y_va, M_va = X_dev[va_idx], Y_dev[va_idx], M_dev[va_idx]
    mu_ff = firm_feature_means(Y_tr, M_tr)
    print(f"cache_loaded seconds={time.time() - t_load:.2f} "
          f"X_tr={X_tr.shape} X_va={X_va.shape}", flush=True)

    if is_booster:
        t_ridge = time.time()
        ridge_oof_tr, ridge_va, ridge_oof_valid = _compute_ridge_predictions_for_fold(
            X_tr, Y_tr, M_tr, X_va,
        )
        print(f"ridge_precompute seconds={time.time() - t_ridge:.2f} "
              f"honest_oof_rows={int(ridge_oof_valid.sum())}/{X_tr.shape[0]}", flush=True)
        # Replay the post-2026-06-19 worker convention: drop fallback rows.
        if ridge_oof_valid.sum() == 0:
            raise SystemExit("no honest Ridge OOF rows; cannot replay booster fit")
        X_tr = X_tr[ridge_oof_valid]
        Y_tr = Y_tr[ridge_oof_valid]
        M_tr = M_tr[ridge_oof_valid]
        base_tr = ridge_oof_tr[ridge_oof_valid]
        base_va = ridge_va
    else:
        base_tr = np.broadcast_to(mu_ff[None, :, :], Y_tr.shape)
        base_va = np.broadcast_to(mu_ff[None, :, :], Y_va.shape)

    if p["FEATURE_X_SCALE"]:
        feat_x_scale = _per_feature_x_scale(X_tr)
        X_tr_in = X_tr / feat_x_scale[None, None, :, None]
        X_va_in = X_va / feat_x_scale[None, None, :, None]
    else:
        X_tr_in = X_tr
        X_va_in = X_va

    Y_tr_cent = (Y_tr - base_tr) * M_tr
    if p["FEATURE_TARGET_SCALE"]:
        feat_sse = np.sum((Y_tr_cent ** 2) * M_tr, axis=(0, 1))
        feat_n = np.sum(M_tr, axis=(0, 1))
        feat_scale = np.sqrt(feat_sse / (feat_n + 1e-8))
        feat_scale = np.where(
            np.isfinite(feat_scale) & (feat_scale > 1e-8), feat_scale, 1.0,
        ).astype(Y_tr.dtype)
    else:
        feat_scale = np.ones(Y_tr.shape[2], dtype=Y_tr.dtype)
    Y_tr_scaled = Y_tr_cent / feat_scale[None, None, :]

    if p["USE_RMS_SCALING"]:
        y_obs = Y_tr_scaled[M_tr > 0]
        y_rms = float(np.sqrt(np.mean(y_obs ** 2))) if y_obs.size else 1.0
        Y_target = Y_tr_scaled / (y_rms + 1e-8)
    else:
        y_rms = 1.0
        Y_target = Y_tr_scaled

    print(f"cp_fit_start rank={p['RANK_REGRESS']} reg_w={p['REG_W']:.4g} "
          f"gamma={p['GAMMA']:.3f} feat_y={p['FEATURE_TARGET_SCALE']} "
          f"feat_x={p['FEATURE_X_SCALE']} rms={p['USE_RMS_SCALING']}", flush=True)
    t_fit = time.time()
    cp = CPRegressor(
        weight_rank=p["RANK_REGRESS"],
        reg_W=p["REG_W"],
        n_iter_max=N_ITER_MAX,
        random_state=SEED,
        verbose=0,
    )
    cp.fit(X_tr_in, Y_target)
    cp_residual = cp.predict(X_va_in) * y_rms * feat_scale[None, None, :]
    y_cp = base_va + p["GAMMA"] * cp_residual
    base_r2 = evaluate_model(Y_va, base_va, M_va)
    cp_r2 = evaluate_model(Y_va, y_cp, M_va)
    fit_seconds = time.time() - t_fit

    row = {
        "objective": args.objective,
        "mode": args.mode,
        "L": args.L,
        "rank_order": trial["rank_order"],
        "trial_number": trial["trial_number"],
        "journal_value": trial["journal_value"],
        "outer_fold": args.outer_fold,
        "base_r2": float(base_r2),
        "cp_r2": float(cp_r2),
        "delta_unclipped": float(cp_r2 - base_r2),
        "fit_seconds": fit_seconds,
        "n_iterations": int(getattr(cp, "n_iterations_", -1)),
        "host": os.uname().nodename,
        "RANK_REGRESS": p["RANK_REGRESS"],
        "REG_W": p["REG_W"],
        "GAMMA": p["GAMMA"],
        "USE_RMS_SCALING": p["USE_RMS_SCALING"],
        "FEATURE_TARGET_SCALE": p["FEATURE_TARGET_SCALE"],
        "FEATURE_X_SCALE": p["FEATURE_X_SCALE"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output, index=False)
    print(f"DONE delta={row['delta_unclipped']:.10f} fit_seconds={fit_seconds:.1f} "
          f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
