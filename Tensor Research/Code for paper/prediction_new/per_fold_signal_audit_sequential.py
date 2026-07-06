#!/usr/bin/env python3
"""Sequential exact top-v3 per-fold audit with immediate CSV writes."""

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

import joblib
import numpy as np
import pandas as pd
import tensorly as tl
from sklearn.model_selection import TimeSeriesSplit
from tensorly.regression.cp_regression import CPRegressor


def decode_param(value, distribution: str):
    try:
        dist = json.loads(distribution)
        if dist.get("name") == "CategoricalDistribution":
            return dist["attributes"]["choices"][int(value)]
    except Exception:
        pass
    return value


def load_top_trials(path: Path, top_k: int) -> list[dict]:
    params: dict[int, dict] = {}
    values: dict[int, float] = {}
    with path.open() as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("op_code") == 5:
                tid = ev["trial_id"]
                params.setdefault(tid, {})[ev["param_name"]] = decode_param(
                    ev["param_value_internal"],
                    ev.get("distribution", ""),
                )
            elif ev.get("op_code") == 6 and ev.get("state") == 1 and ev.get("values"):
                values[ev["trial_id"]] = float(ev["values"][0])
    out = []
    for rank_order, (tid, value) in enumerate(
        sorted(values.items(), key=lambda kv: kv[1], reverse=True)[:top_k],
        start=1,
    ):
        p = params[tid].copy()
        p["RANK_REGRESS"] = int(p["RANK_REGRESS"])
        p["REG_W"] = float(p["REG_W"])
        p["GAMMA"] = float(p["GAMMA"])
        p["USE_RMS_SCALING"] = bool(p["USE_RMS_SCALING"])
        p["FEATURE_TARGET_SCALE"] = bool(p["FEATURE_TARGET_SCALE"])
        p["FEATURE_X_SCALE"] = bool(p["FEATURE_X_SCALE"])
        out.append({"rank_order": rank_order, "trial_number": int(tid), "journal_value": value, "params": p})
    return out


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/student/mcnama53/Projects/Tensor Research"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
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

    studies = [
        ("ridge_delta_v3", "LEVELS", 2, pred_dir / "optuna_journal/study_levels_L2_ridge_delta_v3.log"),
        ("ridge_delta_v3", "LEVELS", 4, pred_dir / "optuna_journal/study_levels_L4_ridge_delta_v3.log"),
        ("residual_delta_v3", "LEVELS", 2, pred_dir / "optuna_journal/study_levels_L2_residual_delta_v3.log"),
        ("residual_delta_v3", "LEVELS", 4, pred_dir / "optuna_journal/study_levels_L4_residual_delta_v3.log"),
    ]
    completed_keys: set[tuple[str, str, int, int, int, int]] = set()
    if args.resume and args.output.exists():
        existing = pd.read_csv(args.output)
        for row in existing.itertuples(index=False):
            completed_keys.add(
                (
                    str(row.objective),
                    str(row.mode),
                    int(row.L),
                    int(row.rank_order),
                    int(row.trial_number),
                    int(row.outer_fold),
                )
            )
        print(f"resume_existing_rows={len(completed_keys)} output={args.output}", flush=True)
    elif args.output.exists():
        args.output.unlink()

    print(f"sequential_audit_start output={args.output}", flush=True)
    for objective, mode, L, journal in studies:
        print(f"study_start objective={objective} mode={mode} L={L}", flush=True)
        cache = joblib.load(cache_path(mode, L))
        X_all, Y_all, M_all = cache["X"], cache["Y"], cache["Mask"]
        split_idx = int(0.8 * len(X_all))
        X_dev, Y_dev, M_dev = X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx]
        fold_packs = []
        for outer_fold, (tr_idx, va_idx) in enumerate(TimeSeriesSplit(n_splits=3).split(X_dev), start=1):
            X_tr, Y_tr, M_tr = X_dev[tr_idx], Y_dev[tr_idx], M_dev[tr_idx]
            X_va, Y_va, M_va = X_dev[va_idx], Y_dev[va_idx], M_dev[va_idx]
            pack = {
                "outer_fold": outer_fold,
                "X_tr": X_tr,
                "Y_tr": Y_tr,
                "M_tr": M_tr,
                "X_va": X_va,
                "Y_va": Y_va,
                "M_va": M_va,
                "mu_ff": firm_feature_means(Y_tr, M_tr),
            }
            if objective == "ridge_delta_v3":
                started = time.time()
                ridge_oof_tr, ridge_va = _compute_ridge_predictions_for_fold(X_tr, Y_tr, M_tr, X_va)
                pack["ridge_oof_tr"] = ridge_oof_tr
                pack["ridge_va"] = ridge_va
                print(
                    f"ridge_precompute objective={objective} mode={mode} L={L} "
                    f"outer_fold={outer_fold} seconds={time.time() - started:.3f}",
                    flush=True,
                )
            fold_packs.append(pack)

        for trial in load_top_trials(journal, args.top_k):
            p = trial["params"]
            for pack in fold_packs:
                key = (
                    objective,
                    mode,
                    L,
                    trial["rank_order"],
                    trial["trial_number"],
                    pack["outer_fold"],
                )
                if key in completed_keys:
                    print(
                        f"fold_skip_existing objective={objective} mode={mode} L={L} "
                        f"rank_order={trial['rank_order']} trial={trial['trial_number']} "
                        f"outer_fold={pack['outer_fold']}",
                        flush=True,
                    )
                    continue
                print(
                    f"fold_start objective={objective} mode={mode} L={L} rank_order={trial['rank_order']} "
                    f"trial={trial['trial_number']} outer_fold={pack['outer_fold']}",
                    flush=True,
                )
                X_tr, Y_tr, M_tr = pack["X_tr"], pack["Y_tr"], pack["M_tr"]
                X_va, Y_va, M_va = pack["X_va"], pack["Y_va"], pack["M_va"]
                if p["FEATURE_X_SCALE"]:
                    feat_x_scale = _per_feature_x_scale(X_tr)
                    X_tr_in = X_tr / feat_x_scale[None, None, :, None]
                    X_va_in = X_va / feat_x_scale[None, None, :, None]
                else:
                    X_tr_in = X_tr
                    X_va_in = X_va

                if objective == "ridge_delta_v3":
                    base_tr = pack["ridge_oof_tr"]
                    base_va = pack["ridge_va"]
                else:
                    base_tr = np.broadcast_to(pack["mu_ff"][None, :, :], Y_tr.shape)
                    base_va = np.broadcast_to(pack["mu_ff"][None, :, :], Y_va.shape)

                Y_tr_cent = (Y_tr - base_tr) * M_tr
                if p["FEATURE_TARGET_SCALE"]:
                    feat_sse = np.sum((Y_tr_cent**2) * M_tr, axis=(0, 1))
                    feat_n = np.sum(M_tr, axis=(0, 1))
                    feat_scale = np.sqrt(feat_sse / (feat_n + 1e-8))
                    feat_scale = np.where(np.isfinite(feat_scale) & (feat_scale > 1e-8), feat_scale, 1.0).astype(Y_tr.dtype)
                else:
                    feat_scale = np.ones(Y_tr.shape[2], dtype=Y_tr.dtype)

                Y_tr_scaled = Y_tr_cent / feat_scale[None, None, :]
                if p["USE_RMS_SCALING"]:
                    y_obs = Y_tr_scaled[M_tr > 0]
                    y_rms = float(np.sqrt(np.mean(y_obs**2))) if y_obs.size else 1.0
                    Y_target = Y_tr_scaled / (y_rms + 1e-8)
                else:
                    y_rms = 1.0
                    Y_target = Y_tr_scaled

                started = time.time()
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
                row = {
                    "objective": objective,
                    "mode": mode,
                    "L": L,
                    "rank_order": trial["rank_order"],
                    "trial_number": trial["trial_number"],
                    "journal_value": trial["journal_value"],
                    "outer_fold": pack["outer_fold"],
                    "base_r2": float(base_r2),
                    "cp_r2": float(cp_r2),
                    "delta_unclipped": float(cp_r2 - base_r2),
                    "fit_seconds": time.time() - started,
                    "n_iterations": int(getattr(cp, "n_iterations_", -1)),
                    "RANK_REGRESS": p["RANK_REGRESS"],
                    "REG_W": p["REG_W"],
                    "GAMMA": p["GAMMA"],
                    "USE_RMS_SCALING": p["USE_RMS_SCALING"],
                    "FEATURE_TARGET_SCALE": p["FEATURE_TARGET_SCALE"],
                    "FEATURE_X_SCALE": p["FEATURE_X_SCALE"],
                }
                append_row(args.output, row)
                print(
                    f"fold_done objective={objective} mode={mode} L={L} rank_order={trial['rank_order']} "
                    f"trial={trial['trial_number']} outer_fold={pack['outer_fold']} "
                    f"delta={row['delta_unclipped']:.10f} seconds={row['fit_seconds']:.3f} "
                    f"iters={row['n_iterations']}",
                    flush=True,
                )
    print("sequential_audit_done", flush=True)


if __name__ == "__main__":
    main()
