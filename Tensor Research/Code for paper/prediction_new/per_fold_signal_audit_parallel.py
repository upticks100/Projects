#!/usr/bin/env python3
"""Parallel top-v3 per-fold signal audit.

Reads completed v3 Optuna journals, selects top-k trials, reruns each requested
trial/fold once, and appends each finished fold to a CSV. This does not modify
the journals or start any Optuna study.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import tensorly as tl
from joblib import Parallel, delayed
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


def load_top_trials(journal_path: Path, top_k: int) -> list[dict]:
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

    trials = []
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
        trials.append(
            {
                "rank_order": rank_order,
                "trial_number": int(tid),
                "journal_value": float(value),
                "params": p,
            }
        )
    return trials


def prepare_api(project_root: Path) -> dict:
    pred_dir = project_root / "Code for paper/prediction_new"
    sys.path.insert(0, str(pred_dir))
    sys.path.insert(0, str(pred_dir.parent))
    from prediction_config import N_ITER_MAX, SEED, cache_path  # noqa: PLC0415
    from worker import (  # noqa: PLC0415
        _compute_ridge_predictions_for_fold,
        _per_feature_x_scale,
        evaluate_model,
        firm_feature_means,
    )

    return {
        "pred_dir": pred_dir,
        "N_ITER_MAX": N_ITER_MAX,
        "SEED": SEED,
        "cache_path": cache_path,
        "_compute_ridge_predictions_for_fold": _compute_ridge_predictions_for_fold,
        "_per_feature_x_scale": _per_feature_x_scale,
        "evaluate_model": evaluate_model,
        "firm_feature_means": firm_feature_means,
    }


def build_fold_packs(mode: str, L: int, is_booster: bool, api: dict) -> list[dict]:
    cache = joblib.load(api["cache_path"](mode, L))
    X_all, Y_all, M_all = cache["X"], cache["Y"], cache["Mask"]
    split_idx = int(0.8 * len(X_all))
    X_dev, Y_dev, M_dev = X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx]

    packs = []
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
            "mu_ff": api["firm_feature_means"](Y_tr, M_tr),
        }
        if is_booster:
            start = time.time()
            ridge_oof_tr, ridge_va = api["_compute_ridge_predictions_for_fold"](
                X_tr,
                Y_tr,
                M_tr,
                X_va,
            )
            pack["ridge_oof_tr"] = ridge_oof_tr
            pack["ridge_va"] = ridge_va
            print(
                f"ridge_precompute mode={mode} L={L} outer_fold={outer_fold} "
                f"seconds={time.time() - start:.3f}",
                flush=True,
            )
        packs.append(pack)
    return packs


def evaluate_task(objective: str, mode: str, L: int, trial: dict, pack: dict, api: dict) -> dict:
    tl.set_backend("numpy")
    np.random.seed(api["SEED"])

    p = trial["params"]
    X_tr, Y_tr, M_tr = pack["X_tr"], pack["Y_tr"], pack["M_tr"]
    X_va, Y_va, M_va = pack["X_va"], pack["Y_va"], pack["M_va"]

    if p["FEATURE_X_SCALE"]:
        feat_x_scale = api["_per_feature_x_scale"](X_tr)
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
        feat_scale = np.where(
            np.isfinite(feat_scale) & (feat_scale > 1e-8),
            feat_scale,
            1.0,
        ).astype(Y_tr.dtype)
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

    start = time.time()
    cp = CPRegressor(
        weight_rank=p["RANK_REGRESS"],
        reg_W=p["REG_W"],
        n_iter_max=api["N_ITER_MAX"],
        random_state=api["SEED"],
    )
    cp.fit(X_tr_in, Y_target)
    cp_residual = cp.predict(X_va_in) * y_rms * feat_scale[None, None, :]
    y_cp = base_va + p["GAMMA"] * cp_residual

    base_r2 = api["evaluate_model"](Y_va, base_va, M_va)
    cp_r2 = api["evaluate_model"](Y_va, y_cp, M_va)
    return {
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
        "fit_seconds": time.time() - start,
        "RANK_REGRESS": p["RANK_REGRESS"],
        "REG_W": p["REG_W"],
        "GAMMA": p["GAMMA"],
        "USE_RMS_SCALING": p["USE_RMS_SCALING"],
        "FEATURE_TARGET_SCALE": p["FEATURE_TARGET_SCALE"],
        "FEATURE_X_SCALE": p["FEATURE_X_SCALE"],
    }


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/student/mcnama53/Projects/Tensor Research"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    api = prepare_api(args.project_root)
    tl.set_backend("numpy")
    np.random.seed(api["SEED"])

    # Booster studies first: their per-fold consistency is the paper-critical
    # question, and the standalone residual studies can take much longer.
    studies = [
        ("ridge_delta_v3", "LEVELS", 2, api["pred_dir"] / "optuna_journal/study_levels_L2_ridge_delta_v3.log"),
        ("ridge_delta_v3", "LEVELS", 4, api["pred_dir"] / "optuna_journal/study_levels_L4_ridge_delta_v3.log"),
        ("residual_delta_v3", "LEVELS", 2, api["pred_dir"] / "optuna_journal/study_levels_L2_residual_delta_v3.log"),
        ("residual_delta_v3", "LEVELS", 4, api["pred_dir"] / "optuna_journal/study_levels_L4_residual_delta_v3.log"),
    ]
    if args.output.exists():
        args.output.unlink()

    print(
        f"per_fold_signal_audit_parallel start output={args.output} n_jobs={args.n_jobs}",
        flush=True,
    )
    t_all = time.time()
    for objective, mode, L, journal in studies:
        print(f"study_start objective={objective} mode={mode} L={L}", flush=True)
        trials = load_top_trials(journal, args.top_k)
        packs = build_fold_packs(mode, L, objective == "ridge_delta_v3", api)
        tasks = [
            (objective, mode, L, trial, pack, api)
            for trial in trials
            for pack in packs
        ]
        parallel = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", verbose=10)
        rows = parallel(delayed(evaluate_task)(*task) for task in tasks)
        completed = []
        for row in rows:
            append_row(args.output, row)
            completed.append(row)
            print(
                f"fold_done objective={row['objective']} mode={row['mode']} L={row['L']} "
                f"rank_order={row['rank_order']} trial={row['trial_number']} fold={row['outer_fold']} "
                f"delta={row['delta_unclipped']:.10f} seconds={row['fit_seconds']:.1f}",
                flush=True,
            )
        df = pd.DataFrame(completed)
        summary = (
            df.groupby(["objective", "mode", "L", "rank_order", "trial_number"], as_index=False)
            .agg(
                journal_value=("journal_value", "first"),
                mean_unclipped=("delta_unclipped", "mean"),
                min_delta=("delta_unclipped", "min"),
                max_delta=("delta_unclipped", "max"),
                positive_folds=("delta_unclipped", lambda s: int((s > 0).sum())),
            )
        )
        print(summary.to_string(index=False), flush=True)
    print(f"per_fold_signal_audit_parallel done seconds={time.time() - t_all:.3f}", flush=True)


if __name__ == "__main__":
    main()
