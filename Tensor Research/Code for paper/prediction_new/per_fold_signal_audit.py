#!/usr/bin/env python3
"""Rerun top v3 CP configurations and log unclipped per-fold deltas.

This is an audit script, not an Optuna worker. It reads existing v3 journal
JSONL files, selects the top-k completed trials, reruns those exact parameter
settings once, and writes per-fold deltas to CSV as rows finish.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

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
                trial_id = ev["trial_id"]
                params.setdefault(trial_id, {})[ev["param_name"]] = decode_param(
                    ev["param_value_internal"],
                    ev.get("distribution", ""),
                )
            elif op == 6 and ev.get("state") == 1 and ev.get("values"):
                values[ev["trial_id"]] = float(ev["values"][0])

    ranked = sorted(values.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    trials = []
    for rank_order, (trial_id, value) in enumerate(ranked, start=1):
        p = params[trial_id].copy()
        p["RANK_REGRESS"] = int(p["RANK_REGRESS"])
        p["REG_W"] = float(p["REG_W"])
        p["GAMMA"] = float(p["GAMMA"])
        p["USE_RMS_SCALING"] = bool(p["USE_RMS_SCALING"])
        p["FEATURE_TARGET_SCALE"] = bool(p["FEATURE_TARGET_SCALE"])
        p["FEATURE_X_SCALE"] = bool(p["FEATURE_X_SCALE"])
        trials.append(
            {
                "rank_order": rank_order,
                "trial_number": int(trial_id),
                "journal_value": float(value),
                "params": p,
            }
        )
    return trials


def prepare_imports(project_root: Path):
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

    fold_packs = []
    for outer_fold, (tr_idx, va_idx) in enumerate(TimeSeriesSplit(n_splits=3).split(X_dev), start=1):
        X_tr, Y_tr, M_tr = X_dev[tr_idx], Y_dev[tr_idx], M_dev[tr_idx]
        X_va, Y_va, M_va = X_dev[va_idx], Y_dev[va_idx], M_dev[va_idx]
        mu_ff = api["firm_feature_means"](Y_tr, M_tr)
        pack = {
            "outer_fold": outer_fold,
            "X_tr": X_tr,
            "Y_tr": Y_tr,
            "M_tr": M_tr,
            "X_va": X_va,
            "Y_va": Y_va,
            "M_va": M_va,
            "mu_ff": mu_ff,
        }
        if is_booster:
            started = time.time()
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
                f"seconds={time.time() - started:.3f}",
                flush=True,
            )
        fold_packs.append(pack)
    return fold_packs


def eval_one_trial(objective: str, mode: str, L: int, trial: dict, fold_packs: list[dict], api: dict) -> list[dict]:
    params = trial["params"]
    is_booster = objective == "ridge_delta_v3"
    rows = []
    for pack in fold_packs:
        X_tr, Y_tr, M_tr = pack["X_tr"], pack["Y_tr"], pack["M_tr"]
        X_va, Y_va, M_va = pack["X_va"], pack["Y_va"], pack["M_va"]
        mu_ff = pack["mu_ff"]

        if params["FEATURE_X_SCALE"]:
            feat_x_scale = api["_per_feature_x_scale"](X_tr)
            X_tr_in = X_tr / feat_x_scale[None, None, :, None]
            X_va_in = X_va / feat_x_scale[None, None, :, None]
        else:
            X_tr_in = X_tr
            X_va_in = X_va

        if is_booster:
            base_tr = pack["ridge_oof_tr"]
            base_va = pack["ridge_va"]
        else:
            base_tr = np.broadcast_to(mu_ff[None, :, :], Y_tr.shape)
            base_va = np.broadcast_to(mu_ff[None, :, :], Y_va.shape)

        Y_tr_cent = (Y_tr - base_tr) * M_tr
        if params["FEATURE_TARGET_SCALE"]:
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
        if params["USE_RMS_SCALING"]:
            y_obs = Y_tr_scaled[M_tr > 0]
            y_rms = float(np.sqrt(np.mean(y_obs**2))) if y_obs.size else 1.0
            Y_target = Y_tr_scaled / (y_rms + 1e-8)
        else:
            y_rms = 1.0
            Y_target = Y_tr_scaled

        started = time.time()
        cp = CPRegressor(
            weight_rank=params["RANK_REGRESS"],
            reg_W=params["REG_W"],
            n_iter_max=api["N_ITER_MAX"],
            random_state=api["SEED"],
        )
        cp.fit(X_tr_in, Y_target)
        cp_residual = cp.predict(X_va_in) * y_rms * feat_scale[None, None, :]
        y_cp = base_va + params["GAMMA"] * cp_residual

        base_r2 = api["evaluate_model"](Y_va, base_va, M_va)
        cp_r2 = api["evaluate_model"](Y_va, y_cp, M_va)
        rows.append(
            {
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
                "RANK_REGRESS": params["RANK_REGRESS"],
                "REG_W": params["REG_W"],
                "GAMMA": params["GAMMA"],
                "USE_RMS_SCALING": params["USE_RMS_SCALING"],
                "FEATURE_TARGET_SCALE": params["FEATURE_TARGET_SCALE"],
                "FEATURE_X_SCALE": params["FEATURE_X_SCALE"],
            }
        )
    return rows


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    pd.DataFrame(rows).to_csv(path, mode="a", header=header, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/student/mcnama53/Projects/Tensor Research"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    api = prepare_imports(args.project_root)
    tl.set_backend("numpy")
    np.random.seed(api["SEED"])

    studies = [
        ("residual_delta_v3", "LEVELS", 2, api["pred_dir"] / "optuna_journal/study_levels_L2_residual_delta_v3.log"),
        ("residual_delta_v3", "LEVELS", 4, api["pred_dir"] / "optuna_journal/study_levels_L4_residual_delta_v3.log"),
        ("ridge_delta_v3", "LEVELS", 2, api["pred_dir"] / "optuna_journal/study_levels_L2_ridge_delta_v3.log"),
        ("ridge_delta_v3", "LEVELS", 4, api["pred_dir"] / "optuna_journal/study_levels_L4_ridge_delta_v3.log"),
    ]

    if args.output.exists():
        args.output.unlink()
    started_all = time.time()
    print(f"per_fold_signal_audit start output={args.output}", flush=True)
    for objective, mode, L, journal in studies:
        print(f"study_start objective={objective} mode={mode} L={L}", flush=True)
        trials = load_top_trials(journal, args.top_k)
        fold_packs = build_fold_packs(mode, L, objective == "ridge_delta_v3", api)
        for trial in trials:
            rows = eval_one_trial(objective, mode, L, trial, fold_packs, api)
            append_rows(args.output, rows)
            deltas = np.asarray([r["delta_unclipped"] for r in rows], dtype=float)
            print(
                f"trial_done objective={objective} mode={mode} L={L} rank_order={trial['rank_order']} "
                f"trial_number={trial['trial_number']} journal={trial['journal_value']:.10f} "
                f"mean_unclipped={deltas.mean():.10f} min={deltas.min():.10f} "
                f"max={deltas.max():.10f} positive_folds={(deltas > 0).sum()}/3",
                flush=True,
            )
    print(f"per_fold_signal_audit done seconds={time.time() - started_all:.3f}", flush=True)


if __name__ == "__main__":
    main()
