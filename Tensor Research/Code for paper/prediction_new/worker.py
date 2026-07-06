"""prediction_new/worker.py

Single Optuna worker for distributed CP regression hyperparameter search.

Connects to a JournalStorage on the shared NFS journal file. Optuna handles
trial assignment/locking; many workers can run concurrently against the same
study from different hosts.

Usage:
    python worker.py --mode SURPRISE --L 2 --n-trials 200
    python worker.py --mode LEVELS   --L 4 --n-trials 200 --time-budget-s 7200
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import optuna  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from optuna.storages import JournalStorage  # noqa: E402
from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock  # noqa: E402

from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
import tensorly as tl  # noqa: E402
from tensorly.regression.cp_regression import CPRegressor  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PARENT_DIR))

from prediction_config import (  # noqa: E402
    LEGACY_WARM_START, MODES, N_ITER_MAX, RANK_RANGE, REG_W_RANGE,
    RDV2_GAMMA_RANGE, RDV2_RANK_RANGE, RDV2_REG_W_RANGE,
    SEED, cache_path, journal_path, study_name,
)

# residual_delta_v3 == residual_delta_v2 + per-feature X scaling toggle.
# ridge_delta_v3   == residual_delta_v3 search space, with the booster
# architecture: CP fits on Ridge OOF residuals and the prediction is
# Ridge + gamma * CP. Score = R2(Ridge + gamma*CP) - R2(Ridge).
OBJECTIVES = (
    "pooled_r2",
    "residual_delta",
    "residual_delta_v2",
    "residual_delta_v3",
    "ridge_delta_v3",
)
# Objectives that subtract a baseline R^2 from R^2(prediction_with_CP).
RESIDUAL_OBJECTIVES = (
    "residual_delta",
    "residual_delta_v2",
    "residual_delta_v3",
    "ridge_delta_v3",
)
V3_OBJECTIVES = ("residual_delta_v3", "ridge_delta_v3")
BOOSTER_OBJECTIVES = ("ridge_delta_v3",)


def get_min_valid_entries(mask: np.ndarray, min_frac: float = 0.05,
                          floor: int = 100, cap: int = 5000) -> int:
    return max(floor, min(cap, int(min_frac * mask.size)))


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    y_t, y_p, m = y_true.flatten(), y_pred.flatten(), mask.flatten()
    valid = m > 0
    if valid.sum() < get_min_valid_entries(mask):
        return None
    clean_t, clean_p = y_t[valid], y_p[valid]
    sst = float(np.sum((clean_t - clean_t.mean()) ** 2))
    if sst <= 1e-8:
        return None
    return 1.0 - float(np.sum((clean_t - clean_p) ** 2)) / sst


def firm_feature_means(Y_tr: np.ndarray, M_tr: np.ndarray) -> np.ndarray:
    denom = M_tr.sum(axis=0)
    mu_ff = (Y_tr * M_tr).sum(axis=0) / (denom + 1e-8)
    denom_feat = M_tr.sum(axis=(0, 1))
    mu_feat = (Y_tr * M_tr).sum(axis=(0, 1)) / (denom_feat + 1e-8)
    missing = denom <= 0
    if missing.any():
        idx_firm, idx_feat = np.where(missing)
        mu_ff[idx_firm, idx_feat] = mu_feat[idx_feat]
    return mu_ff


def objective_journal_path(mode: str, L: int, objective_name: str) -> Path:
    if objective_name == "pooled_r2":
        return journal_path(mode, L)
    base = journal_path(mode, L)
    return base.with_name(f"{base.stem}_{objective_name}{base.suffix}")


def objective_study_name(mode: str, L: int, objective_name: str) -> str:
    if objective_name == "pooled_r2":
        return study_name(mode, L)
    return f"{study_name(mode, L)}_{objective_name}"


def worker_seed(label: str, objective_name: str) -> int:
    raw = f"{SEED}:{label}:{objective_name}:{time.time_ns()}".encode("utf-8")
    return int(hashlib.blake2s(raw, digest_size=4).hexdigest(), 16)


def _per_feature_x_scale(X: np.ndarray) -> np.ndarray:
    """Per-feature RMS scale of X (shape n_windows × n_firms × n_features × L).

    Used to scale X column-wise so high-magnitude features (e.g. log-modulus
    market cap) don't dominate CP's pooled loss. Mirrors the per-feature Y
    scaling already used in residual_delta_v2.
    """
    n_w, n_f, n_feat, n_l = X.shape
    feat_sse = np.sum(X ** 2, axis=(0, 1, 3))
    feat_n = float(n_w * n_f * n_l)
    feat_scale = np.sqrt(feat_sse / (feat_n + 1e-8))
    feat_scale = np.where(
        np.isfinite(feat_scale) & (feat_scale > 1e-8),
        feat_scale,
        1.0,
    ).astype(X.dtype)
    return feat_scale


def _compute_ridge_predictions_for_fold(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    M_tr: np.ndarray,
    X_va: np.ndarray,
    n_inner_splits: int = 3,
    inner_alpha_search_splits: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold Ridge predictions on training, and a Ridge prediction on val.

    Training-side predictions are produced by an inner TimeSeriesSplit so the
    booster CP target Y - Ridge_pred is leakage-free at training time.
    Validation-side prediction is from a single Ridge fit on the full outer
    training set. Training rows whose inner split is too small to honestly
    fit a Ridge baseline (inner_tr_idx.size < MIN_INNER_TR_SIZE) are returned
    as un-initialized. Booster CP training must drop those rows; the evaluator
    at test time must use the same skip rule for replay consistency.
    """
    # Local import keeps the heavy Ridge baseline module out of the import path
    # for non-booster runs.
    from CP_struct_test_new import (  # noqa: E402
        ridge_structured_cp_matched_zero_filled_ts_cv,
    )

    n_tr = X_tr.shape[0]
    ridge_oof_tr = np.zeros_like(Y_tr)
    initialized = np.zeros(n_tr, dtype=bool)

    inner_tscv = TimeSeriesSplit(n_splits=n_inner_splits)
    for inner_tr_idx, inner_va_idx in inner_tscv.split(X_tr):
        # MIN_INNER_TR_SIZE = 5 is load-bearing: it defines what counts as
        # an honest Ridge OOF training row. The evaluate_top_trials_test
        # holdout replay MUST use the same threshold. Pin in both places.
        if inner_tr_idx.size < 5 or inner_va_idx.size == 0:
            continue
        preds = ridge_structured_cp_matched_zero_filled_ts_cv(
            X_tr[inner_tr_idx],
            Y_tr[inner_tr_idx],
            M_tr[inner_tr_idx],
            X_tr[inner_va_idx],
            inner_splits=inner_alpha_search_splits,
        )
        ridge_oof_tr[inner_va_idx] = preds.astype(Y_tr.dtype, copy=False)
        initialized[inner_va_idx] = True

    # Previously: un-initialized rows fell back to firm-feature means. That
    # produced an artifact where booster CP trained on FE residuals in the
    # first outer fold and contributed exact zero on validation when added
    # on top of Ridge. Patch 2026-06-19: leave un-initialized rows zero and
    # let make_objective drop them from CP training.

    ridge_va = ridge_structured_cp_matched_zero_filled_ts_cv(
        X_tr, Y_tr, M_tr, X_va,
        inner_splits=inner_alpha_search_splits,
    ).astype(Y_tr.dtype, copy=False)

    return ridge_oof_tr, ridge_va, initialized


def make_objective(mode: str, L: int, worker_label: str, objective_name: str):
    cache = joblib.load(cache_path(mode, L))
    X_all, Y_all, M_all = cache["X"], cache["Y"], cache["Mask"]
    split_idx = int(0.8 * len(X_all))
    X_dev, Y_dev, M_dev = X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx]
    print(f"[{worker_label}] cache loaded: dev_windows={len(X_dev)} test_windows={len(X_all)-len(X_dev)} "
          f"firms={X_dev.shape[1]} features={X_dev.shape[2]} L={L} objective={objective_name}")

    is_v2 = objective_name == "residual_delta_v2"
    is_v3_like = objective_name in V3_OBJECTIVES
    is_booster = objective_name in BOOSTER_OBJECTIVES

    # Precompute per-fold quantities once per worker startup. For v3 we cache
    # the FE baseline + (optionally) Ridge OOF/val predictions so the Optuna
    # trial loop doesn't pay either cost per trial.
    tscv = TimeSeriesSplit(n_splits=3)
    fold_packs: list[dict] = []
    t_setup = time.time()
    for tr_idx, va_idx in tscv.split(X_dev):
        X_tr, Y_tr, M_tr = X_dev[tr_idx], Y_dev[tr_idx], M_dev[tr_idx]
        X_va, Y_va, M_va = X_dev[va_idx], Y_dev[va_idx], M_dev[va_idx]
        mu_ff = firm_feature_means(Y_tr, M_tr)
        pack = {
            "X_tr": X_tr, "Y_tr": Y_tr, "M_tr": M_tr,
            "X_va": X_va, "Y_va": Y_va, "M_va": M_va,
            "mu_ff": mu_ff,
        }
        if is_booster:
            ridge_oof_tr, ridge_va, ridge_oof_valid = _compute_ridge_predictions_for_fold(
                X_tr, Y_tr, M_tr, X_va,
            )
            pack["ridge_oof_tr"] = ridge_oof_tr
            pack["ridge_va"] = ridge_va
            pack["ridge_oof_valid"] = ridge_oof_valid
        fold_packs.append(pack)
    if is_booster:
        total_rows = sum(p["X_tr"].shape[0] for p in fold_packs)
        kept_rows = sum(int(p["ridge_oof_valid"].sum()) for p in fold_packs)
        print(f"[{worker_label}] Ridge OOF precompute done in {time.time()-t_setup:.1f}s "
              f"across {len(fold_packs)} folds; honest_oof_rows={kept_rows}/{total_rows}")

    def objective(trial: optuna.Trial) -> float:
        if is_v3_like:
            rank_reg = trial.suggest_int("RANK_REGRESS", *RDV2_RANK_RANGE)
            reg_w = trial.suggest_float("REG_W", *RDV2_REG_W_RANGE, log=True)
            gamma = trial.suggest_float("GAMMA", *RDV2_GAMMA_RANGE)
            use_rms = trial.suggest_categorical("USE_RMS_SCALING", [True, False])
            feature_scale_flag = trial.suggest_categorical(
                "FEATURE_TARGET_SCALE", [True, False]
            )
            feature_x_scale_flag = trial.suggest_categorical(
                "FEATURE_X_SCALE", [True, False]
            )
        elif is_v2:
            rank_reg = trial.suggest_int("RANK_REGRESS", *RDV2_RANK_RANGE)
            reg_w = trial.suggest_float("REG_W", *RDV2_REG_W_RANGE, log=True)
            gamma = trial.suggest_float("GAMMA", *RDV2_GAMMA_RANGE)
            use_rms = trial.suggest_categorical("USE_RMS_SCALING", [True, False])
            feature_scale_flag = trial.suggest_categorical(
                "FEATURE_TARGET_SCALE", [True, False]
            )
            feature_x_scale_flag = False
        else:
            rank_reg = trial.suggest_int("RANK_REGRESS", *RANK_RANGE)
            reg_w = trial.suggest_float("REG_W", *REG_W_RANGE, log=True)
            use_rms = trial.suggest_categorical("USE_RMS_SCALING", [True, False])
            gamma = 1.0
            feature_scale_flag = False
            feature_x_scale_flag = False

        scores: list[float] = []
        for pack in fold_packs:
            X_tr_full, Y_tr_full, M_tr_full = pack["X_tr"], pack["Y_tr"], pack["M_tr"]
            X_va, Y_va, M_va = pack["X_va"], pack["Y_va"], pack["M_va"]
            mu_ff = pack["mu_ff"]

            # Booster: drop rows without an honest Ridge OOF target (patch
            # 2026-06-19). For non-booster, train on all rows.
            if is_booster:
                cp_train_rows = pack["ridge_oof_valid"]
                if cp_train_rows.sum() == 0:
                    # No honest Ridge OOF rows in this fold. Fail the whole
                    # trial so Optuna prunes it. This is conservative — it
                    # avoids averaging over a shrinking number of folds
                    # (selection bias toward configs that happen to survive
                    # on the smaller folds).
                    return float("nan")
            else:
                cp_train_rows = np.ones(X_tr_full.shape[0], dtype=bool)

            X_tr = X_tr_full[cp_train_rows]
            Y_tr = Y_tr_full[cp_train_rows]
            M_tr = M_tr_full[cp_train_rows]

            # Per-feature X scaling (v3 toggle). Scale is fit on the rows
            # we actually train on, then applied symmetrically to val.
            if feature_x_scale_flag:
                feat_x_scale = _per_feature_x_scale(X_tr)
                X_tr_in = X_tr / feat_x_scale[None, None, :, None]
                X_va_in = X_va / feat_x_scale[None, None, :, None]
            else:
                X_tr_in = X_tr
                X_va_in = X_va

            # Build CP target. Booster: residual of Ridge OOF (honest rows
            # only). Otherwise: residual of FE.
            if is_booster:
                base_tr = pack["ridge_oof_tr"][cp_train_rows]
                base_va = pack["ridge_va"]
            else:
                base_tr = np.broadcast_to(mu_ff[None, :, :], Y_tr.shape)
                base_va = np.broadcast_to(mu_ff[None, :, :], Y_va.shape)

            Y_tr_cent = (Y_tr - base_tr) * M_tr

            if feature_scale_flag:
                feat_sse = np.sum((Y_tr_cent ** 2) * M_tr, axis=(0, 1))
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

            if use_rms:
                y_obs = Y_tr_scaled[M_tr > 0]
                if y_obs.size <= get_min_valid_entries(M_tr):
                    return float("nan")
                y_rms = float(np.sqrt(np.mean(y_obs ** 2)))
                Y_target = Y_tr_scaled / (y_rms + 1e-8)
            else:
                y_rms = 1.0
                Y_target = Y_tr_scaled

            try:
                cp = CPRegressor(weight_rank=rank_reg, reg_W=reg_w,
                                 n_iter_max=N_ITER_MAX, random_state=SEED)
                cp.fit(X_tr_in, Y_target)
                preds = cp.predict(X_va_in)
                cp_residual = preds * y_rms * feat_scale[None, None, :]
                y_cp = gamma * cp_residual + base_va
                r2 = evaluate_model(Y_va, y_cp, M_va)
                if r2 is None:
                    return float("nan")

                if objective_name in RESIDUAL_OBJECTIVES:
                    base_r2 = evaluate_model(Y_va, base_va, M_va)
                    if base_r2 is None:
                        return float("nan")
                    score = float(r2 - base_r2)
                else:
                    score = float(r2)
                scores.append(max(score, -1.0))
            except Exception as exc:
                print(f"[{worker_label}] trial inner error: {type(exc).__name__}: {exc}")
                return float("nan")

        return float(np.mean(scores)) if scores else float("nan")

    return objective


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=list(MODES))
    p.add_argument("--L", required=True, type=int, choices=[2, 4])
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--time-budget-s", type=float, default=0.0,
                   help="Stop after this many seconds (0 = no limit; --n-trials still applies)")
    p.add_argument("--worker-name", default=os.environ.get("HOSTNAME", "worker"))
    p.add_argument("--objective", choices=OBJECTIVES, default="pooled_r2",
                   help="pooled_r2 is the legacy score; residual_delta optimizes improvement over FE-only")
    p.add_argument("--enqueue-warm-start", action="store_true",
                   help="If first worker, push the legacy CP_PARAMS as a known-good initial trial")
    args = p.parse_args()

    label = f"{args.worker_name}/{args.mode}_L{args.L}/{args.objective}"
    tl.set_backend("numpy")
    np.random.seed(SEED)

    jp = objective_journal_path(args.mode, args.L, args.objective)
    jp.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(jp), lock_obj=JournalFileSymlinkLock(str(jp))))

    seed = worker_seed(label, args.objective)
    sampler = TPESampler(
        seed=seed,
        multivariate=True,
        constant_liar=True,
        n_startup_trials=40,
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=objective_study_name(args.mode, args.L, args.objective),
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )

    if args.enqueue_warm_start and len(study.trials) == 0:
        legacy = LEGACY_WARM_START.get((args.mode, args.L))
        if legacy is not None:
            print(f"[{label}] enqueuing legacy warm-start: {legacy}")
            study.enqueue_trial(legacy)

    objective = make_objective(args.mode, args.L, label, args.objective)

    print(f"[{label}] worker starting (study has {len(study.trials)} prior trials, "
          f"target {args.n_trials} new trials, budget {args.time_budget_s}s, sampler_seed={seed})")
    start = time.time()
    n_done = 0
    while n_done < args.n_trials:
        if args.time_budget_s > 0 and (time.time() - start) > args.time_budget_s:
            print(f"[{label}] time budget reached at trial {n_done}; stopping")
            break
        try:
            study.optimize(objective, n_trials=1, n_jobs=1, gc_after_trial=True)
            n_done += 1
            if n_done % 5 == 0 or n_done == 1:
                try:
                    best = study.best_value
                    bp = study.best_params
                    score_name = (
                        "best_delta"
                        if args.objective in RESIDUAL_OBJECTIVES
                        else "best_cv_r2"
                    )
                    extra = ""
                    if args.objective in ("residual_delta_v2",) + V3_OBJECTIVES:
                        extra = (
                            f"  gamma={bp.get('GAMMA', float('nan')):.3f}"
                            f"  feat_y={bp.get('FEATURE_TARGET_SCALE')}"
                        )
                    if args.objective in V3_OBJECTIVES:
                        extra += f"  feat_x={bp.get('FEATURE_X_SCALE')}"
                    print(
                        f"[{label}] {n_done}/{args.n_trials} trials  {score_name}={best:.5f}  "
                        f"rank={bp.get('RANK_REGRESS')}  reg_w={bp.get('REG_W'):.4g}  "
                        f"rms={bp.get('USE_RMS_SCALING')}{extra}"
                    )
                except (ValueError, KeyError):
                    print(f"[{label}] {n_done}/{args.n_trials} trials  (no completed yet)")
        except KeyboardInterrupt:
            print(f"[{label}] interrupted by user")
            break
        except Exception as exc:
            print(f"[{label}] trial failed: {type(exc).__name__}: {exc}")

    elapsed = time.time() - start
    print(f"[{label}] DONE: {n_done} trials in {elapsed:.0f}s "
          f"({elapsed/max(n_done,1):.1f}s/trial)")


if __name__ == "__main__":
    main()
