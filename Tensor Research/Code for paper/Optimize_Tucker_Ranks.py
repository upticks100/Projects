from __future__ import annotations

import argparse
import time

import numpy as np
import optuna
import pandas as pd
import tensorly as tl
from tensorly.decomposition import partial_tucker

from Build_PrePrediction_Exhibits import (
    build_fundamentals_panel,
    build_fundamentals_tensor,
)
from pre_prediction_config import CACHE_DIR, FEATURE_SPECS, SEED


STUDY_DB = CACHE_DIR / "tucker_rank_optuna.sqlite3"
TRIALS_FILE = CACHE_DIR / "tucker_rank_optuna_trials.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=250)
    parser.add_argument("--study-name", default="tucker-rank-error")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--min-r1", type=int, default=40)
    parser.add_argument("--max-r1", type=int, default=496)
    parser.add_argument("--min-r3", type=int, default=5)
    parser.add_argument("--max-r3", type=int, default=140)
    parser.add_argument("--step-r1", type=int, default=5)
    parser.add_argument("--step-r3", type=int, default=5)
    parser.add_argument("--target-error", type=float, default=0.10)
    parser.add_argument("--startup-trials", type=int, default=100)
    return parser.parse_args()


def observed_relative_error(
    tensor: np.ndarray,
    mask: np.ndarray,
    r1: int,
    r3: int,
    max_iter: int,
    tol: float,
) -> float:
    observed = tensor[mask]
    rms = float(np.sqrt(np.mean(observed**2))) if observed.size else 1.0
    filled = np.nan_to_num(tensor / max(rms, 1e-8), nan=0.0)
    mask_i = mask.astype(np.int8)

    (core, factors), _ = partial_tucker(
        filled,
        rank=[r1, r3],
        modes=[0, 2],
        mask=mask_i,
        n_iter_max=max_iter,
        tol=tol,
        init="svd",
        random_state=SEED,
        verbose=False,
        svd_mask_repeats=1,
    )
    full_factors = [factors[0], np.eye(tensor.shape[1], dtype=np.float64), factors[1]]
    recon = tl.tucker_to_tensor((core, full_factors))
    return float(
        np.linalg.norm((recon - filled) * mask)
        / (np.linalg.norm(filled * mask) + 1e-8)
    )


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tl.set_backend("numpy")

    panel, gics = build_fundamentals_panel(engine=None, force_refresh=False)
    tensor, mask, firms, quarters, _ = build_fundamentals_tensor(panel, gics)
    max_r1 = min(args.max_r1, len(firms))
    max_r3 = min(args.max_r3, len(quarters))

    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        multivariate=True,
        n_startup_trials=args.startup_trials,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        r1 = trial.suggest_int("r1", args.min_r1, max_r1, step=args.step_r1)
        r3 = trial.suggest_int("r3", args.min_r3, max_r3, step=args.step_r3)
        start = time.perf_counter()
        error = observed_relative_error(tensor, mask, r1, r3, args.max_iter, args.tol)
        elapsed = time.perf_counter() - start
        trial.set_user_attr("r2", len(FEATURE_SPECS))
        trial.set_user_attr("elapsed_seconds", elapsed)
        trial.set_user_attr("core_entries", int(r1 * len(FEATURE_SPECS) * r3))
        trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        trials.to_csv(TRIALS_FILE, index=False)
        try:
            best_so_far = min(error, float(study.best_value))
        except ValueError:
            best_so_far = error
        print(
            f"trial={trial.number} rank=[{r1}, {len(FEATURE_SPECS)}, {r3}] "
            f"error={error:.6f} best={best_so_far:.6f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        if error < args.target_error:
            print(f"Trial reached target error below {args.target_error:.2%}.", flush=True)
        return error

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state")).to_csv(
        TRIALS_FILE,
        index=False,
    )
    print("Best trial:", study.best_trial.number, flush=True)
    print("Best params:", study.best_params, flush=True)
    print("Best observed relative error:", study.best_value, flush=True)


if __name__ == "__main__":
    main()
