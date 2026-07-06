from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker

from Build_PrePrediction_Exhibits import build_tensor
from pre_prediction_config import CACHE_DIR, SEED


DEFAULT_ITERS = "1,2,5,10,14,20,30,50,100,200,500"
DEFAULT_RANDOM_SEEDS = "42,43,44"


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Tucker observed-entry error by iteration count."
    )
    parser.add_argument("--r1", type=int, default=67)
    parser.add_argument("--r2", type=int, default=20)
    parser.add_argument("--r3", type=int, default=20)
    parser.add_argument("--iters", default=DEFAULT_ITERS)
    parser.add_argument("--random-seeds", default=DEFAULT_RANDOM_SEEDS)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=CACHE_DIR / "tucker_iteration_audit_67_20_20.csv")
    return parser.parse_args()


def run_one(
    filled: np.ndarray,
    mask: np.ndarray,
    rank: list[int],
    init: str,
    max_iter: int,
    tol: float,
    seed: int,
) -> dict[str, float | int | str]:
    start = time.perf_counter()
    try:
        tucker_tensor, tensorly_errors = tucker(
            filled,
            rank=rank,
            mask=mask.astype(np.float64),
            n_iter_max=max_iter,
            tol=tol,
            init=init,
            random_state=seed,
            return_errors=True,
            verbose=False,
        )
        core, factors = tucker_tensor
        recon = tl.tucker_to_tensor((core, factors))
        observed_error = float(
            np.linalg.norm((recon - filled) * mask)
            / (np.linalg.norm(filled * mask) + 1e-12)
        )
        status = "ok"
        internal_final = float(tensorly_errors[-1]) if tensorly_errors else np.nan
        internal_best = float(np.nanmin(tensorly_errors)) if tensorly_errors else np.nan
    except Exception as exc:  # noqa: BLE001
        observed_error = np.nan
        internal_final = np.nan
        internal_best = np.nan
        status = f"{type(exc).__name__}: {exc}"

    return {
        "r1": rank[0],
        "r2": rank[1],
        "r3": rank[2],
        "init": init,
        "seed": seed,
        "max_iter": max_iter,
        "tol": tol,
        "observed_relative_error": observed_error,
        "tensorly_internal_final": internal_final,
        "tensorly_internal_best": internal_best,
        "elapsed_seconds": time.perf_counter() - start,
        "status": status,
    }


def main() -> None:
    args = parse_args()
    rank = [args.r1, args.r2, args.r3]
    max_iters = parse_csv_ints(args.iters)
    random_seeds = parse_csv_ints(args.random_seeds)

    tl.set_backend("numpy")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tensor, mask = build_tensor()
    observed = tensor[mask]
    scale = float(np.sqrt(np.mean(observed**2))) if observed.size else 1.0
    filled = np.nan_to_num(tensor / max(scale, 1e-8), nan=0.0)

    print(
        f"Auditing Tucker rank={rank} on tensor={tensor.shape}, "
        f"observed_density={mask.mean():.2%}",
        flush=True,
    )
    print(f"iters={max_iters}; random_seeds={random_seeds}; tol={args.tol}", flush=True)

    rows: list[dict[str, float | int | str]] = []
    for max_iter in max_iters:
        jobs = [("svd", SEED)] + [("random", seed) for seed in random_seeds]
        for init, seed in jobs:
            row = run_one(
                filled=filled,
                mask=mask,
                rank=rank,
                init=init,
                max_iter=max_iter,
                tol=args.tol,
                seed=seed,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(
                f"init={init:<6} seed={seed:<3} iter={max_iter:<3} "
                f"obs_err={row['observed_relative_error']:.6f} "
                f"tl_final={row['tensorly_internal_final']:.6f} "
                f"elapsed={row['elapsed_seconds']:.1f}s status={row['status']}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    print(f"\nSaved audit CSV: {args.output}", flush=True)
    ok = df[df["status"] == "ok"].copy()
    if not ok.empty:
        print("\nBest observed-entry errors:", flush=True)
        print(
            ok.nsmallest(10, "observed_relative_error")[
                [
                    "init",
                    "seed",
                    "max_iter",
                    "observed_relative_error",
                    "tensorly_internal_final",
                    "elapsed_seconds",
                ]
            ].to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
