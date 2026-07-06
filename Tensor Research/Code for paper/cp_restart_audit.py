from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Keep each worker modest; launch more workers for parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import tensorly as tl
from joblib import Parallel, delayed
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

from Build_PrePrediction_Exhibits import build_tensor
from pre_prediction_config import CACHE_DIR


DEFAULT_RANKS = "20,40,60,80,100"
DEFAULT_SEEDS = "42,43,44,45,46"


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(part))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CP reconstruction error across random restarts."
    )
    parser.add_argument("--ranks", default=DEFAULT_RANKS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rank/seed jobs already present in --output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CACHE_DIR / "cp_restart_audit.csv",
    )
    return parser.parse_args()


def cp_fit_error(
    filled: np.ndarray,
    mask: np.ndarray,
    base_norm: float,
    rank: int,
    seed: int,
    max_iter: int,
    tol: float,
) -> dict[str, float | int | str]:
    tl.set_backend("numpy")
    start = time.perf_counter()
    try:
        weights, factors = parafac(
            filled,
            rank=rank,
            init="random",
            random_state=seed,
            mask=mask.astype(np.float64),
            n_iter_max=max_iter,
            tol=tol,
            normalize_factors=False,
            verbose=False,
        )
        reconstructed = cp_to_tensor((weights, factors))
        observed_error = float(
            np.linalg.norm((filled - reconstructed) * mask) / (base_norm + 1e-12)
        )
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        observed_error = np.nan
        status = f"{type(exc).__name__}: {exc}"

    return {
        "rank": rank,
        "seed": seed,
        "init": "random",
        "max_iter": max_iter,
        "tol": tol,
        "observed_relative_error": observed_error,
        "elapsed_seconds": time.perf_counter() - start,
        "status": status,
    }


def main() -> None:
    args = parse_args()
    ranks = parse_csv_ints(args.ranks)
    seeds = parse_csv_ints(args.seeds)
    jobs = [(rank, seed) for rank in ranks for seed in seeds]

    existing_rows: list[dict[str, float | int | str]] = []
    done: set[tuple[int, int]] = set()
    if args.resume and args.output.exists():
        existing = pd.read_csv(args.output)
        if {"rank", "seed"}.issubset(existing.columns):
            existing_rows = existing.to_dict("records")
            done = set(zip(existing["rank"].astype(int), existing["seed"].astype(int)))
            jobs = [(rank, seed) for rank, seed in jobs if (rank, seed) not in done]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tl.set_backend("numpy")

    tensor, mask = build_tensor()
    observed = tensor[mask]
    scale = float(np.sqrt(np.mean(observed**2))) if observed.size else 1.0
    filled = np.nan_to_num(tensor / max(scale, 1e-8), nan=0.0)
    base_norm = float(np.linalg.norm(filled[mask]))

    cache_path = CACHE_DIR / "cp_restart_audit_tensor_cache.joblib"
    joblib.dump({"filled": filled, "mask": mask, "base_norm": base_norm}, cache_path)

    print(
        f"CP restart audit on tensor={tensor.shape}, density={mask.mean():.2%}; "
        f"ranks={ranks}; seeds={seeds}; max_iter={args.max_iter}; tol={args.tol}; "
        f"n_jobs={args.n_jobs}; resume={args.resume}; already_done={len(done)}; "
        f"remaining_jobs={len(jobs)}",
        flush=True,
    )
    if not jobs:
        print("No remaining jobs.", flush=True)
        return

    # Use memmaped arrays so child processes do not each copy the full tensor.
    shared = joblib.load(cache_path, mmap_mode="r")
    rows = list(existing_rows)
    parallel_results = Parallel(n_jobs=args.n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(cp_fit_error)(
            shared["filled"],
            shared["mask"],
            float(shared["base_norm"]),
            rank,
            seed,
            args.max_iter,
            args.tol,
        )
        for rank, seed in jobs
    )
    for row in parallel_results:
        rows.append(row)
        pd.DataFrame(rows).sort_values(["rank", "seed"]).to_csv(args.output, index=False)
        print(
            f"finished rank={row['rank']} seed={row['seed']} "
            f"obs_err={row['observed_relative_error']:.6f} "
            f"elapsed={row['elapsed_seconds']:.1f}s status={row['status']}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values(["rank", "seed"])
    df.to_csv(args.output, index=False)
    print(f"\nSaved audit CSV: {args.output}", flush=True)

    ok = df[df["status"] == "ok"].copy()
    if not ok.empty:
        summary = (
            ok.groupby("rank")["observed_relative_error"]
            .agg(["min", "mean", "std", "count"])
            .reset_index()
            .rename(columns={"min": "best", "count": "n_ok"})
        )
        print("\nSummary by rank:", flush=True)
        print(summary.to_string(index=False), flush=True)
        print("\nBest seed by rank:", flush=True)
        print(
            ok.loc[ok.groupby("rank")["observed_relative_error"].idxmin()][
                ["rank", "seed", "observed_relative_error", "elapsed_seconds"]
            ].to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
