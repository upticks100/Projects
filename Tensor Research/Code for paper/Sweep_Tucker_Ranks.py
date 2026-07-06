from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker

from Build_PrePrediction_Exhibits import build_tensor
from pre_prediction_config import CACHE_DIR, FEATURE_SPECS, SEED


OUT_FILE = CACHE_DIR / "tucker_rank_grid_sweep.csv"


def parse_rank_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def stepped_values(start: int, stop: int, step: int, include_stop: bool = True) -> list[int]:
    values = list(range(start, stop + 1, step))
    if include_stop and values[-1] != stop:
        values.append(stop)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-r1", type=int, default=5)
    parser.add_argument("--max-r1", type=int, default=200)
    parser.add_argument("--step-r1", type=int, default=5)
    parser.add_argument(
        "--r1-values",
        default="",
        help="Comma-separated firm ranks. Overrides --min/max/step-r1 if non-empty.",
    )
    parser.add_argument("--min-r2", type=int, default=5)
    parser.add_argument("--max-r2", type=int, default=len(FEATURE_SPECS))
    parser.add_argument("--step-r2", type=int, default=5)
    parser.add_argument(
        "--r2-values",
        default="",
        help="Comma-separated feature ranks. Overrides --min/max/step-r2 if non-empty.",
    )
    parser.add_argument(
        "--r3-values",
        default="5,10,15,20,25,30,40,60,80,100,120,140",
        help="Comma-separated time ranks. Default uses a coarse ladder to keep the grid manageable.",
    )
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--stop-below",
        type=float,
        default=0.0,
        help="Halt when observed error drops below this. 0 disables (run full grid).",
    )
    return parser.parse_args()


def observed_relative_error(
    tensor: np.ndarray,
    mask: np.ndarray,
    r1: int,
    r2: int,
    r3: int,
    max_iter: int,
    tol: float,
) -> float:
    observed = tensor[mask]
    rms = float(np.sqrt(np.mean(observed**2))) if observed.size else 1.0
    filled = np.nan_to_num(tensor / max(rms, 1e-8), nan=0.0)
    mask_i = mask.astype(np.int8)

    core, factors = tucker(
        filled,
        rank=[r1, r2, r3],
        mask=mask_i,
        n_iter_max=max_iter,
        tol=tol,
        init="random",
        random_state=SEED,
        verbose=False,
    )
    recon = tl.tucker_to_tensor((core, factors))
    return float(
        np.linalg.norm((recon - filled) * mask)
        / (np.linalg.norm(filled * mask) + 1e-8)
    )


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tl.set_backend("numpy")
    tensor, mask = build_tensor()
    n_firms, n_features, n_quarters = tensor.shape

    if args.r1_values.strip():
        r1_values = [min(v, n_firms) for v in parse_rank_list(args.r1_values)]
    else:
        r1_values = [
            min(value, n_firms)
            for value in stepped_values(args.min_r1, min(args.max_r1, n_firms), args.step_r1)
        ]
    if args.r2_values.strip():
        r2_values = [min(v, n_features) for v in parse_rank_list(args.r2_values)]
    else:
        r2_values = [
            min(value, n_features)
            for value in stepped_values(args.min_r2, min(args.max_r2, n_features), args.step_r2)
        ]
    r3_values = [min(value, n_quarters) for value in parse_rank_list(args.r3_values)]
    r1_values = sorted(set(r1_values))
    r2_values = sorted(set(r2_values))
    r3_values = sorted(set(r3_values))
    jobs = [(r1, r2, r3) for r1 in r1_values for r2 in r2_values for r3 in r3_values]

    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE)
        done = set(zip(existing["r1"].astype(int), existing["r2"].astype(int), existing["r3"].astype(int)))
        records = existing.to_dict("records")
    else:
        done = set()
        records: list[dict[str, float | int]] = []

    best_error = min(
        (float(record["observed_relative_error"]) for record in records),
        default=float("inf"),
    )

    print(
        f"Sweeping {len(jobs)} rank combinations on tensor {tensor.shape}; "
        f"{len(done)} already complete.",
        flush=True,
    )
    for r1, r2, r3 in jobs:
        if (r1, r2, r3) in done:
            print(f"rank=[{r1}, {r2}, {r3}] cached", flush=True)
            continue

        start = time.perf_counter()
        error = observed_relative_error(tensor, mask, r1, r2, r3, args.max_iter, args.tol)
        elapsed = time.perf_counter() - start
        best_error = min(best_error, error)
        record = {
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "observed_relative_error": error,
            "elapsed_seconds": elapsed,
            "core_entries": int(r1 * r2 * r3),
            "max_iter": args.max_iter,
            "tol": args.tol,
        }
        records.append(record)
        pd.DataFrame.from_records(records).sort_values(["r1", "r2", "r3"]).to_csv(
            OUT_FILE,
            index=False,
        )
        print(
            f"rank=[{r1}, {r2}, {r3}] "
            f"error={error:.6f} best={best_error:.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        if args.stop_below > 0 and error < args.stop_below:
            print(f"Reached target error below {args.stop_below:.2%}; stopping.", flush=True)
            break


if __name__ == "__main__":
    main()
