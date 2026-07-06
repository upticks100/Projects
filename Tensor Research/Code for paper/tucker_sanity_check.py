from __future__ import annotations

import argparse
import time

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker

from Build_PrePrediction_Exhibits import build_tensor
from pre_prediction_config import SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused Tucker sanity check for one rank tuple."
    )
    parser.add_argument("--r1", type=int, default=67)
    parser.add_argument("--r2", type=int, default=20)
    parser.add_argument("--r3", type=int, default=20)
    parser.add_argument("--init", default="svd", choices=["svd", "random"])
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def observed_relative_error(
    tensor: np.ndarray,
    mask: np.ndarray,
    rank: list[int],
    init: str,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> tuple[float, float]:
    observed = tensor[mask]
    scale = float(np.sqrt(np.mean(observed**2))) if observed.size else 1.0
    filled = np.nan_to_num(tensor / max(scale, 1e-8), nan=0.0)
    mask_f = mask.astype(np.float64)

    start = time.perf_counter()
    core, factors = tucker(
        filled,
        rank=rank,
        mask=mask_f,
        n_iter_max=max_iter,
        tol=tol,
        init=init,
        random_state=SEED,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start

    recon = tl.tucker_to_tensor((core, factors))
    numerator = np.linalg.norm((recon - filled) * mask)
    denominator = np.linalg.norm(filled * mask) + 1e-12
    return float(numerator / denominator), elapsed


def main() -> None:
    args = parse_args()
    tl.set_backend("numpy")

    tensor, mask = build_tensor()
    rank = [args.r1, args.r2, args.r3]
    print(
        f"Running Tucker sanity check rank={rank}, init={args.init}, "
        f"max_iter={args.max_iter}, tol={args.tol}",
        flush=True,
    )
    print(f"Tensor shape={tensor.shape}; observed density={mask.mean():.2%}", flush=True)

    error, elapsed = observed_relative_error(
        tensor=tensor,
        mask=mask,
        rank=rank,
        init=args.init,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=args.verbose,
    )
    print(f"OUR_MASKED_OBSERVED_RELATIVE_ERROR={error:.10f}", flush=True)
    print(f"ELAPSED_SECONDS={elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
