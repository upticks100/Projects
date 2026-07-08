"""Block-permutation robustness for the MFI-FCIX independence test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.mfi.build_mfi import _equal_freq_bins, _ln_in


MFI_FCIX_FILE = (
    config.DATA_DIR / "pre_prediction_cache" / "mfi_v2" / "mfi_fcix_quarterly_v2.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=MFI_FCIX_FILE)
    p.add_argument("--out-dir", type=Path,
                   default=config.RESULTS_DIR / "mfi_block_permutation")
    p.add_argument("--bins", type=int, default=config.MFI_NUM_BINS)
    p.add_argument("--n-perm", type=int, default=config.MFI_NUM_PERMUTATIONS)
    p.add_argument("--blocks", default="4,8",
                   help="comma-separated circular block lengths in quarters")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def circular_block_permutation(x: np.ndarray, block: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Permutation made by resampling circular contiguous blocks."""
    n = len(x)
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = []
    for s in starts:
        idx.extend(((s + np.arange(block)) % n).tolist())
    return x[np.asarray(idx[:n])]


def run_one(a: np.ndarray, b: np.ndarray, m: int, n_perm: int,
            block: int, seed: int) -> tuple[dict, pd.DataFrame]:
    a_bin = _equal_freq_bins(a, m)
    b_bin = _equal_freq_bins(b, m)
    ln_obs, in_obs = _ln_in(a_bin, b_bin, m)
    rng = np.random.default_rng(seed + block)
    perms = np.empty((n_perm, 2))
    for i in range(n_perm):
        bb = circular_block_permutation(b_bin, block, rng)
        perms[i] = _ln_in(a_bin, bb, m)
    row = {
        "block_length": block,
        "ln_obs": ln_obs,
        "in_obs": in_obs,
        "n_perm": n_perm,
        "bins": m,
    }
    for name, obs, col in (("ln", ln_obs, 0), ("in", in_obs, 1)):
        null = perms[:, col]
        row[f"{name}_p"] = float((np.sum(null >= obs) + 1) / (n_perm + 1))
        for lvl in (0.01, 0.05, 0.10):
            row[f"{name}_crit_{int(lvl * 100)}pct"] = float(
                np.quantile(null, 1 - lvl))
    return row, pd.DataFrame(perms, columns=["ln", "in"]).assign(
        block_length=block, perm=np.arange(n_perm))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input).dropna(subset=["MFI_v2", "FCIX"])
    a = df["FCIX"].to_numpy(float)
    b = df["MFI_v2"].to_numpy(float)
    blocks = [int(x) for x in args.blocks.split(",") if x.strip()]

    rows, nulls = [], []
    for block in blocks:
        row, perm = run_one(a, b, args.bins, args.n_perm, block, args.seed)
        rows.append(row)
        nulls.append(perm)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "mfi_fcix_block_permutation_summary.csv",
                   index=False)
    pd.concat(nulls, ignore_index=True).to_csv(
        args.out_dir / "mfi_fcix_block_permutation_null.csv", index=False)
    payload = {"n_obs": int(len(df)), "blocks": blocks,
               "rows": summary.to_dict(orient="records")}
    (args.out_dir / "mfi_fcix_block_permutation_summary.json").write_text(
        json.dumps(payload, indent=2))
    print(summary.to_string(index=False))
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
