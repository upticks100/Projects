"""Rebuild the MFI tensor on the clean v2/40 spec (audit Finding 1).

The paper's MFI/FCIX exhibits were generated from the polluted April v1 tensor
(39 features, 12 structurally empty). This script rebuilds everything from the
live pre_prediction_config spec (40 features, 1990Q1-2024Q4,
90-25_Q_Fundamentals_v2.csv):

  1. Full tensor via Build_PrePrediction_Exhibits.build_tensor().
  2. Mask-aware Tucker at rank [67,40,20], RMS-scaled, SVD init
     (Sweep_Tucker_Ranks recipe + 2026-04-29 init decision).
  3. Persist S/F/T factors, core, D_hat, meta -> unlocks anomaly ideas B/C/D/E.
  4. MFI(t) = (1/R3) * sum_k |T[t,k]| ; compare old-vs-new (rho with the v1
     series) and MFI<->FCIX contemporaneous rho.
  5. Gretton-Gyorfi L_n / I_n independence tests vs FCIX, m_n = 6
     equal-frequency bins, 10,000 permutations (paper Table `Independence_Test`
     protocol).

Outputs -> pre_prediction_cache/mfi_v2/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker

from Build_PrePrediction_Exhibits import build_tensor
from pre_prediction_config import (
    CACHE_DIR,
    FCIX_QUARTERLY_FILE,
    FEATURE_SPECS,
    MFI_QUARTERLY_FILE,
    SEED,
)

OUT_DIR = CACHE_DIR / "mfi_v2"
TUCKER_RANK = [67, 40, 20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", default="67,40,20")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--n-perm", type=int, default=10_000)
    p.add_argument("--bins", type=int, default=6)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Independence tests (paper Section 7.3): partition-based L_n and I_n.
# ---------------------------------------------------------------------------
def _equal_freq_bins(x: np.ndarray, m: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0, 1, m + 1)[1:-1])
    return np.searchsorted(edges, x, side="right")


def _ln_in(a_bin: np.ndarray, b_bin: np.ndarray, m: int) -> tuple[float, float]:
    n = a_bin.size
    joint = np.zeros((m, m))
    np.add.at(joint, (a_bin, b_bin), 1.0)
    joint /= n
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    prod = pa * pb
    ln = float(np.abs(joint - prod).sum())
    pos = joint > 0
    i_n = float((joint[pos] * np.log(joint[pos] / prod[pos])).sum())
    return ln, i_n


def independence_tests(fcix: np.ndarray, mfi: np.ndarray, m: int,
                       n_perm: int, seed: int) -> dict:
    a = _equal_freq_bins(fcix, m)
    b = _equal_freq_bins(mfi, m)
    ln_obs, in_obs = _ln_in(a, b, m)

    rng = np.random.default_rng(seed)
    perms = np.empty((n_perm, 2))
    for i in range(n_perm):
        perms[i] = _ln_in(a, rng.permutation(b), m)

    out = {"ln_obs": ln_obs, "in_obs": in_obs, "n_perm": n_perm, "bins": m}
    for name, obs, col in (("ln", ln_obs, 0), ("in", in_obs, 1)):
        null = perms[:, col]
        out[f"{name}_p"] = float((np.sum(null >= obs) + 1) / (n_perm + 1))
        for lvl in (0.01, 0.05, 0.10):
            out[f"{name}_crit_{int(lvl * 100)}pct"] = float(
                np.quantile(null, 1 - lvl))
    return out, pd.DataFrame(perms, columns=["ln", "in"]).assign(
        perm=np.arange(n_perm))[["perm", "ln", "in"]]


def main() -> None:
    args = parse_args()
    rank = [int(v) for v in args.rank.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tl.set_backend("numpy")

    # ---- 1. tensor -------------------------------------------------------
    tensor, mask = build_tensor()
    n_firms, n_feats, n_q = tensor.shape
    assert n_feats == len(FEATURE_SPECS), "feature count drifted from spec"
    quarters = pd.period_range("1990Q1", periods=n_q, freq="Q").astype(str)

    # ---- 2. mask-aware Tucker (RMS recipe, SVD init) ---------------------
    observed = tensor[mask]
    rms = float(np.sqrt(np.mean(observed ** 2)))
    filled = np.nan_to_num(tensor.astype(np.float64) / rms, nan=0.0)
    print(f"Tucker rank={rank} on {tensor.shape}, density {mask.mean():.2%}, "
          f"rms={rms:.4g}", flush=True)
    t0 = time.perf_counter()
    core, factors = tucker(
        filled, rank=rank, mask=mask.astype(np.int8),
        n_iter_max=args.max_iter, tol=args.tol,
        init="svd", random_state=SEED, verbose=False,
    )
    recon = tl.tucker_to_tensor((core, factors))
    rel_err = float(np.linalg.norm((recon - filled) * mask)
                    / (np.linalg.norm(filled * mask) + 1e-8))
    print(f"observed relative error {rel_err:.4f} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    S, F, T = factors
    d_hat = (recon * rms).astype(np.float32)
    joblib.dump(
        {
            "core": core, "S": S, "F": F, "T": T,
            "d_hat": d_hat, "mask": mask,
            "rms": rms, "rank": rank, "rel_err": rel_err,
            "feature_names": [s.label for s in FEATURE_SPECS],
            "quarters": list(quarters),
            "n_firms": n_firms,
        },
        OUT_DIR / "tucker_v2_decomposition.joblib",
        compress=3,
    )

    # ---- 3. MFI v2 -------------------------------------------------------
    mfi_v2 = np.mean(np.abs(T), axis=1)
    pd.DataFrame({"quarter": quarters, "MFI": mfi_v2}).to_csv(
        OUT_DIR / "mfi_quarterly_v2.csv", index=False)

    old = pd.read_csv(MFI_QUARTERLY_FILE)
    fcix = pd.read_csv(FCIX_QUARTERLY_FILE)
    merged = (pd.DataFrame({"quarter": quarters, "MFI_v2": mfi_v2})
              .merge(old.rename(columns={"MFI": "MFI_v1"}), on="quarter",
                     how="left")
              .merge(fcix, on="quarter", how="inner"))
    merged.to_csv(OUT_DIR / "mfi_fcix_quarterly_v2.csv", index=False)

    rho_v1v2 = float(merged["MFI_v2"].corr(merged["MFI_v1"]))
    rho_fcix_v2 = float(merged["MFI_v2"].corr(merged["FCIX"]))
    rho_fcix_v1 = float(merged["MFI_v1"].corr(merged["FCIX"]))

    # ---- 4. independence tests -------------------------------------------
    sub = merged.dropna(subset=["MFI_v2", "FCIX"])
    stats, perms = independence_tests(
        sub["FCIX"].to_numpy(), sub["MFI_v2"].to_numpy(),
        args.bins, args.n_perm, SEED)
    perms.to_csv(OUT_DIR / "independence_permutation_v2.csv", index=False)

    summary = {
        "tensor_shape": [int(n_firms), int(n_feats), int(n_q)],
        "density": float(mask.mean()),
        "tucker_rank": rank, "observed_relative_error": rel_err,
        "n_quarters_merged": int(len(sub)),
        "rho_mfi_v1_v2": rho_v1v2,
        "rho_fcix_mfi_v2": rho_fcix_v2,
        "rho_fcix_mfi_v1": rho_fcix_v1,
        **stats,
    }
    (OUT_DIR / "mfi_v2_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    for name in ("ln", "in"):
        verdict = "REJECT at 1%" if stats[f"{name}_obs"] > \
            stats[f"{name}_crit_1pct"] else (
            "reject at 5%" if stats[f"{name}_obs"] > stats[f"{name}_crit_5pct"]
            else "no rejection")
        print(f"{name.upper()}: obs={stats[f'{name}_obs']:.4f} "
              f"crit1%={stats[f'{name}_crit_1pct']:.4f} "
              f"p={stats[f'{name}_p']:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
