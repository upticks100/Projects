"""MFI robustness appendix helpers.

This module is intentionally small and reproducible. It reads the clean v2 MFI
artifacts and writes two appendix tables:

1. Alternative MFI definitions from the saved v2 time factors.
2. Lightweight rank/normalization variants plus the prior init audit.

The expensive canonical rebuild remains `src.mfi.build_mfi`; this script is a
stability check, not a replacement for the paper's main MFI construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker

import config
from src.mfi.build_mfi import build_tensor


DEFAULT_MFI_DIR = config.DATA_DIR / "pre_prediction_cache" / "mfi_v2"
DEFAULT_AUDIT = config.DATA_DIR / "pre_prediction_cache" / "tucker_iteration_audit_67_20_20.csv"


def _mfi_defs(T: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean_abs_time_factors": np.mean(np.abs(T), axis=1),
        "rms_time_factors": np.sqrt(np.mean(T ** 2, axis=1)),
        "first_factor_abs": np.abs(T[:, 0]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mfi-dir", type=Path, default=DEFAULT_MFI_DIR)
    p.add_argument("--out-dir", type=Path,
                   default=config.RESULTS_DIR / "mfi_robustness")
    p.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--run-light-variants", action="store_true",
                   help="run SVD rank/normalization variants from the raw tensor")
    p.add_argument("--max-iter", type=int, default=80)
    return p.parse_args()


def definition_robustness(mfi_dir: Path, out_dir: Path) -> pd.DataFrame:
    decomp = joblib.load(mfi_dir / "tucker_v2_decomposition.joblib")
    T = np.asarray(decomp["T"], dtype=float)
    quarters = list(decomp["quarters"])
    variants = _mfi_defs(T)
    base = variants["mean_abs_time_factors"]
    rows = []
    series = {"quarter": quarters}
    for name, values in variants.items():
        series[name] = values
        rows.append({
            "variant": name,
            "kind": "index_definition",
            "rank": ",".join(map(str, decomp["rank"])),
            "normalization": "rms",
            "init": "svd",
            "seed": config.SEED,
            "observed_relative_error": decomp["rel_err"],
            "corr_with_baseline": float(np.corrcoef(base, values)[0, 1]),
            "std_ratio_vs_baseline": float(np.std(values) / np.std(base)),
        })
    pd.DataFrame(series).to_csv(out_dir / "mfi_definition_series.csv", index=False)
    return pd.DataFrame(rows)


def audit_summary(audit_csv: Path) -> pd.DataFrame:
    if not audit_csv.exists():
        return pd.DataFrame()
    audit = pd.read_csv(audit_csv)
    rows = []
    for _, r in audit.iterrows():
        if int(r["max_iter"]) not in {50, 200, 500}:
            continue
        rows.append({
            "variant": f"{r['init']}_seed{int(r['seed'])}_iter{int(r['max_iter'])}",
            "kind": "init_audit",
            "rank": f"{int(r['r1'])},{int(r['r2'])},{int(r['r3'])}",
            "normalization": "rms",
            "init": r["init"],
            "seed": int(r["seed"]),
            "observed_relative_error": float(r["observed_relative_error"]),
            "corr_with_baseline": np.nan,
            "std_ratio_vs_baseline": np.nan,
        })
    return pd.DataFrame(rows)


def light_variants(out_dir: Path, max_iter: int) -> pd.DataFrame:
    tl.set_backend("numpy")
    tensor, mask = build_tensor()
    observed = tensor[mask].astype(float)
    rms = float(np.sqrt(np.mean(observed ** 2)))
    base = pd.read_csv(DEFAULT_MFI_DIR / "mfi_quarterly_v2.csv")["MFI"].to_numpy()
    rows = []
    for rank, normalization in [
        ((67, 40, 16), "rms"),
        ((67, 40, 20), "none"),
        ((67, 40, 24), "rms"),
    ]:
        scale = rms if normalization == "rms" else 1.0
        filled = np.nan_to_num(tensor.astype(np.float64) / scale, nan=0.0)
        core, factors = tucker(
            filled,
            rank=list(rank),
            mask=mask.astype(np.int8),
            n_iter_max=max_iter,
            tol=1e-5,
            init="svd",
            random_state=config.SEED,
            verbose=False,
        )
        recon = tl.tucker_to_tensor((core, factors))
        rel_err = float(np.linalg.norm((recon - filled) * mask)
                        / (np.linalg.norm(filled * mask) + 1e-8))
        mfi = np.mean(np.abs(np.asarray(factors[2])), axis=1)
        pd.DataFrame({
            "quarter": pd.period_range(config.MFI_START_QUARTER,
                                       config.MFI_END_QUARTER, freq="Q").astype(str),
            "MFI": mfi,
        }).to_csv(out_dir / f"mfi_variant_rank_{rank[0]}_{rank[1]}_{rank[2]}_{normalization}.csv",
                 index=False)
        rows.append({
            "variant": f"rank_{rank[0]}_{rank[1]}_{rank[2]}_{normalization}",
            "kind": "rank_normalization",
            "rank": ",".join(map(str, rank)),
            "normalization": normalization,
            "init": "svd",
            "seed": config.SEED,
            "observed_relative_error": rel_err,
            "corr_with_baseline": float(np.corrcoef(base, mfi)[0, 1]),
            "std_ratio_vs_baseline": float(np.std(mfi) / np.std(base)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        definition_robustness(args.mfi_dir, args.out_dir),
        audit_summary(args.audit_csv),
    ]
    if args.run_light_variants:
        parts.append(light_variants(args.out_dir, args.max_iter))

    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    out.to_csv(args.out_dir / "mfi_stability_summary.csv", index=False)
    summary = {
        "n_rows": int(len(out)),
        "definition_corr_min": float(out.loc[
            out["kind"].eq("index_definition"), "corr_with_baseline"].min()),
        "rank_norm_corr_min": (
            None if not out["kind"].eq("rank_normalization").any()
            else float(out.loc[
                out["kind"].eq("rank_normalization"), "corr_with_baseline"].min())
        ),
    }
    (args.out_dir / "mfi_stability_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> {args.out_dir / 'mfi_stability_summary.csv'}")


if __name__ == "__main__":
    main()
