"""Analyse the v3 holdout: per-cell summary + regime-tagged per-window analysis.

Reads the CSV pair produced by `evaluate_top_trials_test.py` for each
(objective, mode, L) cell from a holdout output directory and produces:

1. `<holdout_dir>/aggregate_summary.csv` — one row per trial across all cells.
2. `<holdout_dir>/aggregate_per_window.csv` — concatenated per-window with
   regime indicators attached.
3. `<holdout_dir>/regime_summary.csv` — for each (cell, trial), mean delta
   by regime tercile of each indicator, plus the linear-regression slope
   `delta = alpha + beta * indicator + eps`.
4. Printed report.

Regime indicators (all computed from the tensor cache so they are
reproducible and leakage-free relative to the model):

- `y_disp`: cross-sectional std of Y values across firms, averaged over
  features, observed cells only. Computed from each TEST window itself
  (not from training data). High = dispersed regime.
- `mask_density`: fraction of observed cells in the test window. Low =
  high-imputation regime.
- `window_index`: position of the test window within the 16-window test
  block (0 = first test window).

Note: `y_disp` of the test window itself is not a forecastable feature
(it uses Y_test for tagging), so this analysis is descriptive — it
characterises WHERE on the test set CP added value, not a strategy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from prediction_config import (  # noqa: E402
    TEST_START_TARGET_QUARTER,
    cache_path,
    meta_path,
)


def _calendar_split_idx(L: int, n_windows: int) -> int:
    """Calendar-fixed boundary (mirrors evaluate_top_trials_test)."""
    meta = joblib.load(meta_path())
    quarters = [str(q) for q in meta["quarters"]]
    qi = quarters.index(str(TEST_START_TARGET_QUARTER))
    split_idx = qi - L
    if not (0 < split_idx < n_windows):
        raise SystemExit(
            f"calendar split_idx={split_idx} out of range (L={L}, "
            f"n_windows={n_windows}, qi={qi})"
        )
    return split_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "holdout_dir",
        type=Path,
        help="Path to results/v3_holdout_<ts>/ directory.",
    )
    parser.add_argument(
        "--terciles",
        type=int,
        default=3,
        help="Number of quantile buckets for regime tagging (3 = terciles).",
    )
    return parser.parse_args()


def discover_cell_csvs(holdout_dir: Path) -> list[tuple[str, str, int, Path, Path]]:
    """Discover (objective, mode, L, summary_csv, per_window_csv) tuples."""
    cells: list[tuple[str, str, int, Path, Path]] = []
    for summary_csv in sorted(holdout_dir.glob("*.csv")):
        if "_per_window" in summary_csv.stem or summary_csv.stem == "aggregate_summary":
            continue
        pw_csv = summary_csv.with_name(summary_csv.stem + "_per_window.csv")
        if not pw_csv.exists():
            print(f"WARN: missing per-window companion for {summary_csv.name}, skipping")
            continue
        head = pd.read_csv(summary_csv, nrows=1)
        if head.empty:
            print(f"WARN: {summary_csv.name} is empty, skipping")
            continue
        obj = str(head["objective"].iloc[0])
        mode = str(head["mode"].iloc[0])
        L = int(head["L"].iloc[0])
        cells.append((obj, mode, L, summary_csv, pw_csv))
    return cells


def compute_window_indicators(mode: str, L: int) -> pd.DataFrame:
    """Return one row per test window with regime indicators."""
    cache = joblib.load(cache_path(mode, L))
    Y_all, M_all = cache["Y"], cache["Mask"]
    split_idx = _calendar_split_idx(L, len(Y_all))
    Y_test, M_test = Y_all[split_idx:], M_all[split_idx:]

    rows = []
    for w in range(Y_test.shape[0]):
        Y_w = Y_test[w]
        M_w = M_test[w]
        observed = M_w > 0

        if observed.any():
            # cross-sectional std of Y across firms, per feature, observed only,
            # then averaged across features
            feat_disps = []
            for f in range(Y_w.shape[1]):
                col = Y_w[:, f][observed[:, f]]
                if col.size >= 2:
                    feat_disps.append(float(np.std(col, ddof=1)))
            y_disp = float(np.mean(feat_disps)) if feat_disps else np.nan
        else:
            y_disp = np.nan

        mask_density = float(observed.mean())
        rows.append(
            {
                "mode": mode,
                "L": int(L),
                "window_index": w,
                "global_window_index": split_idx + w,
                "y_disp": y_disp,
                "mask_density": mask_density,
            }
        )
    return pd.DataFrame(rows)


def regime_tag(series: pd.Series, n_buckets: int) -> pd.Series:
    if series.nunique() < n_buckets:
        return pd.Series([np.nan] * len(series), index=series.index)
    labels = [f"Q{i + 1}" for i in range(n_buckets)]
    return pd.qcut(series, n_buckets, labels=labels, duplicates="drop").astype(str)


def per_trial_regime_table(per_window_aug: pd.DataFrame,
                           indicators: list[str]) -> pd.DataFrame:
    """For each (objective, mode, L, trial) and each indicator, compute mean
    delta per tercile + linear regression slope."""
    rows = []
    group_cols = ["objective", "mode", "L", "rank_order", "trial_number"]
    for (obj, mode, L, rank, trial), grp in per_window_aug.groupby(group_cols):
        for ind in indicators:
            tag_col = f"{ind}_q"
            if tag_col not in grp.columns or grp[tag_col].isna().all():
                continue
            means = grp.groupby(tag_col, observed=True)["delta"].mean().to_dict()
            x = grp[ind].to_numpy(dtype=float)
            y = grp["delta"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3 and np.std(x[mask]) > 0:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            else:
                slope, intercept, corr = np.nan, np.nan, np.nan
            rows.append(
                {
                    "objective": obj,
                    "mode": mode,
                    "L": int(L),
                    "rank_order": int(rank),
                    "trial_number": int(trial),
                    "indicator": ind,
                    "mean_delta_overall": float(y[mask].mean()) if mask.any() else np.nan,
                    "mean_delta_Q1": float(means.get("Q1", np.nan)),
                    "mean_delta_Q2": float(means.get("Q2", np.nan)),
                    "mean_delta_Q3": float(means.get("Q3", np.nan)),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "corr": corr,
                    "n_windows": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    holdout_dir = args.holdout_dir.resolve()
    if not holdout_dir.is_dir():
        sys.exit(f"holdout_dir does not exist: {holdout_dir}")

    cells = discover_cell_csvs(holdout_dir)
    if not cells:
        sys.exit(f"No (summary, per_window) CSV pairs found in {holdout_dir}")

    print(f"Discovered {len(cells)} cells:")
    for obj, mode, L, s, _ in cells:
        print(f"  {obj} {mode} L={L}  ({s.name})")

    summary_dfs = []
    per_window_dfs = []
    for obj, mode, L, s_csv, pw_csv in cells:
        s_df = pd.read_csv(s_csv)
        pw_df = pd.read_csv(pw_csv)
        ind_df = compute_window_indicators(mode, L)
        pw_aug = pw_df.merge(
            ind_df,
            on=["mode", "L", "window_index", "global_window_index"],
            how="left",
        )
        for ind in ("y_disp", "mask_density", "window_index"):
            pw_aug[f"{ind}_q"] = pw_aug.groupby(["objective", "mode", "L"])[ind].transform(
                lambda v: regime_tag(v, args.terciles)
            )

        summary_dfs.append(s_df)
        per_window_dfs.append(pw_aug)

    agg_summary = pd.concat(summary_dfs, ignore_index=True).sort_values(
        ["objective", "mode", "L", "rank_order"]
    )
    agg_per_window = pd.concat(per_window_dfs, ignore_index=True)
    regime_table = per_trial_regime_table(
        agg_per_window, indicators=["y_disp", "mask_density", "window_index"],
    )

    out_summary = holdout_dir / "aggregate_summary.csv"
    out_pw = holdout_dir / "aggregate_per_window.csv"
    out_regime = holdout_dir / "regime_summary.csv"
    agg_summary.to_csv(out_summary, index=False)
    agg_per_window.to_csv(out_pw, index=False)
    regime_table.to_csv(out_regime, index=False)

    print(f"\nWrote {out_summary}")
    print(f"Wrote {out_pw}")
    print(f"Wrote {out_regime}\n")

    print("=== Aggregate summary (per trial) ===")
    cols = ["objective", "mode", "L", "rank_order", "trial_number",
            "base_test_r2", "ensemble_test_r2", "test_delta",
            "cp_train_windows", "dropped_oof_rows"]
    cols = [c for c in cols if c in agg_summary.columns]
    print(agg_summary[cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== Per-cell rank-1 summary (best trial per cell) ===")
    best = agg_summary.loc[agg_summary.groupby(["objective", "mode", "L"])["test_delta"].idxmax()]
    best_cols = ["objective", "mode", "L", "trial_number",
                 "base_test_r2", "ensemble_test_r2", "test_delta"]
    best_cols = [c for c in best_cols if c in best.columns]
    print(best[best_cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== Regime breakdown (rank-1 trial per cell) ===")
    best_keys = set(
        zip(best["objective"], best["mode"], best["L"], best["trial_number"])
    )
    regime_best = regime_table[
        regime_table.apply(
            lambda r: (r["objective"], r["mode"], r["L"], r["trial_number"]) in best_keys,
            axis=1,
        )
    ]
    for ind in ("y_disp", "mask_density", "window_index"):
        sub = regime_best[regime_best["indicator"] == ind]
        if sub.empty:
            continue
        print(f"\n  {ind}:")
        print(
            sub[
                ["objective", "mode", "L",
                 "mean_delta_Q1", "mean_delta_Q2", "mean_delta_Q3",
                 "slope", "corr"]
            ].to_string(index=False, float_format=lambda v: f"{v:+.5f}")
        )


if __name__ == "__main__":
    main()
