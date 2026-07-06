"""Aggregate per-feature R² CSVs + tag features by stationarity.

Reads `<out_dir>/cell_*.csv` files emitted by the distributed
`evaluate_per_feature.py` runs, attaches per-feature stationarity
metrics computed from the training portion of the tensor cache (no
test leakage), and emits a per-cell breakdown of:

  - per-feature R² delta (CP help)
  - stationarity bucket (Q1 = most stationary, Q3 = least)
  - linear regression of delta on each metric
  - win/loss counts per bucket

Outputs:
  - `<out_dir>/per_feature_aggregate.csv`   one row per (cell, trial, feature)
  - `<out_dir>/feature_stationarity.csv`    one row per (mode, L, feature)
  - `<out_dir>/per_feature_summary.txt`     printed report

Stationarity metrics (per feature, computed on TRAIN portion only of
each (mode, L) cache so the tagging is leakage-free):

  - `vr_stat`: variance ratio of LEVEL vs DIFF cross-sectional means.
    Computed as std(diff(x_f)) / std(x_f), where x_f[t] = mean over
    firms of Y[t, :, f]. Low values = level dominates differences =
    persistent / non-stationary trend. High values ≈ stationary.
  - `cv`: coefficient of variation of x_f over time (std/|mean|).
    Higher = more variable around its mean over time.
  - `trend_slope`: |OLS slope of x_f on t / std(x_f)|. Higher = stronger
    monotonic drift.

`vr_stat` is the primary metric reported in the bucket analysis: low
buckets ≈ non-stationary features, high buckets ≈ stationary features.
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

from prediction_config import cache_path, meta_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "out_dir",
        type=Path,
        help="Directory containing cell_*.csv files from evaluate_per_feature.",
    )
    p.add_argument("--buckets", type=int, default=3, help="Stationarity quantile buckets.")
    return p.parse_args()


def discover_cells(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("cell_*.csv"))


def feature_stationarity_for_cell(mode: str, L: int) -> pd.DataFrame:
    """Compute per-feature stationarity metrics from TRAIN portion of cache.

    The split is the same 80/20 used everywhere else (`int(0.8 * len(X))`).
    """
    cache = joblib.load(cache_path(mode, L))
    Y_all, M_all = cache["Y"], cache["Mask"]
    split_idx = int(0.8 * len(Y_all))
    Y_tr = Y_all[:split_idx]
    M_tr = M_all[:split_idx]
    T, F, K = Y_tr.shape  # windows, firms, features

    rows = []
    for f in range(K):
        Y_f = Y_tr[:, :, f]
        M_f = (M_tr[:, :, f] > 0)
        # per-window cross-sectional mean across firms, observed only
        x_f = np.full(T, np.nan)
        for t in range(T):
            obs = Y_f[t][M_f[t]]
            if obs.size:
                x_f[t] = float(obs.mean())
        valid = np.isfinite(x_f)
        if valid.sum() < 5:
            rows.append(
                {"mode": mode, "L": int(L), "feature_index": f,
                 "vr_stat": np.nan, "cv": np.nan, "trend_slope": np.nan,
                 "n_valid_windows": int(valid.sum())}
            )
            continue
        xs = x_f[valid]
        ts = np.where(valid)[0].astype(float)

        # variance ratio: std(diff) / std(level)
        sd_lvl = float(np.std(xs, ddof=1)) if xs.size >= 2 else np.nan
        diff = np.diff(xs)
        sd_diff = float(np.std(diff, ddof=1)) if diff.size >= 2 else np.nan
        vr_stat = (sd_diff / sd_lvl) if (sd_lvl > 1e-12 and np.isfinite(sd_lvl) and np.isfinite(sd_diff)) else np.nan

        # coefficient of variation (in absolute mean units)
        mu = float(np.mean(xs))
        cv = (sd_lvl / abs(mu)) if abs(mu) > 1e-12 else np.nan

        # |normalized trend slope|
        if ts.size >= 3 and sd_lvl > 1e-12:
            slope, _ = np.polyfit(ts, xs, 1)
            trend_slope = abs(float(slope)) / sd_lvl
        else:
            trend_slope = np.nan

        rows.append(
            {
                "mode": mode,
                "L": int(L),
                "feature_index": f,
                "vr_stat": vr_stat,
                "cv": cv,
                "trend_slope": trend_slope,
                "n_valid_windows": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def bucket_tag(series: pd.Series, n_buckets: int) -> pd.Series:
    if series.nunique(dropna=True) < n_buckets:
        return pd.Series([np.nan] * len(series), index=series.index)
    labels = [f"Q{i + 1}" for i in range(n_buckets)]
    return pd.qcut(series, n_buckets, labels=labels, duplicates="drop").astype(str)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    cell_csvs = discover_cells(out_dir)
    if not cell_csvs:
        sys.exit(f"no cell_*.csv files in {out_dir}")

    print(f"Discovered {len(cell_csvs)} per-cell CSVs in {out_dir}")

    meta = joblib.load(meta_path())
    feature_names = list(meta["feature_names"])

    per_feature_dfs = []
    for csv in cell_csvs:
        try:
            df = pd.read_csv(csv)
        except pd.errors.EmptyDataError:
            print(f"WARN: {csv.name} is empty, skipping")
            continue
        if df.empty:
            continue
        per_feature_dfs.append(df)

    if not per_feature_dfs:
        sys.exit("no per-feature data found")
    perfeat = pd.concat(per_feature_dfs, ignore_index=True)

    # Stationarity tables per (mode, L) cell.
    keys = perfeat[["mode", "L"]].drop_duplicates().itertuples(index=False)
    stat_dfs = [feature_stationarity_for_cell(str(m), int(L)) for m, L in keys]
    stat = pd.concat(stat_dfs, ignore_index=True)
    stat = stat.merge(
        pd.DataFrame({"feature_index": range(len(feature_names)),
                      "feature_name": feature_names}),
        on="feature_index", how="left",
    )

    metrics = ["vr_stat", "cv", "trend_slope"]
    for metric in metrics:
        stat[f"{metric}_q"] = stat.groupby(["mode", "L"])[metric].transform(
            lambda v: bucket_tag(v, args.buckets)
        )

    merged = perfeat.merge(stat, on=["mode", "L", "feature_index"], how="left",
                           suffixes=("", "_dup"))
    # drop dup feature_name if both sides had it
    for c in list(merged.columns):
        if c.endswith("_dup"):
            merged.drop(columns=c, inplace=True)

    out_perfeat = out_dir / "per_feature_aggregate.csv"
    out_stat = out_dir / "feature_stationarity.csv"
    merged.to_csv(out_perfeat, index=False)
    stat.to_csv(out_stat, index=False)
    print(f"\nWrote {out_perfeat}")
    print(f"Wrote {out_stat}")

    # ===== reporting =====
    lines: list[str] = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit()
    emit("=== Pooled per-cell summary (mean ± std of per-trial mean delta across features) ===")
    for (obj, mode, L), grp in merged.groupby(["objective", "mode", "L"]):
        per_trial = grp.groupby("rank_order")["delta"].mean()
        emit(f"  {obj} {mode} L={L}: per-trial mean-feat-delta = "
             f"{per_trial.mean():+.5f} ± {per_trial.std():.5f} "
             f"(n_trials={per_trial.size}, n_features={grp['feature_index'].nunique()})")

    emit()
    emit("=== Win/loss counts (across all trials in each cell) ===")
    for (obj, mode, L), grp in merged.groupby(["objective", "mode", "L"]):
        wins = int((grp["delta"] > 0).sum())
        losses = int((grp["delta"] < 0).sum())
        evaluable = int(grp["delta"].notna().sum())
        emit(f"  {obj} {mode} L={L}: {wins} wins / {losses} losses "
             f"({evaluable} evaluable trial×feature pairs)")

    for metric in metrics:
        emit()
        emit(f"=== Stationarity-bucket breakdown by {metric} (mean delta per bucket; Q1=low metric, Q3=high) ===")
        for (obj, mode, L), grp in merged.groupby(["objective", "mode", "L"]):
            tag_col = f"{metric}_q"
            if tag_col not in grp.columns or grp[tag_col].isna().all():
                continue
            means = grp.groupby(tag_col, observed=True)["delta"].mean().to_dict()
            q1 = means.get("Q1", np.nan)
            q2 = means.get("Q2", np.nan)
            q3 = means.get("Q3", np.nan)
            x = grp[metric].to_numpy(dtype=float)
            y = grp["delta"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3 and np.std(x[mask]) > 0:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            else:
                slope, corr = np.nan, np.nan
            emit(f"  {obj} {mode} L={L}: Q1={q1:+.5f}  Q2={q2:+.5f}  Q3={q3:+.5f}  "
                 f"slope={slope:+.5f}  corr={corr:+.3f}  (n_pairs={int(mask.sum())})")

    emit()
    emit("=== Top-10 highest-delta features (booster cells, averaged across all trials) ===")
    booster = merged[merged["objective"] == "ridge_delta_v3"].copy()
    if not booster.empty:
        avg = (booster.groupby(["mode", "L", "feature_index", "feature_name"])
               .agg(mean_delta=("delta", "mean"),
                    pos_rate=("delta", lambda s: float((s > 0).mean())),
                    mean_base_r2=("base_r2", "mean"),
                    mean_ens_r2=("ensemble_r2", "mean"),
                    n_trials=("delta", "count"))
               .reset_index())
        for (mode, L), grp in avg.groupby(["mode", "L"]):
            emit(f"\n  {mode} L={L} top 10 by mean delta:")
            top = grp.sort_values("mean_delta", ascending=False).head(10)
            emit(top[["feature_name", "mean_base_r2", "mean_ens_r2", "mean_delta",
                      "pos_rate", "n_trials"]].to_string(
                index=False, float_format=lambda v: f"{v:+.5f}"))
            emit(f"\n  {mode} L={L} bottom 5 by mean delta:")
            bot = grp.sort_values("mean_delta").head(5)
            emit(bot[["feature_name", "mean_base_r2", "mean_ens_r2", "mean_delta",
                      "pos_rate", "n_trials"]].to_string(
                index=False, float_format=lambda v: f"{v:+.5f}"))

    (out_dir / "per_feature_summary.txt").write_text("\n".join(lines))
    print(f"\nWrote {out_dir / 'per_feature_summary.txt'}")


if __name__ == "__main__":
    main()
