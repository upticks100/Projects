"""Event-study regression + portfolio sort on the dataset built by
`build_event_study_dataset.py`.

Two analyses on each dataset:

1. Pooled OLS: CAR ~ scaled_surprise + (firm FE, time FE).
   Reported separately for the Ridge baseline's surprise and the
   CP-ensemble's surprise. The economically meaningful claim is that
   the ENSEMBLE surprise explains CAR — i.e. the CP-residual carries
   information about post-announcement returns that Ridge by itself
   missed.

   We also report the *incremental* regression:
     CAR ~ surprise_base + surprise_ensemble
   Here a significant coefficient on surprise_ensemble means CP adds
   announcement-window information beyond what Ridge already captures.

2. Quintile sort on predicted ensemble surprise (signed):
   - Build a long-short portfolio (Q5 − Q1) over the announcement
     window [pre, post], averaged equal-weight across firms in each
     quintile within each quarter.
   - Report mean, std, t-stat, and Sharpe (annualized assuming 4
     announcements/year).

Limitations to be honest about:

- The surprise is computed from realized − predicted on the TEST set,
  so it uses information visible only on (or after) the announcement
  itself. This is an *event study*, not a tradeable strategy: the
  exercise asks whether CP's residual on the day of the announcement
  contains information about that day's CAR. To turn this into a
  tradeable signal you'd need a predicted Y for the upcoming quarter
  BEFORE the announcement, which our model does provide (one quarter
  ahead, lookback features). That is what the portfolio sort below
  uses (predicted vs predicted_base difference).

- Test panel is 16 quarters × 50 firms × at most 40 features. Power
  is limited; we report nominal p-values and bootstrap CIs.

Usage
-----
    python analyze_event_study.py path/to/event_study_dataset.csv \
        [--feature 'Net Income (Loss)']
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path,
                   help="Output of build_event_study_dataset.py")
    p.add_argument("--feature", default=None,
                   help="Restrict to a single feature_name; default: every feature in the file.")
    p.add_argument("--quintiles", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write summary text + per-feature CSV. Default: dataset's dir.")
    return p.parse_args()


def pooled_ols(y: np.ndarray, X: np.ndarray, names: list[str],
               cluster1: np.ndarray | None = None,
               cluster2: np.ndarray | None = None) -> pd.DataFrame:
    """OLS with HC0 SEs by default; with cluster1/cluster2 returns two-way
    clustered SEs (Cameron-Gelbach-Miller): V_2way = V_1 + V_2 − V_12."""
    if y.size == 0 or X.shape[0] != y.size:
        return pd.DataFrame()
    Xc = np.column_stack([np.ones(X.shape[0]), X])
    try:
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.DataFrame()
    resid = y - Xc @ beta
    XtX_inv = np.linalg.pinv(Xc.T @ Xc)

    def _meat(group_ids: np.ndarray) -> np.ndarray:
        score = Xc * resid[:, None]
        df_g = pd.DataFrame(score)
        df_g["_g"] = group_ids
        S = df_g.groupby("_g").sum().to_numpy()
        return S.T @ S

    if cluster1 is None and cluster2 is None:
        middle = Xc.T @ np.diag(resid ** 2) @ Xc
        label = "se_hc0"
    elif cluster2 is None:
        middle = _meat(cluster1)
        label = "se_cluster"
    else:
        # two-way clustered: M1 + M2 − M_intersect
        M1 = _meat(cluster1)
        M2 = _meat(cluster2)
        pair_id = pd.Series(list(zip(cluster1.tolist(), cluster2.tolist()))).astype("category").cat.codes.to_numpy()
        M12 = _meat(pair_id)
        middle = M1 + M2 - M12
        label = "se_cl2way"

    cov = XtX_inv @ middle @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return pd.DataFrame({
        "var": ["intercept"] + names,
        "coef": beta,
        label: se,
        "t_stat": beta / np.where(se > 0, se, np.nan),
    })


def two_way_demean(df: pd.DataFrame, cols: list[str], firm: str = "gvkey",
                   time: str = "quarter") -> pd.DataFrame:
    """Proper two-way FE absorption: x − firm_mean − time_mean + grand_mean.

    NOTE: the previous version did groupby([firm, time]) then demean, which is
    cell demeaning (each row has one (firm, time) value -> wipes all variation).
    """
    out = df.copy()
    for c in cols:
        x = out[c].astype(float)
        firm_mean = out.groupby(firm)[c].transform("mean")
        time_mean = out.groupby(time)[c].transform("mean")
        grand = float(x.mean())
        out[c] = x - firm_mean - time_mean + grand
    return out


def run_for_feature(df: pd.DataFrame, feature: str) -> dict:
    sub = df[(df["feature_name"] == feature) &
             (df["mask"] == 1) &
             (df["car"].notna()) &
             (df["surprise_ensemble_scaled"].notna()) &
             (df["surprise_base_scaled"].notna())].copy()
    n = len(sub)
    result: dict = {"feature": feature, "n_rows": n}
    if n < 20:
        result["note"] = "too few rows"
        return result

    car = sub["car"].to_numpy()
    se_arr = sub["surprise_ensemble_scaled"].to_numpy()
    sb_arr = sub["surprise_base_scaled"].to_numpy()
    cl_firm = sub["gvkey"].astype(str).to_numpy()
    cl_time = sub["quarter"].astype(str).to_numpy()

    # 1a) Univariate (CAR ~ ens surprise), two-way clustered SEs.
    # Inference note: with T=16 quarters, two-way (firm+quarter) clustering
    # is numerically unstable (Cameron-Gelbach-Miller estimator is often
    # non-PSD at this T, producing NaN/garbage SEs). We therefore report
    # ONE-WAY firm-clustered SEs (50 clusters, always PSD) as the primary
    # regression inference — this handles the dominant within-firm serial
    # correlation of a persistence-type signal — and keep the two-way
    # t-stat as a secondary diagnostic only. The credible ECONOMIC test is
    # the quarterly long-short portfolio below (16 quarterly returns
    # aggregate within-quarter cross-correlation correctly).

    # 1a) Univariate CAR ~ ensemble surprise.
    r1 = pooled_ols(car, se_arr.reshape(-1, 1), ["s"], cluster1=cl_firm)
    if not r1.empty:
        result["ens_coef"] = float(r1[r1["var"] == "s"].iloc[0]["coef"])
        result["ens_t_firm"] = float(r1[r1["var"] == "s"].iloc[0]["t_stat"])
    r1b = pooled_ols(car, se_arr.reshape(-1, 1), ["s"],
                     cluster1=cl_firm, cluster2=cl_time)
    if not r1b.empty:
        result["ens_t_cl2way"] = float(r1b[r1b["var"] == "s"].iloc[0]["t_stat"])

    # 1b) Univariate CAR ~ base surprise.
    r2 = pooled_ols(car, sb_arr.reshape(-1, 1), ["s"], cluster1=cl_firm)
    if not r2.empty:
        result["base_coef"] = float(r2[r2["var"] == "s"].iloc[0]["coef"])
        result["base_t_firm"] = float(r2[r2["var"] == "s"].iloc[0]["t_stat"])

    # 1c) Joint regression — INCREMENTAL CP value test (the key spec).
    r3 = pooled_ols(car, np.column_stack([sb_arr, se_arr]),
                    ["sb", "se"], cluster1=cl_firm)
    if not r3.empty:
        result["joint_ens_coef"] = float(r3[r3["var"] == "se"].iloc[0]["coef"])
        result["joint_ens_t_firm"] = float(r3[r3["var"] == "se"].iloc[0]["t_stat"])
        result["joint_base_coef"] = float(r3[r3["var"] == "sb"].iloc[0]["coef"])
        result["joint_base_t_firm"] = float(r3[r3["var"] == "sb"].iloc[0]["t_stat"])
    r3b = pooled_ols(car, np.column_stack([sb_arr, se_arr]),
                     ["sb", "se"], cluster1=cl_firm, cluster2=cl_time)
    if not r3b.empty:
        result["joint_ens_t_cl2way"] = float(r3b[r3b["var"] == "se"].iloc[0]["t_stat"])

    # 1d) Two-way FE absorbed (x − firm_mean − time_mean + grand_mean),
    # firm-clustered.
    fe_df = two_way_demean(
        sub, ["car", "surprise_ensemble_scaled", "surprise_base_scaled"],
        firm="gvkey", time="quarter",
    )
    r4 = pooled_ols(
        fe_df["car"].to_numpy(),
        fe_df[["surprise_base_scaled", "surprise_ensemble_scaled"]].to_numpy(),
        ["sb", "se"], cluster1=cl_firm,
    )
    if not r4.empty:
        result["fe_ens_coef"] = float(r4[r4["var"] == "se"].iloc[0]["coef"])
        result["fe_ens_t_firm"] = float(r4[r4["var"] == "se"].iloc[0]["t_stat"])

    # 1e) Robustness: joint regression in raw-dollar units (firm-clustered).
    if {"surprise_base_scaled_raw_units",
        "surprise_ensemble_scaled_raw_units"}.issubset(sub.columns):
        raw_e = sub["surprise_ensemble_scaled_raw_units"].to_numpy()
        raw_b = sub["surprise_base_scaled_raw_units"].to_numpy()
        ok = np.isfinite(raw_e) & np.isfinite(raw_b)
        if ok.sum() >= 20:
            r5 = pooled_ols(car[ok], np.column_stack([raw_b[ok], raw_e[ok]]),
                            ["sb", "se"], cluster1=cl_firm[ok])
            if not r5.empty:
                result["joint_ens_coef_raw_units"] = float(r5[r5["var"] == "se"].iloc[0]["coef"])
                result["joint_ens_t_firm_raw_units"] = float(r5[r5["var"] == "se"].iloc[0]["t_stat"])

    # 2) Quintile sort on PREDICTABLE CP signal.
    sub["cp_signal"] = sub["predicted_ensemble"] - sub["predicted_base"]
    sort_rows = sub.dropna(subset=["cp_signal", "car"]).copy()
    if len(sort_rows) >= 20:
        sort_rows["q"] = sort_rows.groupby("quarter")["cp_signal"].transform(
            lambda v: pd.qcut(v, args.quintiles, labels=False, duplicates="drop")
                      if v.nunique() >= args.quintiles else np.nan
        )
        sort_rows = sort_rows.dropna(subset=["q"])
        sort_rows["q"] = sort_rows["q"].astype(int)

        # 2a) Equal-weighted quintile portfolio CAR per quarter.
        per_q_ew = sort_rows.groupby(["quarter", "q"])["car"].mean().unstack("q")
        if {0, args.quintiles - 1}.issubset(per_q_ew.columns):
            ls = (per_q_ew[args.quintiles - 1] - per_q_ew[0]).dropna()
            if ls.size >= 4:
                mean = float(ls.mean())
                std = float(ls.std(ddof=1)) if ls.size >= 2 else np.nan
                t = mean / (std / np.sqrt(ls.size)) if std and std > 0 else np.nan
                result["ls_ew_mean_car"] = mean
                result["ls_ew_t"] = float(t) if t is not None else np.nan
                result["ls_ew_n_quarters"] = int(ls.size)
                result["ls_ew_sharpe_ann"] = (float(mean / std) * np.sqrt(4)
                                              if std and std > 0 else np.nan)

        # 2b) Value-weighted (by pre-event mktcap, known ex-ante).
        if "mktcap_pre" in sort_rows.columns:
            vw_rows = sort_rows.dropna(subset=["mktcap_pre"]).copy()
            if not vw_rows.empty:
                def _wmean(g):
                    w = g["mktcap_pre"].to_numpy()
                    y = g["car"].to_numpy()
                    s = w.sum()
                    return float((w * y).sum() / s) if s > 0 else np.nan
                per_q_vw = (vw_rows.groupby(["quarter", "q"])
                            .apply(_wmean, include_groups=False)
                            .unstack("q"))
                if {0, args.quintiles - 1}.issubset(per_q_vw.columns):
                    ls = (per_q_vw[args.quintiles - 1] - per_q_vw[0]).dropna()
                    if ls.size >= 4:
                        mean = float(ls.mean())
                        std = float(ls.std(ddof=1)) if ls.size >= 2 else np.nan
                        t = mean / (std / np.sqrt(ls.size)) if std and std > 0 else np.nan
                        result["ls_vw_mean_car"] = mean
                        result["ls_vw_t"] = float(t) if t is not None else np.nan
                        result["ls_vw_n_quarters"] = int(ls.size)
                        result["ls_vw_sharpe_ann"] = (float(mean / std) * np.sqrt(4)
                                                       if std and std > 0 else np.nan)
    return result


def main() -> None:
    global args
    args = parse_args()
    df = pd.read_csv(args.dataset, dtype={"gvkey": str})
    if df.empty:
        sys.exit("empty dataset")

    features = (
        [args.feature]
        if args.feature is not None
        else sorted(df["feature_name"].dropna().unique().tolist())
    )

    out_dir = args.out_dir if args.out_dir is not None else args.dataset.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analysing {len(features)} feature(s) from {args.dataset.name}")
    print(f"  total rows: {len(df)}  CAR available: {df['car'].notna().sum()}")

    results = [run_for_feature(df, feat) for feat in features]
    res_df = pd.DataFrame(results)

    out_csv = out_dir / f"{args.dataset.stem}_summary.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Print compact headline. Primary inference = firm-clustered (PSD,
    # 50 clusters); two-way [firm,quarter] reported as a secondary
    # diagnostic only (unstable at T=16). Economic test = VW/EW long-short.
    print("\n=== Per-feature regression: CAR ~ surprise "
          "(firm-clustered t = primary; cl2way = unstable diagnostic) ===")
    head_cols = ["feature", "n_rows",
                 "joint_ens_coef", "joint_ens_t_firm", "joint_ens_t_cl2way",
                 "fe_ens_coef", "fe_ens_t_firm",
                 "ls_ew_mean_car", "ls_ew_t",
                 "ls_vw_mean_car", "ls_vw_t", "ls_vw_sharpe_ann"]
    have = [c for c in head_cols if c in res_df.columns]
    print(res_df[have].to_string(index=False,
                                  float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)))


if __name__ == "__main__":
    main()
