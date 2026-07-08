"""Multi-target event-study analysis with FDR control (Part 2 redesign v2).

Consumes the multi-target panel from `build_event_study_dataset.py`. For every
(feature signal x outcome target) pair it computes a battery of tests built
around the EX-ANTE CP increment

    cp_increment = predicted_ensemble - predicted_base          (signed)
    |cp_increment|                                              (magnitude)

This is the clean, transparent "incremental CP" object (audit fix): regressing a
target on [base_surprise, surprise_ensemble] is collinear because
surprise_ensemble - surprise_base = -(cp_increment); we instead regress on
[base_surprise, cp_increment].

Per (feature, target):
  reg     : pooled OLS  target ~ base + incr, firm-clustered SE (descriptive).
  within  : firm-demeaned pooled OLS (removes persistent firm effects) -- this is
            the headline for MAGNITUDE targets, killing the "high-vol firms are
            high-|signal| firms" mechanical confound (audit fix).
  fm      : Fama-MacBeth -- quarterly cross-sectional slope of target on incr,
            t on the mean across quarters. Headline for SIGNED targets.
  ic      : rank-IC -- quarterly Spearman(incr, target), mean + t.
  ls      : quintile long-short on incr (Q5-Q1 of target per quarter); for
            magnitude the target is firm-demeaned first. Block-bootstrap p too.

Signals are scaled per feature by std only (NO mean-centering), so magnitude
predictors |incr| do not peek at the test-panel mean (audit fix).

Multiple testing: q-values (BH, BY, Holm) are computed over the PRIMARY test of
each pair (FM for signed, within for magnitude), both across the whole grid
(exploratory) and within a small PRE-REGISTERED headline family.

Usage
-----
    python analyze_event_study_multi.py panel.csv [--features ALL]
        [--targets ...] [--fdr 0.10] [--out-dir DIR]
        [--headline-features f1,f2,...] [--headline-targets t1,t2,...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def pooled_ols(y: np.ndarray, X: np.ndarray, names: list[str],
               cluster1: np.ndarray | None = None,
               cluster2: np.ndarray | None = None) -> pd.DataFrame:
    """OLS with HC0 SEs by default; with cluster1/cluster2 returns two-way
    clustered SEs (Cameron-Gelbach-Miller): V_2way = V_1 + V_2 - V_12.
    Ported verbatim from the original analyze_event_study.py."""
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
        M1 = _meat(cluster1)
        M2 = _meat(cluster2)
        pair_id = pd.Series(
            list(zip(cluster1.tolist(), cluster2.tolist()))
        ).astype("category").cat.codes.to_numpy()
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


try:
    from scipy import stats as _stats

    def _two_sided_p(t, dfree):
        if t is None or not np.isfinite(t) or dfree is None or dfree < 1:
            return np.nan
        return float(2.0 * _stats.t.sf(abs(t), dfree))

    def _spearman(a, b):
        if len(a) < 4:
            return np.nan
        r = _stats.spearmanr(a, b).correlation
        return float(r) if r is not None and np.isfinite(r) else np.nan
except Exception:  # pragma: no cover
    def _two_sided_p(t, dfree):
        if t is None or not np.isfinite(t):
            return np.nan
        from math import erfc, sqrt
        return float(erfc(abs(t) / sqrt(2.0)))

    def _spearman(a, b):
        a = pd.Series(a).rank().to_numpy()
        b = pd.Series(b).rank().to_numpy()
        if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])


# Outcome target registry: name -> "signed" | "magnitude".
DEFAULT_TARGETS = {
    "car_m1_p1": "signed",
    "car_p2_p10": "signed",
    "car_p2_p30": "signed",
    "car_p2_p60": "signed",
    "ff3_car_m1_p1": "signed",
    "ff3_car_p2_p10": "signed",   # audit fix: include p2_p10 horizon
    "ff3_car_p2_p30": "signed",
    "ff3_car_p2_p60": "signed",
    "abs_car_p2_p30": "magnitude",
    "downside_car_p2_p30": "signed",
    "ff3_downside_car_p2_p30": "signed",
    "realized_vol_p2_p30": "magnitude",
    "idio_vol_p2_p30": "magnitude",
    "abn_vol_m1_p1": "magnitude",
    "abn_vol_p2_p30": "magnitude",
    "max_drawdown_p2_p60": "signed",
}

# Vol targets that get the INCREMENTAL-TO-LAGGED-VOL battery: target -> the
# strictly ex-ante pre-event vol control emitted by the builder. The economic
# question is whether the CP increment forecasts post-event vol BEYOND the firm's
# own recent vol (Bar 1: incremental to known vol predictors). The straddle-PnL
# proxy is (realized - lagged), i.e. buying a straddle priced off lagged vol.
VOL_CONTROL = {
    "realized_vol_p2_p30": "pre_vol",
    "idio_vol_p2_p30": "pre_idio_vol",
}

# Small PRE-REGISTERED headline family (canonical fundamentals x focused targets).
HEADLINE_FEATURES = [
    "Quarterly Income Before Extraordinary Items",
    "Sales/Turnover (Net)",
    "Operating Activities - Net Cash Flow",
    "Pretax Income",
    "Earnings Per Share (Diluted)",
]
HEADLINE_TARGETS = [
    "car_m1_p1", "car_p2_p30", "ff3_car_p2_p30",
    "abs_car_p2_p30", "idio_vol_p2_p30",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("--targets", default=None)
    p.add_argument("--features", default="ALL")
    p.add_argument("--quintiles", type=int, default=5)
    p.add_argument("--fdr", type=float, default=0.10)
    p.add_argument("--min-rows", type=int, default=20)
    p.add_argument("--min-q-firms", type=int, default=10,
                   help="min firms per quarter for FM / IC / sorts")
    p.add_argument("--winsor", type=float, default=0.01,
                   help="two-sided winsorization fraction for the robustness "
                        "re-estimate (per (feature,target) pool). 0 disables.")
    p.add_argument("--trim", type=float, default=0.01,
                   help="two-sided fraction of most-extreme |target| obs dropped "
                        "for the trimmed robustness re-estimate. 0 disables.")
    p.add_argument("--mad-k", type=float, default=5.0,
                   help="outlier-detection threshold in robust (MAD) units.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--headline-features", default=None)
    p.add_argument("--headline-targets", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


# ----------------------------- multiplicity --------------------------------
def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    return _step_q(pvals, kind="bh")


def by_qvalues(pvals: np.ndarray) -> np.ndarray:
    return _step_q(pvals, kind="by")


def holm_qvalues(pvals: np.ndarray) -> np.ndarray:
    return _step_q(pvals, kind="holm")


def _step_q(pvals: np.ndarray, kind: str) -> np.ndarray:
    q = np.full(pvals.shape, np.nan)
    finite = np.where(np.isfinite(pvals))[0]
    if finite.size == 0:
        return q
    p = pvals[finite]
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    if kind == "holm":
        adj = ranked * (m - np.arange(m))
        adj = np.maximum.accumulate(adj)
    else:
        c = 1.0 if kind == "bh" else float(np.sum(1.0 / (np.arange(m) + 1)))  # BY factor
        adj = ranked * m * c / (np.arange(m) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty(m)
    out[order] = adj
    q[finite] = out
    return q


# ----------------------------- helpers -------------------------------------
def _firm_demean(df: pd.DataFrame, cols, firm="gvkey") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(float) - out.groupby(firm)[c].transform("mean")
    return out


def _winsorize(s: pd.Series, p: float) -> pd.Series:
    if p <= 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _mad_z(s: pd.Series) -> pd.Series:
    """Robust z-score using median and MAD (scaled to ~N(0,1))."""
    x = s.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad <= 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - med) / (1.4826 * mad)


def _primary_stat(sub: pd.DataFrame, kind: str, args) -> tuple:
    """Compute the PRIMARY test (within for magnitude, FM for signed) on a frame
    that already has _y, _base, _incr. Returns (t, p)."""
    if kind == "magnitude":
        cl_firm = sub["gvkey"].astype(str).to_numpy()
        dm = _firm_demean(sub, ["_y", "_base", "_incr"], firm="gvkey")
        r = pooled_ols(dm["_y"].to_numpy(), dm[["_base", "_incr"]].to_numpy(),
                       ["b", "i"], cluster1=cl_firm)
        if r.empty:
            return np.nan, np.nan
        row = r[r["var"] == "i"].iloc[0]
        return float(row["t_stat"]), _two_sided_p(float(row["t_stat"]),
                                                   pd.unique(cl_firm).size - 1)
    d = _fm_slope(sub, "_y", "_incr", args.min_q_firms)
    return d.get("fm_t", np.nan), d.get("fm_p", np.nan)


def _fm_slope(sub: pd.DataFrame, ycol: str, xcol: str, min_q_firms: int):
    """Fama-MacBeth: quarterly cross-sectional slope of y on x (with intercept)."""
    slopes = []
    for _, g in sub.groupby("quarter"):
        gg = g[[ycol, xcol]].dropna()
        if len(gg) < min_q_firms:
            continue
        x = gg[xcol].to_numpy()
        if np.std(x) < 1e-12:
            continue
        X = np.column_stack([np.ones(len(gg)), x])
        beta, *_ = np.linalg.lstsq(X, gg[ycol].to_numpy(), rcond=None)
        slopes.append(beta[1])
    slopes = np.asarray(slopes, dtype=float)
    if slopes.size < 4:
        return {}
    mean = float(slopes.mean())
    se = float(slopes.std(ddof=1) / np.sqrt(slopes.size))
    t = mean / se if se > 0 else np.nan
    return {"fm_slope": mean, "fm_t": t, "fm_p": _two_sided_p(t, slopes.size - 1),
            "fm_nq": int(slopes.size)}


def _rank_ic(sub: pd.DataFrame, sig: str, tgt: str, min_q_firms: int):
    ics = []
    for _, g in sub.groupby("quarter"):
        gg = g[[sig, tgt]].dropna()
        if len(gg) < min_q_firms:
            continue
        ic = _spearman(gg[sig].to_numpy(), gg[tgt].to_numpy())
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.asarray(ics, dtype=float)
    if ics.size < 4:
        return {}
    mean = float(ics.mean())
    se = float(ics.std(ddof=1) / np.sqrt(ics.size))
    t = mean / se if se > 0 else np.nan
    return {"ic_mean": mean, "ic_t": t, "ic_p": _two_sided_p(t, ics.size - 1),
            "ic_nq": int(ics.size)}


def _resid_on(y: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Residual of y after a linear (intercept + c) projection."""
    C = np.column_stack([np.ones(len(c)), c])
    beta, *_ = np.linalg.lstsq(C, y, rcond=None)
    return y - C @ beta


def _partial_rank_ic(sub: pd.DataFrame, sig: str, tgt: str, ctrl: str,
                     min_q_firms: int):
    """Quarterly Spearman(sig, tgt) AFTER linearly partialling the control
    (lagged vol) out of BOTH sig and tgt. Outlier-robust incremental test."""
    ics = []
    for _, g in sub.groupby("quarter"):
        gg = g[[sig, tgt, ctrl]].dropna()
        if len(gg) < min_q_firms or gg[ctrl].std() < 1e-12:
            continue
        rs = _resid_on(gg[sig].to_numpy(), gg[ctrl].to_numpy())
        rt = _resid_on(gg[tgt].to_numpy(), gg[ctrl].to_numpy())
        ic = _spearman(rs, rt)
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.asarray(ics, dtype=float)
    if ics.size < 4:
        return {}
    mean = float(ics.mean())
    se = float(ics.std(ddof=1) / np.sqrt(ics.size))
    t = mean / se if se > 0 else np.nan
    return {"pic_mean": mean, "pic_t": t, "pic_p": _two_sided_p(t, ics.size - 1),
            "pic_nq": int(ics.size)}


def _incr_control_battery(sub: pd.DataFrame, target: str, ctrl: str,
                          prefix: str, args, rng) -> dict:
    """Does |CP increment| forecast post-event vol incremental to `ctrl`
    (an expected-vol benchmark: lagged realized vol or option-implied vol)?
      {prefix}ctrl_t  : pooled OLS  vol ~ ctrl + |incr|, firm-clustered
      {prefix}pic_t   : partial rank-IC (ctrl partialled out of both), robust
      {prefix}surp_ls_*: straddle-PnL-proxy LS on |incr|, target = vol - ctrl
    """
    out: dict = {}
    if not ctrl or ctrl not in sub.columns:
        return out
    cc = sub[sub[ctrl].notna()].copy()
    if len(cc) < args.min_rows or cc[ctrl].std() < 1e-12:
        return out
    out[f"{prefix}_n"] = int(len(cc))
    clc = cc["gvkey"].astype(str).to_numpy()
    Xc = np.column_stack([cc[ctrl].to_numpy(), cc["_incr"].to_numpy()])
    rc = pooled_ols(cc["_y"].to_numpy(), Xc, ["c", "i"], cluster1=clc)
    if not rc.empty:
        row = rc[rc["var"] == "i"].iloc[0]
        out[f"{prefix}ctrl_coef"] = float(row["coef"])
        out[f"{prefix}ctrl_t"] = float(row["t_stat"])
        out[f"{prefix}ctrl_p"] = _two_sided_p(out[f"{prefix}ctrl_t"],
                                              pd.unique(clc).size - 1)
    for k, v in _partial_rank_ic(cc, "_incr", target, ctrl,
                                 args.min_q_firms).items():
        out[k.replace("pic", f"{prefix}pic")] = v
    cc["_surp"] = cc["_y"].astype(float) - cc[ctrl].astype(float)
    for k, v in _ls_portfolio(cc, "_incr", "_surp", args.quintiles,
                              args.min_q_firms, args.n_boot, rng).items():
        out[f"{prefix}surp_{k}"] = v
    return out


def _block_bootstrap_p(series: np.ndarray, n_boot: int, rng) -> float:
    """Two-sided moving-block-bootstrap p for H0: mean(series)=0."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return np.nan
    bl = max(2, int(round(n ** (1.0 / 3.0))))
    n_blocks = int(np.ceil(n / bl))
    starts_pool = np.arange(0, n)  # circular
    means = np.empty(n_boot)
    xx = np.concatenate([x, x])  # wrap for circular blocks
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + bl) for s in starts])[:n]
        means[i] = xx[idx].mean()
    frac_le = float(np.mean(means <= 0.0))
    frac_ge = float(np.mean(means >= 0.0))
    return float(min(1.0, 2.0 * min(frac_le, frac_ge)))


def _ls_portfolio(sub: pd.DataFrame, sig: str, tgt: str, quintiles: int,
                  min_q_firms: int, n_boot: int, rng) -> dict:
    srt = sub[[sig, tgt, "quarter"]].dropna().copy()
    if len(srt) < 4 * quintiles:
        return {}
    srt["q"] = srt.groupby("quarter")[sig].transform(
        lambda v: pd.qcut(v, quintiles, labels=False, duplicates="drop")
        if v.nunique() >= quintiles else np.nan)
    srt = srt.dropna(subset=["q"])
    if srt.empty:
        return {}
    srt["q"] = srt["q"].astype(int)
    per_q = srt.groupby(["quarter", "q"])[tgt].mean().unstack("q")
    if not {0, quintiles - 1}.issubset(per_q.columns):
        return {}
    ls = (per_q[quintiles - 1] - per_q[0]).dropna()
    if ls.size < 4:
        return {}
    mean = float(ls.mean())
    sd = float(ls.std(ddof=1))
    t = mean / (sd / np.sqrt(ls.size)) if sd > 0 else np.nan
    return {"ls_mean": mean, "ls_t": float(t) if np.isfinite(t) else np.nan,
            "ls_p": _two_sided_p(t, ls.size - 1),
            "ls_boot_p": _block_bootstrap_p(ls.to_numpy(), n_boot, rng),
            "ls_nq": int(ls.size)}


def _add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature std-scaled (UNCENTERED) base surprise + CP increment."""
    df = df.copy()
    df["cp_increment"] = df["predicted_ensemble"] - df["predicted_base"]
    df["base_z"] = np.nan
    df["incr_z"] = np.nan
    for _, g in df.groupby("feature_name"):
        obs = g[g["mask"] == 1]
        sb = obs["surprise_base_raw"]
        si = obs["cp_increment"]
        sd_b = float(sb.std(ddof=1)) if sb.notna().sum() >= 5 else np.nan
        sd_i = float(si.std(ddof=1)) if si.notna().sum() >= 5 else np.nan
        if sd_b and sd_b > 1e-12:
            df.loc[g.index, "base_z"] = g["surprise_base_raw"] / sd_b
        if sd_i and sd_i > 1e-12:
            df.loc[g.index, "incr_z"] = g["cp_increment"] / sd_i
    return df


def analyze_pair(df: pd.DataFrame, feature: str, target: str, kind: str,
                 args, rng) -> dict:
    sub = df[(df["feature_name"] == feature) & (df["mask"] == 1) &
             df[target].notna() & df["base_z"].notna() &
             df["incr_z"].notna()].copy()
    res = {"feature": feature, "target": target, "kind": kind,
           "n_rows": int(len(sub))}
    if len(sub) < args.min_rows:
        res["note"] = "too few rows"
        return res

    # predictor: signed uses signed signal; magnitude uses |signal| (uncentered).
    if kind == "magnitude":
        sub["_base"] = sub["base_z"].abs()
        sub["_incr"] = sub["incr_z"].abs()
    else:
        sub["_base"] = sub["base_z"]
        sub["_incr"] = sub["incr_z"]
    sub["_y"] = sub[target].astype(float)
    cl_firm = sub["gvkey"].astype(str).to_numpy()

    # reg: pooled firm-clustered  y ~ base + incr  (descriptive)
    r = pooled_ols(sub["_y"].to_numpy(),
                   sub[["_base", "_incr"]].to_numpy(), ["b", "i"],
                   cluster1=cl_firm)
    if not r.empty:
        row = r[r["var"] == "i"].iloc[0]
        res["reg_coef"] = float(row["coef"])
        res["reg_t"] = float(row["t_stat"])
        res["reg_p"] = _two_sided_p(res["reg_t"], pd.unique(cl_firm).size - 1)

    # within: firm-demeaned pooled firm-clustered (confound-robust)
    dm = _firm_demean(sub, ["_y", "_base", "_incr"], firm="gvkey")
    rw = pooled_ols(dm["_y"].to_numpy(), dm[["_base", "_incr"]].to_numpy(),
                    ["b", "i"], cluster1=cl_firm)
    if not rw.empty:
        row = rw[rw["var"] == "i"].iloc[0]
        res["within_coef"] = float(row["coef"])
        res["within_t"] = float(row["t_stat"])
        res["within_p"] = _two_sided_p(res["within_t"], pd.unique(cl_firm).size - 1)

    # Fama-MacBeth + rank-IC (cross-sectional)
    res.update(_fm_slope(sub, "_y", "_incr", args.min_q_firms))
    res.update(_rank_ic(sub, "_incr", target, args.min_q_firms))

    # long-short on the ex-ante signal; magnitude uses firm-demeaned target.
    ls_sub = sub.copy()
    ls_tgt = target
    if kind == "magnitude":
        ls_sub["_y_dm"] = _firm_demean(ls_sub, [target], firm="gvkey")[target]
        ls_tgt = "_y_dm"
    res.update(_ls_portfolio(ls_sub, "_incr", ls_tgt, args.quintiles,
                             args.min_q_firms, args.n_boot, rng))

    # ---- INCREMENTAL-VOL batteries (vol targets only) ----
    # Bar 1: does |CP increment| forecast post-event vol BEYOND a benchmark
    # expected-vol? Two benchmarks: the firm's own LAGGED realized vol (prefix
    # "v", data in hand) and option-IMPLIED vol (prefix "iv", the market's
    # expected vol -- the proper straddle benchmark). Straddle-PnL proxy is
    # (realized - benchmark).
    if target in VOL_CONTROL:
        res.update(_incr_control_battery(sub, target, VOL_CONTROL[target],
                                         "v", args, rng))
        res.update(_incr_control_battery(sub, target, "pre_iv",
                                         "iv", args, rng))

    # PRIMARY test for multiplicity: FM for signed, within for magnitude.
    if kind == "magnitude":
        res["primary_test"] = "within"
        res["primary_t"] = res.get("within_t", np.nan)
        res["primary_p"] = res.get("within_p", np.nan)
    else:
        res["primary_test"] = "fm"
        res["primary_t"] = res.get("fm_t", np.nan)
        res["primary_p"] = res.get("fm_p", np.nan)

    # ---- Outlier robustness of the primary test ----
    # (1) winsorized: clip y + signals at [winsor, 1-winsor] (pooled).
    if args.winsor and args.winsor > 0:
        w = sub.copy()
        for c in ("_y", "_base", "_incr"):
            w[c] = _winsorize(w[c], args.winsor)
        res["primary_t_wins"], res["primary_p_wins"] = _primary_stat(w, kind, args)
    # (2) trimmed: drop the most-extreme |y| obs (raw target tail).
    if args.trim and args.trim > 0 and len(sub) > 40:
        thr = sub["_y"].abs().quantile(1 - args.trim)
        t = sub[sub["_y"].abs() <= thr]
        if len(t) >= args.min_rows:
            res["n_trim_dropped"] = int(len(sub) - len(t))
            res["primary_t_trim"], res["primary_p_trim"] = _primary_stat(t, kind, args)
    # (3) outlier load: share of |y| mass in the single most-extreme obs, and
    # count of robust-MAD outliers in the target.
    yabs = sub["_y"].abs()
    res["y_top1_share"] = float(yabs.max() / yabs.sum()) if yabs.sum() > 0 else np.nan
    res["n_mad_outliers"] = int((_mad_z(sub["_y"]).abs() > args.mad_k).sum())
    return res


def _apply_fdr(res: pd.DataFrame, pcol: str, prefix: str) -> pd.DataFrame:
    if pcol in res.columns:
        p = res[pcol].to_numpy(dtype=float)
        res[f"{prefix}_bh"] = bh_qvalues(p)
        res[f"{prefix}_by"] = by_qvalues(p)
        res[f"{prefix}_holm"] = holm_qvalues(p)
    return res


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.dataset, dtype={"gvkey": str})
    if df.empty:
        sys.exit("empty dataset")
    df = _add_signals(df)

    targets = ([t.strip() for t in args.targets.split(",") if t.strip()]
               if args.targets else list(DEFAULT_TARGETS.keys()))
    targets = [t for t in targets if t in df.columns]
    if not targets:
        sys.exit("no requested targets present in dataset columns")
    kinds = {t: DEFAULT_TARGETS.get(t, "signed") for t in targets}

    features = (sorted(df["feature_name"].dropna().unique().tolist())
                if args.features.strip().upper() == "ALL"
                else [s.strip() for s in args.features.split(",") if s.strip()])

    hl_feats = (HEADLINE_FEATURES if args.headline_features is None
                else [s.strip() for s in args.headline_features.split(",")])
    hl_tgts = (HEADLINE_TARGETS if args.headline_targets is None
               else [s.strip() for s in args.headline_targets.split(",")])

    out_dir = args.out_dir or args.dataset.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Multi-target scan: {len(features)} feature(s) x {len(targets)} target(s)")
    print(f"  primary test: FM (signed) / within-firm (magnitude); "
          f"BH=exploratory, BY/Holm=robustness")

    results = [analyze_pair(df, f, t, kinds[t], args, rng)
               for f in features for t in targets]
    res = pd.DataFrame(results)
    res["headline"] = res.apply(
        lambda r: (r["feature"] in hl_feats) and (r["target"] in hl_tgts), axis=1)

    # FDR over the whole grid (exploratory) on the PRIMARY p.
    res = _apply_fdr(res, "primary_p", "grid")
    # FDR within the pre-registered headline family.
    hmask = res["headline"] & res["primary_p"].notna()
    res["hl_bh"] = np.nan
    res["hl_by"] = np.nan
    if hmask.any():
        sub_h = res.loc[hmask].copy()
        sub_h = _apply_fdr(sub_h, "primary_p", "hl")
        for c in ("hl_bh", "hl_by", "hl_holm"):
            res.loc[hmask, c] = sub_h[c].to_numpy()

    out_csv = out_dir / f"{args.dataset.stem}_multitarget_summary.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(res)} pairs)")

    cols = ["feature", "target", "kind", "n_rows", "primary_test",
            "primary_t", "primary_p", "grid_bh", "grid_by",
            "primary_t_wins", "primary_t_trim", "y_top1_share", "n_mad_outliers",
            "ic_t", "within_t", "ls_t", "ls_boot_p"]
    cols = [c for c in cols if c in res.columns]

    def _fmt(d):
        return d.to_string(index=False,
                           float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v))

    # ---- PRE-REGISTERED HEADLINE ----
    print("\n" + "=" * 70)
    print("PRE-REGISTERED HEADLINE FAMILY (primary test, headline-FDR)")
    print("=" * 70)
    hl = res[res["headline"]].copy()
    if hl.empty:
        print("  (no headline pairs present in dataset)")
    else:
        hcols = [c for c in cols + ["hl_bh", "hl_by"] if c in hl.columns]
        hl = hl.reindex(hl["primary_t"].abs().sort_values(ascending=False).index)
        print(_fmt(hl[hcols]))
        surv = hl[hl["hl_by"] < args.fdr]
        print(f"\n  headline survivors at BY<{args.fdr}: {len(surv)}")

    # ---- EXPLORATORY GRID ----
    print("\n" + "=" * 70)
    print("EXPLORATORY GRID (top 20 by |primary_t|; grid-FDR)")
    print("=" * 70)
    d = res.dropna(subset=["primary_t"]).copy()
    d = d.reindex(d["primary_t"].abs().sort_values(ascending=False).index)
    print(_fmt(d[cols].head(20)))
    for fam in ("grid_bh", "grid_by"):
        if fam in d.columns:
            print(f"  survivors at {fam}<{args.fdr}: {int((d[fam] < args.fdr).sum())}")

    # ---- OUTLIER ROBUSTNESS of the exploratory survivors ----
    if "primary_t_wins" in res.columns:
        print("\n" + "=" * 70)
        print("OUTLIER ROBUSTNESS of grid survivors (raw vs winsor vs trimmed)")
        print("=" * 70)
        surv = res[(res["grid_by"] < args.fdr)].copy()
        if surv.empty:
            print("  (no grid survivors)")
        else:
            surv = surv.reindex(surv["primary_t"].abs().sort_values(ascending=False).index)
            ocols = [c for c in ["feature", "target", "kind", "primary_t",
                                 "primary_t_wins", "primary_t_trim",
                                 "y_top1_share", "n_mad_outliers", "ls_t"]
                     if c in surv.columns]
            print(_fmt(surv[ocols].head(25)))
            # how many survivors keep |t|>=2 after winsorization?
            if "primary_t_wins" in surv.columns:
                keep = int((surv["primary_t_wins"].abs() >= 2.0).sum())
                print(f"\n  grid survivors with |winsorized primary t| >= 2: "
                      f"{keep} / {len(surv)}")

    # ---- VOL-RISK: incremental-to-(lagged & implied)-vol batteries ----
    if "vctrl_t" in res.columns or "ivctrl_t" in res.columns:
        print("\n" + "=" * 70)
        print("VOL-RISK: is |CP increment| incremental to EXPECTED vol?")
        print("  v* = vs LAGGED realized vol (pre_vol) ; iv* = vs option-IMPLIED")
        print("  *ctrl_t = pooled OLS vol~bench+|incr| ; *pic_t = partial rank-IC")
        print("  *surp_ls_* = straddle-PnL-proxy LS on (realized - benchmark)")
        print("=" * 70)
        anycol = "ivctrl_t" if "ivctrl_t" in res.columns else "vctrl_t"
        vt = res[res["target"].isin(VOL_CONTROL.keys()) &
                 res[anycol].notna()].copy()
        if vt.empty:
            print("  (no vol-control results)")
        else:
            vcols = [c for c in ["feature", "target", "within_t",
                                 "vctrl_t", "vpic_t", "vsurp_ls_boot_p",
                                 "ivctrl_t", "ivpic_t",
                                 "ivsurp_ls_mean", "ivsurp_ls_t",
                                 "ivsurp_ls_boot_p"]
                     if c in vt.columns]
            vt = vt.reindex(vt[anycol].abs().sort_values(ascending=False).index)
            print(_fmt(vt[vcols].head(20)))
            n = len(vt)
            for label, tcol, pcol in [
                ("lagged-vol control (vctrl)", "vctrl_t", None),
                ("lagged-vol partial-IC (vpic)", "vpic_t", None),
                ("IMPLIED-vol control (ivctrl)", "ivctrl_t", None),
                ("IMPLIED-vol partial-IC (ivpic)", "ivpic_t", None),
            ]:
                if tcol in vt.columns:
                    k = int((vt[tcol].abs() >= 2.0).sum())
                    print(f"  |t|>=2 after {label}: {k} / {n}")
            for label, bcol in [("lagged", "vsurp_ls_boot_p"),
                                ("IMPLIED", "ivsurp_ls_boot_p")]:
                if bcol in vt.columns:
                    k = int((vt[bcol] < args.fdr).sum())
                    print(f"  straddle-proxy LS ({label} benchmark) boot "
                          f"p<{args.fdr}: {k} / {n}")

    # ---- per-target outlier detection ----
    print("\n=== per-target outlier detection (robust MAD > "
          f"{args.mad_k}) ===")
    odet = []
    for tgt in targets:
        col = df[tgt] if tgt in df.columns else None
        if col is None:
            continue
        # one row per event (use the first feature to avoid duplication)
        one = df[df["feature_name"] == features[0]]
        v = one[tgt].dropna()
        if v.empty:
            continue
        z = _mad_z(v)
        n_out = int((z.abs() > args.mad_k).sum())
        worst_idx = v.index[np.argmax(np.abs(z.to_numpy()))] if len(z) else None
        worst = one.loc[worst_idx] if worst_idx is not None else None
        odet.append({"target": tgt, "n_events": int(len(v)),
                     "n_mad_out": n_out,
                     "worst_gvkey": (worst["gvkey"] if worst is not None else ""),
                     "worst_quarter": (worst["quarter"] if worst is not None else ""),
                     "worst_value": float(v.loc[worst_idx]) if worst_idx is not None else np.nan})
    if odet:
        print(pd.DataFrame(odet).to_string(
            index=False,
            float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)))

    # ---- per-target best ----
    print("\n=== per-target best feature (by |primary_t|) ===")
    rows = []
    for tgt in targets:
        dd = res[(res["target"] == tgt) & res["primary_t"].notna()]
        if dd.empty:
            continue
        b = dd.reindex(dd["primary_t"].abs().sort_values(ascending=False).index).iloc[0]
        rows.append({"target": tgt, "kind": kinds[tgt],
                     "best_feature": b["feature"],
                     "primary_t": b["primary_t"], "primary_p": b["primary_p"],
                     "grid_by": b.get("grid_by", np.nan),
                     "ls_t": b.get("ls_t", np.nan),
                     "ls_boot_p": b.get("ls_boot_p", np.nan)})
    if rows:
        print(_fmt(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()
