"""Veer anomaly experiment (Master Idea List #2 + #3).

Part 2 established: signed surprise -> returns is a robust NULL, and
|cp_increment| -> vol is real but subsumed by option-implied vol. This
experiment tests the object neither of those touched: the DIRECTIONAL,
feature-specific, PERSISTENT structure of the model's forecast errors
("veering off course"), routed to slower / non-return channels.

Stage 1 — veer panel (per cell dump, leakage-safe)
    e[w,i,k] = realized - predicted_ensemble   (log-modulus space, mask==1)
    z[w,i,k] = (e - med_past_firm_ik) / (1.4826 * MAD_past_pooled_k)
    Scale/center come from PAST test quarters only (expanding, 4-q burn-in);
    the firm-level median removes each firm's persistent bias.
    Themed signed scores (5 themes over the 40 features) + overall RMS,
    plus a 3-quarter persistence (drift) variant per theme.

Stage 2 — targets around the announcement (rdq -> next trading day)
    d_dd    : naive Merton distance-to-default (Bharath-Shumway 2008),
              DD(+63td) - DD(-2td). Debt = as-of ANNOUNCED dlcq + 0.5*dlttq.
    d_logpe : log(mktcap / trailing-4q ibq), +63td vs -2td (as-of announced
              earnings; only for positive trailing earnings).
    d_iv    : OptionMetrics ATM 30d implied vol, +63td vs -2td (placebo-ish
              channel: options should already price expected moves).
    Each channel carries its own "already priced" controls: the -2td level
    and the pre-event delta over [-65td, -2td].

Stage 3 — routing tests per (signal x target)
    Fama-MacBeth WITH controls (primary, BH/BY/Holm over the grid),
    partial rank-IC (controls residualized within quarter), asymmetry
    split (negative vs positive veer part), and an expanding-window
    elastic-net B(X): OOS R2(controls+veers) - OOS R2(controls only).

Stage 4 — error-clustering (#3)
    Hierarchical clustering of firms by error-profile correlation vs the
    GICS sector partition (adjusted Rand), plus top-PC variance share of
    the cross-firm error correlation (common-factor threat gauge).

Guardrails: expanding-only scaling (no peeking), per-channel priced
benchmarks as controls, 50-name panel framed as hypothesis-generation.

Usage
-----
    python veer_anomaly_experiment.py <holdout_dir>
        [--cells ridge_delta_v3:2,ridge_delta_v3:4,residual_delta_v3:2,residual_delta_v3:4]
        [--burn-in 4] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from analyze_event_study_multi import (  # noqa: E402
    _spearman, _two_sided_p, _step_q,
)

EVENT_DIR = PARENT_DIR / "pre_prediction_cache" / "event_study_extended"
FUNDAMENTALS = PARENT_DIR / "90-26_Q_Fundamentals_v2_extended.csv"
GICS_FILE = PARENT_DIR / "gvkeys_to_gics.csv"

TRADING_DAYS_YR = 252
POST_OFFSET = 63          # +63 trading days (~1 quarter) after announcement
PRE_OFFSET = -2           # strictly ex-ante read
PRE_DELTA_OFFSET = -65    # start of the pre-event control delta window

# ---- feature -> theme map (all 40 tensor features) -------------------------
THEMES = {
    "leverage": [
        "Debt in Current Liabilities",
        "Long-Term Debt - Total",
        "Long-Term Debt - Issuance",
        "Financing Activities - Net Cash Flow",
        "Sale of Common and Preferred Stock",
        "Cash Dividends",
        "Preferred/Preference Stock (Capital) - Total",
        "Excess Tax Benefit of Stock Options",
    ],
    "earnings": [
        "Quarterly Income Before Extraordinary Items",
        "Annual Income Before Extraordinary Items",
        "Pretax Income",
        "Operating Income Before Depreciation",
        "Earnings Per Share (Basic)",
        "Earnings Per Share (Diluted)",
        "Comprehensive Income - Total",
        "Income Taxes",
        "Non-Operating Income (Expense) - Total",
        "Special Items",
        "Extraordinary Items",
        "Sales/Turnover (Net)",
        "Cost of Goods Sold",
    ],
    "investment": [
        "Acquisitions",
        "Capital Expenditures",
        "Investing Activities - Net Cash Flow",
        "Investing Activities - Other",
        "Sale of Investments",
        "Sale of PPE and Investments - Gain/Loss",
        "Intangible Assets - Total",
    ],
    "liquidity_bs": [
        "Assets - Other - Total",
        "Assets and Liabilities",
        "Cash and Short-Term Investments",
        "Short-Term Investments - Total",
        "Receivables - Total",
        "Inventories - Total",
        "Liabilities Netting Other Adjustments",
        "Noncontrolling Interest",
        "Stockholders Equity",
        "Common/Ordinary Equity - Total",
    ],
    "cashflow": [
        "Operating Activities - Net Cash Flow",
        "Funds from Operations - Other",
    ],
}

# target -> (level control, pre-delta control)
TARGETS = {
    "d_dd":    ("dd_pre", "d_dd_pre"),
    "d_logpe": ("logpe_pre", "d_logpe_pre"),
    "d_iv":    ("iv_pre", "d_iv_pre"),
}

DEFAULT_CELLS = ("ridge_delta_v3:2,ridge_delta_v3:4,"
                 "residual_delta_v3:2,residual_delta_v3:4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("holdout_dir", type=Path)
    p.add_argument("--cells", default=DEFAULT_CELLS,
                   help="comma-separated objective:L pairs")
    p.add_argument("--burn-in", type=int, default=4,
                   help="test quarters used only to seed the expanding scale")
    p.add_argument("--min-theme-obs", type=int, default=2,
                   help="min observed features for a themed score")
    p.add_argument("--min-q-firms", type=int, default=10)
    p.add_argument("--min-past-firm", type=int, default=2,
                   help="min past obs for the firm-level error median")
    p.add_argument("--enet-start", type=int, default=8,
                   help="first OOS quarter index for the elastic net")
    p.add_argument("--fundamentals", type=Path, default=FUNDAMENTALS)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ======================= Stage 1: veer panel ================================
def build_veer_panel(dump: dict, burn_in: int, min_theme_obs: int,
                     min_past_firm: int) -> pd.DataFrame:
    """Firm-quarter veer signals from one prediction dump (leakage-safe)."""
    realized = np.asarray(dump["realized"], dtype=float)
    pred = np.asarray(dump["predicted_ensemble"], dtype=float)
    mask = np.asarray(dump["mask"], dtype=float) > 0
    gvkeys = [str(g) for g in dump["firm_gvkeys"]]
    feats = list(dump["feature_names"])
    quarters = list(dump["quarters_test"])
    W, N, K = realized.shape

    err = np.where(mask, realized - pred, np.nan)

    theme_idx = {t: [feats.index(f) for f in fl if f in feats]
                 for t, fl in THEMES.items()}
    unmapped = set(feats) - {f for fl in THEMES.values() for f in fl}
    if unmapped:
        print(f"  WARNING: {len(unmapped)} features not in any theme: "
              f"{sorted(unmapped)}")

    z = np.full_like(err, np.nan)
    for w in range(burn_in, W):
        past = err[:w]                                   # (w, N, K) PAST only
        # firm-level expanding median (persistent-bias removal)
        n_past_firm = np.sum(np.isfinite(past), axis=0)  # (N, K)
        with np.errstate(all="ignore"):
            med_firm = np.nanmedian(past, axis=0)        # (N, K)
        med_pool = np.nanmedian(past.reshape(w * N, K), axis=0)  # (K,)
        centre = np.where(n_past_firm >= min_past_firm, med_firm,
                          med_pool[None, :])
        # pooled per-feature MAD of past centred errors
        past_dev = past - centre[None, :, :]
        mad = np.nanmedian(np.abs(past_dev).reshape(w * N, K), axis=0)  # (K,)
        scale = 1.4826 * mad
        scale[~np.isfinite(scale) | (scale <= 0)] = np.nan
        z[w] = (err[w] - centre) / scale[None, :]

    rows = []
    for w in range(burn_in, W):
        for i in range(N):
            zi = z[w, i]                       # (K,)
            row = {"gvkey": gvkeys[i], "quarter": quarters[w]}
            n_obs = int(np.isfinite(zi).sum())
            if n_obs == 0:
                continue
            row["veer_rms"] = float(np.sqrt(np.nanmean(zi ** 2)))
            row["n_feat_obs"] = n_obs
            for t, idx in theme_idx.items():
                zt = zi[idx]
                if np.isfinite(zt).sum() >= min_theme_obs:
                    row[f"veer_{t}"] = float(np.nanmean(zt))
                else:
                    row[f"veer_{t}"] = np.nan
            rows.append(row)
    panel = pd.DataFrame(rows)

    # 3-quarter persistence (drift) per theme: mean of current + 2 prior z's.
    panel = panel.sort_values(["gvkey", "quarter"]).reset_index(drop=True)
    for t in THEMES:
        panel[f"drift_{t}"] = (
            panel.groupby("gvkey")[f"veer_{t}"]
                 .transform(lambda s: s.rolling(3, min_periods=2).mean())
        )
    return panel, z, quarters, gvkeys, feats


# ======================= Stage 2: targets ===================================
def _load_link() -> pd.DataFrame:
    link = pd.read_csv(EVENT_DIR / "link_table.csv", dtype={"gvkey": str})
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce")
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2100-01-01"))
    return link


def _lookup_permno(link: pd.DataFrame, gv: str, dt: pd.Timestamp):
    if pd.isna(dt):
        return None
    cand = link[(link["gvkey"] == gv) & (link["linkdt"] <= dt) &
                (dt <= link["linkenddt"])]
    if cand.empty:
        return None
    if (cand["linkprim"] == "P").any():
        cand = cand[cand["linkprim"] == "P"]
    return int(cand.iloc[0]["permno"])


def _daily_frames() -> dict[int, pd.DataFrame]:
    """Per-permno daily frame with E ($M), sigma_E (ann.), mu (12m), iv30."""
    rets = pd.read_csv(EVENT_DIR / "daily_returns.csv",
                       usecols=["permno", "date", "ret", "prc", "shrout"])
    rets["date"] = pd.to_datetime(rets["date"])
    rets = rets.dropna(subset=["ret"])
    rets["permno"] = rets["permno"].astype(int)
    rets = rets.sort_values(["permno", "date"])
    rets["E"] = rets["prc"].abs() * rets["shrout"] / 1000.0   # $ millions

    iv = pd.read_csv(EVENT_DIR / "optionmetrics_iv.csv",
                     usecols=["permno", "date", "iv_30d"])
    iv["date"] = pd.to_datetime(iv["date"])
    iv["permno"] = iv["permno"].astype(int)
    iv = iv.dropna(subset=["iv_30d"]).drop_duplicates(["permno", "date"])
    rets = rets.merge(iv, on=["permno", "date"], how="left")

    out = {}
    for p, g in rets.groupby("permno", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        r = g["ret"].to_numpy(dtype=float)
        sig = (pd.Series(r).rolling(252, min_periods=126).std()
               .to_numpy() * np.sqrt(TRADING_DAYS_YR))
        mu = (pd.Series(np.log1p(r)).rolling(252, min_periods=126).sum()
              .pipe(np.expm1).to_numpy())
        g["sigmaE"] = sig
        g["mu"] = mu
        g["iv_30d"] = g["iv_30d"].ffill(limit=5)
        out[int(p)] = g[["date", "E", "sigmaE", "mu", "iv_30d"]]
    return out


def _naive_dd(E: np.ndarray, F: np.ndarray, sigE: np.ndarray,
              mu: np.ndarray) -> np.ndarray:
    """Bharath-Shumway (2008) naive Merton distance-to-default, T = 1y."""
    with np.errstate(all="ignore"):
        sigD = 0.05 + 0.25 * sigE
        V = E + F
        sigV = (E / V) * sigE + (F / V) * sigD
        dd = (np.log(V / F) + (mu - 0.5 * sigV ** 2)) / sigV
        dd = np.where((F > 0) & (E > 0) & (sigV > 0), dd, np.nan)
    return dd


def build_targets(panel_quarters: pd.DataFrame, fundamentals: Path,
                  gvkeys: list[str]) -> pd.DataFrame:
    """Event-aligned d_dd / d_logpe / d_iv + controls for each firm-quarter.

    panel_quarters: DataFrame with unique (gvkey, quarter) rows to score.
    """
    fund = pd.read_csv(fundamentals, dtype={"gvkey": str},
                       usecols=["gvkey", "datadate", "rdq",
                                "dlcq", "dlttq", "ibq"],
                       low_memory=False)
    fund = fund[fund["gvkey"].isin(gvkeys)]
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    fund["rdq"] = pd.to_datetime(fund["rdq"])
    fund["quarter"] = fund["datadate"].dt.to_period("Q").astype(str)
    fund = (fund.sort_values(["gvkey", "quarter", "datadate"])
                .drop_duplicates(["gvkey", "quarter"], keep="last"))
    # trailing 4-quarter earnings (needs all 4)
    fund = fund.sort_values(["gvkey", "datadate"])
    fund["ib4"] = (fund.groupby("gvkey")["ibq"]
                       .transform(lambda s: s.rolling(4, min_periods=4).sum()))
    fund["F_debt"] = fund["dlcq"].fillna(0.0) + 0.5 * fund["dlttq"].fillna(0.0)

    link = _load_link()
    daily = _daily_frames()
    market = pd.read_csv(EVENT_DIR / "daily_market.csv", usecols=["date"])
    calendar = pd.DatetimeIndex(
        sorted(pd.to_datetime(market["date"]).unique()))

    # per-gvkey announced-asof frame (rdq-sorted) for merge_asof
    asof_frames = {}
    for gv, g in fund.dropna(subset=["rdq"]).groupby("gvkey"):
        asof_frames[gv] = (g.sort_values("rdq")
                            [["rdq", "F_debt", "ib4"]].reset_index(drop=True))

    dd_cache: dict[tuple, pd.DataFrame] = {}

    def firm_series(gv: str, permno: int) -> pd.DataFrame | None:
        key = (gv, permno)
        if key in dd_cache:
            return dd_cache[key]
        if permno not in daily or gv not in asof_frames:
            dd_cache[key] = None
            return None
        d = daily[permno].copy()
        d = pd.merge_asof(d.sort_values("date"), asof_frames[gv],
                          left_on="date", right_on="rdq",
                          direction="backward")
        d["dd"] = _naive_dd(d["E"].to_numpy(), d["F_debt"].to_numpy(),
                            d["sigmaE"].to_numpy(), d["mu"].to_numpy())
        with np.errstate(all="ignore"):
            pe = d["E"].to_numpy() / d["ib4"].to_numpy()
            d["logpe"] = np.where(np.isfinite(pe) & (pe > 0), np.log(pe),
                                  np.nan)
        dd_cache[key] = d
        return d

    ev = panel_quarters.merge(
        fund[["gvkey", "quarter", "rdq"]], on=["gvkey", "quarter"],
        how="left")
    cal_arr = calendar.to_numpy()

    rows = []
    for gv, q, rdq in zip(ev["gvkey"], ev["quarter"], ev["rdq"]):
        out = {"gvkey": gv, "quarter": q}
        rows.append(out)
        if pd.isna(rdq):
            continue
        ci = np.searchsorted(cal_arr, np.datetime64(rdq), side="left")
        if ci >= len(cal_arr):
            continue
        ann = cal_arr[ci]
        out["ann_date"] = pd.Timestamp(ann)
        permno = _lookup_permno(link, gv, pd.Timestamp(ann))
        if permno is None:
            continue
        d = firm_series(gv, permno)
        if d is None:
            continue
        dates = d["date"].to_numpy()
        j = np.searchsorted(dates, ann)
        if j >= len(dates) or dates[j] != ann:
            continue
        jpre, jpost = j + PRE_OFFSET, j + POST_OFFSET
        jpre2 = j + PRE_DELTA_OFFSET
        if jpre < 0:
            continue

        def col(name, k):
            if 0 <= k < len(dates):
                v = d[name].iloc[k]
                return float(v) if np.isfinite(v) else np.nan
            return np.nan

        for name, key in (("dd", "dd"), ("logpe", "logpe"),
                          ("iv_30d", "iv")):
            pre = col(name, jpre)
            post = col(name, jpost)
            pre2 = col(name, jpre2)
            out[f"{key}_pre"] = pre
            if np.isfinite(pre) and np.isfinite(post):
                out[f"d_{key}"] = post - pre
            if np.isfinite(pre) and np.isfinite(pre2):
                out[f"d_{key}_pre"] = pre - pre2
    return pd.DataFrame(rows)


# ======================= Stage 3: routing tests =============================
def _fm_multi(sub: pd.DataFrame, ycol: str, xcols: list[str],
              report_cols: list[str], min_q_firms: int) -> dict:
    """Fama-MacBeth with controls: per-quarter OLS, t on the mean slope(s)."""
    slopes = {c: [] for c in report_cols}
    for _, g in sub.groupby("quarter"):
        gg = g[[ycol] + xcols].dropna()
        if len(gg) < max(min_q_firms, len(xcols) + 2):
            continue
        X = np.column_stack([np.ones(len(gg))] +
                            [gg[c].to_numpy(dtype=float) for c in xcols])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, gg[ycol].to_numpy(dtype=float),
                                   rcond=None)
        for c in report_cols:
            slopes[c].append(beta[1 + xcols.index(c)])
    out = {}
    for c in report_cols:
        s = np.asarray(slopes[c], dtype=float)
        if s.size < 4:
            continue
        se = s.std(ddof=1) / np.sqrt(s.size)
        t = s.mean() / se if se > 0 else np.nan
        out[c] = {"slope": float(s.mean()), "t": float(t),
                  "p": _two_sided_p(t, s.size - 1), "nq": int(s.size)}
    return out


def _partial_rank_ic(sub: pd.DataFrame, sig: str, ycol: str,
                     controls: list[str], min_q_firms: int) -> dict:
    """Quarterly Spearman of signal vs target, controls residualized within
    each quarter from BOTH sides."""
    ics = []
    for _, g in sub.groupby("quarter"):
        gg = g[[sig, ycol] + controls].dropna()
        if len(gg) < max(min_q_firms, len(controls) + 3):
            continue
        C = np.column_stack([np.ones(len(gg))] +
                            [gg[c].to_numpy(dtype=float) for c in controls])

        def resid(v):
            b, *_ = np.linalg.lstsq(C, v, rcond=None)
            return v - C @ b

        ic = _spearman(resid(gg[sig].to_numpy(dtype=float)),
                       resid(gg[ycol].to_numpy(dtype=float)))
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.asarray(ics, dtype=float)
    if ics.size < 4:
        return {}
    se = ics.std(ddof=1) / np.sqrt(ics.size)
    t = ics.mean() / se if se > 0 else np.nan
    return {"pic_mean": float(ics.mean()), "pic_t": float(t),
            "pic_p": _two_sided_p(t, ics.size - 1), "pic_nq": int(ics.size)}


def _enet_increment(df: pd.DataFrame, ycol: str, controls: list[str],
                    veers: list[str], start: int, seed: int) -> dict:
    """Expanding-window elastic net: OOS R2 with vs without veer block.

    OOS R2 uses the TRAIN mean as the baseline predictor (proper OOS
    convention). Returns dR2 and veer-selection frequencies."""
    from sklearn.linear_model import ElasticNetCV

    quarters = sorted(df["quarter"].unique())
    if len(quarters) <= start:
        return {}
    preds_c, preds_cv, actual, base = [], [], [], []
    sel_count = {v: 0 for v in veers}
    n_folds = 0
    for ti in range(start, len(quarters)):
        tr = df[df["quarter"].isin(quarters[:ti])]
        te = df[df["quarter"] == quarters[ti]]

        def fit_predict(cols):
            trg = tr[[ycol] + cols].dropna()
            teg = te[[ycol] + cols].dropna()
            if len(trg) < 60 or len(teg) < 5:
                return None
            Xtr = trg[cols].to_numpy(dtype=float)
            Xte = teg[cols].to_numpy(dtype=float)
            m, s = Xtr.mean(0), Xtr.std(0)
            s[s <= 0] = 1.0
            en = ElasticNetCV(l1_ratio=[0.5, 0.9], n_alphas=30, cv=3,
                              max_iter=5000, random_state=seed)
            en.fit((Xtr - m) / s, trg[ycol].to_numpy(dtype=float))
            return (en, en.predict((Xte - m) / s),
                    teg[ycol].to_numpy(dtype=float),
                    float(trg[ycol].mean()))

        rc = fit_predict(controls)
        rcv = fit_predict(controls + veers)
        if rc is None or rcv is None or len(rc[2]) != len(rcv[2]):
            continue
        preds_c.append(rc[1])
        preds_cv.append(rcv[1])
        actual.append(rc[2])
        base.append(np.full_like(rc[2], rc[3]))
        en_cv = rcv[0]
        for k, v in enumerate(controls + veers):
            if v in sel_count and abs(en_cv.coef_[k]) > 1e-10:
                sel_count[v] += 1
        n_folds += 1
    if n_folds < 3:
        return {}
    y = np.concatenate(actual)
    b = np.concatenate(base)
    sse_b = float(np.sum((y - b) ** 2))
    r2_c = 1.0 - float(np.sum((y - np.concatenate(preds_c)) ** 2)) / sse_b
    r2_cv = 1.0 - float(np.sum((y - np.concatenate(preds_cv)) ** 2)) / sse_b
    return {"r2_controls": r2_c, "r2_controls_veers": r2_cv,
            "dR2": r2_cv - r2_c, "n_pred": int(len(y)),
            "n_folds": n_folds,
            "picked": {v: c / n_folds for v, c in sel_count.items()}}


# ======================= Stage 4: error clustering ==========================
def error_clustering(z: np.ndarray, quarters: list[str], gvkeys: list[str],
                     feats: list[str], burn_in: int) -> dict:
    """Cluster firms by error profile; compare to GICS; common-factor gauge."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import adjusted_rand_score

    zu = z[burn_in:]                                  # (Wq, N, K)
    Wq, N, K = zu.shape
    prof = zu.transpose(1, 0, 2).reshape(N, Wq * K)   # firm x (quarter*feat)

    # pairwise correlation with min overlap
    corr = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(i, N):
            ok = np.isfinite(prof[i]) & np.isfinite(prof[j])
            if ok.sum() >= 20:
                a, b = prof[i, ok], prof[j, ok]
                if a.std() > 0 and b.std() > 0:
                    corr[i, j] = corr[j, i] = float(np.corrcoef(a, b)[0, 1])
    np.fill_diagonal(corr, 1.0)
    corr_f = np.where(np.isfinite(corr), corr, 0.0)

    gics = pd.read_csv(GICS_FILE, dtype={"gvkey": str})
    gics = (gics.sort_values(["gvkey", "datadate"])
                .drop_duplicates("gvkey", keep="last")
                .set_index("gvkey")["gsector"])
    sector = np.array([gics.get(g, -1) for g in gvkeys])

    out = {}
    dist = squareform(np.clip(1.0 - corr_f, 0.0, 2.0), checks=False)
    Zl = linkage(dist, method="average")
    valid = sector >= 0
    for k in (11, 5):
        labels = fcluster(Zl, t=k, criterion="maxclust")
        out[f"ari_k{k}"] = float(
            adjusted_rand_score(sector[valid], labels[valid]))
    # common-factor gauge: top-PC variance share of the error correlation
    eig = np.linalg.eigvalsh(corr_f)
    out["top_pc_share"] = float(eig[-1] / np.sum(np.abs(eig)))
    for t, fl in THEMES.items():
        idx = [feats.index(f) for f in fl if f in feats]
        proft = zu[:, :, idx].transpose(1, 0, 2).reshape(N, Wq * len(idx))
        ct = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                ok = np.isfinite(proft[i]) & np.isfinite(proft[j])
                if ok.sum() >= 8:
                    a, b = proft[i, ok], proft[j, ok]
                    if a.std() > 0 and b.std() > 0:
                        ct[i, j] = ct[j, i] = float(np.corrcoef(a, b)[0, 1])
        np.fill_diagonal(ct, 1.0)
        e = np.linalg.eigvalsh(ct)
        out[f"top_pc_share_{t}"] = float(e[-1] / np.sum(np.abs(e)))
    out["mean_abs_corr"] = float(np.nanmean(np.abs(
        corr[np.triu_indices(N, 1)])))
    return out


# ======================= per-cell driver ====================================
def run_cell(holdout_dir: Path, objective: str, L: int, args,
             targets_cache: dict) -> str:
    pkl = holdout_dir / f"predictions_{objective}_L{L}_rank1.pkl"
    if not pkl.exists():
        return f"[{objective} L{L}] SKIPPED: missing {pkl.name}\n"
    dump = joblib.load(pkl)
    panel, z, quarters, gvkeys, feats = build_veer_panel(
        dump, args.burn_in, args.min_theme_obs, args.min_past_firm)

    # targets are model-independent -> compute once per (gvkey, quarter) set
    key = tuple(sorted(set(zip(panel["gvkey"], panel["quarter"]))))
    if key not in targets_cache:
        targets_cache[key] = build_targets(
            panel[["gvkey", "quarter"]].drop_duplicates(),
            args.fundamentals, gvkeys)
    df = panel.merge(targets_cache[key], on=["gvkey", "quarter"], how="left")

    out_dir = args.out_dir or holdout_dir
    df.to_csv(out_dir / f"veer_panel_{objective}_L{L}.csv", index=False)

    theme_sigs = [f"veer_{t}" for t in THEMES]
    drift_sigs = [f"drift_{t}" for t in THEMES]
    all_sigs = theme_sigs + drift_sigs + ["veer_rms"]

    lines = [f"VEER ANOMALY EXPERIMENT — {objective} L{L}",
             "=" * 70,
             f"panel: {df['quarter'].nunique()} quarters "
             f"({df['quarter'].min()}..{df['quarter'].max()}), "
             f"{df['gvkey'].nunique()} firms, {len(df)} firm-quarters"]
    for tgt in TARGETS:
        n = int(df[tgt].notna().sum()) if tgt in df.columns else 0
        lines.append(f"  {tgt:8s}: {n} scored events")
    lines.append("")

    # ---- routing grid: FM with controls (primary) + partial rank-IC ----
    grid = []
    for tgt, (lvl, pre) in TARGETS.items():
        if tgt not in df.columns:
            continue
        controls = [c for c in (lvl, pre) if c in df.columns]
        for sig in all_sigs:
            sub = df[["quarter", "gvkey", sig, tgt] + controls].copy()
            sub = sub.rename(columns={sig: "_s", tgt: "_y"})
            rec = {"signal": sig, "target": tgt}
            fm = _fm_multi(sub, "_y", controls + ["_s"], ["_s"],
                           args.min_q_firms)
            if "_s" in fm:
                rec.update({"fm_slope": fm["_s"]["slope"],
                            "fm_t": fm["_s"]["t"], "fm_p": fm["_s"]["p"],
                            "fm_nq": fm["_s"]["nq"]})
            rec.update(_partial_rank_ic(sub, "_s", "_y", controls,
                                        args.min_q_firms))
            n = int(sub[["_s", "_y"]].dropna().shape[0])
            rec["n"] = n
            grid.append(rec)
    grid = pd.DataFrame(grid)
    if not grid.empty and "fm_p" in grid.columns:
        p = grid["fm_p"].to_numpy(dtype=float)
        grid["q_bh"] = _step_q(p, "bh")
        grid["q_by"] = _step_q(p, "by")
        grid["q_holm"] = _step_q(p, "holm")
    grid.to_csv(out_dir / f"veer_grid_{objective}_L{L}.csv", index=False)

    lines.append("ROUTING GRID — FM slope WITH per-channel controls "
                 "(primary), partial rank-IC")
    lines.append("-" * 70)
    if not grid.empty:
        show = grid.sort_values("fm_p").head(15)
        cols = ["signal", "target", "fm_slope", "fm_t", "fm_p", "q_bh",
                "q_by", "pic_mean", "pic_t", "n"]
        cols = [c for c in cols if c in show.columns]
        lines.append(show[cols].to_string(index=False,
                                          float_format=lambda v: f"{v:+.4f}"))
        nt = int((grid["fm_p"] < 0.05).sum())
        lines.append(f"\n  raw fm_p<0.05: {nt} / {len(grid)}")
        for qc in ("q_bh", "q_by", "q_holm"):
            if qc in grid.columns:
                lines.append(f"  survivors {qc}<0.1: "
                             f"{int((grid[qc] < 0.1).sum())} / {len(grid)}")
        both = grid[(grid["fm_p"] < 0.05) & (grid.get("pic_p", 1) < 0.05)]
        lines.append(f"  FM & partial-IC both p<0.05: {len(both)}"
                     + (f" -> {[(r.signal, r.target) for r in both.itertuples()]}"
                        if len(both) else ""))
    lines.append("")

    # ---- asymmetry: negative vs positive veer part (themed signals) ----
    lines.append("ASYMMETRY — FM slopes on neg-part vs pos-part "
                 "(deterioration vs improvement veers)")
    lines.append("-" * 70)
    for tgt, (lvl, pre) in TARGETS.items():
        if tgt not in df.columns:
            continue
        controls = [c for c in (lvl, pre) if c in df.columns]
        for sig in theme_sigs:
            sub = df[["quarter", "gvkey", sig, tgt] + controls].copy()
            sub["_neg"] = np.minimum(sub[sig], 0.0)
            sub["_pos"] = np.maximum(sub[sig], 0.0)
            sub = sub.rename(columns={tgt: "_y"})
            fm = _fm_multi(sub, "_y", controls + ["_neg", "_pos"],
                           ["_neg", "_pos"], args.min_q_firms)
            if "_neg" in fm and "_pos" in fm:
                lines.append(
                    f"  {sig:22s} -> {tgt:8s}  "
                    f"neg: t={fm['_neg']['t']:+.2f} (p={fm['_neg']['p']:.3f})"
                    f"  pos: t={fm['_pos']['t']:+.2f} "
                    f"(p={fm['_pos']['p']:.3f})")
    lines.append("")

    # ---- elastic-net incremental OOS R2 ----
    lines.append("ELASTIC-NET B(X) — expanding OOS R2, controls vs "
                 "controls+veers (train-mean baseline)")
    lines.append("-" * 70)
    for tgt, (lvl, pre) in TARGETS.items():
        if tgt not in df.columns:
            continue
        controls = [c for c in (lvl, pre) if c in df.columns]
        res = _enet_increment(df, tgt, controls, all_sigs,
                              args.enet_start, args.seed)
        if not res:
            lines.append(f"  {tgt:8s}: insufficient data")
            continue
        picked = {k: v for k, v in res["picked"].items() if v > 0}
        top = sorted(picked.items(), key=lambda kv: -kv[1])[:5]
        lines.append(
            f"  {tgt:8s}: R2(ctrl)={res['r2_controls']:+.4f}  "
            f"R2(ctrl+veer)={res['r2_controls_veers']:+.4f}  "
            f"dR2={res['dR2']:+.4f}  (n={res['n_pred']}, "
            f"folds={res['n_folds']})")
        if top:
            lines.append("           picked: " +
                         ", ".join(f"{k} {v:.0%}" for k, v in top))
    lines.append("")

    # ---- error clustering ----
    lines.append("ERROR-CLUSTERING vs GICS + common-factor gauge")
    lines.append("-" * 70)
    cl = error_clustering(z, quarters, gvkeys, feats, args.burn_in)
    lines.append(f"  adjusted Rand vs GICS sector: k=11 {cl['ari_k11']:+.3f}"
                 f"   k=5 {cl['ari_k5']:+.3f}")
    lines.append(f"  mean |error-profile corr|: {cl['mean_abs_corr']:.3f}")
    lines.append(f"  top-PC share (overall): {cl['top_pc_share']:.3f}")
    lines.append("  top-PC share per theme: " +
                 ", ".join(f"{t}={cl[f'top_pc_share_{t}']:.3f}"
                           for t in THEMES))
    lines.append("")

    report = "\n".join(lines) + "\n"
    rpt = out_dir / f"veer_report_{objective}_L{L}.txt"
    rpt.write_text(report)
    print(f"[{objective} L{L}] wrote {rpt.name}")
    return report


def main() -> None:
    args = parse_args()
    holdout_dir = args.holdout_dir.resolve()
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    targets_cache: dict = {}
    reports = []
    for cell in args.cells.split(","):
        obj, l_str = cell.strip().rsplit(":", 1)
        reports.append(run_cell(holdout_dir, obj, int(l_str), args,
                                targets_cache))
    print("\n\n".join(reports))


if __name__ == "__main__":
    main()
