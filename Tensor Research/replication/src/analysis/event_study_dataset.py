"""Join predicted/realized fundamentals with CRSP returns around `rdq` and
emit a MULTI-TARGET outcome panel in one pass (Part 2 redesign, 2026-06-29).

Inputs
------
- `--preds`: a predictions pickle produced by `dump_test_predictions.py`.
- `--fundamentals`: path to `90-25_Q_Fundamentals_v2.csv` (default).
- `--link-table`: gvkey<->permno from `pre_prediction_cache/event_study/link_table.csv`.
- `--daily-returns`: CRSP daily returns (permno, date, ret, retx, prc, vol, shrout, ...).
- `--daily-market`: CRSP daily VW market return (date, vwretd, ...).
- `--ff3`: Ken French daily factors (date, mkt_rf, smb, hml, rf), decimal units.
- `--features`: comma-separated feature names to compute surprises on.
  Default: one income feature. Pass `ALL` for every feature in the
  predictions tensor.

Signal (per gvkey x quarter x feature)
--------------------------------------
- `surprise_*_raw`        = realized - predicted   (transformed log-modulus units)
- `surprise_*_raw_units`  = realized - predicted   (inverse-transformed $/share units)
- `surprise_*_scaled[_raw_units]` = per-feature z-score of the above across the panel
- `cp_signal` (computed downstream) = predicted_ensemble - predicted_base

Outcome targets (per gvkey x quarter, repeated across features)
---------------------------------------------------------------
All event windows are trading-day offsets relative to `ann_date` (the next
trading day on/after rdq). Returns accumulate over the inclusive window.

  Market-adjusted CARs (ret - vwretd):
    car_m1_p1   [-1,+1]    3-day announcement CAR  (== legacy `car`)
    car_p2_p10  [+2,+10]   short drift
    car_p2_p30  [+2,+30]   medium drift
    car_p2_p60  [+2,+60]   long drift
    abs_car_p2_p30          repricing intensity = |car_p2_p30|
    downside_car_p2_p30     worst running cumulative abret over [+2,+30] (<=0)

  FF3 market-model abnormal returns (per-event betas from [-250,-30],
  AR_d = (ret_d - rf_d) - (alpha + b_mkt*mkt_rf + b_smb*smb + b_hml*hml)):
    ff3_car_m1_p1, ff3_car_p2_p10, ff3_car_p2_p30, ff3_car_p2_p60
    ff3_abs_car_p2_p30, ff3_downside_car_p2_p30
    ff3_beta_mkt, ff3_beta_smb, ff3_beta_hml, ff3_alpha, ff3_n_est

  Volatility (over [+2,+30]):
    realized_vol_p2_p30     std of daily raw returns (ddof=1)
    idio_vol_p2_p30         std of daily FF3 abnormal returns (ddof=1)

  Attention / information content (turnover = vol/shrout vs pre-event baseline):
    abn_vol_m1_p1   log( mean turnover[-1,+1]  / mean turnover[baseline] )
    abn_vol_p2_p30  log( mean turnover[+2,+30] / mean turnover[baseline] )
    (baseline window default [-30,-6]; ratio is unit-robust within a firm-quarter)

  Downside risk:
    max_drawdown_p2_p60     min peak-to-trough of the gross-return path (<=0)

Each target has an `_n` companion column with the number of valid trading
days used (0 / NaN => target unavailable for that event).

Legacy columns kept for back-compat with the single-target analyzer:
  car (= car_m1_p1), car_abs, pre_window, post_window, n_event_days, mktcap_pre.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config

DEFAULT_FUNDAMENTALS = config.FUNDAMENTALS_FILE
DEFAULT_EVENT_DIR = config.PRE_PRED_CACHE / "event_study"
DEFAULT_FEATURE = "Quarterly Income Before Extraordinary Items"

# Event windows (trading-day offsets relative to ann_date, inclusive).
RET_WINDOWS = {
    "m1_p1": (-1, 1),
    "p2_p10": (2, 10),
    "p2_p30": (2, 30),
    "p2_p60": (2, 60),
}
VOL_WINDOW = (2, 30)        # realized/idio vol estimated over [+2,+30]
PRE_VOL_WINDOW = (-31, -2)  # pre-event (lagged) realized/idio vol, ends before
                            # the [-1,+1] window so it is strictly ex-ante. Same
                            # ~30-day length as VOL_WINDOW for a like-for-like
                            # "did vol rise vs what was already known?" control.
DRAWDOWN_WINDOW = (2, 60)   # max drawdown over [+2,+60]
DOWNSIDE_WINDOW = (2, 30)   # downside CAR over [+2,+30]
ABNVOL_WINDOWS = {"m1_p1": (-1, 1), "p2_p30": (2, 30)}
BASELINE_WINDOW = (-30, -6)  # pre-event turnover baseline (excludes run-up)
FF3_EST_WINDOW = (-250, -30)  # FF3 beta estimation window
FF3_EST_MIN = 60             # min valid days to estimate FF3 betas
# CRSP daily source break: legacy crsp.dsf ends 2024-12-31; CIZ v2 (wrds_dsfv2)
# begins 2025-01-02. Daily volume units can differ across this boundary, so a
# turnover RATIO straddling it is flagged/voided for abnormal-volume targets.
VOL_SOURCE_BOUNDARY = np.datetime64("2025-01-01")
PRE_IV_OFFSET = -2          # pre-event implied vol read at day -2 (strictly ex-ante)
TRADING_DAYS_YR = 252.0     # annualized IV -> daily, to match realized_vol units


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, required=True,
                   help="predictions pickle from dump_test_predictions.py")
    p.add_argument("--fundamentals", type=Path, default=DEFAULT_FUNDAMENTALS)
    p.add_argument("--link-table", type=Path,
                   default=DEFAULT_EVENT_DIR / "link_table.csv")
    p.add_argument("--daily-returns", type=Path,
                   default=DEFAULT_EVENT_DIR / "daily_returns.csv")
    p.add_argument("--daily-market", type=Path,
                   default=DEFAULT_EVENT_DIR / "daily_market.csv")
    p.add_argument("--ff3", type=Path,
                   default=DEFAULT_EVENT_DIR / "ff3_daily.csv",
                   help="Ken French daily factors (date,mkt_rf,smb,hml,rf).")
    p.add_argument("--iv", type=Path,
                   default=DEFAULT_EVENT_DIR / "optionmetrics_iv.csv",
                   help="OptionMetrics ATM implied vol (permno,date,iv_30d,iv_60d). "
                        "Optional; if absent, pre_iv columns are NaN.")
    p.add_argument("--features", default=DEFAULT_FEATURE,
                   help="Comma-separated feature names. 'ALL' = every feature.")
    # legacy single-window flags (kept; control the legacy `car` column)
    p.add_argument("--pre", type=int, default=-1)
    p.add_argument("--post", type=int, default=+1)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def inv_log_modulus(y):
    """Inverse of sign(x)*log1p(|x|) (build_prediction_caches.py:81)."""
    return np.sign(y) * np.expm1(np.abs(y))


def trading_calendar(market_df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.to_datetime(market_df["date"]).unique()))


def map_to_trading_day(ann_dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Map each rdq to the next trading day (incl. itself)."""
    arr = pd.to_datetime(ann_dates).to_numpy()
    cal_arr = calendar.to_numpy()
    idx = np.searchsorted(cal_arr, arr, side="left")
    mapped = np.where(idx < len(cal_arr),
                      cal_arr[np.minimum(idx, len(cal_arr) - 1)],
                      np.datetime64("NaT"))
    too_late = idx >= len(cal_arr)
    out = pd.Series(mapped, index=ann_dates.index)
    out[too_late | ann_dates.isna()] = pd.NaT
    return out


# ----------------------------- window helpers ------------------------------
def _slice(arr: np.ndarray, centre: int, a: int, b: int):
    """Return arr[centre+a : centre+b+1] if fully in-range & finite, else None."""
    lo, hi = centre + a, centre + b
    if lo < 0 or hi >= arr.shape[0]:
        return None
    sl = arr[lo:hi + 1]
    if not np.isfinite(sl).all():
        return None
    return sl


def _win_sum(arr, centre, a, b):
    sl = _slice(arr, centre, a, b)
    if sl is None:
        return np.nan, 0
    return float(sl.sum()), int(sl.size)


def _win_std(arr, centre, a, b):
    sl = _slice(arr, centre, a, b)
    if sl is None or sl.size < 2:
        return np.nan, (0 if sl is None else int(sl.size))
    return float(sl.std(ddof=1)), int(sl.size)


def _running_trough(arr, centre, a, b):
    """Downside (left-tail) measure: the most-negative point of the running
    cumulative sum over the window, FLOORED at 0. A path that never dips below
    its starting point has downside 0 (audit fix: was returning a positive
    value when all cumulative returns were positive, contradicting the
    documented `<= 0` semantics)."""
    sl = _slice(arr, centre, a, b)
    if sl is None:
        return np.nan, 0
    trough = float(np.minimum.accumulate(np.cumsum(sl)).min())
    return min(0.0, trough), int(sl.size)


def _max_drawdown(ret_arr, centre, a, b):
    """Min peak-to-trough of the gross cumulative return path (<= 0).

    The path is prepended with entry wealth 1.0 so a drop on the FIRST in-window
    day registers as drawdown (audit fix: peak previously started at the first
    in-window gross return, hiding a first-day decline)."""
    sl = _slice(ret_arr, centre, a, b)
    if sl is None:
        return np.nan, 0
    gross = np.concatenate([[1.0], np.cumprod(1.0 + sl)])
    peak = np.maximum.accumulate(gross)
    dd = gross / peak - 1.0
    return float(dd.min()), int(sl.size)


def _abn_volume(turn_arr, centre, a, b, base_a, base_b, min_base=10):
    """log( mean turnover[event] / mean turnover[baseline] ).

    Audit fix: the EVENT window must be fully in-range and finite (consistent
    with the CAR/vol windows, which use `_slice`) so the headline measure is not
    computed on a clipped partial window. The pre-event baseline may be partial
    but must have at least `min_base` valid (finite, positive) days.
    """
    evt = _slice(turn_arr, centre, a, b)
    if evt is None or not bool((evt > 0).all()):
        return np.nan, 0
    lo = max(centre + base_a, 0)
    hi = min(centre + base_b, turn_arr.shape[0] - 1)
    if hi < lo:
        return np.nan, 0
    base = turn_arr[lo:hi + 1]
    base = base[np.isfinite(base) & (base > 0)]
    if base.size < min_base:
        return np.nan, 0
    return float(np.log(evt.mean() / base.mean())), int(evt.size)


def _vol_source_straddle(dates, centre, lo_off, hi_off) -> bool:
    """True if the [lo_off, hi_off] window around `centre` spans the CRSP daily
    source break (legacy crsp.dsf ends 2024-12-31; CIZ v2 starts 2025-01-02).
    Turnover units can differ across the break, so an abnormal-volume RATIO whose
    baseline and event sit on opposite sides is contaminated (audit fix)."""
    lo = max(centre + lo_off, 0)
    hi = min(centre + hi_off, dates.shape[0] - 1)
    if hi < lo:
        return False
    span = dates[lo:hi + 1]
    return bool((span < VOL_SOURCE_BOUNDARY).any() and (span >= VOL_SOURCE_BOUNDARY).any())


def _ff3_betas(exret, F, centre):
    """OLS exret ~ [1, mkt_rf, smb, hml] over the estimation window.

    Returns (alpha, b_mkt, b_smb, b_hml, n_est) or (nan*4, n_est) if too few
    valid days.
    """
    a, b = FF3_EST_WINDOW
    lo, hi = centre + a, centre + b
    if lo < 0 or hi >= exret.shape[0]:
        # allow a left-truncated estimation window if enough days remain
        lo = max(lo, 0)
        hi = min(hi, exret.shape[0] - 1)
    if hi <= lo:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    y = exret[lo:hi + 1]
    Xf = F[lo:hi + 1]
    ok = np.isfinite(y) & np.isfinite(Xf).all(axis=1)
    n = int(ok.sum())
    if n < FF3_EST_MIN:
        return (np.nan, np.nan, np.nan, np.nan, n)
    yk = y[ok]
    Xk = np.column_stack([np.ones(n), Xf[ok]])
    beta, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
    return (float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3]), n)


def compute_event_targets(centre, ret, abret_mkt, exret, F, turnover, dates,
                          iv30=None) -> dict:
    """All outcome targets for one (gvkey, quarter) event at index `centre`."""
    out: dict = {}

    # FF3 market-model abnormal returns over the whole series (per-event betas).
    alpha, bm, bs, bh, n_est = _ff3_betas(exret, F, centre)
    out.update(ff3_alpha=alpha, ff3_beta_mkt=bm, ff3_beta_smb=bs,
               ff3_beta_hml=bh, ff3_n_est=n_est)
    if np.isfinite([alpha, bm, bs, bh]).all():
        ar_ff3 = exret - (alpha + bm * F[:, 0] + bs * F[:, 1] + bh * F[:, 2])
    else:
        ar_ff3 = np.full_like(exret, np.nan)

    # Multi-horizon CARs (market-adjusted + FF3).
    for name, (a, b) in RET_WINDOWS.items():
        v, n = _win_sum(abret_mkt, centre, a, b)
        out[f"car_{name}"] = v
        out[f"car_{name}_n"] = n
        vf, nf = _win_sum(ar_ff3, centre, a, b)
        out[f"ff3_car_{name}"] = vf
        out[f"ff3_car_{name}_n"] = nf

    out["abs_car_p2_p30"] = abs(out["car_p2_p30"]) if np.isfinite(out["car_p2_p30"]) else np.nan
    out["ff3_abs_car_p2_p30"] = abs(out["ff3_car_p2_p30"]) if np.isfinite(out["ff3_car_p2_p30"]) else np.nan

    # Downside CARs (worst running cumulative abnormal return).
    da, db = DOWNSIDE_WINDOW
    v, n = _running_trough(abret_mkt, centre, da, db)
    out["downside_car_p2_p30"] = v
    out["downside_car_p2_p30_n"] = n
    vf, nf = _running_trough(ar_ff3, centre, da, db)
    out["ff3_downside_car_p2_p30"] = vf
    out["ff3_downside_car_p2_p30_n"] = nf

    # Volatility (post-event [+2,+30]).
    va, vb = VOL_WINDOW
    rv, nrv = _win_std(ret, centre, va, vb)
    out["realized_vol_p2_p30"] = rv
    out["realized_vol_p2_p30_n"] = nrv
    iv, niv = _win_std(ar_ff3, centre, va, vb)
    out["idio_vol_p2_p30"] = iv
    out["idio_vol_p2_p30_n"] = niv

    # Pre-event (lagged) volatility over [-31,-2]: strictly ex-ante control so we
    # can ask whether the CP increment forecasts post-event vol INCREMENTAL to the
    # firm's own recent vol (and a straddle-PnL proxy: realized - lagged).
    pa, pb = PRE_VOL_WINDOW
    prv, nprv = _win_std(ret, centre, pa, pb)
    out["pre_vol"] = prv
    out["pre_vol_n"] = nprv
    piv, npiv = _win_std(ar_ff3, centre, pa, pb)
    out["pre_idio_vol"] = piv
    out["pre_idio_vol_n"] = npiv

    # Pre-event option-IMPLIED vol (ATM 30d), read at day -2, converted from
    # annualized to a daily std comparable to realized_vol. This is the market's
    # expected vol benchmark -> straddle-PnL proxy = realized - implied.
    out["pre_iv"] = np.nan
    out["pre_iv_ann"] = np.nan
    if iv30 is not None:
        j = centre + PRE_IV_OFFSET
        if 0 <= j < iv30.shape[0]:
            iv_ann = float(iv30[j])
            if np.isfinite(iv_ann) and iv_ann > 0:
                out["pre_iv_ann"] = iv_ann
                out["pre_iv"] = iv_ann / np.sqrt(TRADING_DAYS_YR)

    # Drawdown.
    da2, db2 = DRAWDOWN_WINDOW
    dd, ndd = _max_drawdown(ret, centre, da2, db2)
    out["max_drawdown_p2_p60"] = dd
    out["max_drawdown_p2_p60_n"] = ndd

    # Abnormal turnover. Voided when the baseline<->event span straddles the
    # CRSP daily source break (turnover-unit discontinuity); flagged for audit.
    ba, bb = BASELINE_WINDOW
    for name, (a, b) in ABNVOL_WINDOWS.items():
        straddle = _vol_source_straddle(dates, centre, ba, b)
        out[f"abn_vol_{name}_src_straddle"] = int(straddle)
        if straddle:
            out[f"abn_vol_{name}"] = np.nan
            out[f"abn_vol_{name}_n"] = 0
        else:
            v, n = _abn_volume(turnover, centre, a, b, ba, bb)
            out[f"abn_vol_{name}"] = v
            out[f"abn_vol_{name}_n"] = n

    return out


def main() -> None:
    args = parse_args()

    # ---- Load predictions ----
    preds = joblib.load(args.preds)
    pred_ens = preds["predicted_ensemble"]      # (W_test, F, K)
    pred_base = preds["predicted_base"]
    realized = preds["realized"]
    mask = preds["mask"]
    gvkeys = [str(g) for g in preds["firm_gvkeys"]]
    feat_names = list(preds["feature_names"])
    quarters_test = [str(q) for q in preds["quarters_test"]]
    trial = preds["trial_meta"]

    W, F, K = pred_ens.shape
    assert F == len(gvkeys), (F, len(gvkeys))
    assert K == len(feat_names), (K, len(feat_names))
    assert W == len(quarters_test), (W, len(quarters_test))

    requested_feats = (
        feat_names if args.features.strip().upper() == "ALL"
        else [s.strip() for s in args.features.split(",") if s.strip()]
    )
    missing = [f for f in requested_feats if f not in feat_names]
    if missing:
        sys.exit(f"requested features not in predictions: {missing[:5]}")
    feat_to_idx = {f: i for i, f in enumerate(feat_names)}
    feat_indices = [feat_to_idx[f] for f in requested_feats]

    # ---- Load fundamentals to get rdq ----
    fund_cols = ["gvkey", "datadate", "fyearq", "fqtr", "rdq", "tic", "conm"]
    fund = pd.read_csv(args.fundamentals, dtype={"gvkey": str},
                       usecols=fund_cols, low_memory=False)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    fund["rdq"] = pd.to_datetime(fund["rdq"])
    fund["quarter"] = fund["datadate"].dt.to_period("Q").astype(str)
    fund = fund[fund["gvkey"].isin(gvkeys)]
    fund = fund.sort_values(["gvkey", "quarter", "datadate"]).drop_duplicates(
        ["gvkey", "quarter"], keep="last"
    )

    # ---- Guard: fundamentals must cover every test-target quarter ----
    # (audit fix) Catch the stale-data class of bug where the predictions span
    # quarters the fundamentals file does not contain, which would silently emit
    # blank-rdq / zero-target rows for the missing quarters.
    fund_qmax = fund["quarter"].max()
    missing_q = sorted(q for q in set(quarters_test)
                       if pd.Period(q, "Q") > pd.Period(fund_qmax, "Q"))
    if missing_q:
        sys.exit(
            "FATAL: fundamentals file "
            f"{args.fundamentals.name} ends at {fund_qmax}, but predictions "
            f"cover later test quarters {missing_q}. Pass the extended "
            "--fundamentals (e.g. 90-26_Q_Fundamentals_v2_extended.csv) and the "
            "matching --daily-* / --ff3 from event_study_extended."
        )
    # Soft warning: any test quarter with zero matched fundamentals rows.
    empty_q = [q for q in quarters_test
               if fund[fund["quarter"] == q].empty]
    if empty_q:
        print(f"  WARNING: {len(empty_q)} test quarters have 0 fundamentals "
              f"rows (no events): {empty_q}")

    # ---- Load link table, daily returns, market, FF3 ----
    link = pd.read_csv(args.link_table, dtype={"gvkey": str})
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce")
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2100-01-01"))

    market = pd.read_csv(args.daily_market)
    market["date"] = pd.to_datetime(market["date"])
    calendar = trading_calendar(market)
    market = market[["date", "vwretd"]].set_index("date").sort_index()

    ff3 = pd.read_csv(args.ff3)
    ff3["date"] = pd.to_datetime(ff3["date"])
    ff3 = ff3[["date", "mkt_rf", "smb", "hml", "rf"]].set_index("date").sort_index()

    rets = pd.read_csv(args.daily_returns)
    rets["date"] = pd.to_datetime(rets["date"])
    keep = ["permno", "date", "ret", "prc", "shrout"]
    if "vol" in rets.columns:
        keep.append("vol")
    rets = rets[keep].dropna(subset=["ret"])
    rets["permno"] = rets["permno"].astype(int)
    rets = rets.sort_values(["permno", "date"])

    # CRSP convention: prc < 0 = bid/ask midpoint -> use |prc| for mktcap.
    rets["mktcap"] = rets["prc"].abs() * rets["shrout"]
    if "vol" in rets.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            rets["turnover"] = np.where(rets["shrout"] > 0,
                                        rets["vol"] / rets["shrout"], np.nan)
    else:
        rets["turnover"] = np.nan

    rets = rets.merge(market, left_on="date", right_index=True, how="left")
    rets = rets.merge(ff3, left_on="date", right_index=True, how="left")
    rets["abret_mkt"] = rets["ret"] - rets["vwretd"]
    rets["exret"] = rets["ret"] - rets["rf"]

    # ---- Optional: OptionMetrics ATM 30d implied vol, aligned to the daily grid.
    # Left-merge on (permno, date) then forward-fill within firm (limit 5 trading
    # days) so a missing IV day uses only PAST information (stays ex-ante).
    rets["iv_30d"] = np.nan
    if args.iv is not None and Path(args.iv).exists():
        iv = pd.read_csv(args.iv, usecols=lambda c: c in
                         {"permno", "date", "iv_30d"})
        iv["date"] = pd.to_datetime(iv["date"])
        iv["permno"] = iv["permno"].astype(int)
        iv = iv.dropna(subset=["iv_30d"]).drop_duplicates(["permno", "date"])
        rets = rets.drop(columns=["iv_30d"]).merge(
            iv[["permno", "date", "iv_30d"]], on=["permno", "date"], how="left")
        rets = rets.sort_values(["permno", "date"])
        rets["iv_30d"] = rets.groupby("permno")["iv_30d"].ffill(limit=5)
        cov = float(rets["iv_30d"].notna().mean())
        print(f"  implied vol merged: {iv['permno'].nunique()} permnos, "
              f"daily-grid coverage {cov*100:.1f}%", flush=True)
    else:
        print(f"  NOTE: no IV file at {args.iv}; pre_iv will be NaN", flush=True)

    # Per-permno aligned arrays (sorted by date).
    permno_arrays: dict[int, dict] = {}
    for p, grp in rets.groupby("permno", sort=False):
        permno_arrays[int(p)] = {
            "dates": grp["date"].to_numpy(),
            "ret": grp["ret"].to_numpy(dtype=float),
            "abret_mkt": grp["abret_mkt"].to_numpy(dtype=float),
            "exret": grp["exret"].to_numpy(dtype=float),
            "F": grp[["mkt_rf", "smb", "hml"]].to_numpy(dtype=float),
            "turnover": grp["turnover"].to_numpy(dtype=float),
            "mktcap": grp["mktcap"].to_numpy(dtype=float),
            "iv30": grp["iv_30d"].to_numpy(dtype=float),
        }

    # ---- ann_date + date-aware permno (active on ANN_DATE) ----
    fund["ann_date"] = map_to_trading_day(fund["rdq"], calendar)

    def lookup_permno(gv: str, dt: pd.Timestamp):
        if pd.isna(dt):
            return np.nan
        cand = link[(link["gvkey"] == gv) &
                    (link["linkdt"] <= dt) &
                    (dt <= link["linkenddt"])]
        if cand.empty:
            return np.nan
        if (cand["linkprim"] == "P").any():
            cand = cand[cand["linkprim"] == "P"]
        return int(cand.iloc[0]["permno"])

    permno_anndt = [lookup_permno(g, a) if pd.notna(a) else lookup_permno(g, d)
                    for g, d, a in zip(fund["gvkey"], fund["datadate"], fund["ann_date"])]
    permno_datadate = [lookup_permno(g, d) for g, d in zip(fund["gvkey"], fund["datadate"])]
    fund["permno"] = permno_anndt
    fund["permno_datadate"] = permno_datadate
    diff_n = int(sum(1 for a, d in zip(permno_anndt, permno_datadate)
                     if (a is not None) and (d is not None)
                     and not (pd.isna(a) and pd.isna(d))
                     and a != d))
    if diff_n:
        print(f"  NOTE: {diff_n} gvkey-quarter rows have permno_anndt != "
              f"permno_datadate (link change between quarter-end and rdq).")

    # Target column order (for stable output schema).
    target_cols = []
    for name in RET_WINDOWS:
        target_cols += [f"car_{name}", f"car_{name}_n",
                        f"ff3_car_{name}", f"ff3_car_{name}_n"]
    target_cols += ["abs_car_p2_p30", "ff3_abs_car_p2_p30",
                    "downside_car_p2_p30", "downside_car_p2_p30_n",
                    "ff3_downside_car_p2_p30", "ff3_downside_car_p2_p30_n",
                    "realized_vol_p2_p30", "realized_vol_p2_p30_n",
                    "idio_vol_p2_p30", "idio_vol_p2_p30_n",
                    "pre_vol", "pre_vol_n", "pre_idio_vol", "pre_idio_vol_n",
                    "pre_iv", "pre_iv_ann",
                    "max_drawdown_p2_p60", "max_drawdown_p2_p60_n"]
    for name in ABNVOL_WINDOWS:
        target_cols += [f"abn_vol_{name}", f"abn_vol_{name}_n",
                        f"abn_vol_{name}_src_straddle"]
    target_cols += ["ff3_alpha", "ff3_beta_mkt", "ff3_beta_smb",
                    "ff3_beta_hml", "ff3_n_est"]

    def empty_targets() -> dict:
        return {c: (0 if c.endswith("_n") or c == "ff3_n_est"
                    or c.endswith("_src_straddle") else np.nan)
                for c in target_cols}

    # ---- Walk test windows x features ----
    rows = []
    pre, post = int(args.pre), int(args.post)
    for w_idx, qstr in enumerate(quarters_test):
        sub = fund[fund["quarter"] == qstr].copy()
        sub_by_gv = sub.set_index("gvkey")
        for i, gv in enumerate(gvkeys):
            if gv not in sub_by_gv.index:
                row_meta = {"datadate": pd.NaT, "rdq": pd.NaT, "ann_date": pd.NaT,
                            "permno": np.nan, "tic": "", "conm": ""}
            else:
                r = sub_by_gv.loc[gv]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                row_meta = {"datadate": r["datadate"], "rdq": r["rdq"],
                            "ann_date": r["ann_date"], "permno": r["permno"],
                            "tic": r.get("tic", ""), "conm": r.get("conm", "")}

            permno = row_meta["permno"]
            ann = row_meta["ann_date"]
            tgt = empty_targets()
            mktcap_pre = np.nan
            if pd.notna(permno) and pd.notna(ann) and int(permno) in permno_arrays:
                arr = permno_arrays[int(permno)]
                dates_p = arr["dates"]
                centre = int(np.searchsorted(dates_p, np.datetime64(ann)))
                if centre < len(dates_p) and dates_p[centre] == np.datetime64(ann):
                    tgt = compute_event_targets(
                        centre, arr["ret"], arr["abret_mkt"], arr["exret"],
                        arr["F"], arr["turnover"], dates_p, arr.get("iv30"),
                    )
                    # mktcap one trading day before the announcement window.
                    weight_idx = max(centre + pre - 1, 0)
                    wc = arr["mktcap"][weight_idx]
                    if np.isfinite(wc) and wc > 0:
                        mktcap_pre = float(wc)

            # legacy single-window car columns
            car = tgt.get("car_m1_p1", np.nan)
            n_event = tgt.get("car_m1_p1_n", 0)

            for f_idx in feat_indices:
                fname = feat_names[f_idx]
                base = {
                    "objective": trial.get("objective"),
                    "mode": trial.get("mode"),
                    "L": int(trial.get("L", 0)),
                    "rank_order": int(trial.get("rank_order", 0)),
                    "trial_number": int(trial.get("trial_number", 0)),
                    "gvkey": gv,
                    "permno": (int(permno) if pd.notna(permno) else np.nan),
                    "conm": row_meta["conm"],
                    "tic": row_meta["tic"],
                    "quarter": qstr,
                    "datadate": row_meta["datadate"],
                    "rdq": row_meta["rdq"],
                    "ann_date": row_meta["ann_date"],
                    "feature_name": fname,
                    "predicted_base": float(pred_base[w_idx, i, f_idx]),
                    "predicted_ensemble": float(pred_ens[w_idx, i, f_idx]),
                    "realized": float(realized[w_idx, i, f_idx]),
                    "mask": int(mask[w_idx, i, f_idx]),
                    "surprise_base_raw": float(
                        realized[w_idx, i, f_idx] - pred_base[w_idx, i, f_idx]),
                    "surprise_ensemble_raw": float(
                        realized[w_idx, i, f_idx] - pred_ens[w_idx, i, f_idx]),
                    "predicted_base_raw_units": float(
                        inv_log_modulus(pred_base[w_idx, i, f_idx])),
                    "predicted_ensemble_raw_units": float(
                        inv_log_modulus(pred_ens[w_idx, i, f_idx])),
                    "realized_raw_units": float(
                        inv_log_modulus(realized[w_idx, i, f_idx])),
                    "surprise_base_raw_units": float(
                        inv_log_modulus(realized[w_idx, i, f_idx])
                        - inv_log_modulus(pred_base[w_idx, i, f_idx])),
                    "surprise_ensemble_raw_units": float(
                        inv_log_modulus(realized[w_idx, i, f_idx])
                        - inv_log_modulus(pred_ens[w_idx, i, f_idx])),
                    # legacy
                    "car": car,
                    "mktcap_pre": mktcap_pre,
                    "n_event_days": n_event,
                    "pre_window": pre,
                    "post_window": post,
                }
                base.update(tgt)
                rows.append(base)

    df = pd.DataFrame(rows)

    # Per-feature z-scored surprises (transformed + raw units).
    for kind in ("base", "ensemble"):
        for unit_suffix, src_col in (("scaled", f"surprise_{kind}_raw"),
                                     ("scaled_raw_units",
                                      f"surprise_{kind}_raw_units")):
            z = pd.Series(np.nan, index=df.index)
            for fname, grp in df.groupby("feature_name"):
                obs = grp[(grp["mask"] == 1) & grp[src_col].notna()][src_col]
                if obs.size >= 5:
                    sd = float(obs.std(ddof=1))
                    if sd > 1e-12:
                        z.loc[grp.index] = (grp[src_col] - obs.mean()) / sd
            df[f"surprise_{kind}_{unit_suffix}"] = z

    df["car_abs"] = df["car"].abs()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {args.out}")
    print(f"  rows: {len(df)}  features: {df['feature_name'].nunique()}  "
          f"quarters: {df['quarter'].nunique()}  gvkeys: {df['gvkey'].nunique()}")
    # Per-target availability (mask=1 rows of one feature = unique events).
    one_feat = df[df["feature_name"] == requested_feats[0]]
    n_events = len(one_feat)
    print(f"  unique events (one feature): {n_events}")
    print("  target availability (finite / events):")
    headline = ["car_m1_p1", "car_p2_p30", "car_p2_p60", "ff3_car_p2_p30",
                "abs_car_p2_p30", "realized_vol_p2_p30", "idio_vol_p2_p30",
                "abn_vol_m1_p1", "abn_vol_p2_p30", "max_drawdown_p2_p60",
                "downside_car_p2_p30"]
    for c in headline:
        if c in one_feat.columns:
            frac = one_feat[c].notna().mean()
            print(f"    {c:<26s} {one_feat[c].notna().sum():>5d} ({frac:.0%})")
    if "ff3_n_est" in one_feat.columns:
        ok = (one_feat["ff3_n_est"] >= FF3_EST_MIN).mean()
        print(f"  FF3 betas estimable: {ok:.0%} of events "
              f"(median n_est={one_feat['ff3_n_est'].median():.0f})")


if __name__ == "__main__":
    main()
