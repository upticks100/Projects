"""Veer -> forward EQUITY RETURN test (declared exploratory, log 2026-07-07).

Part 2's return null covered the CP-increment signal; the veer/drift signals
were never routed to returns. Declared design (before running):

  target   fwd market-adjusted buy-hold return over [+2td, +63td] after the
           announcement (log-sum of firm rets minus log-sum of vwretd over
           the same calendar span)
  control  pre-event abnormal return over [-65td, -2td] (momentum guard)
  signal   drift_cashflow; expected sign POSITIVE
  tests    (a) FM slope with control
           (b) within-quarter tercile long-short portfolio (top - bottom
               drift_cashflow), equal weight: mean, t, annualized Sharpe
  slices   full panel + within-quarter dd_pre terciles (risky = low DD)
  cells    all 4

Usage:
    python veer_return_test.py results/v3_holdout_499_20260706 --tag 499
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from veer_anomaly_experiment import _fm_multi  # noqa: E402

PROJECT = ROOT_DIR.parent
EVENT_DIR = PROJECT / "pre_prediction_cache" / "event_study_499"
POST_START, POST_END = 2, 63
PRE_START, PRE_END = -65, -2
SIG = "drift_cashflow"
CELLS = (("ridge_delta_v3", 2), ("ridge_delta_v3", 4),
         ("residual_delta_v3", 2), ("residual_delta_v3", 4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("holdout_dir", type=Path)
    p.add_argument("--tag", default="499")
    p.add_argument("--min-q-firms", type=int, default=10)
    return p.parse_args()


def load_link_499() -> pd.DataFrame:
    # NOT veer_anomaly_experiment._load_link: that resolves against the
    # module's default (50-firm) event dir when imported.
    link = pd.read_csv(EVENT_DIR / "link_table.csv", dtype={"gvkey": str})
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"], errors="coerce")
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2100-01-01"))
    return link


def load_daily():
    rets = pd.read_csv(EVENT_DIR / "daily_returns.csv.gz",
                       usecols=["permno", "date", "ret"])
    rets["date"] = pd.to_datetime(rets["date"])
    rets = rets.dropna(subset=["ret"]).sort_values(["permno", "date"])
    firms = {}
    for p, g in rets.groupby("permno", sort=False):
        firms[int(p)] = (g["date"].to_numpy(),
                         np.log1p(g["ret"].to_numpy(dtype=float)))
    mkt = pd.read_csv(EVENT_DIR / "daily_market.csv",
                      usecols=["date", "vwretd"]).dropna()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date")
    m_dates = mkt["date"].to_numpy()
    m_clog = np.concatenate([[0.0], np.cumsum(np.log1p(
        mkt["vwretd"].to_numpy(dtype=float)))])
    return firms, m_dates, m_clog


def _mkt_logret(m_dates, m_clog, d0, d1) -> float:
    """Market log buy-hold over calendar span [d0, d1] inclusive."""
    i0 = np.searchsorted(m_dates, d0)
    i1 = np.searchsorted(m_dates, d1, side="right")
    return float(m_clog[i1] - m_clog[i0])


def build_return_targets(panel: pd.DataFrame, link: pd.DataFrame,
                         firms: dict, m_dates, m_clog) -> pd.DataFrame:
    from veer_anomaly_experiment import _lookup_permno
    rows = []
    for gv, q, ann in zip(panel["gvkey"].astype(str), panel["quarter"],
                          pd.to_datetime(panel["ann_date"])):
        out = {"gvkey": gv, "quarter": q}
        rows.append(out)
        if pd.isna(ann):
            continue
        permno = _lookup_permno(link, gv, ann)
        if permno is None or permno not in firms:
            continue
        dates, lr = firms[permno]
        j = np.searchsorted(dates, np.datetime64(ann))
        if j >= len(dates) or dates[j] != np.datetime64(ann):
            continue
        # forward window [+2, +63]
        a, b = j + POST_START, j + POST_END
        if b < len(dates):
            firm_log = float(lr[a:b + 1].sum())
            out["fwd_abret"] = firm_log - _mkt_logret(
                m_dates, m_clog, dates[a], dates[b])
        # pre window [-65, -2]
        a2, b2 = j + PRE_START, j + PRE_END
        if a2 >= 0:
            firm_log = float(lr[a2:b2 + 1].sum())
            out["pre_abret"] = firm_log - _mkt_logret(
                m_dates, m_clog, dates[a2], dates[b2])
    return pd.DataFrame(rows)


def ls_portfolio(sub: pd.DataFrame, sig: str, ycol: str) -> dict:
    """Within-quarter tercile long-short: top - bottom, equal weight."""
    rets = []
    for q, g in sub.groupby("quarter"):
        g = g.dropna(subset=[sig, ycol])
        if len(g) < 15:
            continue
        terc = pd.qcut(g[sig], 3, labels=False, duplicates="drop")
        if terc.nunique() < 3:
            continue
        rets.append(float(g.loc[terc == 2, ycol].mean()
                          - g.loc[terc == 0, ycol].mean()))
    r = np.asarray(rets, dtype=float)
    if r.size < 4:
        return {}
    se = r.std(ddof=1) / np.sqrt(r.size)
    return {"mean_q": float(r.mean()), "t": float(r.mean() / se),
            "sharpe_ann": float(r.mean() / r.std(ddof=1) * 2.0),
            "nq": int(r.size), "hit": float((r > 0).mean())}


def main() -> int:
    args = parse_args()
    link = load_link_499()
    firms, m_dates, m_clog = load_daily()
    print(f"daily returns for {len(firms)} permnos\n")

    report = []
    for obj, L in CELLS:
        panel = pd.read_csv(
            args.holdout_dir / f"veer_panel_{obj}_L{L}_{args.tag}.csv",
            dtype={"gvkey": str})
        tg = build_return_targets(panel[["gvkey", "quarter", "ann_date"]],
                                  link, firms, m_dates, m_clog)
        df = panel.merge(tg, on=["gvkey", "quarter"], how="left")
        base = df.dropna(subset=[SIG, "fwd_abret", "pre_abret"]).copy()
        base["dd_terc"] = (base.groupby("quarter")["dd_pre"]
                           .transform(lambda s: pd.qcut(
                               s, 3, labels=["risky", "mid", "safe"])
                               if s.notna().sum() >= 15 else pd.Series(
                                   pd.NA, index=s.index)))

        print(f"=== {obj} L{L}: {len(base)} events, "
              f"{base['quarter'].nunique()} quarters ===")
        row = {"objective": obj, "L": L, "n": len(base)}
        slices = [("full", base),
                  ("risky_dd", base[base["dd_terc"] == "risky"]),
                  ("safe_dd", base[base["dd_terc"] == "safe"])]
        for name, s in slices:
            fm = _fm_multi(s, "fwd_abret", [SIG, "pre_abret"], [SIG],
                           args.min_q_firms).get(SIG)
            ls = ls_portfolio(s, SIG, "fwd_abret")
            if fm:
                row[f"{name}_fm_slope"] = fm["slope"]
                row[f"{name}_fm_t"] = fm["t"]
                row[f"{name}_fm_p"] = fm["p"]
                print(f"  [{name:8s}] FM slope={fm['slope']:+.5f} "
                      f"t={fm['t']:+.2f} p={fm['p']:.3f} nq={fm['nq']}", end="")
            else:
                print(f"  [{name:8s}] FM insufficient", end="")
            if ls:
                row[f"{name}_ls_mean_q"] = ls["mean_q"]
                row[f"{name}_ls_t"] = ls["t"]
                row[f"{name}_ls_sharpe"] = ls["sharpe_ann"]
                print(f" | LS {ls['mean_q']:+.3%}/q t={ls['t']:+.2f} "
                      f"SR={ls['sharpe_ann']:+.2f} hit={ls['hit']:.0%} "
                      f"nq={ls['nq']}")
            else:
                print(" | LS insufficient")
        report.append(row)
        print()

    out = args.holdout_dir / f"veer_return_test_{args.tag}.csv"
    pd.DataFrame(report).to_csv(out, index=False)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
