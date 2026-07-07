"""H1 economic translation on market CDS spreads (pre-registered extension).

RESEARCH_LOG 2026-07-07: the confirmed drift_cashflow -> d_dd effect gets its
magnitude translated into market credit-spread units. Declared before running:

  target    d_logcds = log parspread(+63td) - log parspread(-2td) around the
            SAME announcement dates already in the veer panels
  controls  logcds_pre (level at -2td), d_logcds_pre (log change -65td..-2td)
  test      FM slope of drift_cashflow with controls, all 4 cells
  sign      expected NEGATIVE (cash-flow over-performance -> tightening)
  output    bp and % tightening per 1-sd drift_cashflow at the median spread

Data: pre_prediction_cache/event_study_499/cds_markit.csv.gz (Markit 5Y USD
SNRFOR XR14, 189 matched gvkeys). Trading-day offsets on the CRSP market
calendar; CDS quotes forward-filled onto that calendar with a 5-day limit.
Ticker `BR` is excluded: the link audit shows Markit's BR is Burlington
Resources (dead ticker reused by Broadridge) — confirmed false match.

Usage:
    python cds_h1_translation.py results/v3_holdout_499_20260706
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from veer_anomaly_experiment import (  # noqa: E402
    _fm_multi,
    _partial_rank_ic,
)

PROJECT = ROOT_DIR.parent
EVENT_DIR = PROJECT / "pre_prediction_cache" / "event_study_499"
POST_OFFSET, PRE_OFFSET, PRE_DELTA_OFFSET = 63, -2, -65
FFILL_LIMIT = 5
BAD_TICKERS = {"BR"}          # confirmed false ticker-reuse match
CELLS = (("ridge_delta_v3", 2), ("ridge_delta_v3", 4),
         ("residual_delta_v3", 2), ("residual_delta_v3", 4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("holdout_dir", type=Path)
    p.add_argument("--tag", default="499")
    p.add_argument("--min-q-firms", type=int, default=10)
    return p.parse_args()


def load_cds_on_calendar() -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """Per-gvkey log-spread series reindexed to the CRSP trading calendar."""
    market = pd.read_csv(EVENT_DIR / "daily_market.csv", usecols=["date"])
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(market["date"]).unique()))

    cds = pd.read_csv(EVENT_DIR / "cds_markit.csv.gz",
                      usecols=["gvkey", "ticker", "date", "parspread"],
                      dtype={"gvkey": str})
    cds = cds[~cds["ticker"].isin(BAD_TICKERS)]
    cds["date"] = pd.to_datetime(cds["date"])

    frames = {}
    for gv, g in cds.groupby("gvkey"):
        s = (g.set_index("date")["parspread"]
              .reindex(calendar)
              .ffill(limit=FFILL_LIMIT))
        frames[gv] = np.log(s.to_numpy())
    return frames, calendar.to_numpy()


def build_cds_targets(panel: pd.DataFrame, logcds: dict[str, np.ndarray],
                      cal: np.ndarray) -> pd.DataFrame:
    rows = []
    for gv, q, ann in zip(panel["gvkey"].astype(str), panel["quarter"],
                          pd.to_datetime(panel["ann_date"])):
        out = {"gvkey": gv, "quarter": q}
        rows.append(out)
        if pd.isna(ann) or gv not in logcds:
            continue
        j = np.searchsorted(cal, np.datetime64(ann))
        if j >= len(cal) or cal[j] != np.datetime64(ann):
            continue
        s = logcds[gv]

        def at(k):
            if 0 <= k < len(s) and np.isfinite(s[k]):
                return float(s[k])
            return np.nan

        pre = at(j + PRE_OFFSET)
        post = at(j + POST_OFFSET)
        pre2 = at(j + PRE_DELTA_OFFSET)
        out["logcds_pre"] = pre
        if np.isfinite(pre) and np.isfinite(post):
            out["d_logcds"] = post - pre
        if np.isfinite(pre) and np.isfinite(pre2):
            out["d_logcds_pre"] = pre - pre2
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    logcds, cal = load_cds_on_calendar()
    print(f"CDS series for {len(logcds)} gvkeys on {len(cal)}-day calendar\n")

    controls = ["logcds_pre", "d_logcds_pre"]
    sig = "drift_cashflow"
    report_rows = []
    for obj, L in CELLS:
        panel_path = (args.holdout_dir /
                      f"veer_panel_{obj}_L{L}_{args.tag}.csv")
        panel = pd.read_csv(panel_path, dtype={"gvkey": str})
        tg = build_cds_targets(panel[["gvkey", "quarter", "ann_date"]],
                               logcds, cal)
        df = panel.merge(tg, on=["gvkey", "quarter"], how="left")
        sub = df.dropna(subset=[sig, "d_logcds"] + controls)

        fm = _fm_multi(sub, "d_logcds", [sig] + controls, [sig],
                       args.min_q_firms).get(sig)
        ic = _partial_rank_ic(sub, sig, "d_logcds", controls,
                              args.min_q_firms)

        sd_sig = float(sub[sig].std())
        med_bp = float(np.exp(sub["logcds_pre"].median()) * 1e4)
        row = {"objective": obj, "L": L,
               "n_events": len(sub), "n_firms": sub["gvkey"].nunique(),
               "n_quarters": sub["quarter"].nunique(),
               "sd_signal": sd_sig, "median_spread_bp": med_bp}
        if fm:
            dlog_1sd = fm["slope"] * sd_sig
            pct = float(np.exp(dlog_1sd) - 1)
            row.update({
                "fm_slope": fm["slope"], "fm_t": fm["t"], "fm_p": fm["p"],
                "fm_nq": fm["nq"],
                "dlog_per_1sd": dlog_1sd, "pct_per_1sd": pct,
                "bp_per_1sd_at_median": pct * med_bp,
            })
        if ic:
            row.update({"pic_mean": ic["pic_mean"], "pic_t": ic["pic_t"],
                        "pic_p": ic["pic_p"]})
        report_rows.append(row)

        print(f"=== {obj} L{L}: {len(sub)} events, "
              f"{row['n_firms']} firms, {row['n_quarters']} quarters ===")
        if fm:
            print(f"  FM slope {fm['slope']:+.5f} (t={fm['t']:+.2f}, "
                  f"p={fm['p']:.4f}, nq={fm['nq']}) "
                  f"[expected sign: NEGATIVE]")
            print(f"  per +1sd {sig} (sd={sd_sig:.2f}): "
                  f"dlog={row['dlog_per_1sd']:+.4f} => "
                  f"{row['pct_per_1sd']:+.2%} of spread "
                  f"= {row['bp_per_1sd_at_median']:+.2f} bp at the "
                  f"median spread ({med_bp:.0f} bp)")
        else:
            print("  FM: insufficient data")
        if ic:
            print(f"  partial rank-IC {ic['pic_mean']:+.4f} "
                  f"(t={ic['pic_t']:+.2f}, p={ic['pic_p']:.4f}, "
                  f"nq={ic['pic_nq']})")
        print()

    out = args.holdout_dir / f"cds_h1_translation_{args.tag}.csv"
    pd.DataFrame(report_rows).to_csv(out, index=False)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
