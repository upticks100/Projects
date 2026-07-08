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

import config
from src.analysis.veer import _fm_multi, _partial_rank_ic

EVENT_DIR = config.EVENT_DIR_499
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
    p.add_argument("--event-dir", type=Path, default=EVENT_DIR,
                   help="event-data dir with daily_market.csv + CDS file")
    p.add_argument("--cds-file", default="cds_markit.csv.gz",
                   help="CDS csv(.gz) filename inside --event-dir")
    p.add_argument("--robust-battery", action="store_true",
                   help="write HY/appendix robustness variants instead of only the headline spec")
    p.add_argument("--robust-out", type=Path, default=None,
                   help="optional output CSV for --robust-battery")
    return p.parse_args()


def load_cds_on_calendar(event_dir: Path = EVENT_DIR,
                         cds_file: str = "cds_markit.csv.gz",
                         ) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """Per-gvkey log-spread series reindexed to the CRSP trading calendar."""
    market = pd.read_csv(event_dir / "daily_market.csv", usecols=["date"])
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(market["date"]).unique()))

    cds = pd.read_csv(event_dir / cds_file, dtype={"gvkey": str})
    if "ticker" in cds.columns:   # 499 pull was ticker-matched at pull time
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
                      cal: np.ndarray, post_offset: int = POST_OFFSET,
                      pre_offset: int = PRE_OFFSET,
                      pre_delta_offset: int = PRE_DELTA_OFFSET) -> pd.DataFrame:
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

        pre = at(j + pre_offset)
        post = at(j + post_offset)
        pre2 = at(j + pre_delta_offset)
        out["logcds_pre"] = pre
        if np.isfinite(pre) and np.isfinite(post):
            out["d_logcds"] = post - pre
            out["spread_pre_bp"] = float(np.exp(pre) * 1e4)
            out["spread_post_bp"] = float(np.exp(post) * 1e4)
            out["d_spread_bp"] = out["spread_post_bp"] - out["spread_pre_bp"]
        if np.isfinite(pre) and np.isfinite(pre2):
            out["d_logcds_pre"] = pre - pre2
            out["spread_pre_delta_start_bp"] = float(np.exp(pre2) * 1e4)
            out["d_spread_pre_bp"] = (
                out.get("spread_pre_bp", float(np.exp(pre) * 1e4))
                - out["spread_pre_delta_start_bp"]
            )
    return pd.DataFrame(rows)


def _winsorize_frame(df: pd.DataFrame, cols: list[str], pct: float) -> pd.DataFrame:
    """Two-sided quantile winsorization for sensitivity checks."""
    if pct <= 0:
        return df
    out = df.copy()
    for col in cols:
        lo, hi = out[col].quantile([pct, 1 - pct])
        out[col] = out[col].clip(lo, hi)
    return out


def _score_spec(df: pd.DataFrame, sig: str, target_kind: str,
                winsor_pct: float, min_q_firms: int) -> dict:
    if target_kind == "log":
        ycol = "d_logcds"
        controls = ["logcds_pre", "d_logcds_pre"]
    elif target_kind == "raw_bp":
        ycol = "d_spread_bp"
        controls = ["spread_pre_bp", "d_spread_pre_bp"]
    else:
        raise ValueError(target_kind)

    sub = df.dropna(subset=[sig, ycol] + controls).copy()
    sub = _winsorize_frame(sub, [sig, ycol] + controls, winsor_pct)
    fm = _fm_multi(sub, ycol, [sig] + controls, [sig], min_q_firms).get(sig)
    ic = _partial_rank_ic(sub, sig, ycol, controls, min_q_firms)

    sd_sig = float(sub[sig].std()) if len(sub) else np.nan
    med_bp = (float(sub["spread_pre_bp"].median())
              if "spread_pre_bp" in sub and len(sub) else np.nan)
    row = {
        "n_events": len(sub),
        "n_firms": sub["gvkey"].nunique() if "gvkey" in sub else 0,
        "n_quarters": sub["quarter"].nunique() if "quarter" in sub else 0,
        "sd_signal": sd_sig,
        "median_spread_bp": med_bp,
    }
    if fm:
        effect_1sd = fm["slope"] * sd_sig
        if target_kind == "log":
            pct = float(np.exp(effect_1sd) - 1)
            bp = pct * med_bp
            row.update({"dlog_per_1sd": effect_1sd,
                        "pct_per_1sd": pct,
                        "bp_per_1sd_at_median": bp})
        else:
            row.update({"bp_per_1sd_at_median": effect_1sd})
        row.update({"fm_slope": fm["slope"], "fm_t": fm["t"],
                    "fm_p": fm["p"], "fm_nq": fm["nq"]})
    if ic:
        row.update({"pic_mean": ic["pic_mean"], "pic_t": ic["pic_t"],
                    "pic_p": ic["pic_p"], "pic_nq": ic["pic_nq"]})
    return row


def _robustness_battery(args: argparse.Namespace,
                        logcds: dict[str, np.ndarray], cal: np.ndarray) -> Path:
    """Appendix battery for the HY CDS result.

    Variants cover raw-vs-log spread changes, alternate post-event windows,
    winsorization, leave-one-quarter-out stability, a 2022 tightening-period
    drop, and conditioning on the pre-event spread tercile.
    """
    sig = "drift_cashflow"
    rows: list[dict] = []

    def add_row(base: dict, df: pd.DataFrame, target_kind: str,
                winsor_pct: float = 0.0):
        scored = _score_spec(df, sig, target_kind, winsor_pct, args.min_q_firms)
        rows.append({**base, **scored})

    for obj, L in CELLS:
        panel_path = args.holdout_dir / f"veer_panel_{obj}_L{L}_{args.tag}.csv"
        panel = pd.read_csv(panel_path, dtype={"gvkey": str})

        target_cache: dict[int, pd.DataFrame] = {}
        for post in (21, 42, 63):
            tg = build_cds_targets(
                panel[["gvkey", "quarter", "ann_date"]], logcds, cal,
                post_offset=post,
            )
            target_cache[post] = panel.merge(tg, on=["gvkey", "quarter"],
                                             how="left")

        for post in (21, 42, 63):
            df = target_cache[post]
            add_row({"objective": obj, "L": L, "variant": f"log_post{post}",
                     "target_kind": "log", "post_offset": post,
                     "winsor_pct": 0.0, "drop_rule": "none",
                     "spread_tercile": "all"}, df, "log")

        add_row({"objective": obj, "L": L, "variant": "raw_bp_post63",
                 "target_kind": "raw_bp", "post_offset": 63,
                 "winsor_pct": 0.0, "drop_rule": "none",
                 "spread_tercile": "all"}, target_cache[63], "raw_bp")

        for pct in (0.01, 0.05):
            add_row({"objective": obj, "L": L,
                     "variant": f"log_post63_winsor_{pct:g}",
                     "target_kind": "log", "post_offset": 63,
                     "winsor_pct": pct, "drop_rule": "none",
                     "spread_tercile": "all"}, target_cache[63], "log", pct)

        # No HY test quarter overlaps the 2020 COVID shock, but keep the drop
        # rule explicit so the appendix documents that it is non-binding here.
        for label, predicate in (
            ("drop_covid_2020", lambda q: str(q).startswith("2020")),
            ("drop_2022_tightening", lambda q: str(q).startswith("2022")),
        ):
            df = target_cache[63]
            keep = ~df["quarter"].astype(str).map(predicate)
            add_row({"objective": obj, "L": L, "variant": label,
                     "target_kind": "log", "post_offset": 63,
                     "winsor_pct": 0.0, "drop_rule": label,
                     "spread_tercile": "all"}, df[keep], "log")

        for q in sorted(target_cache[63]["quarter"].dropna().astype(str).unique()):
            df = target_cache[63]
            add_row({"objective": obj, "L": L, "variant": f"loo_{q}",
                     "target_kind": "log", "post_offset": 63,
                     "winsor_pct": 0.0, "drop_rule": f"leave_out_{q}",
                     "spread_tercile": "all"}, df[df["quarter"].astype(str) != q],
                    "log")

        tertile_df = target_cache[63].dropna(subset=["spread_pre_bp"]).copy()
        if not tertile_df.empty:
            tertile_df["spread_tercile"] = pd.qcut(
                tertile_df["spread_pre_bp"], 3,
                labels=["low", "middle", "high"], duplicates="drop")
            for tercile, g in tertile_df.groupby("spread_tercile"):
                add_row({"objective": obj, "L": L,
                         "variant": "spread_pre_tercile",
                         "target_kind": "log", "post_offset": 63,
                         "winsor_pct": 0.0, "drop_rule": "none",
                         "spread_tercile": str(tercile)}, g, "log")

    out = args.robust_out or (
        args.holdout_dir / f"cds_h1_robustness_{args.tag}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> int:
    args = parse_args()
    logcds, cal = load_cds_on_calendar(args.event_dir, args.cds_file)
    print(f"CDS series for {len(logcds)} gvkeys on {len(cal)}-day calendar\n")
    if args.robust_battery:
        out = _robustness_battery(args, logcds, cal)
        print(f"-> {out}")
        return 0

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
