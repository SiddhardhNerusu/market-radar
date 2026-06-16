"""HONEST 8-K item-conditioned drift test (published_at calendar, not scored_at batch).

KEY FIX over edge_screen_v2/analyze_8k_items: those day-demean by ss.scored_at, but
the SEC corpus is BACKFILLED -- 90k+ 8-Ks were all scored on 2026-05-12 while their
real event dates (rs.published_at == so.price_at_flag_ts) span 2024-07 .. 2026-05
(~480 distinct trading days). Demeaning a 2024-07 event against a 2026-05 batch mean
strips the WRONG day's beta and collapses everything to ~19 fake days. This harness
uses published_at as the event calendar and day-demeans WITHIN the same-day 8-K
universe (strip the beta common to all 8-K filers that day -> item-specific alpha).

Honesty controls:
  - dedup to one bet per (ticker, publish-day)
  - day-demean vs same-publish-day clean 8-K bets (>=N_MIN_DAY peers, else drop day)
  - net of realistic round-trip cost (_round_trip_cost_frac)
  - report median + 10% trimmed mean + mean (outlier-robust), win%, N, distinct days
  - chronological OOS split (first half of calendar vs second half)
  - bootstrap-ish: also report result after dropping the 2 best & 2 worst bets

Usage: PYTHONPATH=src .venv/bin/python scripts/edge_8k_drift_honest.py [--horizon 5|1|20]
"""
from __future__ import annotations

import argparse
import pathlib
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.ingestors.sec_item_codes import (
    ITEM_LABELS, SUBTYPE_TO_EVENT, BEARISH_8K_EVENTS, dominant_subtype,
    extract_item_codes,
)
from market_radar.storage import get_connection

N_MIN_DAY = 5  # require >=5 same-day 8-K peers to form a meaningful day-mean


def trimmed_mean(xs, frac=0.10):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(len(s) * frac)
    s = s[k: len(s) - k] if len(s) - 2 * k > 0 else s
    return sum(s) / len(s)


def summary(xs):
    """Return dict of robust stats on a list of net-alpha FRACTIONS (->%)."""
    if not xs:
        return None
    n = len(xs)
    mean = sum(xs) / n * 100
    med = st.median(xs) * 100
    tm = trimmed_mean(xs) * 100
    win = sum(1 for x in xs if x > 0) / n * 100
    # drop 2 best + 2 worst
    if n > 6:
        s = sorted(xs)[2:-2]
        robust = sum(s) / len(s) * 100
    else:
        robust = float("nan")
    return dict(n=n, mean=mean, med=med, tm=tm, win=win, robust=robust)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", choices=["1", "5", "20"], default="5")
    ap.add_argument("--min-bets", type=int, default=40)
    args = ap.parse_args()
    rcol = {"1": "return_1d_pct", "5": "return_5d_pct", "20": "return_20d_pct"}[args.horizon]

    clean = (f"so.{rcol} IS NOT NULL AND COALESCE(so.data_corrupt,0)=0 "
             f"AND so.price_at_flag>=1 AND so.price_at_flag<=2000 AND ABS(so.{rcol})<=100")

    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT so.ticker t, substr(rs.published_at,1,10) d, so.price_at_flag px,
                       so.{rcol} r, rs.body body
                FROM raw_signals rs
                JOIN signal_scores ss ON ss.signal_id=rs.id
                JOIN signal_outcomes so ON so.score_id=ss.id
                WHERE rs.source LIKE 'sec%8-k%' AND rs.body LIKE '%Item %'
                  AND rs.published_at IS NOT NULL AND {clean}""").fetchall()

    # 1. dedup to one bet per (ticker, publish-day); union item codes.
    bets: dict = {}
    for x in rows:
        key = (x["t"], x["d"])
        b = bets.setdefault(key, {"rets": [], "px": x["px"], "codes": set(), "d": x["d"]})
        b["rets"].append(x["r"])
        b["codes"].update(extract_item_codes(x["body"]))
    for b in bets.values():
        b["ret"] = sum(b["rets"]) / len(b["rets"])

    # 2. same-day 8-K mean (the beta to strip). Require >=N_MIN_DAY peers.
    byday = defaultdict(list)
    for b in bets.values():
        byday[b["d"]].append(b["ret"])
    daymean = {d: sum(v) / len(v) for d, v in byday.items() if len(v) >= N_MIN_DAY}
    days = sorted(daymean)
    mid = days[len(days) // 2]

    # 3. net day-demeaned alpha per bet, attach to every item code it carries.
    percode = defaultdict(lambda: {"all": [], "early": [], "late": [], "days": set()})
    allbets = []
    for b in bets.values():
        if b["d"] not in daymean:
            continue
        net = (b["ret"] - daymean[b["d"]]) / 100.0 - _round_trip_cost_frac(b["px"])
        allbets.append(net)
        bucket = "early" if b["d"] < days[len(days) // 2] else "late"
        for code in (b["codes"] or {"(none)"}):
            pc = percode[code]
            pc["all"].append(net)
            pc[bucket].append(net)
            pc["days"].add(b["d"])

    print(f"horizon={args.horizon}d  clean 8-K bets(dedup ticker,pubday)={len(bets)}  "
          f"usable bets(day>= {N_MIN_DAY} peers)={len(allbets)}  distinct pub-days={len(days)} "
          f"({days[0]}..{days[-1]})  split at {days[len(days)//2]}")
    base = summary(allbets)
    print(f"  ALL-8K baseline net-alpha: mean={base['mean']:+.3f}% med={base['med']:+.3f}% "
          f"trim={base['tm']:+.3f}% (by construction ~0 -- demeaned)\n")

    hdr = (f"{'item':6} {'label':38} {'bets':>5} {'days':>4} "
           f"{'mean%':>7} {'med%':>7} {'trim%':>7} {'robust%':>8} {'win%':>6} "
           f"{'OOSe_med%':>9} {'OOSl_med%':>9}")
    print(hdr)
    out = []
    for code, pc in percode.items():
        if len(pc["all"]) < args.min_bets:
            continue
        s = summary(pc["all"])
        e = summary(pc["early"]) or {"med": float("nan")}
        l = summary(pc["late"]) or {"med": float("nan")}
        out.append((s["med"], code, s, len(pc["days"]), e["med"], l["med"]))
    for medv, code, s, nd, em, lm in sorted(out, reverse=True):
        label = ITEM_LABELS.get(code, code)
        print(f"{code:6} {label:38} {s['n']:5d} {nd:4d} {s['mean']:+7.3f} {s['med']:+7.3f} "
              f"{s['tm']:+7.3f} {s['robust']:+8.3f} {s['win']:6.1f} {em:+9.3f} {lm:+9.3f}")

    # 4. BEARISH OVERLAY: pool all 8-Ks whose dominant subtype maps to a bearish event.
    print("\n--- AVOID-LONG / SHORT OVERLAY (dominant subtype -> bearish event) ---")
    bearish = defaultdict(lambda: {"all": [], "early": [], "late": [], "days": set()})
    pooled = {"all": [], "early": [], "late": [], "days": set()}
    for b in bets.values():
        if b["d"] not in daymean:
            continue
        sub = dominant_subtype(sorted(b["codes"]))
        ev = SUBTYPE_TO_EVENT.get(sub or "")
        net = (b["ret"] - daymean[b["d"]]) / 100.0 - _round_trip_cost_frac(b["px"])
        bucket = "early" if b["d"] < days[len(days) // 2] else "late"
        if ev in BEARISH_8K_EVENTS:
            g = bearish[ev]
            g["all"].append(net); g[bucket].append(net); g["days"].add(b["d"])
            pooled["all"].append(net); pooled[bucket].append(net); pooled["days"].add(b["d"])
    print(hdr.replace("item  ", "event ").replace("label", "(bearish family)"))
    rowsb = []
    for ev, g in bearish.items():
        if len(g["all"]) < 10:
            continue
        s = summary(g["all"]); e = summary(g["early"]) or {"med": float("nan")}
        l = summary(g["late"]) or {"med": float("nan")}
        rowsb.append((s["med"], ev, s, len(g["days"]), e["med"], l["med"]))
    for medv, ev, s, nd, em, lm in sorted(rowsb):
        print(f"{ev:6} {'':38} {s['n']:5d} {nd:4d} {s['mean']:+7.3f} {s['med']:+7.3f} "
              f"{s['tm']:+7.3f} {s['robust']:+8.3f} {s['win']:6.1f} {em:+9.3f} {lm:+9.3f}")
    if pooled["all"]:
        s = summary(pooled["all"]); e = summary(pooled["early"]); l = summary(pooled["late"])
        print(f"{'POOL':6} {'all bearish 8-Ks':38} {s['n']:5d} {len(pooled['days']):4d} "
              f"{s['mean']:+7.3f} {s['med']:+7.3f} {s['tm']:+7.3f} {s['robust']:+8.3f} "
              f"{s['win']:6.1f} {e['med']:+9.3f} {l['med']:+9.3f}")
        print("\nNOTE: SHORT pnl ~= -1 * these long net-alphas (minus a 2nd round-trip "
              "cost + borrow). Median is the honest center; mean is outlier-distorted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
