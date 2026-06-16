"""Measure 8-K Item-code-conditioned drift (deep-dive #3 payoff) + backfill event_subtype.

For each 8-K Item code, compute the day-demeaned (beta-stripped), net-of-cost,
deduped-to-(ticker,day) forward return on CLEAN resolved outcomes — the honest test
of which 8-K event classes drift. With --backfill, also writes the dominant
event_subtype into llm_classifications for 8-K signals (turning the 100%-NULL
column into a real taxonomy).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/analyze_8k_items.py             # measure only
  PYTHONPATH=src .venv/bin/python scripts/analyze_8k_items.py --backfill  # + write subtype
"""
import argparse
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.ingestors.sec_item_codes import (
    ITEM_LABELS, dominant_subtype, extract_item_codes,
)
from market_radar.storage import get_connection

CLEAN = ("so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0 "
         "AND so.price_at_flag>=1 AND so.price_at_flag<=2000 AND ABS(so.return_5d_pct)<=100")


def _avg(v):
    return sum(v) / len(v) * 100 if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    args = ap.parse_args()

    with get_connection() as conn:
        # TRUE event-day calendar (COALESCE(published_at, scored_at)) for BOTH the
        # benchmark and the 8-K bets. The SEC corpus is backfilled — 90k+ filings all
        # share one scored_at (2026-05-12) while their real event dates (published_at ==
        # price_at_flag_ts, the return anchor) span 2024-07..2026-05. Keying off scored_at
        # collapses ~462 event-days into ~19 fake days and demeans against the wrong day's
        # beta. The benchmark MUST use the same key as the bets or daymean.get() misses and
        # the demean silently becomes 0 — hence the raw_signals join here too. See
        # edge_8k_drift_honest.py for the reference pattern.
        allrows = conn.execute(
            f"""SELECT so.ticker t,
                       COALESCE(substr(rs.published_at,1,10), substr(ss.scored_at,1,10)) d,
                       so.return_5d_pct r
                FROM signal_scores ss JOIN signal_outcomes so ON so.score_id=ss.id
                JOIN raw_signals rs ON rs.id=ss.signal_id
                WHERE {CLEAN}""").fetchall()
        k8 = conn.execute(
            f"""SELECT rs.id sid, so.ticker t,
                       COALESCE(substr(rs.published_at,1,10), substr(ss.scored_at,1,10)) d,
                       so.price_at_flag px, so.return_5d_pct r, rs.body body
                FROM raw_signals rs JOIN signal_scores ss ON ss.signal_id=rs.id
                JOIN signal_outcomes so ON so.score_id=ss.id
                WHERE rs.source LIKE 'sec%' AND rs.body LIKE '%Item %' AND {CLEAN}""").fetchall()

    # Day-mean benchmark over ALL clean deduped (ticker,day) bets (the beta to strip).
    allb = defaultdict(list)
    for x in allrows:
        allb[(x["t"], x["d"])].append(x["r"])
    byday = defaultdict(list)
    for (t, d), rs in allb.items():
        byday[d].append(sum(rs) / len(rs))
    daymean = {d: sum(v) / len(v) for d, v in byday.items()}
    days = sorted(byday)
    mid = days[len(days) // 2] if days else None

    # 8-K bets deduped to (ticker,day), union of item codes.
    bets: dict = {}
    for x in k8:
        key = (x["t"], x["d"])
        b = bets.setdefault(key, {"rets": [], "px": x["px"], "codes": set()})
        b["rets"].append(x["r"])
        b["codes"].update(extract_item_codes(x["body"]))

    percode = defaultdict(lambda: {"net": [], "days": set(), "early": [], "late": []})
    for (t, d), b in bets.items():
        ret = sum(b["rets"]) / len(b["rets"])
        net = (ret - daymean.get(d, 0.0)) / 100.0 - _round_trip_cost_frac(b["px"])
        for code in b["codes"]:
            pc = percode[code]
            pc["net"].append(net)
            pc["days"].add(d)
            (pc["early"] if (mid and d < mid) else pc["late"]).append(net)

    print(f"8-K deduped (ticker,day) bets={len(bets)}  distinct days={len(days)}  "
          f"(item codes found on {sum(1 for b in bets.values() if b['codes'])} bets)")
    print(f"{'item':6} {'label':34} {'bets':>5} {'days':>4} {'netA%':>7} {'win%':>6} {'OOSe%':>7} {'OOSl%':>7}")
    out = [(_avg(pc["net"]), c, len(pc["net"]), len(pc["days"]),
            sum(1 for x in pc["net"] if x > 0) / len(pc["net"]) * 100,
            _avg(pc["early"]), _avg(pc["late"]))
           for c, pc in percode.items() if len(pc["net"]) >= 30]
    for net, code, n, nd, win, e, l in sorted(out, reverse=True):
        print(f"{code:6} {ITEM_LABELS.get(code,'?'):34} {n:5d} {nd:4d} {net:7.3f} {win:6.1f} {e:7.3f} {l:7.3f}")
    print("\nnetA% = day-demeaned, net-of-cost, deduped per (ticker,day). <15 days => UNPROVEN.")

    if args.backfill:
        wrote = 0
        with get_connection() as conn:
            for x in k8:
                sub = dominant_subtype(extract_item_codes(x["body"]))
                if not sub:
                    continue
                cur = conn.execute(
                    "UPDATE llm_classifications SET event_subtype=? "
                    "WHERE signal_id=? AND event_subtype IS NULL",
                    (sub, x["sid"]))
                wrote += cur.rowcount
        print(f"\n[backfill] wrote event_subtype to {wrote} llm_classifications rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
