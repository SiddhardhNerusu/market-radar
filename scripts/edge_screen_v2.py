"""Beta-neutral, deduplicated, cost-aware edge screen (ingestion deep-dive keystone).

The prior screens conflated market BETA with alpha and counted the same bet up to
21x. This harness makes every verdict trustworthy:
  1. DEDUP to one bet per (ticker, scored-day)   -> kills pseudo-replication.
  2. DAY-DEMEAN (return minus that day's mean across all bets) -> strips market beta,
     leaving alpha vs the day's own signal universe.
  3. NET of realistic round-trip cost (_round_trip_cost_frac).
  4. OOS split (earlier days vs later days) -> does the edge persist?
  5. Flags when distinct-day count < 15 (cannot conclude — the current ceiling).

A real candidate must be positive day-demeaned, net of cost, AND persist OOS on
>=15 distinct days. Run after scripts/recover_holdout.py to use the OOS sample.

Usage:  PYTHONPATH=src .venv/bin/python scripts/edge_screen_v2.py [--by event|source]
"""
import argparse
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.storage import get_connection

MIN_DAYS_TO_CONCLUDE = 15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["event", "source"], default="event")
    ap.add_argument("--min-n", type=int, default=40, help="min deduped bets per family")
    args = ap.parse_args()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT rs.source AS source, COALESCE(ss.event_type,'(none)') AS et,
                   so.ticker AS ticker,
                   -- TRUE event-day calendar. The SEC corpus is BACKFILLED: 90k+
                   -- filings were all scored on one day (scored_at=2026-05-12) while
                   -- their real event dates (published_at == price_at_flag_ts, the
                   -- return-window anchor) span 2024-07..2026-05. Keying the day-demean
                   -- and OOS split off scored_at strips the WRONG day's beta and
                   -- collapses ~462 event-days into ~19 fake batch-days. Fall back to
                   -- scored_at only when published_at is NULL. See edge_8k_drift_honest.py.
                   COALESCE(substr(rs.published_at,1,10), substr(ss.scored_at,1,10)) AS day,
                   so.price_at_flag AS px, so.return_5d_pct AS r5
            FROM signal_scores ss
            JOIN signal_outcomes so ON so.score_id = ss.id
            JOIN raw_signals rs ON rs.id = ss.signal_id
            WHERE so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0
              AND so.price_at_flag >= 1 AND so.price_at_flag <= 2000
              AND ABS(so.return_5d_pct) <= 100
            """
        ).fetchall()

    # 1. dedup to one bet per (ticker, day) — mean return; keep px + family label.
    bets: dict = {}
    for r in rows:
        key = (r["ticker"], r["day"])
        b = bets.setdefault(key, {"rets": [], "px": r["px"],
                                  "source": r["source"], "et": r["et"], "day": r["day"]})
        b["rets"].append(r["r5"])
    for b in bets.values():
        b["ret"] = sum(b["rets"]) / len(b["rets"])

    # 2. day mean over DEDUPED bets (the beta to strip).
    by_day = defaultdict(list)
    for b in bets.values():
        by_day[b["day"]].append(b["ret"])
    day_mean = {d: sum(v) / len(v) for d, v in by_day.items()}
    days_sorted = sorted(by_day)
    mid = days_sorted[len(days_sorted) // 2] if days_sorted else None

    print(f"clean deduped bets={len(bets)}  distinct days={len(days_sorted)}  "
          f"({days_sorted[0] if days_sorted else '?'}..{days_sorted[-1] if days_sorted else '?'})")
    if len(days_sorted) < MIN_DAYS_TO_CONCLUDE:
        print(f"⚠ only {len(days_sorted)} distinct days (< {MIN_DAYS_TO_CONCLUDE}) — "
              f"NOTHING here is conclusive; run recover_holdout.py to add OOS days.\n")

    # 3. aggregate per family: day-demeaned, net-of-cost alpha + OOS split.
    fam = defaultdict(lambda: {"net": [], "days": set(), "early": [], "late": []})
    keyf = (lambda b: b["et"]) if args.by == "event" else (lambda b: b["source"])
    for b in bets.values():
        demeaned_pct = b["ret"] - day_mean[b["day"]]           # alpha, %
        net_alpha = demeaned_pct / 100.0 - _round_trip_cost_frac(b["px"])  # fraction
        f = fam[keyf(b)]
        f["net"].append(net_alpha)
        f["days"].add(b["day"])
        (f["early"] if (mid and b["day"] < mid) else f["late"]).append(net_alpha)

    def _avg(v):
        return (sum(v) / len(v) * 100) if v else float("nan")

    out = []
    for name, f in fam.items():
        if len(f["net"]) < args.min_n:
            continue
        out.append((
            _avg(f["net"]), name, len(f["net"]), len(f["days"]),
            sum(1 for x in f["net"] if x > 0) / len(f["net"]) * 100,
            _avg(f["early"]), _avg(f["late"]),
        ))
    print(f"{'family':30} {'bets':>5} {'days':>4} {'netA%':>7} {'win%':>6} "
          f"{'OOS_early%':>10} {'OOS_late%':>9}  verdict")
    for net, name, n, nd, win, early, late in sorted(out, reverse=True):
        if nd >= MIN_DAYS_TO_CONCLUDE and net > 0 and early > 0 and late > 0:
            v = "*** persists ***"
        elif net > 0:
            v = "positive (unproven)" if nd < MIN_DAYS_TO_CONCLUDE else "in-sample only"
        else:
            v = "no edge"
        print(f"{name:30} {n:5d} {nd:4d} {net:7.3f} {win:6.1f} {early:10.3f} {late:9.3f}  {v}")
    print("\nnetA% = day-demeaned (beta-stripped) return, net of round-trip cost, per "
          "deduped (ticker,day) bet. Real edge = positive netA AND OOS_early>0 AND "
          f"OOS_late>0 across >={MIN_DAYS_TO_CONCLUDE} days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
