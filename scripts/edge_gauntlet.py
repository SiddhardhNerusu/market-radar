"""Full 7-test gauntlet for a durable, net-of-cost, OOS edge. Read-only.

A candidate family passes ONLY if it clears ALL of:
  (1) DEDUP to one bet per (ticker, event_day)  [event_day = COALESCE(published_at, scored_at)[:10]]
  (2) DAY-DEMEAN returns (strip that day's universe mean = market beta)
  (3) NET of round-trip cost (_round_trip_cost_frac, bucketed by price)
  (4) OOS split: earliest 70% of days TRAIN, latest 30% TEST -> positive in BOTH
  (5) DROP TOP-5 winners -> still positive (no outlier dependence)
  (6) PERMUTATION p < 0.05 (shuffle which bets belong to the family, 2000 draws)
  (7) >= 15 distinct event-days

Usage:
  PYTHONPATH=src .venv/bin/python scripts/edge_gauntlet.py --by event
  PYTHONPATH=src .venv/bin/python scripts/edge_gauntlet.py --by source
  PYTHONPATH=src .venv/bin/python scripts/edge_gauntlet.py --by item8k   # 8-K item codes
  PYTHONPATH=src .venv/bin/python scripts/edge_gauntlet.py --live-only   # exclude backfilled
  PYTHONPATH=src .venv/bin/python scripts/edge_gauntlet.py --side short  # bet the family DOWN
"""
import argparse
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.storage import get_connection

MIN_DAYS = 15
N_PERM = 2000
random.seed(7)

CLEAN = ("so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0 "
         "AND so.price_at_flag>=1 AND so.price_at_flag<=2000 AND ABS(so.return_5d_pct)<=100")


def _avg(v):
    return sum(v) / len(v) if v else float("nan")


def load(by, live_only):
    item_codes = None
    if by == "item8k":
        from market_radar.ingestors.sec_item_codes import extract_item_codes
        item_codes = extract_item_codes
    live_filter = (" AND ABS(julianday(ss.scored_at)-julianday(COALESCE(rs.published_at,ss.scored_at)))<=2"
                   if live_only else "")
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT rs.source src, COALESCE(ss.event_type,'(none)') et, so.ticker tk,
                       COALESCE(substr(rs.published_at,1,10), substr(ss.scored_at,1,10)) day,
                       so.price_at_flag px, so.return_5d_pct r5,
                       ABS(julianday(ss.scored_at)-julianday(COALESCE(rs.published_at,ss.scored_at)))<=2 AS live,
                       rs.body body
                FROM signal_scores ss
                JOIN signal_outcomes so ON so.score_id=ss.id
                JOIN raw_signals rs ON rs.id=ss.signal_id
                WHERE {CLEAN}{live_filter}""").fetchall()
    return rows, item_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["event", "source", "item8k"], default="event")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--live-only", action="store_true")
    ap.add_argument("--side", choices=["long", "short"], default="long")
    args = ap.parse_args()
    sign = 1.0 if args.side == "long" else -1.0

    rows, item_codes = load(args.by, args.live_only)

    # 1. dedup to one bet per (ticker, day): mean raw return, keep px + family labels.
    bets = {}
    for r in rows:
        key = (r["tk"], r["day"])
        b = bets.setdefault(key, {"r": [], "px": r["px"], "day": r["day"], "fams": set()})
        b["r"].append(r["r5"])
        if args.by == "source":
            b["fams"].add(r["src"])
        elif args.by == "event":
            b["fams"].add(r["et"])
        else:  # item8k
            if r["src"].startswith("sec") and r["body"] and "Item " in r["body"]:
                b["fams"].update(item_codes(r["body"]))
    for b in bets.values():
        b["ret"] = _avg(b["r"])

    # 2. day-mean over deduped bets = beta to strip.
    byday = defaultdict(list)
    for b in bets.values():
        byday[b["day"]].append(b["ret"])
    daymean = {d: _avg(v) for d, v in byday.items()}
    days_sorted = sorted(byday)
    # 70/30 OOS split on the day axis
    split_idx = int(len(days_sorted) * 0.70)
    train_days = set(days_sorted[:split_idx])

    # 3. per-bet net day-demeaned alpha (signed by side), plus family membership.
    blist = []
    for b in bets.values():
        demeaned = (b["ret"] - daymean[b["day"]]) / 100.0
        net = sign * demeaned - _round_trip_cost_frac(b["px"])
        blist.append({"net": net, "day": b["day"], "fams": b["fams"],
                      "train": b["day"] in train_days})

    # aggregate per family
    fam = defaultdict(lambda: {"net": [], "days": set(), "tr": [], "te": []})
    for b in blist:
        for f in b["fams"]:
            fa = fam[f]
            fa["net"].append(b["net"])
            fa["days"].add(b["day"])
            (fa["tr"] if b["train"] else fa["te"]).append(b["net"])

    # permutation null: for a family of size n, mean net of n random bets.
    all_net = [b["net"] for b in blist]
    N = len(all_net)

    def perm_p(observed_mean, n):
        if n == 0 or n > N:
            return float("nan")
        hits = 0
        for _ in range(N_PERM):
            s = sum(random.choice(all_net) for _ in range(n)) / n
            if s >= observed_mean:
                hits += 1
        return (hits + 1) / (N_PERM + 1)

    print(f"side={args.side} live_only={args.live_only} by={args.by}  "
          f"total deduped bets={N}  distinct days={len(days_sorted)} "
          f"({days_sorted[0]}..{days_sorted[-1]})  split@{days_sorted[split_idx-1] if split_idx else '?'}")
    hdr = f"{'family':28} {'bets':>5} {'days':>4} {'netA%':>7} {'win%':>5} {'TR%':>7} {'TE%':>7} {'drop5%':>7} {'permp':>6}  verdict"
    print(hdr)
    print("-" * len(hdr))

    out = []
    for name, fa in fam.items():
        n = len(fa["net"])
        if n < args.min_n:
            continue
        netA = _avg(fa["net"])
        # drop top-5 winners
        d5 = sorted(fa["net"])[:-5] if n > 5 else fa["net"]
        out.append((netA, name, n, len(fa["days"]),
                    sum(1 for x in fa["net"] if x > 0) / n,
                    _avg(fa["tr"]), _avg(fa["te"]), _avg(d5)))

    for netA, name, n, nd, win, tr, te, d5 in sorted(out, reverse=True):
        p = perm_p(netA, n)
        passes = (nd >= MIN_DAYS and netA > 0 and tr > 0 and te > 0 and d5 > 0 and p < 0.05)
        if passes:
            v = "*** SURVIVES ***"
        elif netA <= 0:
            v = "no edge (neg)"
        elif nd < MIN_DAYS:
            v = f"unproven (<{MIN_DAYS}d)"
        elif tr <= 0 or te <= 0:
            v = "fails OOS"
        elif d5 <= 0:
            v = "outlier-dependent"
        elif p >= 0.05:
            v = f"not signif (p={p:.3f})"
        else:
            v = "?"
        print(f"{name:28} {n:5d} {nd:4d} {netA*100:7.3f} {win*100:5.1f} "
              f"{tr*100:7.3f} {te*100:7.3f} {d5*100:7.3f} {p:6.3f}  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
