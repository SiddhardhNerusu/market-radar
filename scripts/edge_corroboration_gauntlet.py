"""Full-gauntlet test of the MULTI-SOURCE CORROBORATION hypothesis.

Two definitions of "high corroboration" are tested independently:
  A) FIELD: the pipeline's own ss.corroboration_count (# independent signals on
     same ticker in last 4h), thresholds >=2 and >=3. This is what the original
     audit found (+1.57% demeaned at >=3, tiny-n).
  B) DISTINCT-SOURCES: rebuilt from the data — # of DISTINCT rs.source feeds that
     flagged the same (ticker, event_day). thresholds >=2 and >=3. This is the
     literal reading of the hypothesis ("distinct sources within a short window").

Gauntlet (a candidate is REAL only if it passes ALL):
  1. DEDUP to one bet per (ticker, event_day); mean the dup returns.
  2. DAY-DEMEAN returns vs the WHOLE deduped universe that event_day (strip beta).
  3. NET of round-trip cost (_round_trip_cost_frac by price).
  4. OOS split: earliest 70% of distinct days TRAIN vs latest 30% TEST -> +ve BOTH.
  5. DROP TOP-5 winners -> stays positive.
  6. PERMUTATION p<0.05 (2000 shuffles: draw n random bets from the deduped universe).
  7. >=15 distinct event-days.

Contamination guard: report the full set AND the live-timed-only subset
(scored within 2d of published_at), since corroboration is ~96% a live-window
phenomenon and the backfilled SEC tail sits in 2024-2026 thin days that distort
the OOS split.

Usage: PYTHONPATH=src .venv/bin/python scripts/edge_corroboration_gauntlet.py
"""
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.storage import get_connection

random.seed(12345)
MIN_DAYS = 15
N_PERM = 2000

CLEAN = """
  so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0
  AND so.price_at_flag>=1 AND so.price_at_flag<=2000 AND ABS(so.return_5d_pct)<=100
"""


def load_rows():
    with get_connection() as conn:
        return conn.execute(
            f"""
            SELECT so.ticker AS ticker,
                   substr(COALESCE(rs.published_at, ss.scored_at),1,10) AS day,
                   rs.source AS source,
                   ss.corroboration_count AS cc,
                   so.price_at_flag AS px,
                   so.return_5d_pct AS r5,
                   CASE WHEN ABS(julianday(ss.scored_at)-julianday(rs.published_at))<=2
                        THEN 1 ELSE 0 END AS live
            FROM signal_scores ss
            JOIN signal_outcomes so ON so.score_id = ss.id
            JOIN raw_signals rs ON rs.id = ss.signal_id
            WHERE {CLEAN}
            """
        ).fetchall()


def build_universe(rows):
    """Dedup ALL clean rows to one bet per (ticker, day). Returns the full
    deduped universe with per-bet attrs: ret, px, day, max corr, distinct srcs,
    live fraction."""
    agg = {}
    for r in rows:
        key = (r["ticker"], r["day"])
        b = agg.setdefault(key, {"rets": [], "px": r["px"], "day": r["day"],
                                 "cc": 0, "srcs": set(), "live_any": 0, "n_live": 0, "n": 0})
        b["rets"].append(r["r5"])
        b["cc"] = max(b["cc"], r["cc"] or 0)
        b["srcs"].add(r["source"])
        b["n"] += 1
        b["n_live"] += (r["live"] or 0)
    for b in agg.values():
        b["ret"] = sum(b["rets"]) / len(b["rets"])
        b["nsrc"] = len(b["srcs"])
        # bet is "live" if a majority of its underlying rows are live-timed
        b["live"] = 1 if b["n_live"] >= (b["n"] / 2.0) else 0
    return agg


def day_demean(universe):
    by_day = defaultdict(list)
    for b in universe.values():
        by_day[b["day"]].append(b["ret"])
    return {d: sum(v) / len(v) for d, v in by_day.items()}


def net_alpha(b, day_mean):
    demeaned = b["ret"] - day_mean[b["day"]]            # in pct
    return demeaned / 100.0 - _round_trip_cost_frac(b["px"])  # fraction


def gauntlet(universe, day_mean, selector, label):
    """selector(b)->bool picks the family. universe values already have ret/px/day."""
    fam = [b for b in universe.values() if selector(b)]
    all_net = [net_alpha(b, day_mean) for b in universe.values()]
    if not fam:
        return {"label": label, "n": 0, "days": 0, "survives": False, "verdict": "empty family"}

    nets = [net_alpha(b, day_mean) for b in fam]
    days = sorted({b["day"] for b in fam})
    n = len(fam)
    nd = len(days)
    mean_net = sum(nets) / n * 100  # pct

    # OOS split on the day axis: earliest 70% days train, latest 30% test
    all_days = days
    cut = all_days[int(len(all_days) * 0.70)] if len(all_days) >= 2 else None
    train = [net_alpha(b, day_mean) for b in fam if cut and b["day"] < cut]
    test = [net_alpha(b, day_mean) for b in fam if cut and b["day"] >= cut]
    oos_early = (sum(train) / len(train) * 100) if train else float("nan")
    oos_late = (sum(test) / len(test) * 100) if test else float("nan")

    # DROP TOP-5 winners
    drop5 = sorted(nets)[:-5] if n > 5 else []
    drop5_mean = (sum(drop5) / len(drop5) * 100) if drop5 else float("nan")

    # PERMUTATION: draw n random bets from the WHOLE deduped universe, 2000x
    obs = sum(nets) / n
    ge = 0
    for _ in range(N_PERM):
        samp = random.sample(all_net, n)
        if (sum(samp) / n) >= obs:
            ge += 1
    perm_p = (ge + 1) / (N_PERM + 1)

    win = sum(1 for x in nets if x > 0) / n * 100

    survives = (
        nd >= MIN_DAYS
        and mean_net > 0
        and (oos_early > 0)
        and (oos_late > 0)
        and (drop5_mean > 0)
        and perm_p < 0.05
    )
    return {
        "label": label, "n": n, "days": nd,
        "net_alpha_pct": round(mean_net, 4),
        "oos_early_pct": round(oos_early, 4) if oos_early == oos_early else None,
        "oos_late_pct": round(oos_late, 4) if oos_late == oos_late else None,
        "drop5_pct": round(drop5_mean, 4) if drop5_mean == drop5_mean else None,
        "perm_p": round(perm_p, 4),
        "win_pct": round(win, 1),
        "survives": survives,
    }


def report(universe, tag):
    dm = day_demean(universe)
    print(f"\n========== {tag} ==========")
    print(f"deduped universe bets={len(universe)}  distinct days={len(dm)}")
    families = [
        ("FIELD corr_count>=2", lambda b: b["cc"] >= 2),
        ("FIELD corr_count>=3", lambda b: b["cc"] >= 3),
        ("DISTINCT-src>=2", lambda b: b["nsrc"] >= 2),
        ("DISTINCT-src>=3", lambda b: b["nsrc"] >= 3),
    ]
    results = []
    hdr = f"{'family':22} {'n':>5} {'days':>4} {'netA%':>8} {'OOSe%':>8} {'OOSl%':>8} {'drop5%':>8} {'permp':>7} {'win%':>5}  survives"
    print(hdr)
    for label, sel in families:
        res = gauntlet(universe, dm, sel, label)
        results.append(res)
        if res["n"] == 0:
            print(f"{label:22} EMPTY")
            continue
        print(f"{label:22} {res['n']:5d} {res['days']:4d} {res['net_alpha_pct']:8.3f} "
              f"{(res['oos_early_pct'] if res['oos_early_pct'] is not None else float('nan')):8.3f} "
              f"{(res['oos_late_pct'] if res['oos_late_pct'] is not None else float('nan')):8.3f} "
              f"{(res['drop5_pct'] if res['drop5_pct'] is not None else float('nan')):8.3f} "
              f"{res['perm_p']:7.4f} {res['win_pct']:5.1f}  {res['survives']}")
    return results


def main():
    rows = load_rows()
    print(f"loaded clean rows={len(rows)}")
    full = build_universe(rows)
    # live-only universe: rebuild from live rows only
    live_rows = [r for r in rows if (r["live"] or 0) == 1]
    live = build_universe(live_rows)

    report(full, "FULL (all clean, incl backfilled SEC tail)")
    report(live, "LIVE-ONLY (rows scored within 2d of published_at)")


if __name__ == "__main__":
    sys.exit(main())
