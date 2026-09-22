"""DILUTION / FORCED-SELLING SHORT gauntlet.

Hypothesis: names with active dilution -- SEC 424B5/424B3 prospectus-supplements,
reverse-split language, going-concern, at-the-market (ATM) / registered-direct /
public-offering language in raw_signals.title|body -- UNDERPERFORM forward 5d.
A robust NEGATIVE forward return = a SHORT/avoid edge.

We bet the family DOWN. SHORT net alpha per bet =
    -( demeaned_return_frac )  -  round_trip_cost_frac(px)  [ - borrow_cost ]
i.e. we profit from the negative (beta-stripped) drift but still pay a round-trip;
the borrow variant adds a per-day hard-to-borrow fee that the base cost model omits.

THE GAUNTLET (a candidate is REAL only if it passes ALL):
  1. DEDUP one bet per (ticker, event_day = COALESCE(published_at, scored_at)[:10])
  2. DAY-DEMEAN vs the WHOLE clean deduped universe on the same day-key (strip beta)
  3. NET of round-trip cost (_round_trip_cost_frac)
  4. OOS split: earliest 70% days TRAIN vs latest 30% TEST -- positive SHORT alpha in BOTH
  5. DROP TOP-5 short winners -- stays positive (no outlier dependence)
  6. PERMUTATION p<0.05 (2000 random n-draws from universe; one-sided on SHORT alpha)
  7. >=15 distinct event-days

CONTAMINATION: prefer live-timed rows (|scored_at - published_at| <= 2d). Backfilled
SEC rows are scored ~300d after the event; we run a LIVE-ONLY pass (the headline) and
a FULL-CLEAN pass (flagged) for comparison.

Usage: PYTHONPATH=src .venv/bin/python scripts/edge_dilution_short.py [--mode live|full] [--borrow-bps-per-day 25]
"""
from __future__ import annotations

import argparse
import pathlib
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.storage import get_connection

MIN_DAYS = 15
HOLD_DAYS = 5  # 5d horizon -> borrow accrues over ~5 calendar days held

# Dilution keyword / form filter on raw_signals title|body.
DILUTION_SQL = """(
    rs.title LIKE '%424B5%' OR rs.title LIKE '%424B3%'
    OR rs.body LIKE '%424B5%' OR rs.body LIKE '%424B3%'
    OR LOWER(rs.title) LIKE '%reverse split%'        OR LOWER(rs.body) LIKE '%reverse split%'
    OR LOWER(rs.title) LIKE '%reverse stock split%'  OR LOWER(rs.body) LIKE '%reverse stock split%'
    OR LOWER(rs.title) LIKE '%going concern%'         OR LOWER(rs.body) LIKE '%going concern%'
    OR LOWER(rs.title) LIKE '%at-the-market%'         OR LOWER(rs.body) LIKE '%at-the-market%'
    OR LOWER(rs.title) LIKE '%at the market offering%' OR LOWER(rs.body) LIKE '%at the market offering%'
    OR LOWER(rs.title) LIKE '%registered direct%'     OR LOWER(rs.body) LIKE '%registered direct%'
    OR LOWER(rs.title) LIKE '%public offering%'       OR LOWER(rs.body) LIKE '%public offering%'
)"""

CLEAN_SQL = ("so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0 "
             "AND so.price_at_flag BETWEEN 1 AND 2000 AND ABS(so.return_5d_pct)<=100")


def build_universe(conn):
    """Return (universe_bets, dilution_keys, day_mean).

    universe = ALL clean deduped (ticker, event_day) bets in the DB (the demean pool).
    dilution_keys = subset of (ticker, event_day) keys that are dilution + live-timed
    (or full-clean, per caller filter applied separately).
    """
    rows = conn.execute(
        f"""
        SELECT so.ticker AS t,
               substr(COALESCE(rs.published_at, ss.scored_at),1,10) AS d,
               so.price_at_flag AS px, so.return_5d_pct AS r5
        FROM signal_scores ss
        JOIN signal_outcomes so ON so.score_id=ss.id
        JOIN raw_signals rs ON rs.id=ss.signal_id
        WHERE {CLEAN_SQL}
        """
    ).fetchall()
    bets = {}
    for x in rows:
        key = (x["t"], x["d"])
        b = bets.setdefault(key, {"rets": [], "px": x["px"], "d": x["d"]})
        b["rets"].append(x["r5"])
    for b in bets.values():
        b["ret"] = sum(b["rets"]) / len(b["rets"])
    byday = defaultdict(list)
    for b in bets.values():
        byday[b["d"]].append(b["ret"])
    day_mean = {d: sum(v) / len(v) for d, v in byday.items()}
    return bets, day_mean


def dilution_keys(conn, mode):
    """Set of (ticker, event_day) keys flagged as dilution. mode in {live,full}."""
    live_clause = ""
    if mode == "live":
        # live-timed: scored within ~2 calendar days of published_at
        live_clause = (
            "AND rs.published_at IS NOT NULL "
            "AND ABS(julianday(substr(ss.scored_at,1,10)) "
            "        - julianday(substr(rs.published_at,1,10))) <= 2"
        )
    rows = conn.execute(
        f"""
        SELECT DISTINCT so.ticker AS t,
               substr(COALESCE(rs.published_at, ss.scored_at),1,10) AS d
        FROM signal_scores ss
        JOIN signal_outcomes so ON so.score_id=ss.id
        JOIN raw_signals rs ON rs.id=ss.signal_id
        WHERE {CLEAN_SQL} AND {DILUTION_SQL} {live_clause}
        """
    ).fetchall()
    return {(x["t"], x["d"]) for x in rows}


def short_alpha(b, day_mean, borrow_frac):
    """SHORT net alpha (fraction) for one deduped bet.

    Long demeaned return = ret - day_mean. Short pnl = -(that). Then subtract a
    round-trip cost (charged regardless of direction) and an optional borrow fee.
    """
    demeaned = (b["ret"] - day_mean[b["d"]]) / 100.0
    return (-demeaned) - _round_trip_cost_frac(b["px"]) - borrow_frac


def stats(xs):
    n = len(xs)
    if not n:
        return None
    mean = sum(xs) / n
    med = st.median(xs)
    win = sum(1 for x in xs if x > 0) / n
    return dict(n=n, mean=mean, med=med, win=win)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "full"], default="live")
    ap.add_argument("--borrow-bps-per-day", type=float, default=0.0,
                    help="hard-to-borrow fee in bps/day over the 5d hold (shorts only)")
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args()
    borrow_frac = args.borrow_bps_per_day / 1e4 * HOLD_DAYS
    random.seed(42)

    with get_connection() as conn:
        universe, day_mean = build_universe(conn)
        dkeys = dilution_keys(conn, args.mode)

    # family bets = universe bets whose key is flagged dilution
    fam = [b for k, b in universe.items() if k in dkeys]
    fam_days = sorted({b["d"] for b in fam})
    print(f"=== DILUTION SHORT gauntlet (mode={args.mode}, "
          f"borrow={args.borrow_bps_per_day}bps/day over {HOLD_DAYS}d) ===")
    print(f"universe deduped bets={len(universe)}  distinct days={len({b['d'] for b in universe.values()})}")
    print(f"dilution family bets={len(fam)}  distinct days={len(fam_days)} "
          f"({fam_days[0] if fam_days else '?'}..{fam_days[-1] if fam_days else '?'})")

    fam_alpha = [short_alpha(b, day_mean, borrow_frac) for b in fam]
    s = stats(fam_alpha)
    if not s:
        print("no bets — abort")
        return 1
    print(f"\nSHORT net alpha: mean={s['mean']*100:+.3f}%  median={s['med']*100:+.3f}%  "
          f"win%={s['win']*100:.1f}  n={s['n']}")

    # --- OOS 70/30 day split on the FAMILY's own day axis ---
    cut_idx = int(len(fam_days) * 0.70)
    cut_day = fam_days[cut_idx] if cut_idx < len(fam_days) else fam_days[-1]
    train = [short_alpha(b, day_mean, borrow_frac) for b in fam if b["d"] < cut_day]
    test = [short_alpha(b, day_mean, borrow_frac) for b in fam if b["d"] >= cut_day]
    ts, te = stats(train), stats(test)
    n_train_days = len({b["d"] for b in fam if b["d"] < cut_day})
    n_test_days = len({b["d"] for b in fam if b["d"] >= cut_day})
    print(f"OOS split at {cut_day}: "
          f"TRAIN mean={ts['mean']*100:+.3f}% (n={ts['n']}, {n_train_days}d)  "
          f"TEST mean={te['mean']*100:+.3f}% (n={te['n']}, {n_test_days}d)" if ts and te
          else "OOS: one side empty")

    # --- DROP TOP-5 short winners (the 5 best short alphas) ---
    dropped = sorted(fam_alpha, reverse=True)[5:]
    sd = stats(dropped)
    print(f"DROP top-5 short winners: mean={sd['mean']*100:+.3f}%  median={sd['med']*100:+.3f}%  n={sd['n']}")

    # --- PERMUTATION: random n-draws from universe, short alpha; p = P(rand mean >= obs) ---
    ukeys = list(universe.keys())
    uni_alpha = {k: short_alpha(universe[k], day_mean, borrow_frac) for k in ukeys}
    uni_vals = list(uni_alpha.values())
    obs_mean = s["mean"]
    n_fam = len(fam)
    ge = 0
    for _ in range(args.perm):
        draw = random.sample(uni_vals, n_fam)
        if sum(draw) / n_fam >= obs_mean:
            ge += 1
    p_perm = (ge + 1) / (args.perm + 1)
    uni_mean = sum(uni_vals) / len(uni_vals)
    print(f"PERMUTATION ({args.perm} draws of n={n_fam}): universe-mean short alpha="
          f"{uni_mean*100:+.3f}%  obs={obs_mean*100:+.3f}%  p={p_perm:.4f}")

    # --- VERDICT ---
    passes = {
        "dedup": True,
        "day_demean": True,
        "net_cost": True,
        ">=15 days": len(fam_days) >= MIN_DAYS,
        "net positive (short)": s["mean"] > 0,
        "OOS both positive": bool(ts and te and ts["mean"] > 0 and te["mean"] > 0),
        "drop-top5 positive": sd["mean"] > 0,
        "perm p<0.05": p_perm < 0.05,
    }
    print("\n--- GAUNTLET ---")
    for k, v in passes.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    survives = all(passes.values())
    print(f"\nSURVIVES = {survives}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
