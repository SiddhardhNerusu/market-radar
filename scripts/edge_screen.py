"""Honest cost-aware edge screen + frozen out-of-sample validation (P2).

For each event_type on CLEAN data, simulate a long-only catalyst trade
(entry = price_at_flag, ATR-proxy TP/SL, day-1 stop, realistic price-bucketed
round-trip cost) and report NET expectancy split into IN-SAMPLE (earlier dates)
vs FROZEN OUT-OF-SAMPLE (later dates), plus a 1.5x cost stress on the OOS slice.

A candidate is only worth building if it is positive IN-SAMPLE *and* positive
OUT-OF-SAMPLE *and* survives the cost stress on a tradeable sample size. This is
the gate that separates a real edge from in-sample / multiple-testing luck.

Usage:  PYTHONPATH=src .venv/bin/python scripts/edge_screen.py
"""
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.storage import get_connection

ATR = 0.02
TP = 2.25 * ATR
SL = 0.75 * ATR
OOS_FRACTION = 0.35      # last 35% of clean rows (chronological) are frozen OOS
MIN_OOS_N = 100          # below this, OOS verdict is "insufficient"


def net_return(price, r1, r5, cost_mult=1.0):
    if r1 is not None and r1 <= -SL:
        exit_ = -SL
    elif r5 >= TP:
        exit_ = TP
    elif r5 <= -SL:
        exit_ = -SL
    else:
        exit_ = r5
    return exit_ - _round_trip_cost_frac(price) * cost_mult


def _avg(vals):
    return (sum(vals) / len(vals) * 100) if vals else 0.0


def main() -> int:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT ss.event_type AS et, ss.scored_at AS ts, so.price_at_flag AS px,
                   so.return_1d_pct AS r1, so.return_5d_pct AS r5
            FROM signal_scores ss
            JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE so.return_5d_pct IS NOT NULL
              AND COALESCE(so.data_corrupt, 0) = 0
              AND so.price_at_flag IS NOT NULL AND so.price_at_flag >= 1
            ORDER BY ss.scored_at
            """
        ).fetchall()

    # Chronological cutoff for the frozen OOS slice.
    cut_idx = int(len(rows) * (1 - OOS_FRACTION))
    cutoff_ts = rows[cut_idx]["ts"] if rows else ""
    is_days = len({r["ts"][:10] for r in rows[:cut_idx]})
    oos_days = len({r["ts"][:10] for r in rows[cut_idx:]})
    print(f"clean rows={len(rows)}  IS<{cutoff_ts[:10]} ({is_days} days)  "
          f"OOS>={cutoff_ts[:10]} ({oos_days} days)\n")

    in_s: dict[str, list] = defaultdict(list)
    oos: dict[str, list] = defaultdict(list)
    oos_stress: dict[str, list] = defaultdict(list)
    for i, r in enumerate(rows):
        et = r["et"] or "(none)"
        a = (r["r1"] / 100.0 if r["r1"] is not None else None)
        b = r["r5"] / 100.0
        bucket = in_s if i < cut_idx else oos
        bucket[et].append(net_return(r["px"], a, b))
        if i >= cut_idx:
            oos_stress[et].append(net_return(r["px"], a, b, cost_mult=1.5))

    print(f"{'event_type':26} {'IS_n':>6} {'IS_avg%':>8} {'OOS_n':>6} "
          f"{'OOS_avg%':>9} {'OOS@1.5x%':>10}  verdict")
    cand = sorted(set(in_s) | set(oos),
                  key=lambda e: _avg(oos.get(e, [])), reverse=True)
    for et in cand:
        isv, ov, ost = in_s.get(et, []), oos.get(et, []), oos_stress.get(et, [])
        if len(isv) + len(ov) < 200:
            continue
        is_avg, oos_avg, oos_st = _avg(isv), _avg(ov), _avg(ost)
        if len(ov) < MIN_OOS_N:
            verdict = "insufficient OOS n"
        elif is_avg > 0 and oos_avg > 0 and oos_st > 0:
            verdict = "*** SURVIVES ***"
        elif oos_avg <= 0 < is_avg:
            verdict = "overfit (IS+ OOS-)"
        else:
            verdict = "no edge"
        print(f"{et:26} {len(isv):6d} {is_avg:8.3f} {len(ov):6d} "
              f"{oos_avg:9.3f} {oos_st:10.3f}  {verdict}")
    print("\nSURVIVES = positive in-sample AND out-of-sample AND under 1.5x cost, "
          f"with OOS n>={MIN_OOS_N}. Anything else is not a tradeable edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
