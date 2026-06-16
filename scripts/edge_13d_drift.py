"""Honest event-study test of 13D activist-INITIATION drift (Brav/Jiang).

CRITICAL DATA NOTE: these SEC filings were scored in one bulk backfill
(scored_at=2026-05-12) but the OUTCOME is event-anchored: price_at_flag_ts ==
published_at (the filing date), and return_5d/20d are the true post-filing
returns. So we MUST demean on the EVENT day (published_at), NOT scored_at
(edge_screen_v2 keys on scored_at, which would collapse all 350 into one day).

Honest rules applied:
  - clean: return IS NOT NULL, data_corrupt=0, price_at_flag>=1
  - lag filter: |price_at_flag_ts - published_at| <= 3 days (drop stale-price
    contaminations where the fetcher grabbed a much later bar)
  - dedup to one bet per (ticker, event-day)
  - EVENT-DAY demean: subtract that event-day's mean across ALL clean SEC 13x
    bets (strip the market/backfill-universe beta for that day)
  - net of realistic round-trip cost (_round_trip_cost_frac)
  - split: SC 13D (initiation) vs SC 13D/A (amend) vs SC 13G vs SC 13G/A
  - report mean, MEDIAN, 10% trimmed mean, win%, n, distinct event-days
  - OOS: early-half vs late-half by event-date
"""
import pathlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.storage import get_connection

FORMS = ("SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A")
HORIZONS = ("return_5d_pct", "return_20d_pct")


def _trimmed_mean(xs, frac=0.10):
    xs = sorted(xs)
    k = int(len(xs) * frac)
    core = xs[k: len(xs) - k] if len(xs) - 2 * k >= 1 else xs
    return sum(core) / len(core)


def fetch(horizon):
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT json_extract(rs.raw_payload,'$.form') AS form,
                   so.ticker AS ticker,
                   substr(rs.published_at,1,10) AS eday,
                   so.price_at_flag AS px,
                   so.{horizon} AS ret,
                   julianday(substr(so.price_at_flag_ts,1,10))
                     - julianday(substr(rs.published_at,1,10)) AS lag
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE json_extract(rs.raw_payload,'$.form') IN ('SC 13D','SC 13D/A','SC 13G','SC 13G/A')
              AND so.{horizon} IS NOT NULL
              AND COALESCE(so.data_corrupt,0)=0
              AND so.price_at_flag >= 1
              AND rs.published_at IS NOT NULL
              AND ABS({horizon}) <= 100
            """
        ).fetchall()
    # lag filter (entry price must reflect the filing date)
    rows = [r for r in rows if r["lag"] is not None and abs(r["lag"]) <= 3]
    return rows


def analyze(horizon):
    rows = fetch(horizon)
    # 1. dedup to one bet per (form, ticker, event-day) — keep px + form
    bets = {}
    for r in rows:
        key = (r["form"], r["ticker"], r["eday"])
        b = bets.setdefault(key, {"rets": [], "px": r["px"], "form": r["form"], "eday": r["eday"]})
        b["rets"].append(r["ret"])
    for b in bets.values():
        b["ret"] = sum(b["rets"]) / len(b["rets"])

    # 2. EVENT-DAY demean across ALL clean 13x bets that day (strip beta)
    by_day = defaultdict(list)
    for b in bets.values():
        by_day[b["eday"]].append(b["ret"])
    day_mean = {d: sum(v) / len(v) for d, v in by_day.items()}
    days_sorted = sorted(by_day)
    mid = days_sorted[len(days_sorted) // 2]

    # 3. per-form: day-demeaned, net-of-cost alpha (fraction)
    fam = defaultdict(lambda: {"net": [], "raw": [], "days": set(), "early": [], "late": []})
    for b in bets.values():
        demeaned_pct = b["ret"] - day_mean[b["eday"]]
        net = demeaned_pct / 100.0 - _round_trip_cost_frac(b["px"])
        f = fam[b["form"]]
        f["net"].append(net * 100)        # store as %
        f["raw"].append(b["ret"])         # raw (non-demeaned) %
        f["days"].add(b["eday"])
        (f["early"] if b["eday"] < mid else f["late"]).append(net * 100)

    print(f"\n========== HORIZON {horizon} ==========")
    print(f"clean deduped bets={len(bets)}  distinct EVENT-days={len(days_sorted)} "
          f"({days_sorted[0]}..{days_sorted[-1]})")
    print(f"{'form':10} {'n':>4} {'days':>4} {'rawMean%':>8} {'netMean%':>8} "
          f"{'netMed%':>8} {'netTrim%':>8} {'win%':>5} {'early%':>7} {'late%':>7}")
    for form in FORMS:
        f = fam[form]
        if not f["net"]:
            continue
        n = len(f["net"])
        net_mean = sum(f["net"]) / n
        net_med = statistics.median(f["net"])
        net_trim = _trimmed_mean(f["net"])
        raw_mean = sum(f["raw"]) / len(f["raw"])
        win = sum(1 for x in f["net"] if x > 0) / n * 100
        em = (sum(f["early"]) / len(f["early"])) if f["early"] else float("nan")
        lm = (sum(f["late"]) / len(f["late"])) if f["late"] else float("nan")
        print(f"{form:10} {n:4d} {len(f['days']):4d} {raw_mean:8.3f} {net_mean:8.3f} "
              f"{net_med:8.3f} {net_trim:8.3f} {win:5.1f} {em:7.3f} {lm:7.3f}")
    return fam, len(days_sorted)


if __name__ == "__main__":
    for h in HORIZONS:
        analyze(h)
    print("\nnet% = event-day-demeaned (beta-stripped) return minus round-trip cost, "
          "per deduped (ticker, event-day) bet. Real edge needs net>0 AND median>0 "
          "AND trimmed>0 AND early>0 AND late>0.")
