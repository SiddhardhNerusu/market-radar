"""Gauntlet: short-term reversal + calendar effects on the live-timed universe.

Hypotheses:
  (a) REVERSAL — do the biggest 1-day movers reverse over the next days,
      net-of-cost? Sort variable = return_1d_pct (the day-1 move). Forward
      reversal return = the day1->day5 leg = (1+r5)/(1+r1)-1, i.e. the part of
      the 5d window that happens AFTER the day-1 move we condition on. We test:
        - SHORT the biggest UP movers (top decile r1): P&L = -fwd
        - LONG  the biggest DOWN movers (bottom decile r1): P&L = +fwd
      Day-demean the forward leg on the SAME (ticker,day) key universe, net of
      _round_trip_cost_frac (long) or +borrow for shorts is NOT modeled but
      flagged. Gauntlet all candidates.
  (b) CALENDAR — day-of-week and time-of-day (UTC hour bucket) effects on the
      deduped forward 5d return, day-demeaned + net of cost.

Gauntlet (must pass ALL): dedup, day-demean, net-of-cost, OOS 70/30 day-split
both positive, drop-top-5 stays positive, permutation p<0.05, >=15 distinct days.

Usage: PYTHONPATH=src .venv/bin/python scripts/edge_reversal_seasonality.py
"""
from __future__ import annotations

import pathlib
import sys
import random
from collections import defaultdict
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.storage import get_connection

random.seed(42)
MIN_DAYS = 15
N_PERM = 5000


def fetch_bets():
    """Deduped (ticker, day) bets in the live-timed universe.

    day = COALESCE(published_at, scored_at)[:10]. Live-timed = scored within 2d
    of published. Each bet carries mean r1, mean r5, px (median-ish via first),
    weekday, utc-hour (mode of the dupes' published hour).
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT so.ticker AS ticker,
                   substr(COALESCE(rs.published_at,ss.scored_at),1,10) AS day,
                   so.price_at_flag AS px,
                   so.return_1d_pct AS r1,
                   so.return_5d_pct AS r5,
                   CAST(substr(rs.published_at,12,2) AS INTEGER) AS hr
            FROM signal_scores ss
            JOIN signal_outcomes so ON so.score_id = ss.id
            JOIN raw_signals rs ON rs.id = ss.signal_id
            WHERE so.return_5d_pct IS NOT NULL AND so.return_1d_pct IS NOT NULL
              AND COALESCE(so.data_corrupt,0) = 0
              AND so.price_at_flag BETWEEN 1 AND 2000
              AND ABS(so.return_5d_pct) <= 100
              AND rs.published_at IS NOT NULL
              AND ABS(julianday(ss.scored_at) - julianday(rs.published_at)) <= 2
            """
        ).fetchall()

    agg = {}
    for r in rows:
        key = (r["ticker"], r["day"])
        b = agg.setdefault(key, {"r1": [], "r5": [], "px": r["px"],
                                 "ticker": r["ticker"], "day": r["day"],
                                 "hrs": []})
        b["r1"].append(r["r1"])
        b["r5"].append(r["r5"])
        if r["hr"] is not None:
            b["hrs"].append(r["hr"])
    bets = []
    for b in agg.values():
        r1 = sum(b["r1"]) / len(b["r1"])
        r5 = sum(b["r5"]) / len(b["r5"])
        # forward (day1 -> day5) leg in fraction terms
        denom = 1.0 + r1 / 100.0
        if denom <= 0:
            continue  # degenerate (>=100% day-1 loss); excluded
        fwd = (1.0 + r5 / 100.0) / denom - 1.0  # fraction
        wd = date.fromisoformat(b["day"]).weekday()  # 0=Mon
        hr = max(set(b["hrs"]), key=b["hrs"].count) if b["hrs"] else None
        bets.append({"ticker": b["ticker"], "day": b["day"], "px": b["px"],
                     "r1": r1, "r5": r5, "fwd": fwd, "wd": wd, "hr": hr})
    return bets


def day_split(bets):
    days = sorted({b["day"] for b in bets})
    cut = days[int(len(days) * 0.70)] if len(days) > 1 else None
    return set(days), cut


def gauntlet(name, member_alphas_by_bet, universe_net, label):
    """member_alphas_by_bet: list of (net_alpha_fraction, day) for the family.
    universe_net: list of net_alpha for the WHOLE deduped universe under the same
    P&L convention (for the permutation null). label: 'long'/'short' note.
    Returns dict of gauntlet metrics + survives bool."""
    n = len(member_alphas_by_bet)
    if n == 0:
        return None
    alphas = [a for a, _ in member_alphas_by_bet]
    days_member = sorted({d for _, d in member_alphas_by_bet})
    nd = len(days_member)
    net = sum(alphas) / n * 100  # pct

    # OOS split on the global day axis
    all_days = sorted({d for _, d in universe_net_days})
    cut = all_days[int(len(all_days) * 0.70)]
    early = [a for a, d in member_alphas_by_bet if d < cut]
    late = [a for a, d in member_alphas_by_bet if d >= cut]
    oos_e = (sum(early) / len(early) * 100) if early else float("nan")
    oos_l = (sum(late) / len(late) * 100) if late else float("nan")

    # drop top-5 winners
    srt = sorted(alphas, reverse=True)
    kept = srt[5:] if len(srt) > 5 else []
    drop5 = (sum(kept) / len(kept) * 100) if kept else float("nan")

    # permutation: draw n random bets from the universe (same P&L convention)
    obs = sum(alphas) / n
    pool = [a for a, _ in universe_net_days]
    ge = 0
    for _ in range(N_PERM):
        s = sum(random.sample(pool, n)) / n if n <= len(pool) else sum(pool) / len(pool)
        if s >= obs:
            ge += 1
    pval = (ge + 1) / (N_PERM + 1)

    survives = (nd >= MIN_DAYS and net > 0 and oos_e > 0 and oos_l > 0
                and drop5 > 0 and pval < 0.05)
    return {"name": name, "label": label, "n": n, "days": nd, "netA": net,
            "oos_early": oos_e, "oos_late": oos_l, "drop5": drop5, "p": pval,
            "survives": survives}


# globals filled in main (permutation pool)
universe_net_days = []


def run():
    bets = fetch_bets()
    days = sorted({b["day"] for b in bets})
    print(f"LIVE deduped bets={len(bets)}  distinct days={len(days)}  "
          f"({days[0]}..{days[-1]})\n")

    # ---- day-demean the FORWARD leg across the whole universe ----
    by_day_fwd = defaultdict(list)
    for b in bets:
        by_day_fwd[b["day"]].append(b["fwd"])
    day_mean_fwd = {d: sum(v) / len(v) for d, v in by_day_fwd.items()}

    # also day-demean the raw 5d return for the calendar tests
    by_day_r5 = defaultdict(list)
    for b in bets:
        by_day_r5[b["day"]].append(b["r5"] / 100.0)
    day_mean_r5 = {d: sum(v) / len(v) for d, v in by_day_r5.items()}

    # ============================================================
    # (a) REVERSAL
    # ============================================================
    # rank bets by r1 WITHIN each day (cross-sectional decile), so "biggest mover"
    # is relative to the day's own dispersion (consistent with day-demean logic).
    by_day_bets = defaultdict(list)
    for b in bets:
        by_day_bets[b["day"]].append(b)

    # assign within-day decile rank on r1
    for d, lst in by_day_bets.items():
        lst.sort(key=lambda x: x["r1"])
        m = len(lst)
        for i, b in enumerate(lst):
            b["r1_pct_rank"] = i / max(m - 1, 1)  # 0..1

    # ---- SHORT top-decile up-movers: P&L = -(fwd - day_mean_fwd), net cost ----
    # ---- LONG bottom-decile down-movers: P&L = +(fwd - day_mean_fwd), net cost
    # Build the permutation universes per convention.
    global universe_net_days

    def build_family(pred, side):
        """side 'short' => alpha = -(demeaned fwd) - cost ; 'long' => +(demeaned fwd) - cost.
        Returns (family bets list, full-universe list under same convention)."""
        fam, uni = [], []
        for b in bets:
            dem = b["fwd"] - day_mean_fwd[b["day"]]
            cost = _round_trip_cost_frac(b["px"])
            a = (-dem if side == "short" else dem) - cost
            uni.append((a, b["day"]))
            if pred(b):
                fam.append((a, b["day"]))
        return fam, uni

    results = []

    # Decile thresholds
    configs = [
        ("reversal_SHORT_top10pct_upmovers", lambda b: b["r1_pct_rank"] >= 0.90, "short"),
        ("reversal_SHORT_top20pct_upmovers", lambda b: b["r1_pct_rank"] >= 0.80, "short"),
        ("reversal_LONG_bot10pct_downmovers", lambda b: b["r1_pct_rank"] <= 0.10, "long"),
        ("reversal_LONG_bot20pct_downmovers", lambda b: b["r1_pct_rank"] <= 0.20, "long"),
        # absolute-move version: biggest |r1| up moves (>= +10% on the day)
        ("reversal_SHORT_abs_up_ge10pct", lambda b: b["r1"] >= 10.0, "short"),
        ("reversal_LONG_abs_down_le_neg10pct", lambda b: b["r1"] <= -10.0, "long"),
    ]
    for name, pred, side in configs:
        fam, uni = build_family(pred, side)
        universe_net_days = uni
        r = gauntlet(name, fam, uni, side)
        if r:
            results.append(r)

    # ============================================================
    # (b) CALENDAR — day-of-week (long the family, net cost, day-demeaned r5)
    # ============================================================
    def build_calendar_family(pred):
        fam, uni = [], []
        for b in bets:
            dem = (b["r5"] / 100.0) - day_mean_r5[b["day"]]
            cost = _round_trip_cost_frac(b["px"])
            a = dem - cost
            uni.append((a, b["day"]))
            if pred(b):
                fam.append((a, b["day"]))
        return fam, uni

    wd_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    for wd, nm in wd_names.items():
        fam, uni = build_calendar_family(lambda b, w=wd: b["wd"] == w)
        universe_net_days = uni
        r = gauntlet(f"dow_LONG_{nm}", fam, uni, "long")
        if r:
            results.append(r)

    # time-of-day buckets (UTC). Market: pre-open <13z, open 13-15z, midday 15-18z,
    # close 18-21z, after >=21z (approx ET = UTC-4 in summer; 13z=9am ET).
    tod = [
        ("tod_LONG_preopen_lt13z", lambda b: b["hr"] is not None and b["hr"] < 13),
        ("tod_LONG_open_13to15z", lambda b: b["hr"] is not None and 13 <= b["hr"] < 15),
        ("tod_LONG_midday_15to18z", lambda b: b["hr"] is not None and 15 <= b["hr"] < 18),
        ("tod_LONG_close_18to21z", lambda b: b["hr"] is not None and 18 <= b["hr"] < 21),
        ("tod_LONG_afterhrs_ge21z", lambda b: b["hr"] is not None and b["hr"] >= 21),
    ]
    for name, pred in tod:
        fam, uni = build_calendar_family(pred)
        universe_net_days = uni
        r = gauntlet(name, fam, uni, "long")
        if r:
            results.append(r)

    # ---- print ----
    hdr = (f"{'family':38} {'side':6} {'bets':>6} {'days':>4} {'netA%':>7} "
           f"{'OOSe%':>7} {'OOSl%':>7} {'drop5%':>7} {'perm_p':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: -x["netA"]):
        v = "*** SURVIVES ***" if r["survives"] else "dead"
        print(f"{r['name']:38} {r['label']:6} {r['n']:6d} {r['days']:4d} "
              f"{r['netA']:7.3f} {r['oos_early']:7.3f} {r['oos_late']:7.3f} "
              f"{r['drop5']:7.3f} {r['p']:7.4f}  {v}")

    print("\nNotes: netA% = day-demeaned forward leg (reversal) or 5d (calendar), "
          "net of _round_trip_cost_frac. SHORT families do NOT include borrow cost "
          "(>0.2%/day on HTBs) -> any short netA is optimistic. All on LIVE-timed "
          f"data only ({days[0]}..{days[-1]}, {len(days)} days).")
    return results


if __name__ == "__main__":
    run()
