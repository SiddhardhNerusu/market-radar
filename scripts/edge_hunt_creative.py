"""Creative-edge hunt: cross-sectional ranking, ensembles, coverage drift, etc.

Reuses the edge_screen_v2 honest methodology EXACTLY:
  - dedup to one bet per (ticker, scored-day)
  - DAY-DEMEAN (strip market beta)
  - net of realistic round-trip cost (_round_trip_cost_frac)
  - OOS early/late day split
  - clean rule: COALESCE(data_corrupt,0)=0 AND price_at_flag>=1
Adds: median, 10% trimmed mean, top-outlier share, and a cross-sectional
long-short (top vs bottom decile, market-neutral by construction since both
legs are same-day).
"""
import pathlib
import sys
import statistics as st
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.storage import get_connection


def load_bets():
    """One row per (ticker, scored-day) with enrichment attached."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT so.ticker tk, substr(ss.scored_at,1,10) day,
                   AVG(so.return_5d_pct) r5, AVG(so.return_1d_pct) r1,
                   AVG(so.price_at_flag) px,
                   AVG(ss.composite_score) comp, AVG(ss.model_p_5d) mp,
                   AVG(ss.sentiment_magnitude) smag,
                   MAX(CASE WHEN ss.event_type='insider_buy' THEN 1 ELSE 0 END) ev_insbuy,
                   MAX(CASE WHEN ss.event_type='leadership_change' THEN 1 ELSE 0 END) ev_lead,
                   MAX(CASE WHEN ss.event_type='analyst_upgrade' THEN 1 ELSE 0 END) ev_upg,
                   MAX(CASE WHEN ss.event_type='buyback' THEN 1 ELSE 0 END) ev_buyback,
                   COUNT(DISTINCT rs.source) n_sources,
                   GROUP_CONCAT(DISTINCT rs.source) sources,
                   COUNT(*) n_signals
            FROM signal_scores ss
            JOIN signal_outcomes so ON so.score_id=ss.id
            JOIN raw_signals rs ON rs.id=ss.signal_id
            WHERE so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0
              AND so.price_at_flag>=1 AND so.price_at_flag<=2000
              AND ABS(so.return_5d_pct)<=100
            GROUP BY so.ticker, day
            """
        ).fetchall()
        bets = [dict(r) for r in rows]

        # enrichment: recent insider net buying (7d), days_to_cover snapshot,
        # recent earnings surprise (10d)
        ins = defaultdict(list)
        for r in conn.execute(
            "SELECT ticker,report_date,transaction_code,is_acquired,role_score,is_10b5_1,shares,price "
            "FROM insider_transactions"
        ):
            ins[r["ticker"]].append(dict(r))
        si = {r["ticker"]: r["days_to_cover"]
              for r in conn.execute("SELECT ticker,days_to_cover FROM short_interest")}
        earn = defaultdict(list)
        for r in conn.execute(
            "SELECT ticker,report_date,eps_surprise_pct FROM earnings_data "
            "WHERE eps_surprise_pct IS NOT NULL"
        ):
            earn[r["ticker"]].append(dict(r))

    for b in bets:
        day = b["day"]
        # net insider $ acquired in prior 7d (P/A codes acquired vs S/D disposed)
        net_buy_usd = 0.0
        has_open_market_buy = 0
        for t in ins.get(b["tk"], []):
            if not (t["report_date"] and t["report_date"] <= day
                    and t["report_date"] >= _minus(day, 7)):
                continue
            sh = (t["shares"] or 0) * (t["price"] or 0)
            if t["transaction_code"] == "P":  # open-market purchase
                net_buy_usd += sh
                has_open_market_buy = 1
            elif t["transaction_code"] == "S":
                net_buy_usd -= sh
        b["ins_net_buy_usd"] = net_buy_usd
        b["ins_open_buy"] = has_open_market_buy
        b["dtc"] = si.get(b["tk"])
        es = [e["eps_surprise_pct"] for e in earn.get(b["tk"], [])
              if e["report_date"] and _minus(day, 10) <= e["report_date"] <= day]
        b["eps_surprise"] = es[0] if es else None
    return bets


def _minus(day, n):
    from datetime import date, timedelta
    y, m, d = map(int, day.split("-"))
    return (date(y, m, d) - timedelta(days=n)).isoformat()


def demean_net(bets):
    """Attach day-demeaned, net-of-cost alpha (fraction) to each bet."""
    by_day = defaultdict(list)
    for b in bets:
        by_day[b["day"]].append(b["r5"])
    day_mean = {d: sum(v) / len(v) for d, v in by_day.items()}
    for b in bets:
        b["alpha"] = (b["r5"] - day_mean[b["day"]]) / 100.0 - _round_trip_cost_frac(b["px"])
    return sorted(by_day)


def stats(vals):
    if not vals:
        return None
    n = len(vals)
    s = sorted(vals)
    mean = sum(s) / n
    med = st.median(s)
    k = int(n * 0.1)
    trimmed = s[k:n - k] if n - 2 * k > 0 else s
    tmean = sum(trimmed) / len(trimmed)
    win = sum(1 for x in s if x > 0) / n
    # outlier dependence: mean with top-1 and top-3 removed
    mean_drop3 = (sum(s[:-3]) / (n - 3)) if n > 4 else mean
    return dict(n=n, mean=mean * 100, median=med * 100, tmean=tmean * 100,
                win=win * 100, mean_drop3=mean_drop3 * 100)


def report(name, sel, bets, days_sorted, min_n=30):
    sub = [b for b in bets if sel(b)]
    if len(sub) < min_n:
        return f"{name:42} n={len(sub):4d}  <min_n, skip"
    mid = days_sorted[len(days_sorted) // 2]
    early = [b["alpha"] for b in sub if b["day"] < mid]
    late = [b["alpha"] for b in sub if b["day"] >= mid]
    nd = len(set(b["day"] for b in sub))
    a = stats([b["alpha"] for b in sub])
    e, l = stats(early), stats(late)
    verdict = "no edge"
    if a["mean"] > 0 and a["median"] > 0 and e and l and e["mean"] > 0 and l["mean"] > 0 and nd >= 15:
        verdict = "*** persists (median+) ***"
    elif a["mean"] > 0 and a["median"] > 0:
        verdict = "positive (med+, check OOS/days)"
    elif a["mean"] > 0:
        verdict = "mean+ but median<=0 (outlier-driven)"
    return (f"{name:42} n={a['n']:4d} d={nd:2d} mean={a['mean']:6.2f} med={a['median']:6.2f} "
            f"tmean={a['tmean']:6.2f} drop3={a['mean_drop3']:6.2f} win={a['win']:4.1f} "
            f"E={e['mean'] if e else float('nan'):6.2f} L={l['mean'] if l else float('nan'):6.2f}  {verdict}")


def decile_ls(name, score_key, bets, days_sorted, min_per_day=8):
    """Per-day cross-sectional long top-decile minus short bottom-decile.
    Market-neutral by construction (same-day long & short). Net of cost both legs.
    Returns daily L-S alpha series."""
    by_day = defaultdict(list)
    for b in bets:
        if b.get(score_key) is not None:
            by_day[b["day"]].append(b)
    daily = []
    for day, lst in by_day.items():
        if len(lst) < min_per_day:
            continue
        lst = sorted(lst, key=lambda b: b[score_key])
        k = max(1, len(lst) // 10)
        bot = lst[:k]
        top = lst[-k:]
        # net-of-cost raw return each leg (cross-sectional, no demean needed: it's
        # a same-day spread so the day mean cancels)
        long_r = sum(b["r5"] / 100 - _round_trip_cost_frac(b["px"]) for b in top) / len(top)
        short_r = sum(b["r5"] / 100 + _round_trip_cost_frac(b["px"]) for b in bot) / len(bot)
        daily.append((day, long_r - short_r))
    if len(daily) < 10:
        return f"{name:42} only {len(daily)} usable days, skip"
    vals = [v for _, v in daily]
    a = stats(vals)
    mid = days_sorted[len(days_sorted) // 2]
    e = stats([v for d, v in daily if d < mid])
    l = stats([v for d, v in daily if d >= mid])
    verdict = "no edge"
    if a["mean"] > 0 and a["median"] > 0 and e and l and e["mean"] > 0 and l["mean"] > 0:
        verdict = "*** persists ***"
    elif a["mean"] > 0 and a["median"] > 0:
        verdict = "positive (med+)"
    elif a["mean"] > 0:
        verdict = "mean+ median<=0 (fragile)"
    return (f"{name:42} days={a['n']:3d} mean={a['mean']:6.2f} med={a['median']:6.2f} "
            f"tmean={a['tmean']:6.2f} win={a['win']:4.1f} "
            f"E={e['mean'] if e else float('nan'):6.2f} L={l['mean'] if l else float('nan'):6.2f}  {verdict}")


def main():
    bets = load_bets()
    days = demean_net(bets)
    print(f"loaded {len(bets)} deduped bets, {len(days)} clean days ({days[0]}..{days[-1]})\n")
    print("== alpha = day-demeaned 5d return, net round-trip cost (fraction*100=%) ==")
    print("== mean/med/tmean in %/trade; E/L = OOS early/late mean; drop3 = mean w/ top-3 removed ==\n")

    print("--- H1: signal ENSEMBLE (multiple distinct sources agree on same ticker/day) ---")
    print(report("1 source only", lambda b: b["n_sources"] == 1, bets, days))
    print(report(">=2 distinct sources agree", lambda b: b["n_sources"] >= 2, bets, days))
    print(report(">=3 distinct sources agree", lambda b: b["n_sources"] >= 3, bets, days))
    print(report(">=4 distinct sources agree", lambda b: b["n_sources"] >= 4, bets, days))

    print("\n--- H2: ENSEMBLE = filing/insider event + price-action breakout same day ---")
    pa = lambda b: b["sources"] and "price_action" in b["sources"]
    sec = lambda b: b["sources"] and ("sec_edgar" in b["sources"] or "sec_" in b["sources"])
    print(report("price-action breakout present", pa, bets, days))
    print(report("SEC filing + price-action same day", lambda b: pa(b) and sec(b), bets, days))
    print(report("insider open-market buy (7d) present", lambda b: b["ins_open_buy"] == 1, bets, days))
    print(report("insider buy + price-action same day", lambda b: b["ins_open_buy"] == 1 and pa(b), bets, days))

    print("\n--- H3: CROSS-SECTIONAL ranking (long top-decile vs short bottom-decile, mkt-neutral) ---")
    print(decile_ls("by composite_score", "comp", bets, days))
    print(decile_ls("by model_p_5d", "mp", bets, days))
    print(decile_ls("by sentiment_magnitude", "smag", bets, days))
    print(decile_ls("by #distinct sources (ensemble rank)", "n_sources", bets, days))
    print(decile_ls("by insider net-buy $ (7d)", "ins_net_buy_usd", bets, days))

    print("\n--- H4: UNDER-COVERED microcaps (price proxy: cheaper = less covered) drift ---")
    print(report("px < $5 (micro)", lambda b: b["px"] < 5, bets, days))
    print(report("$5 <= px < $20", lambda b: 5 <= b["px"] < 20, bets, days))
    print(report("px >= $20 (liquid)", lambda b: b["px"] >= 20, bets, days))
    print(report("micro + >=2 sources (under-covered + confirmation)",
                 lambda b: b["px"] < 5 and b["n_sources"] >= 2, bets, days))

    print("\n--- H5: short-squeeze proxy: high days-to-cover + bullish event ---")
    print(report("days_to_cover >= 3", lambda b: (b["dtc"] or 0) >= 3, bets, days))
    print(report("days_to_cover >= 5", lambda b: (b["dtc"] or 0) >= 5, bets, days))
    print(report("dtc>=3 + price-action breakout", lambda b: (b["dtc"] or 0) >= 3 and pa(b), bets, days))

    print("\n--- H6: earnings post-surprise drift (PEAD), thin coverage ---")
    print(report("recent positive EPS surprise (10d)", lambda b: (b["eps_surprise"] or 0) > 5, bets, days, min_n=15))
    print(report("recent negative EPS surprise (10d)", lambda b: (b["eps_surprise"] or 0) < -5, bets, days, min_n=15))


if __name__ == "__main__":
    main()
