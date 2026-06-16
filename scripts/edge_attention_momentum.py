"""Honest re-test of the attention / cross-sectional-momentum hypothesis.

Naive stocktwits_trending was +4.71%/trade = beta + 21x dupes. This strips both:
  - DEDUP to one bet per (ticker, scored-day).
  - DAY-DEMEAN (subtract that day's mean across ALL clean deduped bets) = strip beta.
  - NET of realistic round-trip cost (_round_trip_cost_frac), by px bucket.
  - Slice by an ATTENTION-INTENSITY rank (stocktwits mention count / day, cross-sectional
    quantile) and by ABNORMAL-VOLUME breakout flags (price_action vol_mult / atr_break).
  - Robustness: mean + MEDIAN + 10%-trimmed mean; outlier-drop sensitivity; OOS early/late.
  - Liquid filter knob (px >= $5) to answer "is it just penny-stock beta".

Real edge => positive net-of-cost day-demeaned, persists early AND late, median agrees
with mean (not 1-2 outliers), on >=15 distinct days. Default verdict otherwise: artifact.
"""
import pathlib, sys, json, statistics
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from market_radar.backtest.replay import _round_trip_cost_frac
from market_radar.storage import get_connection

MIN_DAYS = 15


def trimmed_mean(xs, frac=0.10):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(len(s) * frac)
    s = s[k: len(s) - k] if len(s) - 2 * k > 0 else s
    return sum(s) / len(s)


def fetch():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT rs.source src, so.ticker tk, substr(ss.scored_at,1,10) day,
                   so.price_at_flag px, so.return_1d_pct r1, so.return_5d_pct r5,
                   rs.raw_payload payload, ss.sentiment sent
            FROM signal_scores ss
            JOIN signal_outcomes so ON so.score_id = ss.id
            JOIN raw_signals rs ON rs.id = ss.signal_id
            WHERE so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0
              AND so.price_at_flag >= 1 AND so.price_at_flag <= 2000
              AND ABS(so.return_5d_pct) <= 100
            """
        ).fetchall()
    return rows


def build_bets(rows, min_px=1.0):
    """One bet per (ticker, day). Carry attention count (stocktwits mentions that
    day) and best breakout strength seen."""
    bets = {}
    st_count = defaultdict(int)            # (ticker,day) -> stocktwits mention count
    for r in rows:
        if r["px"] < min_px:
            continue
        key = (r["tk"], r["day"])
        b = bets.setdefault(key, {"r1": [], "r5": [], "px": r["px"], "day": r["day"],
                                  "srcs": set(), "vol_mult": 0.0, "atr_break": 0.0,
                                  "trend5": None})
        b["r1"].append(r["r1"]); b["r5"].append(r["r5"]); b["srcs"].add(r["src"])
        if r["src"] == "stocktwits_trending":
            st_count[key] += 1
        if r["src"].startswith("price_action") and r["payload"]:
            try:
                p = json.loads(r["payload"])
                b["vol_mult"] = max(b["vol_mult"], float(p.get("vol_mult") or 0))
                b["atr_break"] = max(b["atr_break"], float(p.get("atr_break_x") or 0))
                if p.get("trend_5bar_pct") is not None:
                    b["trend5"] = float(p["trend_5bar_pct"])
            except Exception:
                pass
    for key, b in bets.items():
        b["r1"] = sum(x for x in b["r1"] if x is not None) / max(1, sum(1 for x in b["r1"] if x is not None)) if any(x is not None for x in b["r1"]) else None
        b["r5"] = sum(b["r5"]) / len(b["r5"])
        b["att"] = st_count.get(key, 0)
    return bets


def day_demean(bets, ret_key):
    by_day = defaultdict(list)
    for b in bets.values():
        v = b[ret_key]
        if v is not None:
            by_day[b["day"]].append(v)
    dm = {d: sum(v) / len(v) for d, v in by_day.items()}
    return dm


def report(name, net_list, days, early, late):
    if not net_list:
        print(f"  {name:34} (empty)"); return
    n = len(net_list)
    mean = sum(net_list) / n * 100
    med = statistics.median(net_list) * 100
    tm = trimmed_mean(net_list) * 100
    win = sum(1 for x in net_list if x > 0) / n * 100
    # outlier sensitivity: drop top+bottom 1 each
    s = sorted(net_list)
    drop2 = (sum(s[1:-1]) / (n - 2) * 100) if n > 4 else float("nan")
    e = (sum(early) / len(early) * 100) if early else float("nan")
    l = (sum(late) / len(late) * 100) if late else float("nan")
    nd = len(days)
    verdict = "no edge"
    if mean > 0 and med > 0 and e > 0 and l > 0 and nd >= MIN_DAYS:
        verdict = "*** persists ***"
    elif mean > 0 and nd < MIN_DAYS:
        verdict = "positive (unproven, <15d)"
    elif mean > 0 and (med <= 0):
        verdict = "outlier-driven (med<=0)"
    elif mean > 0:
        verdict = "in-sample, fails OOS"
    print(f"  {name:34} n={n:5d} d={nd:2d} | meanA%={mean:+.3f} med%={med:+.3f} "
          f"trim%={tm:+.3f} drop2%={drop2:+.3f} win%={win:4.1f} | "
          f"early%={e:+.3f} late%={l:+.3f} | {verdict}")


def run_slice(bets, ret_key, label, min_px):
    dm = day_demean(bets, ret_key)
    days_sorted = sorted(dm)
    if not days_sorted:
        print(f"\n### {label} (px>={min_px}) — no data"); return
    mid = days_sorted[len(days_sorted) // 2]
    # attention quantile per day (cross-sectional rank of stocktwits mention count)
    # build day -> sorted attention thresholds (top tercile)
    by_day_att = defaultdict(list)
    for b in bets.values():
        if b["att"] > 0:
            by_day_att[b["day"]].append(b["att"])
    att_hi = {}
    for d, v in by_day_att.items():
        sv = sorted(v)
        att_hi[d] = sv[int(len(sv) * 0.66)] if sv else 1e9

    buckets = defaultdict(lambda: {"net": [], "days": set(), "early": [], "late": []})

    def add(bucket, b, net):
        x = buckets[bucket]
        x["net"].append(net); x["days"].add(b["day"])
        (x["early"] if b["day"] < mid else x["late"]).append(net)

    for b in bets.values():
        v = b[ret_key]
        if v is None:
            continue
        net = (v - dm[b["day"]]) / 100.0 - _round_trip_cost_frac(b["px"])
        add("ALL clean bets", b, net)
        if b["att"] > 0:
            add("stocktwits mentioned", b, net)
            if b["att"] >= att_hi.get(b["day"], 1e9):
                add("stocktwits TOP-attention tercile", b, net)
            else:
                add("stocktwits lower attention", b, net)
        if b["vol_mult"] >= 3.0:
            add("vol-spike >=3x", b, net)
        if b["atr_break"] >= 0.5:
            add("ATR breakout >=0.5x", b, net)
        if "price_action_donchian_breakout" in b["srcs"]:
            add("donchian breakout", b, net)
        if "price_action_opening_range_breakout" in b["srcs"]:
            add("ORB breakout", b, net)
        # momentum confirmation: breakout AND attention together
        if b["att"] > 0 and (b["vol_mult"] >= 3.0 or b["atr_break"] >= 0.5
                             or any(s.startswith("price_action") for s in b["srcs"])):
            add("attention + breakout combo", b, net)

    print(f"\n### {label} | hold={ret_key} | px>={min_px} | "
          f"{len(days_sorted)} days ({days_sorted[0]}..{days_sorted[-1]}) mid={mid}")
    order = ["ALL clean bets", "stocktwits mentioned", "stocktwits TOP-attention tercile",
             "stocktwits lower attention", "vol-spike >=3x", "ATR breakout >=0.5x",
             "donchian breakout", "ORB breakout", "attention + breakout combo"]
    for k in order:
        if k in buckets and len(buckets[k]["net"]) >= 40:
            report(k, buckets[k]["net"], buckets[k]["days"], buckets[k]["early"], buckets[k]["late"])


def main():
    rows = fetch()
    for min_px in (1.0, 5.0):
        bets = build_bets(rows, min_px=min_px)
        for ret_key in ("r1", "r5"):
            run_slice(bets, ret_key, f"ATTENTION/MOMENTUM", min_px)
    print("\nmeanA% = day-demeaned (beta-stripped) return net of round-trip cost, per "
          "deduped (ticker,day) bet. Real edge => mean>0 AND median>0 AND early>0 AND "
          f"late>0 across >={MIN_DAYS} days; outlier-robust (drop2 holds sign).")


if __name__ == "__main__":
    main()
