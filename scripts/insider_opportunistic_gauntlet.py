"""Gauntlet test: OPPORTUNISTIC INSIDER BUYS (Form-4 transactionCode=P, NOT 10b5-1).

The Cohen-Malloy-Pomorski edge: non-routine open-market insider purchases.
We use the insider_transactions aux table (which carries the actual SEC Form-4
transaction_code, is_10b5_1 flag, and role attributes) joined to signal_outcomes
via signal_id == signal_scores.id == signal_outcomes.score_id.

Gauntlet (a candidate is REAL only if it passes ALL):
  1. DEDUP to one bet per (ticker, event_day). event_day = report_date.
  2. DAY-DEMEAN vs the full clean deduped universe on the same day-key.
  3. NET of round-trip cost (_round_trip_cost_frac by price).
  4. OOS split: earliest 70% days TRAIN vs latest 30% TEST, positive in BOTH.
  5. DROP TOP-5 winners -> stays positive.
  6. PERMUTATION p<0.05 (2000 draws of n random bets from universe).
  7. >=15 distinct event-days.

Direction: LONG (opportunistic buys -> we bet UP).
"""
import sqlite3
import numpy as np

DB = "data/market_radar.db"

def round_trip_cost_frac(px):
    if px is None or px <= 0:
        return 0.040
    if px < 1:  return 0.040
    if px < 3:  return 0.030
    if px < 5:  return 0.020
    if px < 10: return 0.012
    if px < 50: return 0.005
    return 0.002

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

CLEAN = ("so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0 "
         "AND so.price_at_flag BETWEEN 1 AND 2000 AND ABS(so.return_5d_pct)<=100")

# ---------------------------------------------------------------------------
# 1. Build the day-demean benchmark: full clean deduped universe.
#    Universe event_day = COALESCE(published_at, scored_at)[:10].
#    Dedup to (ticker, eday): mean the duplicate returns.
# ---------------------------------------------------------------------------
uni_rows = conn.execute(f"""
  SELECT ss.ticker AS ticker,
         substr(COALESCE(rs.published_at, ss.scored_at),1,10) AS eday,
         so.return_5d_pct AS r5
  FROM signal_scores ss
  JOIN signal_outcomes so ON so.score_id = ss.id
  JOIN raw_signals rs ON rs.id = ss.signal_id
  WHERE {CLEAN}
""").fetchall()

# dedup universe to (ticker, eday) mean
from collections import defaultdict
uni_dd = defaultdict(list)
for r in uni_rows:
    uni_dd[(r["ticker"], r["eday"])].append(r["r5"])
uni_bet = {k: float(np.mean(v)) for k, v in uni_dd.items()}

# universe per-day mean (across deduped bets) -> the demean baseline
day_vals = defaultdict(list)
for (tkr, eday), r in uni_bet.items():
    day_vals[eday].append(r)
day_mean = {d: float(np.mean(v)) for d, v in day_vals.items()}

# For permutation: a flat pool of (eday, demeaned_net) over the WHOLE universe,
# so the null = "n random clean bets" drawn from the same demean+cost process.
def net_of_cost(r5_pct, px):
    # r5_pct is a percent; cost is a fraction -> convert
    return r5_pct - 100.0 * round_trip_cost_frac(px)

# need px for universe to net it; refetch with px (dedup px = mean px)
uni_px_rows = conn.execute(f"""
  SELECT ss.ticker AS ticker,
         substr(COALESCE(rs.published_at, ss.scored_at),1,10) AS eday,
         so.return_5d_pct AS r5, so.price_at_flag AS px
  FROM signal_scores ss
  JOIN signal_outcomes so ON so.score_id = ss.id
  JOIN raw_signals rs ON rs.id = ss.signal_id
  WHERE {CLEAN}
""").fetchall()
uni_dd2 = defaultdict(list)
for r in uni_px_rows:
    uni_dd2[(r["ticker"], r["eday"])].append((r["r5"], r["px"]))
universe_pool = []  # demeaned, net-of-cost alpha for every universe bet
for (tkr, eday), lst in uni_dd2.items():
    r5 = float(np.mean([x[0] for x in lst]))
    px = float(np.mean([x[1] for x in lst]))
    alpha = net_of_cost(r5, px) - net_of_cost(day_mean[eday], px)
    # demean on RAW return then net; but cost is same px so it cancels in demean.
    # Cleaner: demean raw, then subtract cost once.
    universe_pool.append(alpha)
universe_pool = np.array(universe_pool)

# ---------------------------------------------------------------------------
# Helper: run gauntlet on a family given list of (eday, ticker, r5, px)
# ---------------------------------------------------------------------------
def demean_net_alpha(eday, r5, px):
    """day-demeaned, net-of-cost alpha in percent for one bet."""
    dm = r5 - day_mean[eday]            # strip the day's universe mean (beta)
    return dm - 100.0 * round_trip_cost_frac(px)  # net of round-trip cost

def run_gauntlet(rows, label, side="long", n_perm=2000, seed=42):
    # rows: list of dict(ticker, eday, r5, px)
    # dedup to (ticker, eday): mean r5 and px
    dd = defaultdict(list)
    for r in rows:
        dd[(r["ticker"], r["eday"])].append((r["r5"], r["px"]))
    bets = []
    for (tkr, eday), lst in dd.items():
        r5 = float(np.mean([x[0] for x in lst]))
        px = float(np.mean([x[1] for x in lst]))
        a = demean_net_alpha(eday, r5, px)
        if side == "short":
            a = -a  # short flips sign of return; (cost still subtracted -> approx)
        bets.append((eday, a, r5, px))
    n = len(bets)
    days = sorted(set(b[0] for b in bets))
    ndays = len(days)
    alphas = np.array([b[1] for b in bets])
    net = float(alphas.mean()) if n else float("nan")

    # OOS split on day axis: earliest 70% days TRAIN, latest 30% TEST
    split_idx = int(np.ceil(ndays * 0.70))
    train_days = set(days[:split_idx])
    test_days = set(days[split_idx:])
    train_a = np.array([b[1] for b in bets if b[0] in train_days])
    test_a = np.array([b[1] for b in bets if b[0] in test_days])
    train_net = float(train_a.mean()) if len(train_a) else float("nan")
    test_net = float(test_a.mean()) if len(test_a) else float("nan")

    # DROP TOP-5 winners
    order = np.argsort(alphas)[::-1]
    keep = np.ones(n, dtype=bool)
    keep[order[:5]] = False
    drop5_net = float(alphas[keep].mean()) if keep.sum() else float("nan")

    # PERMUTATION: draw n random bets from universe_pool, 2000 times,
    # compare observed net (LONG = same sign as pool). For short, flip pool.
    rng = np.random.default_rng(seed)
    pool = universe_pool if side == "long" else -universe_pool
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = pool[rng.integers(0, len(pool), n)].mean()
    p_perm = float((null >= net).mean())  # one-sided: null beats observed

    return {
        "label": label, "side": side, "n": n, "ndays": ndays,
        "net": net, "train_net": train_net, "test_net": test_net,
        "drop5_net": drop5_net, "p_perm": p_perm,
        "ntrain_days": len(train_days), "ntest_days": len(test_days),
        "ntrain": len(train_a), "ntest": len(test_a),
    }

def fetch_pcode(extra_where=""):
    # NOTE: report_date sometimes carries a tz suffix (e.g. '2026-05-27-05:00');
    # normalize the event-day to the first 10 chars to match the universe key.
    q = f"""
      SELECT it.ticker AS ticker, substr(it.report_date,1,10) AS eday,
             so.return_5d_pct AS r5, so.price_at_flag AS px
      FROM insider_transactions it
      JOIN signal_scores ss ON ss.id = it.signal_id
      JOIN signal_outcomes so ON so.score_id = ss.id
      WHERE it.transaction_code = 'P'
        AND {CLEAN}
        {extra_where}
    """
    rows = [dict(r) for r in conn.execute(q).fetchall()]
    # Drop bets whose event-day has no universe demean baseline (can't strip beta).
    kept, dropped = [], 0
    for r in rows:
        if r["eday"] in day_mean:
            kept.append(r)
        else:
            dropped += 1
    if dropped:
        print(f"  [dropped {dropped} P-code rows on event-days absent from the demean universe]")
    return kept

def show(res):
    survives = (
        res["ndays"] >= 15 and
        res["net"] > 0 and
        (res["train_net"] > 0 and res["test_net"] > 0) and
        res["drop5_net"] > 0 and
        res["p_perm"] < 0.05
    )
    print(f"\n=== {res['label']} ({res['side']}) ===")
    print(f"  n bets (deduped)      : {res['n']}")
    print(f"  distinct event-days   : {res['ndays']}  (>=15? {res['ndays']>=15})")
    print(f"  net alpha %% (demean+cost): {res['net']:+.3f}")
    print(f"  OOS TRAIN net %%        : {res['train_net']:+.3f}  ({res['ntrain_days']} days, {res['ntrain']} bets)")
    print(f"  OOS TEST  net %%        : {res['test_net']:+.3f}  ({res['ntest_days']} days, {res['ntest']} bets)")
    print(f"  drop-top-5 net %%       : {res['drop5_net']:+.3f}")
    print(f"  permutation p         : {res['p_perm']:.4f}")
    print(f"  >>> SURVIVES ALL 7    : {survives}")
    return survives

print("="*70)
print("UNIVERSE (demean benchmark + permutation pool):")
print(f"  deduped bets: {len(universe_pool)}, distinct days: {len(day_mean)}")
print("="*70)

# MAIN: opportunistic = P-code, exclude 10b5-1 planned
rows = fetch_pcode("AND COALESCE(it.is_10b5_1,0)=0")
res_main = run_gauntlet(rows, "Opportunistic P-buy (excl 10b5-1), LONG", "long")
surv_main = show(res_main)

# Sensitivity: ALL P-code (incl the 10 planned)
rows_all = fetch_pcode("")
res_all = run_gauntlet(rows_all, "All P-buy (incl 10b5-1), LONG", "long")
show(res_all)

# Sensitivity: opportunistic AND non-routine role (officer/director/10pct, role_score>=1)
rows_role = fetch_pcode("AND COALESCE(it.is_10b5_1,0)=0 AND COALESCE(it.role_score,0)>=2")
res_role = run_gauntlet(rows_role, "Opportunistic P-buy, role_score>=2, LONG", "long")
show(res_role)

print("\n" + "="*70)
print("PRIMARY VERDICT (opportunistic P-buy excl 10b5-1):", surv_main)
print("="*70)
conn.close()
