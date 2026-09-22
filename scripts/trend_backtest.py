#!/usr/bin/env python
"""Trend-following backtest — 3 PRE-REGISTERED variants on 11 liquid ETFs.

Pure research. Reads data/research_bars.db (etf_bars). Never touches market_radar.db.

PRE-REGISTERED SPEC (no other variants, no parameter tweaks):
  V1  12-1 time-series momentum: long if total return close[t-252] -> close[t-21] > 0.
      Signal at month-end close, trade at next open (first trading day of month).
  V2  Faber 10-month SMA: long if month-end close > SMA of last 10 month-end closes
      (inclusive). Monthly, same execution.
  V3  50/200-day SMA crossover: long if SMA50 > SMA200 (daily closes). Checked at the
      last trading day of each week, trade at next open.

SIZING (single scheme, all variants):
  Among ON assets, weight proportional to 1/trailing-63d vol, normalized to sum 1;
  then scale the whole portfolio so trailing-60d-covariance implied annualized vol
  = 10%, capped at 1.0 (NO leverage). OFF assets / residual sit in cash at 0%.

EXECUTION REALISM:
  Signal on close of day t -> trade at OPEN of t+1. Costs 5 bps per side on traded
  notional (includes the initial position build and vol-target adjustments).

EVALUATION:
  Benchmarks: SPY buy&hold and 60/40 SPY/IEF (monthly rebalance), both cost-free
  (conservative for the strategy). EXCESS = strategy daily return - 60/40 daily
  return. OOS split: eval-start..2021-12-31 vs 2022-01-01..present; excess Sharpe
  must be > 0 in BOTH halves. Significance: stationary block bootstrap
  (Politis-Romano, expected block 21d, 2000 draws, fixed seed) on daily excess
  returns; p(mean <= 0) must be < 0.05/3 = 0.0167. SURVIVES = both conditions.

HONEST WARM-UP NOTE:
  Data starts 2016-01-04 and V1 needs 252 trading days of history, so the common
  evaluation window for ALL variants and benchmarks starts at the first trading day
  of the month after the first month-end with >= 252 prior trading days
  (2017-02-01). The "2016-2021" early half is therefore actually 2017-02..2021-12.
"""

import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT, "data", "research_bars.db")

SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "HYG", "GLD", "DBC", "VNQ"]

COST_PER_SIDE = 0.0005          # 5 bps per side on traded notional
VOL_TARGET = 0.10               # 10% annualized portfolio vol target
VOL_LOOKBACK = 63               # trailing daily-return window for per-asset vol
COV_LOOKBACK = 60               # trailing daily-return window for covariance
MOM_LOOKBACK = 252              # V1: 12 months
MOM_SKIP = 21                   # V1: skip most recent month
SMA_MONTHS = 10                 # V2
SMA_FAST, SMA_SLOW = 50, 200    # V3
OOS_SPLIT = pd.Timestamp("2022-01-01")
N_BOOT = 2000
AVG_BLOCK = 21
BONFERRONI_P = 0.05 / 3
SEED = 42
TRADING_DAYS = 252


# ---------------------------------------------------------------- data loading

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT symbol, date, open, close FROM etf_bars ORDER BY date", conn,
        parse_dates=["date"])
    conn.close()
    opens = df.pivot(index="date", columns="symbol", values="open")[SYMBOLS]
    closes = df.pivot(index="date", columns="symbol", values="close")[SYMBOLS]
    if opens.isna().any().any() or closes.isna().any().any():
        raise SystemExit("FATAL: missing bars — symbols not aligned on common dates")
    if (closes <= 0).any().any() or (opens <= 0).any().any():
        raise SystemExit("FATAL: non-positive prices in data")
    return opens, closes


# ------------------------------------------------------------------- calendars

def period_last_positions(dates, keyfunc):
    """Positions (integer iloc) of the last trading day of each period."""
    keys = [keyfunc(d) for d in dates]
    out, cur = [], None
    for i, k in enumerate(keys):
        if cur is not None and k != cur:
            out.append(i - 1)
        cur = k
    out.append(len(dates) - 1)
    return out


# --------------------------------------------------------------------- signals

def signals_at(variant, closes, month_end_count, s, month_end_positions):
    """Boolean ON/OFF vector for all symbols using info up to close position s."""
    if variant == "V1":
        r = closes.iloc[s - MOM_SKIP].values / closes.iloc[s - MOM_LOOKBACK].values - 1.0
        return r > 0
    if variant == "V2":
        # monthly closes up to and including this month-end
        me = [p for p in month_end_positions if p <= s]
        mcl = closes.iloc[me]
        sma = mcl.iloc[-SMA_MONTHS:].mean().values
        return mcl.iloc[-1].values > sma
    if variant == "V3":
        fast = closes.iloc[s - SMA_FAST + 1: s + 1].mean().values
        slow = closes.iloc[s - SMA_SLOW + 1: s + 1].mean().values
        return fast > slow
    raise ValueError(variant)


# ---------------------------------------------------------------------- sizing

def size_weights(on, rets, s):
    """Pre-registered sizing at signal position s (uses info up to close of s)."""
    n = len(on)
    w = np.zeros(n)
    if not on.any():
        return w
    win_vol = rets.iloc[s - VOL_LOOKBACK + 1: s + 1]
    vols = win_vol.std(ddof=1).values * np.sqrt(TRADING_DAYS)
    ok = on & (vols > 0)
    if not ok.any():
        return w
    inv = np.where(ok, 1.0 / vols, 0.0)
    b = inv / inv.sum()
    cov = rets.iloc[s - COV_LOOKBACK + 1: s + 1].cov().values
    port_var_d = float(b @ cov @ b)
    port_vol_ann = np.sqrt(max(port_var_d, 0.0) * TRADING_DAYS)
    k = min(1.0, VOL_TARGET / port_vol_ann) if port_vol_ann > 0 else 1.0
    return k * b


# ------------------------------------------------------------------- simulator

def simulate(dates, opens, closes, targets, start, end, cost_rate):
    """Dollar-position simulator. targets: {position -> weight vector}.
    Rebalances execute at the OPEN of the target position's day. Returns
    (daily returns Series over [start, end], annualized one-sided turnover)."""
    o, c = opens.values, closes.values
    pos = np.zeros(len(SYMBOLS))
    cash = 1.0
    prev_v = 1.0
    rets_out, idx_out = [], []
    turnover_1s = 0.0
    for i in range(start, end + 1):
        if i in targets:
            if i > start:
                pos = pos * (o[i] / c[i - 1])
            v = pos.sum() + cash
            tgt = targets[i] * v
            traded = np.abs(tgt - pos).sum()
            cost = cost_rate * traded
            turnover_1s += traded / (2.0 * v)
            cash = v - tgt.sum() - cost
            pos = tgt * (c[i] / o[i])
        else:
            pos = pos * (c[i] / c[i - 1])
        v = pos.sum() + cash
        rets_out.append(v / prev_v - 1.0)
        idx_out.append(dates[i])
        prev_v = v
    n_years = len(rets_out) / TRADING_DAYS
    return pd.Series(rets_out, index=pd.DatetimeIndex(idx_out)), turnover_1s / n_years


# --------------------------------------------------------------------- metrics

def perf_stats(r):
    eq = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(TRADING_DAYS) if r.std(ddof=1) > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    annual = (1.0 + r).groupby(r.index.year).prod() - 1.0
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, max_dd=dd, annual=annual)


def excess_sharpe(x):
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def stationary_bootstrap_p(x, n_draws=N_BOOT, avg_block=AVG_BLOCK, seed=SEED):
    """Politis-Romano stationary bootstrap; p = fraction of draws with mean <= 0."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    p_new = 1.0 / avg_block
    starts = rng.integers(0, n, size=(n_draws, n))
    new_block = rng.random(size=(n_draws, n)) < p_new
    idx = np.empty((n_draws, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(new_block[:, t], starts[:, t], cont)
    means = x[idx].mean(axis=1)
    return float((means <= 0).mean())


# ------------------------------------------------------------------------ main

def main():
    opens, closes = load_data()
    dates = closes.index
    rets = closes.pct_change()

    month_ends = period_last_positions(dates, lambda d: (d.year, d.month))
    week_ends = period_last_positions(
        dates, lambda d: (d.isocalendar().year, d.isocalendar().week))

    # Common eval start: first trading day after the first month-end with >= 252
    # prior trading days (so ALL variants + sizing are fully warmed up).
    first_me = next(p for p in month_ends if p >= MOM_LOOKBACK)
    start = first_me + 1
    end = len(dates) - 1
    print(f"Eval window: {dates[start].date()} -> {dates[end].date()} "
          f"({end - start + 1} trading days)")

    # --- build rebalance schedules: {trade position -> weight vector}
    def monthly_targets(variant):
        tg = {}
        for s in month_ends:
            if s < first_me or s + 1 > end:
                continue
            on = signals_at(variant, closes, None, s, month_ends)
            tg[s + 1] = size_weights(on, rets, s)
        return tg

    def weekly_targets():
        tg = {}
        sig_positions = [start - 1] + [s for s in week_ends if s >= start - 1]
        for s in sorted(set(sig_positions)):
            if s + 1 > end:
                continue
            on = signals_at("V3", closes, None, s, month_ends)
            tg[s + 1] = size_weights(on, rets, s)
        return tg

    variants = {
        "V1_tsmom_12_1": monthly_targets("V1"),
        "V2_faber_10m_sma": monthly_targets("V2"),
        "V3_ma_50_200": weekly_targets(),
    }

    # --- benchmarks (cost-free: conservative for the strategies)
    spy_w = np.array([1.0 if s == "SPY" else 0.0 for s in SYMBOLS])
    b6040_w = np.array([0.6 if s == "SPY" else (0.4 if s == "IEF" else 0.0)
                        for s in SYMBOLS])
    spy_r, _ = simulate(dates, opens, closes, {start: spy_w}, start, end, 0.0)
    b6040_tg = {p + 1: b6040_w for p in month_ends if start - 1 <= p and p + 1 <= end}
    b6040_tg[start] = b6040_w
    b6040_r, _ = simulate(dates, opens, closes, b6040_tg, start, end, 0.0)

    spy_stats = perf_stats(spy_r)
    b_stats = perf_stats(b6040_r)

    print("\n=== BENCHMARKS (same window, cost-free) ===")
    for name, st in [("SPY buy&hold", spy_stats), ("60/40 SPY/IEF", b_stats)]:
        print(f"{name:16s} CAGR {st['cagr']*100:6.2f}%  vol {st['vol']*100:5.2f}%  "
              f"Sharpe {st['sharpe']:5.2f}  maxDD {st['max_dd']*100:7.2f}%  "
              f"2022 {st['annual'].get(2022, float('nan'))*100:6.2f}%")

    results = []
    for name, tg in variants.items():
        r, turn = simulate(dates, opens, closes, tg, start, end, COST_PER_SIDE)
        st = perf_stats(r)
        ex = r - b6040_r
        early = ex[ex.index < OOS_SPLIT]
        late = ex[ex.index >= OOS_SPLIT]
        es_e, es_l = excess_sharpe(early), excess_sharpe(late)
        p = stationary_bootstrap_p(ex.values)
        survives = (es_e > 0) and (es_l > 0) and (p < BONFERRONI_P)
        results.append(dict(
            name=name, cagr=st["cagr"], vol=st["vol"], sharpe=st["sharpe"],
            max_dd=st["max_dd"], annual=st["annual"], turnover=turn,
            excess_sharpe_full=excess_sharpe(ex), es_early=es_e, es_late=es_l,
            boot_p=p, survives=survives))

        print(f"\n=== {name} ===")
        print(f"CAGR {st['cagr']*100:6.2f}%  vol {st['vol']*100:5.2f}%  "
              f"Sharpe {st['sharpe']:5.2f}  maxDD {st['max_dd']*100:7.2f}%  "
              f"ann.turnover(1-side) {turn:5.2f}x")
        print("Annual returns: " + "  ".join(
            f"{y}:{v*100:+.1f}%" for y, v in st["annual"].items()))
        print(f"Excess vs 60/40 — Sharpe full {excess_sharpe(ex):+.2f}  "
              f"early({early.index[0].date()}..{early.index[-1].date()}) {es_e:+.2f}  "
              f"late({late.index[0].date()}..) {es_l:+.2f}")
        print(f"Stationary block bootstrap p(mean excess <= 0) = {p:.4f} "
              f"(threshold {BONFERRONI_P:.4f})")
        print(f"SURVIVES: {survives}")

    print("\n=== VERDICT ===")
    n_pass = sum(r["survives"] for r in results)
    print(f"{n_pass}/3 pre-registered variants survive the gauntlet "
          f"(positive excess Sharpe in both halves AND bootstrap p < 0.0167).")
    print("Reminder: 2017-2026 window omits the 2011-2019 trend lost-decade middle; "
          "any pass is PROVISIONAL pending 3-6 months forward paper.")

    out = dict(
        eval_start=str(dates[start].date()), eval_end=str(dates[end].date()),
        benchmarks=dict(
            spy=dict(cagr=spy_stats["cagr"], sharpe=spy_stats["sharpe"],
                     max_dd=spy_stats["max_dd"],
                     annual={int(y): float(v) for y, v in spy_stats["annual"].items()}),
            b6040=dict(cagr=b_stats["cagr"], sharpe=b_stats["sharpe"],
                       max_dd=b_stats["max_dd"],
                       annual={int(y): float(v) for y, v in b_stats["annual"].items()})),
        variants=[dict(
            name=r["name"], cagr=r["cagr"], vol=r["vol"], sharpe=r["sharpe"],
            max_dd=r["max_dd"], turnover_ann=r["turnover"],
            annual={int(y): float(v) for y, v in r["annual"].items()},
            excess_sharpe_full=r["excess_sharpe_full"],
            excess_sharpe_early=r["es_early"], excess_sharpe_late=r["es_late"],
            bootstrap_p=r["boot_p"], survives=r["survives"]) for r in results])
    out_path = os.path.join(PROJECT, "data", "trend_backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON results -> {out_path}")


if __name__ == "__main__":
    main()
