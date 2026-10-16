# Results snapshots

Small output files kept from the research and paper-trading runs. The full
SQLite database, model artefacts and logs were never committed.

- `trend_backtest_results.json` — output of `scripts/trend_backtest.py`:
  trend-following variants against SPY, 2017-02 to 2026-07 (CAGR, Sharpe,
  max drawdown, per-year returns).
- `option_counterfactuals.csv` — hourly tracker comparing each paper debit
  spread's actual exit with what it would have been worth if held
  (`hold_minus_actual`), June 2026.

## Form 4 insider-purchase event study (2026-10-16, SEC data-set history)

`event_study_form4_*.json` — output of `scripts/event_study_form4.py` over the
SEC Insider Transactions Data Sets (2024q1–2026q1, loaded by
`scripts/backfill_form4_datasets.py`). Code-P open-market purchases, one event
per (ticker, event_day), ≥ $10 names, 10b5-1 plan trades excluded, entry at the
event-day close (optimistic), returns to the 5th / 20th close, minus SPY over
the same window, minus the price-tier round-trip cost. Price source: yfinance
daily (no Alpaca keys at the time).

| cut | events | days | 5d mean | 5d median | 5d win | 5d drop-top-10% | 5d 95% CI | 20d median |
|---|---|---|---|---|---|---|---|---|
| all ≥ $25k | 6,624 | 552 | −0.25% | −0.58% | 43.2% | −1.52% | [−0.46, −0.05] | −1.10% |
| senior (director / C-suite) | 4,930 | 538 | −0.29% | −0.59% | 43.3% | −1.49% | [−0.50, −0.06] | −1.16% |
| ≥ $100k | 4,424 | 548 | −0.19% | −0.59% | 43.6% | −1.56% | [−0.44, 0.05] | −1.20% |
| 2026 Q1 only | 897 | 61 | +0.08% | −0.31% | 45.9% | −1.42% | [−0.59, 0.76] | +0.16% |

Every cut FAILS the promotion gate at both horizons: the net median is negative,
the trimmed mean is negative, the mean after dropping the top 10% of winners is
−1.5% to −3%, and for the full sample the day-cluster bootstrap CI of the net
mean sits entirely below zero at 5 days. The pre-cost abnormal mean is slightly
positive and "significant" against the ticker-matched placebo (p 0.01–0.05),
which is exactly the June pattern: a thin average carried by a few large
winners, eaten whole by costs, with the typical bet losing. For a retail
long-only buyer this signal has no tradeable edge on this data.
