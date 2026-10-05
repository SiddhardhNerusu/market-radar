# Results snapshots

Small output files kept from the research and paper-trading runs. The full
SQLite database, model artefacts and logs were never committed.

- `trend_backtest_results.json` — output of `scripts/trend_backtest.py`:
  trend-following variants against SPY, 2017-02 to 2026-07 (CAGR, Sharpe,
  max drawdown, per-year returns).
- `option_counterfactuals.csv` — hourly tracker comparing each paper debit
  spread's actual exit with what it would have been worth if held
  (`hold_minus_actual`), June 2026.
