"""Backtest the live trader's strategy against historical signal_outcomes.

Replays the same gate / size / stop / TP logic the live trader uses, but
against the 156k labeled outcomes in signal_outcomes. Outputs:
  - daily P&L curve
  - hit rate / Sharpe / max drawdown
  - per-event-type breakdown
  - comparison across strategy presets (old gate vs new gate vs new+confluence)

Used to validate the strategy BEFORE risking real money. Run with:

    python scripts/backtest.py --preset new_gate

This is an APPROXIMATE backtest:
  - We use the recorded return_5d_pct as the realised outcome
  - We assume bracket stops fire at the ATR-multiple price (1.5x ATR loss / 2.5x ATR gain)
    when intra-period high/low would have hit them — but we don't have intra-day
    bars in the historical data, so we use end-of-period return as a proxy.
  - Slippage modeled as 5bps per trade (typical for liquid Alpaca symbols).
  - Options simulation uses the theoretical max-gain / max-loss capture rate
    derived from the underlying return.

These approximations are honest about their limits — see the per-section comments.
"""
from .replay import (
    BacktestConfig,
    BacktestResult,
    StrategyPreset,
    PRESETS,
    run_backtest,
)

__all__ = ["BacktestConfig", "BacktestResult", "StrategyPreset", "PRESETS", "run_backtest"]
