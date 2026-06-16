"""Lock the backtest-realism fixes (P1 rebuild): price-bucketed round-trip costs
and the day-1 stop path correction — the two changes that flipped the look-ahead
Sharpe from +6.6 to -5.0. The final re-audit flagged ZERO backtest tests.
"""
from market_radar.backtest.replay import (
    PRESETS,
    BacktestConfig,
    _round_trip_cost_frac,
    _simulate_stock_trade,
)


def test_cost_buckets_microcap_expensive_and_monotonic():
    c = _round_trip_cost_frac
    assert c(0.5) == 0.040 and c(2) == 0.030 and c(4) == 0.020
    assert c(8) == 0.012 and c(20) == 0.005 and c(100) == 0.002
    costs = [c(p) for p in (0.5, 2, 4, 8, 20, 100)]
    assert costs == sorted(costs, reverse=True), "cheaper names must cost more"


def test_cost_handles_none_zero_negative():
    assert _round_trip_cost_frac(None) == 0.040
    assert _round_trip_cost_frac(0) == 0.040
    assert _round_trip_cost_frac(-5) == 0.040


def _cand(return_1d_pct):
    return {"model_p": 0.70, "return_1d_pct": return_1d_pct, "score_id": 1,
            "ticker": "AAA", "scored_at": "2026-06-01", "event_type": "x"}


def test_day1_stop_flips_a_5d_winner_that_was_stopped_first():
    cfg = BacktestConfig(preset=PRESETS["new_gate"])
    # Both ended the 5d window at +10%. The only difference is the day-1 path.
    benign = _simulate_stock_trade(_cand(1.0), cfg, 10_000.0, "buy", 10.0, 0.10)
    stopped = _simulate_stock_trade(_cand(-5.0), cfg, 10_000.0, "buy", 10.0, 0.10)
    assert benign is not None and stopped is not None
    assert benign.pnl_pct > 0, "a real winner with a benign day-1 should net positive"
    assert stopped.pnl_pct < 0, (
        "a position whose day-1 move already breached the stop must exit at the "
        "stop — it cannot ride the 5-day recovery (the look-ahead correction)")
    assert stopped.pnl_pct < benign.pnl_pct
