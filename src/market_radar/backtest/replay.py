"""Backtest engine — replays the live trader's gate + sizing + exit logic
against historical signal_outcomes.

Approximations (be honest about them):
  - Entry price = price_at_flag.
  - Exit price for STOCK trades:
      Use return_5d_pct as the realized 5-day return. Approximate bracket
      hit logic: if hypothetical TP (entry * (1 + tp_atr_mult * atr_pct))
      is reached BEFORE stop (entry * (1 - sl_atr_mult * atr_pct)), exit
      at TP; else exit at stop. We don't have intra-day bars in the
      historical data, so we simulate the outcome by:
        - if final return >= TP threshold: exit at TP
        - elif final return <= -SL threshold: exit at SL
        - else: exit at final return (end-of-period)
      This MILDLY favors strategies with tight stops vs. wide stops.

  - Exit price for OPTIONS spread trades:
      Use a payoff curve. Given an underlying 5-day return r:
        - bull call debit: max_gain captured if r >= (short_strike - entry)/entry
                          0 captured if r <= 0
                          linear interp between
        - similar for bear put
      This is theoretical max-on-expiry; in reality you'd take profits earlier
      via the 50%-of-max rule, but for backtest we use a simple proxy.

  - Round-trip cost is PRICE-BUCKETED (100-400 bps for sub-$5 names; see
    _round_trip_cost_frac), NOT a flat 5 bps — the old 5 bps flattered every
    small-cap result by 1-2 orders of magnitude.
  - Daily P&L aggregated by the day the signal was scored.

UPPER-BOUND WARNING: without intraday bars we can't simulate the true price
PATH, so TP/SL logic uses the final 5-day return as a proxy. That is OPTIMISTIC
(it assumes a winner that ended above TP was never stopped out first). Treat any
positive result here as an UPPER BOUND: a strategy that fails this backtest will
certainly fail live; one that passes still needs a live pilot before scaling.
Corrupt-flagged outcomes (data_corrupt) are excluded.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal, Optional

from ..storage import get_connection

log = logging.getLogger("marketradar.backtest")

Mode = Literal["stock", "options"]

ATR_PCT_ASSUMPTION = 0.02  # 2% of price as ATR proxy if we don't have it

# Option spreads bleed the bid/ask of BOTH legs — model a heavy round-trip on
# the debit. Illiquid single-name verticals routinely give up 10%+ of the debit.
_OPTION_RT_COST_FRAC = float(os.getenv("BACKTEST_OPTION_RT_COST", "0.10"))


def _round_trip_cost_frac(price: float) -> float:
    """Realistic round-trip (entry+exit) transaction cost as a FRACTION of
    notional, bucketed by price (P1 rebuild).

    The old flat 5 bps was 1-2 orders of magnitude too low for the sub-$5
    microcaps this strategy trades: spread + slippage + market impact run
    100-300 bps+ round-trip there (a $0.05 spread on a $5 stock is a 1% round
    trip BEFORE any impact). Cost scales with 1/price because cheaper names have
    wider relative spreads. This is the dominant reason small-cap catalyst
    chasing rarely nets out — making it explicit is the whole point.
    """
    try:
        px = float(price)
    except (TypeError, ValueError):
        px = 0.0
    if px <= 0:
        return 0.040
    if px < 1:
        return 0.040    # 400 bps — sub-$1
    if px < 3:
        return 0.030    # 300 bps
    if px < 5:
        return 0.020    # 200 bps
    if px < 10:
        return 0.012    # 120 bps
    if px < 50:
        return 0.005    # 50 bps
    return 0.002        # 20 bps — liquid large caps


@dataclass
class StrategyPreset:
    name: str
    description: str
    composite_threshold: float = 0.0
    p_buy_min: float = 0.65
    p_sell_max: float = 0.35
    min_source_weight: float = 7.0
    require_factual: bool = True
    block_anti_pump: bool = True
    blocked_event_types: tuple[str, ...] = (
        "other", "proxy_statement", "passive_5pct_stake",
        "ipo_registration", "routine_prospectus", "routine_proxy",
        "material_event_amend", "activist_position",
    )
    # Risk + sizing
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5
    kelly_fraction: float = 0.25
    max_position_pct: float = 5.0
    # Options simulation
    mode: Mode = "stock"
    # Realistic equity start
    starting_equity_usd: float = 6300.0  # ~£5k


PRESETS: dict[str, StrategyPreset] = {
    "old_gate": StrategyPreset(
        name="old_gate",
        description="composite>=7.5 + model_p>=0.62 (the original v1 gate)",
        composite_threshold=7.5,
        p_buy_min=0.62, p_sell_max=0.38,
        min_source_weight=0.0, require_factual=False, block_anti_pump=False,
        blocked_event_types=(),
    ),
    "new_gate": StrategyPreset(
        name="new_gate",
        description="p>=0.65 OR <=0.35 + factual + source_weight>=7 + anti-pump + event filter",
    ),
    "new_gate_options": StrategyPreset(
        name="new_gate_options",
        description="Same as new_gate but using options spreads for the 13 whitelisted underlyings",
        mode="options",
    ),
    "aggressive": StrategyPreset(
        name="aggressive",
        description="Lower p threshold (0.60 / 0.40), more trades, more variance",
        p_buy_min=0.60, p_sell_max=0.40,
    ),
}


@dataclass
class TradeResult:
    score_id: int
    ticker: str
    direction: str
    scored_date: str
    entry: float
    exit_: float
    qty: float
    pnl_usd: float
    pnl_pct: float
    mode: Mode
    event_type: Optional[str]
    model_p: float


@dataclass
class BacktestConfig:
    preset: StrategyPreset
    from_date: Optional[str] = None     # 'YYYY-MM-DD'
    to_date: Optional[str] = None
    max_concurrent_per_day: int = 200   # cap total trades per day (raised for backtest realism)
    options_underlyings: frozenset[str] = field(default_factory=lambda: frozenset({
        "SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD",
        "META", "MSFT", "GOOGL", "AMZN", "COIN", "PLTR",
    }))
    # Options payoff assumptions
    options_width_pct_of_atr: float = 0.5    # spread width ≈ 0.5 * ATR
    options_debit_to_width_ratio: float = 0.40  # debit ≈ 40% of width
    verbose: bool = False


@dataclass
class BacktestResult:
    preset_name: str
    n_trades: int
    n_wins: int
    n_losses: int
    hit_rate: float
    avg_pnl_pct: float
    avg_pnl_usd: float
    total_pnl_usd: float
    starting_equity_usd: float
    ending_equity_usd: float
    total_return_pct: float
    n_days: int
    avg_daily_pnl_usd: float
    median_daily_pnl_usd: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    sharpe_annualized: float
    daily_pnl_curve: list[tuple[str, float, float]]  # (date, daily_pnl, cumulative_equity)
    by_event_type: dict[str, dict]
    by_direction: dict[str, dict]
    config: BacktestConfig


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_backtest(cfg: BacktestConfig) -> BacktestResult:
    candidates = _fetch_candidates(cfg)
    log.info("backtest: %d candidates after gate (preset=%s)",
             len(candidates), cfg.preset.name)

    trades: list[TradeResult] = []
    equity = cfg.preset.starting_equity_usd
    daily_trades: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}
    by_event: dict[str, list[float]] = {}
    by_direction: dict[str, list[float]] = {}

    for cand in candidates:
        scored_date = (cand["scored_at"] or "")[:10]
        if not scored_date:
            continue
        if daily_trades.get(scored_date, 0) >= cfg.max_concurrent_per_day:
            continue
        trade = _simulate_trade(cand, cfg=cfg, equity=equity)
        if trade is None:
            continue
        trades.append(trade)
        daily_trades[scored_date] = daily_trades.get(scored_date, 0) + 1
        daily_pnl[scored_date] = daily_pnl.get(scored_date, 0.0) + trade.pnl_usd
        equity += trade.pnl_usd
        by_event.setdefault(trade.event_type or "?", []).append(trade.pnl_usd)
        by_direction.setdefault(trade.direction, []).append(trade.pnl_usd)

    return _compile_result(trades, daily_pnl, cfg, equity, by_event, by_direction)


# ---------------------------------------------------------------------------
# Candidate fetch (mirrors live_trader._fetch_candidates)
# ---------------------------------------------------------------------------

def _fetch_candidates(cfg: BacktestConfig) -> list[dict]:
    p = cfg.preset
    blocked = p.blocked_event_types
    blocked_clause = ""
    blocked_params: tuple = ()
    if blocked:
        placeholders = ",".join("?" * len(blocked))
        blocked_clause = (
            f"AND (ss.event_type IS NULL "
            f"     OR ss.event_type LIKE 'pa\\_%' ESCAPE '\\' "
            f"     OR ss.event_type NOT IN ({placeholders}))"
        )
        blocked_params = tuple(blocked)
    factual_clause = "AND COALESCE(ss.factual,0) = 1" if p.require_factual else ""
    pump_clause = "AND COALESCE(ss.anti_pump_flag,0) = 0" if p.block_anti_pump else ""
    date_clauses, date_params = [], []
    if cfg.from_date:
        date_clauses.append("ss.scored_at >= ?")
        date_params.append(cfg.from_date + "T00:00:00Z")
    if cfg.to_date:
        date_clauses.append("ss.scored_at <= ?")
        date_params.append(cfg.to_date + "T23:59:59Z")
    date_clause = " AND " + " AND ".join(date_clauses) if date_clauses else ""

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT ss.id AS score_id, ss.ticker, ss.event_type, ss.model_p_5d AS model_p,
                   ss.composite_score, ss.scored_at, ss.factual, ss.source_weight,
                   so.price_at_flag, so.return_5d_pct, so.return_1d_pct
            FROM signal_scores ss
            JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE ss.composite_score >= ?
              AND ss.model_p_5d IS NOT NULL
              AND (ss.model_p_5d >= ? OR ss.model_p_5d <= ?)
              AND COALESCE(ss.source_weight,0) >= ?
              AND so.return_5d_pct IS NOT NULL
              AND COALESCE(so.data_corrupt, 0) = 0
              AND so.return_5d_pct BETWEEN -50 AND 50
              AND so.price_at_flag > 0
              {factual_clause}
              {pump_clause}
              {blocked_clause}
              {date_clause}
            ORDER BY ss.scored_at ASC
            """,
            (p.composite_threshold, p.p_buy_min, p.p_sell_max,
             p.min_source_weight, *blocked_params, *date_params),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Trade simulator
# ---------------------------------------------------------------------------

def _simulate_trade(cand: dict, *, cfg: BacktestConfig, equity: float) -> Optional[TradeResult]:
    p = cfg.preset
    model_p = float(cand["model_p"])
    direction = "buy" if model_p >= p.p_buy_min else "sell"
    entry = float(cand["price_at_flag"])
    if entry <= 0:
        return None
    ret_5d_pct = float(cand["return_5d_pct"]) / 100.0  # convert pct to fraction
    # For sells (short), the trade P&L is inverted relative to underlying return
    realized_pct = ret_5d_pct if direction == "buy" else -ret_5d_pct

    # Decide route: options vs stock
    symbol = (cand["ticker"] or "").upper()
    use_options = (p.mode == "options" and symbol in cfg.options_underlyings)

    if use_options:
        return _simulate_option_spread(cand, cfg, equity, direction, entry, realized_pct)
    return _simulate_stock_trade(cand, cfg, equity, direction, entry, realized_pct)


def _simulate_stock_trade(cand, cfg, equity, direction, entry, realized_pct) -> Optional[TradeResult]:
    p = cfg.preset
    atr_pct = ATR_PCT_ASSUMPTION
    # Bracket trip math:
    tp_threshold = p.tp_atr_mult * atr_pct       # e.g., 2.5 * 0.02 = 0.05
    sl_threshold = p.sl_atr_mult * atr_pct       # e.g., 1.5 * 0.02 = 0.03
    # Partial PATH correction (P1 rebuild): without intraday bars we can't know
    # the true path, but the 1-day return reveals whether the stop was breached
    # EARLY. If day-1 already hit the stop, we were stopped out then and cannot
    # ride the 5-day recovery — the single biggest correction to the old
    # look-ahead that assumed every winner cleanly caught its TP. (Still mildly
    # optimistic on TP timing; a true fix needs intraday bars.)
    r1 = cand.get("return_1d_pct")
    realized_1d = None
    if r1 is not None:
        realized_1d = float(r1) / 100.0
        if direction == "sell":
            realized_1d = -realized_1d
    if realized_1d is not None and realized_1d <= -sl_threshold:
        exit_pct = -sl_threshold          # stopped out on day 1 — can't recover
    elif realized_pct >= tp_threshold:
        exit_pct = tp_threshold
    elif realized_pct <= -sl_threshold:
        exit_pct = -sl_threshold
    else:
        exit_pct = realized_pct
    # Realistic, price-bucketed round-trip cost hits every trade — the dominant
    # reason small-cap catalyst trades don't net out (see _round_trip_cost_frac).
    exit_pct -= _round_trip_cost_frac(entry)

    # Kelly sizing using TP:SL ratio
    b = tp_threshold / sl_threshold
    kelly_raw = (cand["model_p"] * b - (1 - cand["model_p"])) / b if direction == "buy" \
        else ((1 - cand["model_p"]) * b - cand["model_p"]) / b
    if kelly_raw <= 0:
        return None
    size_pct = min(kelly_raw * p.kelly_fraction * 100, p.max_position_pct)
    notional = equity * size_pct / 100
    qty = max(int(notional / entry), 0)
    if qty < 1:
        return None
    actual_notional = qty * entry
    pnl_usd = actual_notional * exit_pct
    exit_price = entry * (1 + exit_pct if direction == "buy" else 1 - exit_pct)
    return TradeResult(
        score_id=cand["score_id"], ticker=cand["ticker"], direction=direction,
        scored_date=(cand["scored_at"] or "")[:10],
        entry=entry, exit_=exit_price, qty=qty,
        pnl_usd=pnl_usd, pnl_pct=exit_pct * 100,
        mode="stock", event_type=cand.get("event_type"), model_p=cand["model_p"],
    )


def _simulate_option_spread(cand, cfg, equity, direction, entry, realized_pct) -> Optional[TradeResult]:
    """Approximate vertical debit spread payoff."""
    p = cfg.preset
    # Spread width as a fraction of entry
    atr_pct = ATR_PCT_ASSUMPTION
    width_pct = cfg.options_width_pct_of_atr * atr_pct
    debit_pct = cfg.options_debit_to_width_ratio * width_pct
    # max_gain_pct = (width - debit) per spread, expressed as fraction of entry
    max_gain_pct = (width_pct - debit_pct)
    max_loss_pct = debit_pct  # in fraction-of-entry terms; on a debit basis this maps to 100% loss
    # Effective payoff:
    # If favorable move >= width_pct: capture max_gain
    # If favorable move <= 0: capture max_loss (full debit lost)
    # In between: linear interpolation
    fav_move = realized_pct if direction == "buy" else -realized_pct
    if fav_move <= 0:
        # Full debit loss (max loss)
        spread_pnl_pct = -max_loss_pct
    elif fav_move >= width_pct:
        spread_pnl_pct = max_gain_pct
    else:
        # Linear interp between -debit (at 0) and +max_gain (at width)
        spread_pnl_pct = -debit_pct + (fav_move / width_pct) * (max_gain_pct + debit_pct)

    # Asymmetric Kelly sizing on the spread
    b = max_gain_pct / max_loss_pct  # reward:risk ratio in this approximation
    kelly_raw = (cand["model_p"] * b - (1 - cand["model_p"])) / b
    if kelly_raw <= 0:
        return None
    size_pct = min(kelly_raw * p.kelly_fraction * 100, p.max_position_pct)
    # Number of contracts ≈ size_pct of equity / (debit_per_spread * 100)
    debit_per_spread = entry * debit_pct  # $/share
    max_loss_per_spread = debit_per_spread * 100  # $/contract
    budget = equity * size_pct / 100
    contracts = max(int(budget / max_loss_per_spread), 0)
    if contracts < 1:
        return None
    total_debit = contracts * max_loss_per_spread
    pnl_usd = total_debit * (spread_pnl_pct / debit_pct)  # scale: -debit -> -total_debit, +max_gain -> +total*gain/loss
    # Slippage proxy: 5bps of notional
    pnl_usd -= total_debit * _OPTION_RT_COST_FRAC
    return TradeResult(
        score_id=cand["score_id"], ticker=cand["ticker"], direction=direction,
        scored_date=(cand["scored_at"] or "")[:10],
        entry=entry, exit_=entry * (1 + realized_pct),
        qty=contracts,
        pnl_usd=pnl_usd, pnl_pct=spread_pnl_pct * 100,
        mode="options", event_type=cand.get("event_type"), model_p=cand["model_p"],
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _compile_result(trades, daily_pnl, cfg, ending_equity,
                    by_event, by_direction) -> BacktestResult:
    p = cfg.preset
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl_usd > 0)
    losses = sum(1 for t in trades if t.pnl_usd < 0)
    total = sum(t.pnl_usd for t in trades)
    starting = p.starting_equity_usd
    days_sorted = sorted(daily_pnl.keys())
    n_days = len(days_sorted)

    # Equity curve + drawdown
    curve: list[tuple[str, float, float]] = []
    eq = starting
    peak = starting
    max_dd_usd = 0.0
    for d in days_sorted:
        pnl = daily_pnl[d]
        eq += pnl
        peak = max(peak, eq)
        dd = peak - eq
        max_dd_usd = max(max_dd_usd, dd)
        curve.append((d, pnl, eq))
    max_dd_pct = (max_dd_usd / peak * 100) if peak > 0 else 0.0

    # Sharpe (daily basis, annualized assuming 252 trading days)
    if n_days > 1:
        daily = [daily_pnl[d] for d in days_sorted]
        mean = sum(daily) / n_days
        var = sum((x - mean) ** 2 for x in daily) / n_days
        sd = math.sqrt(var)
        sharpe = (mean / sd * math.sqrt(252)) if sd > 0 else 0.0
        median = sorted(daily)[n_days // 2]
    else:
        sharpe = 0.0
        median = 0.0

    by_event_summary = {
        et: {
            "n": len(pnls),
            "total_usd": sum(pnls),
            "avg_usd": sum(pnls) / len(pnls),
            "hit_rate": sum(1 for x in pnls if x > 0) / len(pnls) * 100,
        } for et, pnls in by_event.items() if pnls
    }
    by_dir_summary = {
        d: {
            "n": len(pnls), "total_usd": sum(pnls),
            "avg_usd": sum(pnls) / len(pnls),
            "hit_rate": sum(1 for x in pnls if x > 0) / len(pnls) * 100,
        } for d, pnls in by_direction.items() if pnls
    }

    return BacktestResult(
        preset_name=p.name,
        n_trades=n, n_wins=wins, n_losses=losses,
        hit_rate=(wins / n * 100) if n else 0.0,
        avg_pnl_pct=sum(t.pnl_pct for t in trades) / n if n else 0.0,
        avg_pnl_usd=total / n if n else 0.0,
        total_pnl_usd=total,
        starting_equity_usd=starting,
        ending_equity_usd=ending_equity,
        total_return_pct=(ending_equity - starting) / starting * 100,
        n_days=n_days,
        avg_daily_pnl_usd=total / n_days if n_days else 0.0,
        median_daily_pnl_usd=median,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_usd=max_dd_usd,
        sharpe_annualized=sharpe,
        daily_pnl_curve=curve,
        by_event_type=by_event_summary,
        by_direction=by_dir_summary,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Pretty printer (used by CLI)
# ---------------------------------------------------------------------------

def format_result(r: BacktestResult) -> str:
    lines = []
    p = r.config.preset
    lines.append(f"=== Backtest: {p.name} ===")
    lines.append(f"  {p.description}")
    lines.append("")
    lines.append(f"  Trades:        {r.n_trades}  ({r.n_wins} wins / {r.n_losses} losses)")
    lines.append(f"  Hit rate:      {r.hit_rate:.1f}%")
    lines.append(f"  Avg P&L/trade: ${r.avg_pnl_usd:+.2f} ({r.avg_pnl_pct:+.2f}%)")
    lines.append(f"  Total P&L:     ${r.total_pnl_usd:+,.0f}")
    lines.append(f"  Starting equity: ${r.starting_equity_usd:,.0f}")
    lines.append(f"  Ending equity:   ${r.ending_equity_usd:,.0f}")
    lines.append(f"  Total return:    {r.total_return_pct:+.1f}%")
    lines.append(f"  Trading days:    {r.n_days}")
    lines.append(f"  Avg daily P&L:   ${r.avg_daily_pnl_usd:+.2f}")
    lines.append(f"  Median daily:    ${r.median_daily_pnl_usd:+.2f}")
    lines.append(f"  Max drawdown:    ${r.max_drawdown_usd:,.0f} ({r.max_drawdown_pct:.1f}%)")
    lines.append(f"  Sharpe (annual): {r.sharpe_annualized:.2f}")
    lines.append("")
    lines.append("  By direction:")
    for d, s in sorted(r.by_direction.items()):
        lines.append(f"    {d:5s} n={s['n']:5d}  avg=${s['avg_usd']:+7.2f}  "
                     f"hit={s['hit_rate']:5.1f}%  total=${s['total_usd']:+9.0f}")
    lines.append("")
    lines.append("  Top event types by P&L:")
    sorted_events = sorted(r.by_event_type.items(),
                           key=lambda kv: -kv[1]["total_usd"])[:10]
    for et, s in sorted_events:
        lines.append(f"    {et:30s} n={s['n']:5d}  avg=${s['avg_usd']:+7.2f}  "
                     f"hit={s['hit_rate']:5.1f}%  total=${s['total_usd']:+9.0f}")
    return "\n".join(lines)
