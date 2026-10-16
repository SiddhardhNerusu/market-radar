"""LaneConfig — every knob of the insider lane, from ``LANE_*`` env vars.

Defaults encode the sizing doctrine from docs/EDGE_HUNT_2026-06-16.md §2.5:
per-name risk <= 0.5% of the book on a real 2x ATR stop, <= 5 concurrent
names, a 5% portfolio-heat ceiling, and a 3% daily-loss kill switch that
FLATTENS (the old $200 cap could never bind and only blocked new entries).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _f(k: str, d: float) -> float:
    v = os.getenv(k, "").strip()
    try:
        return float(v) if v else d
    except ValueError:
        return d


def _i(k: str, d: int) -> int:
    return int(_f(k, d))


def _b(k: str, d: bool) -> bool:
    v = os.getenv(k, "").strip().lower()
    return d if not v else v in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LaneConfig:
    # --- loop -------------------------------------------------------------
    interval_seconds: int = 60
    dry_run: bool = False

    # --- the book the lane sizes against ------------------------------------
    # Alpaca paper starts at $100k. Set LANE_BOOK_EQUITY_USD to the capital you
    # actually have (e.g. 10000) so every % below means what it says. 0 => use
    # the broker's reported equity.
    book_equity_usd: float = 10_000.0

    # --- signal qualification (what counts as a bet) --------------------------
    min_notional_usd: float = 25_000.0     # summed insider purchase $ that day
    min_price: float = 10.0                # the tier where the tilt was not a cost artifact
    min_role_score: int = 0                # 0 any insider; 2 director+; 3 C-suite
    exclude_10b5_1: bool = True            # plan trades are routine, not informative
    freshness_hours: int = 48              # published_at must be within this window
    ticker_cooldown_days: int = 10         # no second bet on a ticker inside this window

    # --- entry mechanics -------------------------------------------------------
    entry_limit_bps: float = 20.0          # marketable limit: ask * (1 + bps)
    max_spread_bps: float = 50.0           # skip if (ask-bid)/mid wider than this
    no_entry_first_minutes: int = 15       # skip the opening auction chaos
    no_entry_last_minutes: int = 30        # no fresh multi-day bets into the close
    entry_ttl_minutes: int = 60            # cancel an unfilled parent after this

    # --- exits -----------------------------------------------------------------
    stop_atr_mult: float = 2.0             # broker-side stop, 2x ATR(14)
    tp_atr_mult: float = 4.0               # broker-side target, 2R
    max_hold_calendar_days: int = 14       # time stop (~10 trading days)

    # --- risk ------------------------------------------------------------------
    risk_per_trade_pct: float = 0.5        # % of book at risk per bet (entry -> stop)
    max_position_pct: float = 20.0         # notional cap per name, % of book
    max_positions: int = 5
    max_portfolio_heat_pct: float = 5.0    # sum of open risk_usd / book
    daily_loss_pct: float = 3.0            # equity drop vs day start => flatten + halt
    gross_cap_usd: float = 0.0             # 0 => book_equity_usd
    hard_order_notional_usd: float = 2_500.0

    @property
    def candidate_name(self) -> str:
        """Stable label for the edge_ledger cohort this config defines."""
        tag = "insider_P"
        tag += f"_px{int(self.min_price)}"
        tag += f"_notional{int(self.min_notional_usd // 1000)}k"
        if self.min_role_score > 0:
            tag += f"_role{self.min_role_score}"
        if self.exclude_10b5_1:
            tag += "_no10b5"
        return tag

    @classmethod
    def from_env(cls) -> "LaneConfig":
        return cls(
            interval_seconds=_i("LANE_INTERVAL_SECONDS", 60),
            dry_run=_b("LANE_DRY_RUN", False),
            book_equity_usd=_f("LANE_BOOK_EQUITY_USD", 10_000.0),
            min_notional_usd=_f("LANE_MIN_NOTIONAL_USD", 25_000.0),
            min_price=_f("LANE_MIN_PRICE", 10.0),
            min_role_score=_i("LANE_MIN_ROLE_SCORE", 0),
            exclude_10b5_1=_b("LANE_EXCLUDE_10B5_1", True),
            freshness_hours=_i("LANE_FRESHNESS_HOURS", 48),
            ticker_cooldown_days=_i("LANE_TICKER_COOLDOWN_DAYS", 10),
            entry_limit_bps=_f("LANE_ENTRY_LIMIT_BPS", 20.0),
            max_spread_bps=_f("LANE_MAX_SPREAD_BPS", 50.0),
            no_entry_first_minutes=_i("LANE_NO_ENTRY_FIRST_MINUTES", 15),
            no_entry_last_minutes=_i("LANE_NO_ENTRY_LAST_MINUTES", 30),
            entry_ttl_minutes=_i("LANE_ENTRY_TTL_MINUTES", 60),
            stop_atr_mult=_f("LANE_STOP_ATR_MULT", 2.0),
            tp_atr_mult=_f("LANE_TP_ATR_MULT", 4.0),
            max_hold_calendar_days=_i("LANE_MAX_HOLD_CALENDAR_DAYS", 14),
            risk_per_trade_pct=_f("LANE_RISK_PER_TRADE_PCT", 0.5),
            max_position_pct=_f("LANE_MAX_POSITION_PCT", 20.0),
            max_positions=_i("LANE_MAX_POSITIONS", 5),
            max_portfolio_heat_pct=_f("LANE_MAX_PORTFOLIO_HEAT_PCT", 5.0),
            daily_loss_pct=_f("LANE_DAILY_LOSS_PCT", 3.0),
            gross_cap_usd=_f("LANE_GROSS_CAP_USD", 0.0),
            hard_order_notional_usd=_f("LANE_HARD_ORDER_NOTIONAL_USD", 2_500.0),
        )

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty means coherent.
        The trader refuses to start on any problem (fail-closed startup)."""
        p: list[str] = []
        if self.book_equity_usd < 0:
            p.append("LANE_BOOK_EQUITY_USD must be >= 0")
        if not (0 < self.risk_per_trade_pct <= 2.0):
            p.append("LANE_RISK_PER_TRADE_PCT must be in (0, 2]")
        if self.max_portfolio_heat_pct < self.risk_per_trade_pct:
            p.append("portfolio heat cap must be >= per-trade risk")
        if not (0 < self.daily_loss_pct <= 10.0):
            p.append("LANE_DAILY_LOSS_PCT must be in (0, 10]")
        if self.daily_loss_pct < self.risk_per_trade_pct:
            p.append("daily loss cap must be >= per-trade risk")
        if self.stop_atr_mult <= 0 or self.tp_atr_mult <= self.stop_atr_mult:
            p.append("need 0 < LANE_STOP_ATR_MULT < LANE_TP_ATR_MULT")
        if self.max_positions < 1:
            p.append("LANE_MAX_POSITIONS must be >= 1")
        if self.min_price < 5.0:
            p.append("LANE_MIN_PRICE below $5 puts the lane in the measured-negative microcap tier")
        if self.hard_order_notional_usd <= 0:
            p.append("LANE_HARD_ORDER_NOTIONAL_USD must be > 0")
        return p
