"""Live ``RiskManager`` adapter — feeds Alpaca account state into the
existing 7-rule risk evaluator.

The shipped ``RiskManager`` reads P&L and exposure from the local SQLite
DB. That works fine for the signal-research pipeline, but the *live*
trader is the source of truth for what's actually filled at the broker.
So we subclass and override the four lookup helpers with values pulled
from Alpaca.

Rules left untouched (still evaluated from CONFIG):
  1. emergency_stop
  3. drift_block
  4. min_calibrated_p
  5. max_daily_trades (we still use Alpaca order count below)

Rules overridden:
  2. daily_loss_cap          -> Alpaca: equity - last_equity
  6. max_gross_exposure      -> Alpaca: long_market_value + short_market_value
     plus account_equity      -> Alpaca: equity
  7. max_single-sector       -> Alpaca positions joined with SECTOR_MAP
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from ..risk import RiskManager, SECTOR_MAP
from .alpaca_client import AlpacaClient, AlpacaError

log = logging.getLogger("marketradar.execution.live_risk")


class LiveRiskManager(RiskManager):
    """``RiskManager`` whose account/exposure helpers read from Alpaca."""

    def __init__(self, alpaca: AlpacaClient):
        super().__init__()
        self._alpaca = alpaca
        self._cache_at: Optional[datetime] = None
        self._cached_account = None
        self._cached_positions: list = []
        # 5-second cache prevents hammering Alpaca with one call per rule.
        from datetime import timedelta as _td
        self._cache_ttl = _td(seconds=5)

    # ------------------------------------------------------------------
    # Cache refresh
    # ------------------------------------------------------------------
    def _refresh_if_stale(self) -> bool:
        now = datetime.utcnow()
        if (
            self._cached_account is not None
            and self._cache_at is not None
            and (now - self._cache_at) < self._cache_ttl
        ):
            return True
        try:
            self._cached_account = self._alpaca.get_account()
            self._cached_positions = self._alpaca.get_positions()
        except AlpacaError as exc:
            log.warning("LiveRiskManager refresh failed: %s", exc)
            return False
        self._cache_at = now
        return True

    # ------------------------------------------------------------------
    # Overridden helpers
    # ------------------------------------------------------------------
    def _today_realized_pnl_usd(self) -> Optional[float]:
        """Approximate today's P&L from Alpaca's last_equity diff.

        Alpaca's ``last_equity`` is yesterday's closing equity, so
        ``equity - last_equity`` is intraday change (realized + unrealized).
        We use it as the kill-switch input — protecting capital takes
        precedence over the realized/unrealized distinction.
        """
        if not self._refresh_if_stale() or self._cached_account is None:
            return None
        a = self._cached_account
        return float(a.equity - a.last_equity)

    def _today_trade_count(self) -> int:
        """Count today's submitted orders via Alpaca."""
        try:
            since = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat() + "Z"
            orders = self._alpaca.list_orders(
                status="all", limit=500, after=since, nested=False
            )
        except AlpacaError as exc:
            log.warning("list_orders today failed: %s — failing closed", exc)
            # Returning a huge number guarantees the daily-trades rule blocks.
            return 10_000
        return len(orders)

    def _latest_account_equity(self) -> Optional[float]:
        if not self._refresh_if_stale() or self._cached_account is None:
            return None
        return float(self._cached_account.equity)

    def _current_gross_exposure_usd(self) -> Optional[float]:
        if not self._refresh_if_stale():
            return None
        return float(sum(abs(p.market_value) for p in self._cached_positions))

    def _current_sector_exposure_pct(
        self, sector: str, equity_usd: float
    ) -> Optional[float]:
        if equity_usd <= 0:
            return None
        if not self._refresh_if_stale():
            return None
        sector_value = 0.0
        for p in self._cached_positions:
            sym_sector = SECTOR_MAP.get(p.symbol.upper())
            if sym_sector == sector:
                sector_value += abs(p.market_value)
        return (sector_value / equity_usd) * 100.0

    def _hours_since_last_drift_alert(self) -> Optional[float]:
        # Defer to the parent (DB-backed) implementation. The drift detector
        # writes its alerts to SQLite via ml/drift.py.
        return super()._hours_since_last_drift_alert()
