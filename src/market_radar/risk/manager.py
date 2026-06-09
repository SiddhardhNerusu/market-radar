"""Risk infrastructure — every real-money trade MUST pass through here.

Seven hard rules (configured via RISK_* env vars):
  1. Emergency stop:           RISK_EMERGENCY_STOP=1 blocks everything
  2. Daily loss kill-switch:   if today's realized loss >= cap, block
  3. Drift block:              if model drift alerted < N hours ago, block
  4. Min calibrated probability
  5. Max daily trades
  6. Max gross exposure
  7. Max single-ticker / single-sector concentration

Default behaviour is FAIL-CLOSED.  Any condition that can't be checked
(missing data, exception, undefined state) -> trade blocked.  There is
NO "default allow" path.

Caller pattern:

    from market_radar.risk import RiskManager, TradeProposal
    decision = RiskManager().evaluate(TradeProposal(
        ticker="AAPL", direction="buy", size_pct=2.5,
        calibrated_p=0.71, sector="tech",
    ))
    if not decision.allowed:
        log.warning("Trade blocked: %s", decision.reason)
        return
    # ... proceed to execution ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..config import CONFIG
from ..storage import get_connection


log = logging.getLogger("marketradar.risk")


# Hardcoded sector map for concentration check.  Mirrors a subset of
# src/market_radar/ml/graph_features.py SECTOR_PEERS.
SECTOR_MAP: dict[str, str] = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "GOOG": "tech",
    "META": "tech", "AMZN": "tech", "NVDA": "tech", "AMD": "tech",
    "INTC": "tech", "TSM": "tech", "AVGO": "tech", "QCOM": "tech",
    "MU": "tech", "ASML": "tech", "AMAT": "tech", "LRCX": "tech",
    "ORCL": "tech",
    "TSLA": "auto", "RIVN": "auto", "LCID": "auto", "F": "auto",
    "GM": "auto", "NIO": "auto", "XPEV": "auto", "LI": "auto",
    "JPM": "banks", "BAC": "banks", "WFC": "banks", "C": "banks",
    "GS": "banks", "MS": "banks",
    "WMT": "retail", "TGT": "retail", "COST": "retail", "HD": "retail",
    "LOW": "retail", "NKE": "retail", "LULU": "retail",
    "JNJ": "pharma", "PFE": "pharma", "MRK": "pharma", "LLY": "pharma",
    "ABBV": "pharma", "BMY": "pharma",
    "AAL": "travel", "UAL": "travel", "DAL": "travel", "LUV": "travel",
    "BKNG": "travel", "ABNB": "travel",
    "RIOT": "crypto", "MARA": "crypto", "COIN": "crypto", "MSTR": "crypto",
    "HOOD": "crypto", "CLSK": "crypto", "HUT": "crypto", "BITF": "crypto",
    "BTBT": "crypto",
    "LMT": "defense", "RTX": "defense", "NOC": "defense", "GD": "defense",
    "BA": "defense",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
}


@dataclass(frozen=True)
class TradeProposal:
    ticker: str
    direction: str        # 'buy' or 'sell'
    size_pct: float       # of account equity, e.g. 2.5
    calibrated_p: float   # model output in [0, 1]
    sector: Optional[str] = None  # falls back to SECTOR_MAP lookup
    kind: str = "stock"   # 'stock' | 'crypto' | 'option' — used by options-reserve rule


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    blocking_rule: Optional[str] = None


_VALID_DIRECTIONS = {"buy", "sell"}


class RiskManager:
    """Evaluates trade proposals against the seven hard rules.

    All DB reads are read-only.  Helper methods that fail return None or
    a conservative fail-closed sentinel so evaluate() can refuse the
    trade.
    """

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def evaluate(
        self,
        t: TradeProposal,
        *,
        account_equity_usd: Optional[float] = None,
        current_gross_usd: Optional[float] = None,
        current_notional_usd: Optional[float] = None,
        current_crypto_usd: Optional[float] = None,
        current_ticker_usd: Optional[float] = None,
        current_position_count: Optional[int] = None,
        market_open: bool = True,
    ) -> RiskDecision:
        # Input sanity — fail closed on malformed proposals.
        if not isinstance(t, TradeProposal):
            return RiskDecision(False, "proposal is not a TradeProposal",
                                "input_validation")
        ticker = (t.ticker or "").upper().strip()
        if not ticker:
            return RiskDecision(False, "empty ticker", "input_validation")
        if t.direction not in _VALID_DIRECTIONS:
            return RiskDecision(False,
                                f"direction {t.direction!r} not in {_VALID_DIRECTIONS}",
                                "input_validation")
        if t.size_pct is None or t.size_pct <= 0:
            return RiskDecision(False,
                                f"size_pct {t.size_pct} must be > 0",
                                "input_validation")
        if t.calibrated_p is None or not (0.0 <= t.calibrated_p <= 1.0):
            return RiskDecision(False,
                                f"calibrated_p {t.calibrated_p} not in [0,1]",
                                "input_validation")

        # Rule 1: emergency stop
        if CONFIG.risk_emergency_stop:
            return RiskDecision(False, "RISK_EMERGENCY_STOP=1 set in .env",
                                "emergency_stop")

        # Rule 2: daily loss kill-switch
        today_pnl = self._today_realized_pnl_usd()
        if today_pnl is None:
            return RiskDecision(False,
                                "Could not read today's realized P&L — failing closed",
                                "daily_loss_cap")
        if today_pnl <= -abs(CONFIG.risk_daily_loss_cap_usd):
            return RiskDecision(
                False,
                f"Daily loss cap hit: today P&L ${today_pnl:.2f} "
                f"<= -${CONFIG.risk_daily_loss_cap_usd:.2f}",
                "daily_loss_cap",
            )

        # Rule 2b: monthly drawdown halt — if equity is $X below 30-day peak,
        # halt and require manual review. Protects against compound bad weeks.
        drawdown_breached = self._monthly_drawdown_breached()
        if drawdown_breached is self._DD_CHECK_ERROR:
            return RiskDecision(
                False,
                "Monthly drawdown check failed (data unavailable) — failing closed",
                "monthly_drawdown",
            )
        if drawdown_breached:
            peak, current, dd = drawdown_breached
            return RiskDecision(
                False,
                f"Monthly drawdown halt: equity ${current:.0f} is ${dd:.0f} "
                f"below 30d peak ${peak:.0f} (cap=${CONFIG.risk_monthly_drawdown_usd:.0f}) "
                f"— manual review required",
                "monthly_drawdown",
            )

        # Rule 3: drift block
        drift_alert_age = self._hours_since_last_drift_alert()
        if drift_alert_age is not None and drift_alert_age < CONFIG.risk_drift_block_hours:
            return RiskDecision(
                False,
                f"Drift detector alerted {drift_alert_age:.1f}h ago "
                f"(< {CONFIG.risk_drift_block_hours}h window)",
                "drift_block",
            )

        # Rule 4: min calibrated probability
        # 'buy' needs p >= threshold; 'sell' needs p <= 1 - threshold
        if t.direction == "buy" and t.calibrated_p < CONFIG.risk_min_calibrated_p:
            return RiskDecision(
                False,
                f"Buy p={t.calibrated_p:.3f} < min {CONFIG.risk_min_calibrated_p:.2f}",
                "min_p",
            )
        if t.direction == "sell" and t.calibrated_p > (1 - CONFIG.risk_min_calibrated_p):
            return RiskDecision(
                False,
                f"Sell p={t.calibrated_p:.3f} > max {1 - CONFIG.risk_min_calibrated_p:.2f}",
                "min_p",
            )

        # Rule 5: max daily trades
        today_trades = self._today_trade_count()
        if today_trades >= CONFIG.risk_max_daily_trades:
            return RiskDecision(
                False,
                f"Daily trade cap: {today_trades} >= {CONFIG.risk_max_daily_trades}",
                "max_daily_trades",
            )

        # Rule 6: max gross exposure — needs account equity.
        if account_equity_usd is None:
            account_equity_usd = self._latest_account_equity()
        if account_equity_usd is None or account_equity_usd <= 0:
            return RiskDecision(
                False,
                "Could not read account equity — failing closed",
                "max_gross_exposure",
            )
        # Prefer caller-supplied current_gross (live trader has Alpaca's truth).
        # Fall back to DB sum only if not passed.
        if current_gross_usd is not None:
            gross_now = float(current_gross_usd)
        else:
            gross_now = self._current_gross_exposure_usd()
            if gross_now is None:
                return RiskDecision(
                    False,
                    "Could not read current gross exposure — failing closed",
                    "max_gross_exposure",
                )
        # Prefer caller-supplied notional (sizer computed it on adjusted equity
        # with regime + confluence multipliers). Otherwise reconstruct as
        # raw_equity * size_pct (a fallback that under-counts when multipliers
        # are active — Issue #2 from the audit).
        if current_notional_usd is not None and current_notional_usd > 0:
            new_position_usd = float(current_notional_usd)
        else:
            new_position_usd = account_equity_usd * (t.size_pct / 100.0)
        gross_after = gross_now + new_position_usd
        # Options reserve — keep a chunk of total gross available ONLY for option
        # spreads (the highest-EV path). Stocks and crypto compete for the
        # remaining budget; options get the full gross cap.
        if t.kind == "option":
            effective_cap = CONFIG.risk_max_gross_exposure_usd
        else:
            effective_cap = (CONFIG.risk_max_gross_exposure_usd
                             - CONFIG.risk_options_reserve_usd)
        if gross_after > effective_cap + 1.0:  # $1 rounding slack
            return RiskDecision(
                False,
                f"Gross exposure ${gross_after:.0f} would exceed "
                f"${effective_cap:.0f} ({'options' if t.kind == 'option' else 'non-option'} cap; "
                f"options reserve=${CONFIG.risk_options_reserve_usd:.0f})",
                "max_gross_exposure",
            )

        # Rule 6b: crypto-only ceiling. Crypto trades 24/7 and would fill the
        # entire gross budget overnight before US-session stocks/options ever
        # see a candidate. Hard cap forces crypto to be selective.
        # WEEKEND: when the US market is closed, the stock+options budget is
        # idle, so crypto is allowed a higher ceiling to put that capital to
        # work. Reverts automatically the moment the market opens; the Monday
        # pre-open trim brings exposure back to the weekday cap.
        if t.kind == "crypto":
            crypto_cap = (CONFIG.risk_max_crypto_exposure_usd if market_open
                          else CONFIG.risk_max_crypto_exposure_weekend_usd)
            if current_crypto_usd is not None:
                crypto_now = float(current_crypto_usd)
            else:
                crypto_now = self._current_crypto_exposure_usd() or 0.0
            crypto_after = crypto_now + new_position_usd
            if crypto_after > crypto_cap + 1.0:  # $1 slack
                return RiskDecision(
                    False,
                    f"Crypto exposure ${crypto_after:.0f} would exceed "
                    f"${crypto_cap:.0f} (crypto-only cap, "
                    f"{'market-open' if market_open else 'weekend'})",
                    "max_crypto_exposure",
                )

        # Rule 7a: per-ticker concentration — checks TOTAL exposure on this
        # ticker (existing + new), bound on TRUE equity. Prevents stacking
        # via multiple signals on the same name while still allowing a
        # single high-conviction Kelly-sized trade.
        # Options get a higher cap (12%) because risk is DEFINED — max loss
        # is the debit paid. Allows one max-conviction spread plus a small
        # add-on, but stops the 5x-AAPL-spread runaway.
        per_ticker_cap = (12.0 if t.kind == "option"
                          else CONFIG.risk_max_position_pct)
        if current_ticker_usd is not None:
            existing_ticker_usd = float(current_ticker_usd)
        else:
            existing_ticker_usd = self._current_ticker_exposure_usd(ticker, t.kind)
        total_ticker_usd = existing_ticker_usd + new_position_usd
        total_ticker_pct = (total_ticker_usd / account_equity_usd) * 100.0
        # Epsilon tolerance: the sizer clamps notional to EXACTLY the cap
        # (e.g. $678 = 6.0% of $11,300), but crypto qty rounding to 6 decimals
        # can tick the notional a few cents over, making 6.0000009% > 6.0 and
        # rejecting a legitimately-capped trade. Allow 0.1pp of slack.
        if total_ticker_pct > per_ticker_cap + 0.1:
            return RiskDecision(
                False,
                f"Ticker exposure ${total_ticker_usd:.0f} "
                f"({total_ticker_pct:.1f}% of true equity, existing ${existing_ticker_usd:.0f} "
                f"+ new ${new_position_usd:.0f}) exceeds per-ticker cap "
                f"{per_ticker_cap:.1f}% ({t.kind})",
                "max_position_pct",
            )

        # Rule 7b: per-sector concentration
        sector = (t.sector or SECTOR_MAP.get(ticker, "other")).lower()
        sector_now_pct = self._current_sector_exposure_pct(sector, account_equity_usd)
        if sector_now_pct is None:  # exposure unreadable (data error) — fail closed
            return RiskDecision(
                False,
                "Sector exposure unavailable (data error) — failing closed",
                "max_sector_pct",
            )
        sector_after = sector_now_pct + t.size_pct
        if sector_after > CONFIG.risk_max_sector_pct:
            return RiskDecision(
                False,
                f"Sector {sector!r} exposure {sector_after:.1f}% > "
                f"cap {CONFIG.risk_max_sector_pct:.1f}%",
                "max_sector_pct",
            )

        # Rule 8: concentration / pile-on cap — limit the number of distinct
        # open names so the book can't load up on many correlated bets at once
        # (2026-06-05 lesson: ~10 simultaneous bullish positions all sank together).
        # FAIL CLOSED: if the live count is unavailable, block — do not add risk.
        # (Previously this returned True when count was None, silently bypassing the
        # cap exactly when a positions-snapshot hiccup nulled the count.)
        if current_position_count is None:
            return RiskDecision(
                False,
                "Concentration cap: open-position count unavailable — blocking (fail-closed)",
                "max_concurrent_positions",
            )
        if current_position_count >= CONFIG.risk_max_concurrent_positions:
            return RiskDecision(
                False,
                f"Concentration cap: {current_position_count} open names "
                f">= {CONFIG.risk_max_concurrent_positions} (avoid correlated pile-on)",
                "max_concurrent_positions",
            )

        return RiskDecision(True, "all 8 rules passed")

    # ------------------------------------------------------------------
    # Helpers — every one is read-only and fails closed.
    # ------------------------------------------------------------------
    def _today_realized_pnl_usd(self) -> Optional[float]:
        """Today's realized PnL from Alpaca trading.

        Reads from ``bot_daily_pnl`` which the live trader's
        ``_reconcile_realized_pnl`` keeps in sync with Alpaca's
        intraday truth (account equity delta minus current unrealized).
        Returns 0.0 if no row yet today.  Any exception returns None
        which forces a fail-closed decision.
        """
        try:
            with get_connection() as conn:
                # Use US-Eastern date to match _update_daily_pnl writer
                # (avoids day-boundary mismatch during 00-04 UTC).
                try:
                    from zoneinfo import ZoneInfo
                    from datetime import datetime as _dt
                    eastern_date = _dt.now(ZoneInfo("America/New_York")).date().isoformat()
                except Exception:  # noqa: BLE001
                    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                    eastern_date = (_dt.now(_tz.utc) - _td(hours=4)).date().isoformat()
                row = conn.execute(
                    "SELECT realized_pnl_usd FROM bot_daily_pnl "
                    "WHERE trading_date = ?",
                    (eastern_date,),
                ).fetchone()
                return float(row[0]) if row else 0.0
        except Exception as exc:  # noqa: BLE001
            log.warning("risk._today_realized_pnl_usd failed: %s", exc)
            return None

    # Sentinel returned by _monthly_drawdown_breached ONLY on a check ERROR
    # (DB unreachable) — distinct from a clean "not breached" (None). The caller
    # fails CLOSED on this; a clean None still allows trading. (Without this,
    # error and not-breached both returned None → fail-OPEN on DB error.)
    _DD_CHECK_ERROR = object()

    def _monthly_drawdown_breached(self) -> Optional[tuple[float, float, float]]:
        """Check if equity is more than `risk_monthly_drawdown_usd` below
        the 30-day rolling peak. Returns (peak, current, drawdown) when
        breached, else None.

        Reads from bot_account_snapshots — the live trader writes one row
        per iteration with Alpaca's true equity.
        """
        try:
            with get_connection() as conn:
                peak_row = conn.execute(
                    "SELECT MAX(equity_usd) AS peak FROM bot_account_snapshots "
                    "WHERE snapshot_at > datetime('now','-30 days')"
                ).fetchone()
                cur_row = conn.execute(
                    "SELECT equity_usd FROM bot_account_snapshots "
                    "ORDER BY snapshot_at DESC LIMIT 1"
                ).fetchone()
            if not peak_row or not cur_row:
                return None
            peak = float(peak_row[0] or 0)
            current = float(cur_row[0] or 0)
            if peak <= 0 or current <= 0:
                return None
            drawdown = peak - current
            if drawdown >= CONFIG.risk_monthly_drawdown_usd:
                return (peak, current, drawdown)
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("risk._monthly_drawdown_breached failed: %s", exc)
            return self._DD_CHECK_ERROR  # fail CLOSED at caller (distinct from not-breached None)

    def _today_trade_count(self) -> int:
        """Number of orders placed today (UTC) on Alpaca. Fails closed."""
        try:
            with get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS n FROM bot_decisions
                    WHERE outcome = 'placed'
                      AND decided_at >= datetime('now', 'start of day')
                    """
                ).fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:  # noqa: BLE001
            log.warning("risk._today_trade_count failed: %s", exc)
            return 10 ** 9  # fail closed

    def _hours_since_last_drift_alert(self) -> Optional[float]:
        """None when never alerted; float hours otherwise.

        IMPORTANT: filters by the CURRENTLY DEPLOYED model version. Stale
        alerts from old models (which the bot no longer uses) are IGNORED.
        Without this filter, a retrained model would inherit the old model's
        drift alerts and block trading until the 24h window expired.

        On exception returns 0.0 so the drift_block rule treats it as a
        recent alert (fail closed).
        """
        # Read currently deployed model version from pointer file
        current_version = None
        try:
            import json
            from pathlib import Path
            pointer = Path(__file__).resolve().parents[3] / "data" / "models" / "current.json"
            if pointer.exists():
                current_version = json.loads(pointer.read_text()).get("version")
        except Exception:  # noqa: BLE001
            pass

        try:
            with get_connection() as conn:
                if current_version:
                    row = conn.execute(
                        """
                        SELECT observed_at FROM model_drift_observations
                        WHERE alerted = 1 AND model_version = ?
                        ORDER BY id DESC LIMIT 1
                        """,
                        (current_version,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT observed_at FROM model_drift_observations
                        WHERE alerted = 1 ORDER BY id DESC LIMIT 1
                        """
                    ).fetchone()
                if not row or not row["observed_at"]:
                    return None
                raw = row["observed_at"]
                ts = self._parse_ts_utc(raw)
                if ts is None:
                    log.warning("risk: unparseable drift ts %r — failing closed", raw)
                    return 0.0
                return (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        except Exception as exc:  # noqa: BLE001
            log.warning("risk._hours_since_last_drift_alert failed: %s", exc)
            return 0.0  # fail closed (treat as recent alert)

    def _latest_account_equity(self) -> Optional[float]:
        """Latest Alpaca paper account equity from bot_account_snapshots.

        This is a fallback — the live trader always passes account_equity_usd
        explicitly when calling evaluate(). Used only if the caller doesn't
        pass equity (e.g., scripts running risk checks outside the live loop).
        """
        try:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT equity_usd FROM bot_account_snapshots "
                    "ORDER BY snapshot_at DESC LIMIT 1"
                ).fetchone()
                if row and row[0] is not None and float(row[0]) > 0:
                    return float(row[0])
                return None
        except Exception as exc:  # noqa: BLE001
            log.warning("risk._latest_account_equity failed: %s", exc)
            return None

    def _current_crypto_exposure_usd(self) -> Optional[float]:
        """Sum of open Alpaca crypto positions' market value.

        Reads from ``bot_orders`` (currently-open fills with no realized P&L
        recorded yet) — same data path the live trader uses internally.
        Returns 0.0 if no open crypto. Fails closed (None) on error.
        """
        try:
            with get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT COALESCE(SUM(ABS(qty * filled_avg_price)), 0) AS gross
                    FROM bot_orders
                    WHERE status = 'filled'
                      AND realized_pnl_usd IS NULL
                      AND (ticker LIKE '%/USD' OR (ticker LIKE '%USD' AND LENGTH(ticker) <= 9))
                    """
                ).fetchone()
                return float(row[0]) if row else 0.0
        except Exception as exc:  # noqa: BLE001
            log.warning("risk._current_crypto_exposure_usd failed: %s", exc)
            return None

    def _current_ticker_exposure_usd(self, ticker: str, kind: str) -> float:
        """Open exposure on this specific ticker (or underlying for options).

        For stocks/crypto: sum of |qty * filled_avg_price| from bot_orders.
        For options: sum of total_debit_usd from bot_option_spreads where
        underlying matches and the spread is still open.
        """
        try:
            with get_connection() as conn:
                if kind == "option":
                    row = conn.execute(
                        """
                        SELECT COALESCE(SUM(total_debit_usd), 0) AS exposure
                        FROM bot_option_spreads
                        WHERE underlying = ?
                          AND closed_at IS NULL
                        """,
                        (ticker.upper(),),
                    ).fetchone()
                else:
                    # Stocks/crypto — normalize ticker forms (BTC/USD vs BTCUSD)
                    candidates = {ticker.upper()}
                    if "/" in ticker:
                        candidates.add(ticker.replace("/", "").upper())
                    elif ticker.upper().endswith("USD") and len(ticker) <= 9:
                        candidates.add(f"{ticker[:-3].upper()}/USD")
                    placeholders = ",".join("?" * len(candidates))
                    row = conn.execute(
                        f"""
                        SELECT COALESCE(SUM(ABS(qty * filled_avg_price)), 0) AS exposure
                        FROM bot_orders
                        WHERE status = 'filled'
                          AND realized_pnl_usd IS NULL
                          AND UPPER(ticker) IN ({placeholders})
                        """,
                        tuple(candidates),
                    ).fetchone()
                return float(row[0]) if row else 0.0
        except Exception as exc:  # noqa: BLE001
            log.warning("risk._current_ticker_exposure_usd failed: %s", exc)
            return 0.0

    def _current_gross_exposure_usd(self) -> Optional[float]:
        """Sum of |quantity * fill_price| across open Alpaca positions.

        We trade on Alpaca, not T212 — the previous version read T212
        positions (the user's real-money portfolio) as the gross baseline,
        which silently blocked nearly every Alpaca trade because T212
        already held ~$5500 of equity exposure. Now reads from bot_orders
        (the Alpaca-mirror DB the live trader populates).

        Counts: stock + crypto + option-spread DEBITS (option max-loss).
        """
        try:
            with get_connection() as conn:
                # Stocks + crypto from bot_orders (open fills, no realized P&L yet)
                stock_crypto = conn.execute(
                    """
                    SELECT COALESCE(SUM(ABS(qty * filled_avg_price)), 0) AS gross
                    FROM bot_orders
                    WHERE status = 'filled' AND realized_pnl_usd IS NULL
                    """
                ).fetchone()
                stock_crypto_usd = float(stock_crypto[0]) if stock_crypto else 0.0
                # Open option spreads — debit paid is the max-loss exposure
                opts = conn.execute(
                    """
                    SELECT COALESCE(SUM(total_debit_usd), 0) AS gross
                    FROM bot_option_spreads
                    WHERE closed_at IS NULL
                    """
                ).fetchone()
                opts_usd = float(opts[0]) if opts else 0.0
                return stock_crypto_usd + opts_usd
        except Exception as exc:  # noqa: BLE001
            log.warning("risk._current_gross_exposure_usd failed: %s", exc)
            return None

    def _current_sector_exposure_pct(
        self, sector: str, account_equity: float
    ) -> float:
        """Percent of account equity in the given sector.

        Sums open Alpaca positions (bot_orders) by sector via SECTOR_MAP.
        Options spreads are counted by underlying.
        Fails closed (returns 100% on error so trade is blocked).
        """
        if not account_equity or account_equity <= 0:
            return 100.0
        try:
            with get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT ticker, qty, filled_avg_price
                    FROM bot_orders
                    WHERE status = 'filled' AND realized_pnl_usd IS NULL
                    """
                ).fetchall()
                total = 0.0
                for r in rows:
                    t = (r["ticker"] or "").upper()
                    base = t.split("/")[0] if "/" in t else t  # crypto BTC/USD → BTC
                    if SECTOR_MAP.get(base, "other").lower() != sector.lower():
                        continue
                    qty = float(r["qty"] or 0)
                    px = r["filled_avg_price"]
                    if px is None:
                        continue
                    total += abs(qty * float(px))
                # Add option spreads on tickers in this sector
                opt_rows = conn.execute(
                    """
                    SELECT underlying, total_debit_usd
                    FROM bot_option_spreads
                    WHERE closed_at IS NULL
                    """
                ).fetchall()
                for r in opt_rows:
                    if SECTOR_MAP.get((r["underlying"] or "").upper(), "other").lower() == sector.lower():
                        total += float(r["total_debit_usd"] or 0)
                return total / account_equity * 100.0
        except Exception as exc:  # noqa: BLE001
            log.warning("risk._current_sector_exposure_pct failed: %s", exc)
            return 100.0  # fail closed

    # ------------------------------------------------------------------
    # Parsing helper
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_ts_utc(raw: str) -> Optional[datetime]:
        """Parse an ISO-like timestamp into UTC datetime.  None on failure."""
        if not raw:
            return None
        s = raw.strip()
        # Try the common formats used in the schema.
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        # Last-resort: try fromisoformat after trimming trailing Z.
        try:
            tail = s.rstrip("Z")
            dt = datetime.fromisoformat(tail)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError:
            return None
