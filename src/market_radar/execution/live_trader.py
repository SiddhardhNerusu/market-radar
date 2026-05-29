"""Live trader — the top-level loop that turns scored signals into orders.

Flow per iteration
------------------
  1. Snapshot Alpaca account (equity, positions, open orders) → ``bot_account_snapshots``.
  2. Reconcile any open ``bot_orders`` with Alpaca's current order state,
     update fills + realized P&L + ``bot_daily_pnl``.
  3. Pull new ``signal_scores`` rows where:
        composite_score   >= COMPOSITE_THRESHOLD   (default CONFIG.notification_threshold)
        model_p_5d        IS NOT NULL              (ML has predicted)
        scored_at         within FRESHNESS_MINUTES (default 60)
        no row in bot_decisions for this score_id
        ticker not already in an open Alpaca position
        Alpaca market is open OR ticker is crypto
  4. For each candidate:
       a. Run the selective conformal gate (selective_gate.decide).
       b. Run the LiveRiskManager (all 7 hard rules vs live Alpaca state).
       c. Size via Kelly + ATR (sizer.size_trade).
       d. Submit a bracket order via Alpaca with TP + SL legs.
       e. Persist every step to bot_decisions / bot_orders.
  5. Sleep CONFIG.live_trader_interval_s seconds and repeat.

The loop is single-threaded and idempotent — re-running after a crash
will not double-submit any signal because ``bot_decisions(score_id)``
is UNIQUE.

A bracket order has its protective stop + take profit registered with
Alpaca server-side. Even if this bot crashes mid-flight, the stops
still trigger.
"""
from __future__ import annotations

import logging
import signal as _signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..config import CONFIG
from ..ml.selective_gate import decide as gate_decide
from ..risk import SECTOR_MAP, TradeProposal
from ..storage import get_connection
from ..storage.db import utc_now
from .alpaca_client import AlpacaClient, AlpacaError, BracketOrder
from .live_risk import LiveRiskManager
from .options import (
    AlpacaOptionsClient,
    OPTIONS_UNDERLYINGS,
    build_vertical_debit_spread,
)
from .options.spreads import spec_to_legs
from .sizer import SizingResult, get_atr, size_trade

log = logging.getLogger("marketradar.execution.live_trader")


@dataclass
class TraderConfig:
    # Composite score is RETAINED as a soft filter only (default 0 = off) because
    # measured edge analysis shows composite_score is NOT predictive of returns.
    # The bot now gates primarily on model_p_5d, which IS predictive (75% hit rate
    # at p>=0.65 on 208 historical samples).
    composite_threshold: float = 0.0
    freshness_minutes: int = 60
    interval_seconds: int = 45
    max_candidates_per_loop: int = 8
    direction_p_buy_min: float = 0.65            # raised from 0.62 — measured edge inflection point
    direction_p_sell_max: float = 0.35           # symmetric short-side gate
    dry_run: bool = False
    crypto_tickers: tuple[str, ...] = ("BTC/USD", "ETH/USD", "LTC/USD")
    allow_after_hours: bool = False

    # Confluence filters — every "yes" here improves measured win rate
    require_factual: bool = True                 # drop unverified rumors
    min_corroboration: int = 0                   # 0 = off; raise to 2+ for higher precision
    block_anti_pump: bool = True                 # always block obvious pumps
    min_source_weight: float = 7.0               # tier1 facts + price action only by default

    # Allow price-action signals to bypass the corroboration check (they're
    # standalone observations, not news that needs corroboration)
    bypass_corroboration_for_price_action: bool = True

    # Options spread routing — when a signal's underlying is in the options
    # whitelist (~10 mega-liquid tickers) AND options_enabled is True, the
    # signal is routed to the options spread path instead of the stock
    # bracket path. Stock path stays as the fallback for the other 80+ tickers.
    options_enabled: bool = False                # default OFF — opt-in via env
    options_target_dte: int = 1                  # DAY-TRADER: target 1-DTE (was 14, then 7)
    options_min_dte: int = 0                     # allow 0DTE on SPY/QQQ/IWM
    options_max_dte: int = 2                     # cap at 2 DTE — never hold longer
    options_take_profit_pct: float = 0.25        # 25% of max gain — quicker locks
    options_stop_loss_pct: float = 0.50          # close at 50% of max loss (= 50% debit drop)
    options_time_stop_dte: int = 0               # force-close at expiry day
    options_max_hold_hours: int = 4              # DAY-TRADER: max 4h hold
    options_eod_flatten: bool = True             # close ALL spreads at 20:55 BST

    # Day-trading strategy: short holds, tight stops, EOD flatten.
    # Switches signal horizon from 5d → 1d so we predict and trade on
    # intraday-scale outcomes, not weekly drift.
    signal_horizon: str = "1d"                   # '1d' | '5d' | '20d'
    stock_sl_atr_mult: float = 0.75              # tighter than 1.5 — quicker resolution
    stock_tp_atr_mult: float = 1.25              # quicker than 2.5 — same R:R = 1.67
    eod_flatten_minutes_before_close: int = 5    # close everything 5min before market close

    # Daily profit-take auto-flatten. TWO modes:
    #
    # 1. HARD CAP: if intraday >= daily_profit_take_usd, flatten immediately.
    #    Disabled when daily_profit_take_usd = 0.
    #
    # 2. TRAILING (smarter): once intraday clears `daily_tp_arm_at_usd`,
    #    track the peak. If intraday drops `daily_tp_giveback_usd` below
    #    the peak, flatten + halt. This locks target+ once hit but lets
    #    winners run as long as they're trending up.
    #
    # Default config: trail-mode armed at $190 (~£150 target), gives back
    # max $40. Hard cap disabled in favor of trailing.
    daily_profit_take_usd: float = 0.0           # 0 = hard cap disabled
    daily_tp_arm_at_usd: float = 190.0           # ≈ £150 — arm trail once hit
    daily_tp_giveback_usd: float = 40.0          # ≈ £32 max giveback from peak

    # PDT (Pattern Day Trader) awareness — US stock day-trades are capped at
    # 3 per rolling 5 days for accounts under $25k. The bot tracks recent
    # day-trades and STOPS opening new stock positions once at the limit.
    # Crypto + held-overnight options are exempt.
    pdt_enforce: bool = True
    pdt_account_equity_threshold: float = 25_000.0
    pdt_day_trade_limit_per_5d: int = 3
    pdt_safety_margin: int = 1                   # stop at 2 of 3 to leave headroom

    # Override account equity for paper sizing realism.
    # Paper accounts start at $100k of fake money but you have £5k real.
    # Set LIVE_OVERRIDE_EQUITY_USD=6300 so the bot sizes for your actual capital.
    override_equity_usd: float = 0.0             # 0 = use Alpaca's reported equity

    # Macro regime — global gate. When the regime says "panic" the bot halts
    # entirely; otherwise size is multiplied by regime.size_multiplier (0.5x
    # bearish, 1.0x neutral, 1.2x bullish).
    use_macro_regime: bool = True

    # Hard-block list of low-edge LLM event_types — these had measured negative
    # or zero edge on 156k historical outcomes. Saves DB+ML cycles by never
    # routing them to a trade decision.
    blocked_event_types: tuple[str, ...] = (
        "other", "proxy_statement", "passive_5pct_stake",
        "ipo_registration", "routine_prospectus", "routine_proxy",
        "material_event_amend", "activist_position",
    )

    @classmethod
    def from_env(cls) -> "TraderConfig":
        import os
        def _f(k, d): return float(os.getenv(k, d))
        def _i(k, d): return int(os.getenv(k, d))
        def _b(k, d): return os.getenv(k, str(d)).strip().lower() in {"1", "true", "yes"}
        return cls(
            composite_threshold=_f("LIVE_COMPOSITE_THRESHOLD", 0.0),
            freshness_minutes=_i("LIVE_FRESHNESS_MINUTES", 60),
            interval_seconds=_i("LIVE_INTERVAL_SECONDS", 45),
            max_candidates_per_loop=_i("LIVE_MAX_CANDIDATES_PER_LOOP", 8),
            direction_p_buy_min=_f("LIVE_P_BUY_MIN", 0.65),
            direction_p_sell_max=_f("LIVE_P_SELL_MAX", 0.35),
            dry_run=_b("LIVE_DRY_RUN", False),
            allow_after_hours=_b("LIVE_ALLOW_AFTER_HOURS", False),
            require_factual=_b("LIVE_REQUIRE_FACTUAL", True),
            min_corroboration=_i("LIVE_MIN_CORROBORATION", 0),
            block_anti_pump=_b("LIVE_BLOCK_ANTI_PUMP", True),
            min_source_weight=_f("LIVE_MIN_SOURCE_WEIGHT", 7.0),
            options_enabled=_b("LIVE_OPTIONS_ENABLED", False),
            options_target_dte=_i("LIVE_OPT_TARGET_DTE", 1),
            options_min_dte=_i("LIVE_OPT_MIN_DTE", 0),
            options_max_dte=_i("LIVE_OPT_MAX_DTE", 2),
            options_take_profit_pct=_f("LIVE_OPT_TP_PCT", 0.25),
            options_stop_loss_pct=_f("LIVE_OPT_SL_PCT", 0.50),
            options_time_stop_dte=_i("LIVE_OPT_TIME_STOP_DTE", 0),
            options_max_hold_hours=_i("LIVE_OPT_MAX_HOLD_HOURS", 4),
            options_eod_flatten=_b("LIVE_OPT_EOD_FLATTEN", True),
            signal_horizon=os.getenv("LIVE_SIGNAL_HORIZON", "1d").strip().lower(),
            stock_sl_atr_mult=_f("LIVE_STOCK_SL_ATR_MULT", 0.75),
            stock_tp_atr_mult=_f("LIVE_STOCK_TP_ATR_MULT", 1.25),
            eod_flatten_minutes_before_close=_i("LIVE_EOD_FLATTEN_MIN", 5),
            pdt_enforce=_b("LIVE_PDT_ENFORCE", True),
            pdt_day_trade_limit_per_5d=_i("LIVE_PDT_LIMIT", 3),
            pdt_safety_margin=_i("LIVE_PDT_SAFETY_MARGIN", 1),
            override_equity_usd=_f("LIVE_OVERRIDE_EQUITY_USD", 0.0),
            use_macro_regime=_b("LIVE_USE_MACRO_REGIME", True),
            daily_profit_take_usd=_f("LIVE_DAILY_TP_USD", 0.0),
            daily_tp_arm_at_usd=_f("LIVE_DAILY_TP_ARM_USD", 190.0),
            daily_tp_giveback_usd=_f("LIVE_DAILY_TP_GIVEBACK_USD", 40.0),
        )


# ---------------------------------------------------------------------------
# The trader
# ---------------------------------------------------------------------------

class LiveTrader:

    def __init__(
        self,
        *,
        config: Optional[TraderConfig] = None,
        alpaca: Optional[AlpacaClient] = None,
    ):
        self.cfg = config or TraderConfig.from_env()
        self.alpaca = alpaca or AlpacaClient()
        self.risk = LiveRiskManager(self.alpaca)
        # Options client only instantiated if enabled (no API calls at startup)
        self.options = AlpacaOptionsClient(self.alpaca) if self.cfg.options_enabled else None
        if self.options is not None:
            # Preflight: confirm options is actually enabled on this account.
            # First trade will otherwise silently fail with 403 on contract fetch.
            try:
                contracts = self.options.list_contracts("SPY", limit=1)
                if not contracts:
                    log.error(
                        "OPTIONS PREFLIGHT FAILED: list_contracts(SPY) returned empty. "
                        "Your Alpaca paper account likely doesn't have options enabled. "
                        "Visit https://app.alpaca.markets/paper/dashboard/overview "
                        "and click 'Apply Now' on the 'Trade Options Commission Free' panel. "
                        "Disabling options for this run."
                    )
                    self.options = None
                else:
                    log.info("[preflight] options trading enabled (SPY contract resolved: %s)",
                             contracts[0].symbol)
            except Exception as exc:  # noqa: BLE001
                log.error("OPTIONS PREFLIGHT crashed: %s — disabling options for this run", exc)
                self.options = None
        # Current macro regime — set at the top of each run_once(), nullable
        self._current_regime = None
        self._stop = False
        # PERSISTENCE: load daily TP fired flag from DB so restarts respect "halted today"
        self._load_daily_tp_state()
        log.info(
            "LiveTrader ready: composite>=%.1f freshness<=%dm interval=%ds "
            "dry_run=%s mode=%s",
            self.cfg.composite_threshold,
            self.cfg.freshness_minutes,
            self.cfg.interval_seconds,
            self.cfg.dry_run,
            "PAPER" if "paper" in self.alpaca.base_url else "LIVE",
        )

    def _load_daily_tp_state(self) -> None:
        """Restore _daily_tp_fired_on / peak from DB so restarts don't reopen
        after we've already locked the day. Uses bot_daily_pnl.tp_fired column
        (added via migration below)."""
        from datetime import date as _date
        try:
            with get_connection() as conn:
                # Ensure columns exist (idempotent — silent on already-exist)
                for col, ctype in [("tp_fired", "INTEGER"), ("tp_peak_usd", "REAL")]:
                    try:
                        conn.execute(f"ALTER TABLE bot_daily_pnl ADD COLUMN {col} {ctype}")
                    except Exception:  # noqa: BLE001
                        pass
                row = conn.execute(
                    "SELECT tp_fired, tp_peak_usd FROM bot_daily_pnl WHERE trading_date=?",
                    (_us_eastern_date().isoformat(),),
                ).fetchone()
            if row and row[0]:
                self._daily_tp_fired_on = _date.today()
                self._daily_tp_armed_on = _date.today()
                self._daily_tp_peak = row[1] or 0.0
                log.warning("🎯 Loaded TP state from DB: ALREADY FIRED today, peak was $%.2f",
                            self._daily_tp_peak or 0)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load daily TP state: %s", exc)

    def _persist_daily_tp_state(self) -> None:
        """Write daily TP fired flag + peak to DB so restarts pick it up."""
        try:
            with get_connection() as conn:
                conn.execute(
                    """INSERT INTO bot_daily_pnl
                       (trading_date, realized_pnl_usd, trades_count, wins, losses,
                        largest_win, largest_loss, updated_at, tp_fired, tp_peak_usd)
                       VALUES (?, 0, 0, 0, 0, 0, 0, ?, 1, ?)
                       ON CONFLICT(trading_date) DO UPDATE SET
                         tp_fired = 1,
                         tp_peak_usd = COALESCE(excluded.tp_peak_usd, tp_peak_usd),
                         updated_at = excluded.updated_at""",
                    (_us_eastern_date().isoformat(), utc_now(),
                     getattr(self, "_daily_tp_peak", 0.0)),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not persist daily TP state: %s", exc)

    def _reconcile_option_spreads(self) -> None:
        """Sync bot_option_spreads.status with Alpaca reality.

        Each loop, walk every spread row with status='pending_new' (or 'new')
        and query its Alpaca order. Outcomes:
          * filled → mark filled in DB, send Telegram "OPTIONS FILLED" alert
          * canceled / rejected / expired → mark canceled, no notif
          * still pending after 5 min → cancel at Alpaca, mark canceled
          * still pending under 5 min → leave alone, check next iteration

        This keeps the DB coherent with broker reality. Without it,
        unfilled submits sit as phantoms forever, polluting risk math.
        """
        try:
            with get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT id, alpaca_order_id, client_order_id, underlying,
                           direction, contracts, total_debit_usd,
                           max_loss_usd, max_gain_usd, status, submitted_at
                    FROM bot_option_spreads
                    WHERE status IN ('pending_new', 'new', 'accepted', 'partially_filled')
                    """
                ).fetchall()
            if not rows:
                return
            from datetime import datetime as _dt
            for r in rows:
                sp = dict(r)
                order_id = sp["alpaca_order_id"]
                if not order_id:
                    continue
                # Skip cleanup-canceled synthetic IDs (start with 'orphan-' or 'reconcile-')
                if order_id.startswith(("orphan-", "reconcile-")):
                    continue
                try:
                    o = self.alpaca.get_order(order_id)
                except AlpacaError as exc:
                    # 404 → order never existed at Alpaca → mark canceled
                    log.info("[OPT reconcile] %s: lookup failed (%s) — marking canceled",
                             sp["underlying"], str(exc)[:60])
                    with get_connection() as conn:
                        conn.execute(
                            "UPDATE bot_option_spreads SET status='canceled', "
                            "closed_at=?, exit_reason='alpaca_lookup_failed' WHERE id=?",
                            (utc_now(), sp["id"]),
                        )
                    continue
                status = (o.status or "").lower()
                if status == "filled":
                    # Promote to filled — also book it as confirmed and notify
                    with get_connection() as conn:
                        conn.execute(
                            "UPDATE bot_option_spreads SET status='filled', "
                            "filled_at=? WHERE id=? AND status != 'filled'",
                            (o.filled_at or utc_now(), sp["id"]),
                        )
                        # Only notify if we actually updated (avoid dup notifications)
                        if conn.total_changes > 0:
                            log.info(
                                "[OPT reconcile] %s FILLED at Alpaca — spread id=%d "
                                "qty=%d debit=$%.0f max_gain=$%.0f",
                                sp["underlying"], sp["id"], sp["contracts"],
                                sp["total_debit_usd"] or 0, sp["max_gain_usd"] or 0,
                            )
                            try:
                                from ..notifications.realtime import TradeAlert, notify_trade
                                notify_trade(TradeAlert(
                                    kind="OPTIONS_FILLED",
                                    symbol=sp["underlying"],
                                    direction=sp["direction"],
                                    qty=sp["contracts"],
                                    notional_usd=sp["total_debit_usd"],
                                    extra=(
                                        f"FILLED — debit ${sp['total_debit_usd']:.0f} "
                                        f"max_gain ${sp['max_gain_usd']:.0f}"
                                    ),
                                ))
                            except Exception:  # noqa: BLE001
                                pass
                elif status in ("canceled", "rejected", "expired", "suspended"):
                    log.info("[OPT reconcile] %s %s at Alpaca — marking canceled",
                             sp["underlying"], status.upper())
                    with get_connection() as conn:
                        conn.execute(
                            "UPDATE bot_option_spreads SET status='canceled', "
                            "closed_at=?, exit_reason=? WHERE id=?",
                            (utc_now(), f"alpaca_{status}", sp["id"]),
                        )
                else:
                    # Still pending — check age. If > 5 min, cancel.
                    try:
                        sub_dt = _dt.fromisoformat(
                            sp["submitted_at"].replace("Z", "+00:00"))
                        age_min = (_dt.now(sub_dt.tzinfo) - sub_dt).total_seconds() / 60
                    except Exception:  # noqa: BLE001
                        age_min = 0
                    if age_min > 5:
                        log.info("[OPT reconcile] %s still %s after %.1f min — canceling",
                                 sp["underlying"], status, age_min)
                        try:
                            self.alpaca.cancel_order(order_id)
                        except AlpacaError:
                            pass
                        with get_connection() as conn:
                            conn.execute(
                                "UPDATE bot_option_spreads SET status='canceled', "
                                "closed_at=?, exit_reason='stale_pending_cancel' WHERE id=?",
                                (utc_now(), sp["id"]),
                            )
        except Exception as exc:  # noqa: BLE001
            log.warning("_reconcile_option_spreads failed: %s", exc)

    def _reconcile_realized_pnl(self, account) -> None:
        """Sync bot_daily_pnl.realized to match Alpaca's intraday truth.

        Alpaca's account equity minus last_equity is the authoritative
        intraday P&L. Subtract current unrealized to get realized. Sync
        both directions when the gap exceeds $1 — the math is invariant
        under position reopen, so downward sync is safe and necessary
        for losses + external closes (cleanup scripts, EOD retries) to
        appear in the daily report.
        """
        try:
            positions = self.alpaca.get_positions()
            alpaca_intraday = float(account.equity - account.last_equity)
            current_unreal = sum(float(p.unrealized_pl) for p in positions)
            true_realized = alpaca_intraday - current_unreal
            eastern_date_iso = _us_eastern_date().isoformat()
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT realized_pnl_usd FROM bot_daily_pnl WHERE trading_date=?",
                    (eastern_date_iso,),
                ).fetchone()
                current = float(row[0]) if row else 0.0
                if abs(true_realized - current) > 1.0:
                    delta = true_realized - current
                    conn.execute(
                        """INSERT INTO bot_daily_pnl
                           (trading_date, realized_pnl_usd, trades_count, wins, losses,
                            largest_win, largest_loss, updated_at)
                           VALUES (?, ?, 0, 0, 0, 0, 0, ?)
                           ON CONFLICT(trading_date) DO UPDATE SET
                             realized_pnl_usd = ?,
                             updated_at = excluded.updated_at""",
                        (eastern_date_iso, true_realized, utc_now(), true_realized),
                    )
                    log.info(
                        "[reconcile] realized P&L synced: $%.2f → $%.2f (delta $%+.2f)",
                        current, true_realized, delta,
                    )
        except Exception as exc:  # noqa: BLE001
            log.warning("realized P&L reconcile failed: %s", exc)

    # ------------------------------------------------------------------
    # Startup reconciliation — discover orphaned Alpaca positions/orders
    # that aren't tracked in the DB (e.g. after a crash). Alerts user, logs
    # gaps, and marks DB rows for any "lost" positions so risk + sizing
    # don't double-count.
    # ------------------------------------------------------------------
    def adopt_orphan_positions(self) -> int:
        """One-shot: take ownership of any Alpaca positions not in bot_orders.

        Inserts synthetic bot_orders rows so the position monitor + EOD flatten
        know about them. Pulls the existing bracket child orders to populate
        SL/TP. Idempotent.
        """
        try:
            positions = self.alpaca.get_positions()
            open_orders = self.alpaca.list_orders(status="open", limit=200)
        except AlpacaError as exc:
            log.warning("[orphan-adopt] cannot reach Alpaca: %s", exc)
            return 0
        with get_connection() as conn:
            tracked = {
                r[0].upper() for r in conn.execute(
                    "SELECT DISTINCT ticker FROM bot_orders "
                    "WHERE status='filled' AND realized_pnl_usd IS NULL"
                ).fetchall()
            }
        import time as _t_orphan
        adopted = 0
        for p in positions:
            sym = p.symbol.upper()
            if sym in tracked:
                continue
            # Find this position's bracket children if any
            children = [o for o in open_orders if o.symbol.upper() == sym]
            tp = next((float(o.limit_price) for o in children
                       if o.order_type == "limit" and o.limit_price), None)
            sl = next((float(o.stop_price) for o in children
                       if o.stop_price), None)
            direction = "buy" if p.qty >= 0 else "sell"
            with get_connection() as conn:
                try:
                    conn.execute(
                        """
                        INSERT INTO bot_orders
                          (alpaca_order_id, client_order_id, ticker, direction,
                           order_class, qty, submitted_price, stop_loss,
                           take_profit, status, filled_qty, filled_avg_price,
                           submitted_at, filled_at)
                        VALUES (?, ?, ?, ?, 'bracket', ?, ?, ?, ?, 'filled',
                                ?, ?, ?, ?)
                        """,
                        (
                            f"orphan-{sym}-{_t_orphan.time_ns()}",
                            f"orphan-co-{sym}-{_t_orphan.time_ns()}",
                            sym, direction, abs(float(p.qty)),
                            float(p.avg_entry_price), sl, tp,
                            abs(float(p.qty)), float(p.avg_entry_price),
                            utc_now(), utc_now(),
                        ),
                    )
                    adopted += 1
                    log.info("[orphan-adopt] adopted %s qty=%.4f entry=%.2f SL=%s TP=%s",
                             sym, p.qty, p.avg_entry_price, sl, tp)
                except Exception as exc:  # noqa: BLE001
                    log.warning("[orphan-adopt] adopt %s failed: %s", sym, exc)
        if adopted:
            log.info("[orphan-adopt] adopted %d orphan positions", adopted)
        return adopted

    def reconcile_on_startup(self) -> None:
        """Sync Alpaca state ↔ DB on bot boot. Idempotent."""
        log.info("[startup] Reconciling Alpaca state ↔ DB…")
        # First, adopt any orphan positions so the reconcile loop knows them
        try:
            self.adopt_orphan_positions()
        except Exception as exc:  # noqa: BLE001
            log.warning("[startup] adopt_orphan_positions failed: %s", exc)
        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.get_positions()
            open_orders = self.alpaca.list_orders(status="open", limit=200)
        except AlpacaError as exc:
            log.warning("[startup] Cannot reach Alpaca during reconciliation: %s", exc)
            return

        # --- Orphan positions: held at Alpaca, but no matching open bot_order ---
        with get_connection() as conn:
            tracked = {
                r[0].upper() for r in conn.execute(
                    "SELECT ticker FROM bot_orders "
                    "WHERE status='filled' AND realized_pnl_usd IS NULL"
                ).fetchall()
            }
        orphan_symbols = []
        for p in positions:
            sym = p.symbol.upper()
            if sym in tracked:
                continue
            # Skip orphans we know about via options spreads
            with get_connection() as conn:
                opt = conn.execute(
                    "SELECT id FROM bot_option_spreads "
                    "WHERE underlying=? AND closed_at IS NULL LIMIT 1",
                    (sym,),
                ).fetchone()
            if opt:
                continue
            orphan_symbols.append((sym, p.qty, p.market_value, p.unrealized_pl))

        if orphan_symbols:
            log.warning(
                "[startup] %d ORPHAN Alpaca positions (held at broker, not in bot DB):",
                len(orphan_symbols),
            )
            for sym, qty, mv, upnl in orphan_symbols:
                log.warning("  %s qty=%.2f mv=$%.2f u_pnl=$%+.2f",
                            sym, qty, mv, upnl)
            try:
                from ..notifications.realtime import TradeAlert, notify_trade
                notify_trade(TradeAlert(
                    kind="REGIME_HALT",
                    symbol="ORPHAN",
                    extra=(f"Bot found {len(orphan_symbols)} untracked Alpaca position(s) "
                           f"on startup. Inspect manually before relying on bot risk/sizing."),
                ))
            except Exception:  # noqa: BLE001
                pass

        # --- Lost orders: bot_orders status='new/accepted' but no matching open at Alpaca ---
        alpaca_open_ids = {o.id for o in open_orders}
        with get_connection() as conn:
            stale = conn.execute(
                "SELECT id, alpaca_order_id, ticker, status FROM bot_orders "
                "WHERE status NOT IN ('filled','canceled','expired','rejected') "
                "  AND alpaca_order_id IS NOT NULL"
            ).fetchall()
            for row in stale:
                if row["alpaca_order_id"] in alpaca_open_ids:
                    continue
                # Refetch — order may have filled/canceled while bot was down
                try:
                    fresh = self.alpaca.get_order(row["alpaca_order_id"])
                    conn.execute(
                        "UPDATE bot_orders SET status=?, filled_qty=?, "
                        "  filled_avg_price=?, filled_at=?, canceled_at=? "
                        "WHERE id=?",
                        (fresh.status, fresh.filled_qty, fresh.filled_avg_price,
                         fresh.filled_at, fresh.canceled_at, row["id"]),
                    )
                    log.info("[startup] reconciled order %s: %s -> %s",
                             row["alpaca_order_id"], row["status"], fresh.status)
                except AlpacaError as exc:
                    log.warning("[startup] refetch %s failed: %s",
                                row["alpaca_order_id"], exc)

        # --- Pending submits that may have actually landed at Alpaca during a crash ---
        with get_connection() as conn:
            pending = conn.execute(
                "SELECT score_id, outcome_detail FROM bot_decisions "
                "WHERE outcome='pending_submit'"
            ).fetchall()
        if pending:
            log.warning("[startup] %d decisions in 'pending_submit' state — manual review needed",
                        len(pending))
            # Mark for human inspection rather than auto-resolving (safer)

        log.info("[startup] Reconciliation done: %d orphan positions, %d stale orders, %d pending submits",
                 len(orphan_symbols), len(stale), len(pending) if pending else 0)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        _signal.signal(_signal.SIGTERM, self._handle_sigterm)
        _signal.signal(_signal.SIGINT, self._handle_sigterm)
        # Reconcile state before the loop starts
        try:
            self.reconcile_on_startup()
        except Exception as exc:  # noqa: BLE001
            log.exception("startup reconciliation failed (continuing): %s", exc)
        log.info("Entering main loop. Send SIGINT/SIGTERM to stop cleanly.")
        while not self._stop:
            try:
                self.run_once()
            except KeyboardInterrupt:
                break
            except Exception as exc:  # noqa: BLE001
                log.exception("Top-level loop error (continuing): %s", exc)
            slept = 0
            while slept < self.cfg.interval_seconds and not self._stop:
                time.sleep(1)
                slept += 1
        log.info("Stopped cleanly.")

    def _handle_sigterm(self, signum, frame):  # noqa: ARG002
        log.info("Signal %d received — stopping after current iteration.", signum)
        self._stop = True

    # ------------------------------------------------------------------
    # One iteration
    # ------------------------------------------------------------------
    def run_once(self) -> None:
        # Step 1: snapshot
        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.get_positions()
            open_orders = self.alpaca.list_orders(status="open", limit=200)
        except AlpacaError as exc:
            log.warning("Cannot reach Alpaca: %s — skipping iteration", exc)
            return

        self._snapshot_account(account, positions, open_orders)

        if not account.is_tradeable:
            log.warning("Alpaca account not tradeable: status=%s blocked=%s/%s",
                        account.status, account.trading_blocked, account.account_blocked)
            return

        # Step 2: reconcile any of our outstanding orders
        try:
            self._reconcile_orders()
        except Exception as exc:  # noqa: BLE001
            log.exception("Reconcile failed: %s", exc)

        # Step 2-bis: reconcile option spreads. Each pending_new row gets
        # checked against Alpaca's actual order state. Fills → mark filled +
        # notify Telegram. Rejected / canceled / stale → mark canceled so
        # the DB matches reality and no phantom positions pollute risk calcs.
        try:
            self._reconcile_option_spreads()
        except Exception as exc:  # noqa: BLE001
            log.exception("Option spread reconcile failed: %s", exc)

        # Step 2a: reconcile bot's realized P&L against Alpaca truth.
        # Manual closes / cleanup scripts / EOD orders that the poller missed
        # all get retroactively booked here so the dashboard matches reality.
        try:
            self._reconcile_realized_pnl(account)
        except Exception as exc:  # noqa: BLE001
            log.exception("Realized P&L reconcile failed: %s", exc)

        # Step 2b: poll & exit options spreads at TP/SL/time-stop
        try:
            self._poll_option_exits()
        except Exception as exc:  # noqa: BLE001
            log.exception("Options exit poll failed: %s", exc)

        # Step 2b.5: poll & exit crypto positions at SL/TP (Alpaca crypto = no brackets)
        try:
            self._poll_crypto_exits(positions)
        except Exception as exc:  # noqa: BLE001
            log.exception("Crypto exit poll failed: %s", exc)

        # Step 2b.6: crypto reversal flip — close held crypto when strong
        # opposite-direction price-action signal arrives. Frees the symbol so
        # next iteration can re-enter the new direction.
        try:
            self._check_crypto_flips(positions)
        except Exception as exc:  # noqa: BLE001
            log.exception("Crypto flip check failed: %s", exc)

        # Step 2c: EOD flatten — close stock positions before market close
        try:
            self._eod_flatten_if_due(positions)
        except Exception as exc:  # noqa: BLE001
            log.exception("EOD flatten failed: %s", exc)

        # Step 2d: Daily profit-take — if intraday P&L exceeds cap, flatten + halt
        try:
            if self._daily_profit_take_if_due(account, positions):
                log.warning("Daily profit-take fired. Skipping rest of iteration.")
                return
        except Exception as exc:  # noqa: BLE001
            log.exception("Daily profit-take check failed: %s", exc)

        # Step 3: market clock — skip stock trading when closed (unless after-hours allowed)
        market_open = self.alpaca.is_market_open()

        # Step 3b: macro regime gate. PANIC halts everything; bearish/bullish
        # adjust size + direction permissions for the loop.
        regime = None
        if self.cfg.use_macro_regime:
            try:
                from ..signals.macro_regime import get_regime
                regime = get_regime()
                log.info("Macro regime: %s (%s)", regime.bias, regime.reason)
                if regime.halt:
                    log.warning("REGIME HALT: %s — skipping iteration", regime.reason)
                    try:
                        from ..notifications.realtime import TradeAlert, notify_trade
                        notify_trade(TradeAlert(
                            kind="REGIME_HALT", symbol="ALL",
                            extra=regime.reason,
                        ))
                    except Exception:  # noqa: BLE001
                        pass
                    return
            except Exception as exc:  # noqa: BLE001
                log.warning("Macro regime fetch failed (continuing neutral): %s", exc)

        self._current_regime = regime  # used downstream for sizing + direction filter

        # If daily profit-take already fired today, do not open new trades
        from datetime import date as _date
        if getattr(self, "_daily_tp_fired_on", None) == _date.today():
            log.debug("Daily TP fired earlier today — blocking new opens")
            return

        # Step 4: gather candidates
        # Normalize position symbols to the same form _normalize_ticker produces.
        # Alpaca returns crypto as "AAVEUSD" (no slash); signals use "AAVE/USD".
        # Without normalization the exclude_symbols set never matches crypto and
        # the bot stacks positions on the same coin every iteration.
        held_symbols = set()
        for p in positions:
            sym = p.symbol.upper()
            held_symbols.add(sym)
            norm = _normalize_ticker(sym)
            if norm:
                held_symbols.add(norm)
        candidates = self._fetch_candidates(exclude_symbols=held_symbols)
        if not candidates:
            log.debug("No fresh candidates.")
            return
        log.info("Found %d candidate(s).", len(candidates))

        for cand in candidates[: self.cfg.max_candidates_per_loop]:
            try:
                self._process_candidate(cand, account=account,
                                        market_open=market_open,
                                        positions=positions)
            except Exception as exc:  # noqa: BLE001
                log.exception("Candidate score_id=%s crashed: %s",
                              cand.get("score_id"), exc)

    # ------------------------------------------------------------------
    # Step 1: snapshot
    # ------------------------------------------------------------------
    def _snapshot_account(self, account, positions, open_orders) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO bot_account_snapshots
                    (snapshot_at, equity_usd, cash_usd, buying_power_usd,
                     long_market_value, open_positions, open_orders)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (utc_now(), account.equity, account.cash, account.buying_power,
                 account.long_market_value, len(positions), len(open_orders)),
            )

    # ------------------------------------------------------------------
    # Step 2: reconcile fills + daily P&L
    # ------------------------------------------------------------------
    def _reconcile_orders(self) -> None:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, alpaca_order_id, ticker, direction, qty,
                       status, filled_qty, filled_avg_price, submitted_price
                FROM bot_orders
                WHERE status NOT IN ('filled','canceled','expired','rejected')
                   OR realized_pnl_usd IS NULL
                """
            ).fetchall()

        for row in rows:
            order_id = row["alpaca_order_id"]
            # Skip synthetic orphan IDs (generated by adopt_orphan_positions —
            # not real Alpaca order IDs, so get_order() will always 422)
            if order_id and order_id.startswith("orphan-"):
                # Mark as closed if the position is no longer at Alpaca
                continue
            try:
                fresh = self.alpaca.get_order(order_id)
            except AlpacaError as exc:
                log.warning("Reconcile order %s: %s", order_id, exc)
                continue

            prev_status = (row["status"] or "").lower()
            now_status = (fresh.status or "").lower()
            with get_connection() as conn:
                conn.execute(
                    """
                    UPDATE bot_orders SET status = ?, filled_qty = ?,
                           filled_avg_price = ?, filled_at = ?, canceled_at = ?
                    WHERE id = ?
                    """,
                    (fresh.status, fresh.filled_qty, fresh.filled_avg_price,
                     fresh.filled_at, fresh.canceled_at, row["id"]),
                )

                # If it just filled (parent leg), the child legs will get
                # reconciled on a future loop. Realized P&L is computed when
                # one of the child TP/SL legs fills — at that point the
                # parent's fill price + child fill price give us the P&L.
                if fresh.status == "filled" and fresh.legs:
                    # The parent fill - this is an entry, not an exit. Wait for child legs.
                    pass
                elif fresh.status == "filled" and not fresh.legs:
                    # Standalone fill - could be the closing child leg of a parent.
                    self._maybe_realize_pnl(conn, row, fresh)

            # FILL NOTIFICATION — fires once when status transitions from
            # non-filled to filled. Stock/crypto entry brackets only (we
            # detect by presence of child legs OR by the order being a
            # simple crypto market that just filled). Closing/exit fills
            # already have their own notifications via _maybe_realize_pnl.
            if (prev_status != "filled" and now_status == "filled"
                    and fresh.filled_avg_price):
                # Only notify for ENTRIES — closing legs are children of
                # brackets we already entered. Detect by parent-id absence.
                is_closing = bool(
                    getattr(fresh, "legs", None) is None
                    or (isinstance(fresh.legs, list) and len(fresh.legs) == 0
                        and (row["status"] or "").lower() != "new")
                )
                if not is_closing:
                    try:
                        from ..notifications.realtime import TradeAlert, notify_trade
                        notify_trade(TradeAlert(
                            kind="FILLED",
                            symbol=row["ticker"],
                            direction=row["direction"],
                            qty=float(fresh.filled_qty or 0),
                            price=float(fresh.filled_avg_price),
                            notional_usd=(float(fresh.filled_qty or 0)
                                          * float(fresh.filled_avg_price)),
                            extra="entry fill confirmed",
                        ))
                    except Exception:  # noqa: BLE001
                        log.debug("FILLED notification failed")

    def _maybe_realize_pnl(self, conn, row, fresh) -> None:
        """If this filled order is a closing child leg, attribute P&L to its parent.

        Looks up the parent via Alpaca's ``legs`` linkage (the closing leg's
        parent_id, when Alpaca returns nested orders). Falls back to the
        ticker-direction heuristic only if no FK is available.

        Uses ``abs(qty)`` throughout so short fills (where Alpaca returns
        negative qty) don't double-flip the P&L sign.
        """
        # Preferred path: Alpaca returns the parent_id (or legs.parent_id) on the closing leg
        parent_alpaca_id = (
            getattr(fresh, "legs", None) and len(fresh.legs) and fresh.legs[0].get("id")
        )
        # Strict parent lookup via Alpaca order ID — race-safe even when
        # multiple children fill in the same loop iteration.
        parent = None
        if parent_alpaca_id:
            parent = conn.execute(
                """
                SELECT id, qty, filled_avg_price, direction
                FROM bot_orders
                WHERE alpaca_order_id = ?
                  AND order_class = 'bracket'
                  AND realized_pnl_usd IS NULL
                """,
                (parent_alpaca_id,),
            ).fetchone()
        if parent is None:
            # Fallback heuristic — same as before, but only triggered when
            # Alpaca didn't expose the parent linkage.
            parent = conn.execute(
                """
                SELECT id, qty, filled_avg_price, direction
                FROM bot_orders
                WHERE ticker = ? AND direction != ?
                  AND realized_pnl_usd IS NULL
                  AND order_class = 'bracket'
                  AND status = 'filled'
                ORDER BY id DESC LIMIT 1
                """,
                (row["ticker"], row["direction"]),
            ).fetchone()
        if not parent or not parent["filled_avg_price"] or not fresh.filled_avg_price:
            return
        entry = float(parent["filled_avg_price"])
        exit_ = float(fresh.filled_avg_price)
        # abs() so short fills (Alpaca returns negative qty) don't flip sign
        qty = abs(float(parent["qty"]))
        if qty <= 0:
            return
        # parent direction is the entry direction; pnl signed accordingly
        if parent["direction"] == "buy":
            pnl = (exit_ - entry) * qty
        else:
            pnl = (entry - exit_) * qty
        notional = entry * qty
        pct = pnl / notional if notional > 0 else 0.0
        exit_reason = "take_profit" if pnl > 0 else "stop_loss"
        conn.execute(
            """
            UPDATE bot_orders SET realized_pnl_usd = ?, pnl_pct = ?, exit_reason = ?
            WHERE id = ?
            """,
            (pnl, pct, exit_reason, parent["id"]),
        )
        self._update_daily_pnl(conn, pnl)
        log.info(
            "Realized P&L: %s %s qty=%.2f entry=%.2f exit=%.2f -> $%.2f (%.2f%%) [%s]",
            parent["direction"], row["ticker"], qty, entry, exit_, pnl, pct * 100,
            exit_reason,
        )
        # Telegram on every fill (silent fail)
        try:
            from ..notifications.realtime import TradeAlert, notify_trade
            notify_trade(TradeAlert(
                kind="FILLED", symbol=row["ticker"], pnl_usd=pnl,
                extra=f"{parent['direction']} {qty:.0f} @{entry:.2f}→{exit_:.2f} ({exit_reason})",
            ))
        except Exception:  # noqa: BLE001
            pass

    def _update_daily_pnl(self, conn, pnl_usd: float) -> None:
        today = _us_eastern_date().isoformat()
        conn.execute(
            """
            INSERT INTO bot_daily_pnl
                (trading_date, realized_pnl_usd, trades_count, wins, losses,
                 largest_win, largest_loss, updated_at)
            VALUES (?, ?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(trading_date) DO UPDATE SET
                realized_pnl_usd = realized_pnl_usd + excluded.realized_pnl_usd,
                trades_count = trades_count + 1,
                wins = wins + excluded.wins,
                losses = losses + excluded.losses,
                largest_win = MAX(largest_win, excluded.largest_win),
                largest_loss = MIN(largest_loss, excluded.largest_loss),
                updated_at = excluded.updated_at
            """,
            (today, pnl_usd,
             1 if pnl_usd > 0 else 0,
             1 if pnl_usd < 0 else 0,
             max(pnl_usd, 0.0), min(pnl_usd, 0.0), utc_now()),
        )

    # ------------------------------------------------------------------
    # Step 3: candidate query
    # ------------------------------------------------------------------
    def _fetch_candidates(self, *, exclude_symbols: set[str]) -> list[dict]:
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(minutes=self.cfg.freshness_minutes))
        cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Rewritten 2026-05-27 — gate primarily on model_p_5d (measured edge:
        # 75% hit rate at p>=0.65 vs. 37% at composite>=7). The composite
        # filter is now a soft floor (default 0). Add factual / anti-pump /
        # corroboration gates which materially improve precision in the
        # measured data. Price-action signals (source LIKE 'price_action_%')
        # are direct observations so they bypass the corroboration requirement.
        require_factual_clause = (
            "AND COALESCE(ss.factual, 0) = 1" if self.cfg.require_factual else ""
        )
        anti_pump_clause = (
            "AND COALESCE(ss.anti_pump_flag, 0) = 0" if self.cfg.block_anti_pump else ""
        )
        if self.cfg.min_corroboration > 0 and self.cfg.bypass_corroboration_for_price_action:
            corroboration_clause = (
                f"AND (rs.source LIKE 'price_action_%' "
                f"     OR COALESCE(ss.corroboration_count, 0) >= {self.cfg.min_corroboration})"
            )
        elif self.cfg.min_corroboration > 0:
            corroboration_clause = (
                f"AND COALESCE(ss.corroboration_count, 0) >= {self.cfg.min_corroboration}"
            )
        else:
            corroboration_clause = ""

        # Drop low-edge event types entirely. Price-action events (event_type
        # starts with 'pa_') get a free pass — they're not LLM-classified noise.
        if self.cfg.blocked_event_types:
            placeholders = ",".join("?" * len(self.cfg.blocked_event_types))
            blocked_event_clause = (
                f"AND (ss.event_type IS NULL "
                f"     OR ss.event_type LIKE 'pa\\_%' ESCAPE '\\' "
                f"     OR ss.event_type NOT IN ({placeholders}))"
            )
            blocked_event_params = tuple(self.cfg.blocked_event_types)
        else:
            blocked_event_clause = ""
            blocked_event_params = ()

        # Use the configured signal horizon (1d for day-trading, 5d for swing).
        # Falls back to model_p_5d if the chosen horizon's column is NULL.
        horizon_col = {
            "1d": "ss.model_p_1d",
            "5d": "ss.model_p_5d",
            "20d": "ss.model_p_20d",
        }.get(self.cfg.signal_horizon, "ss.model_p_5d")
        # Effective probability — chosen horizon if present, else 5d as fallback.
        p_expr = f"COALESCE({horizon_col}, ss.model_p_5d)"

        # Price-action signals are DIRECT OBSERVATIONS — they have their own
        # built-in direction/composite, not ML predictions. They bypass the
        # extreme-p requirement because the ML model has never seen this
        # event_type during training and would produce garbage predictions.
        # The signal itself IS the edge; we just need composite + sentiment.
        pa_bypass_clause = (
            "OR (rs.source LIKE 'price_action_%' "
            "    AND ss.composite_score >= 6.5 "
            "    AND ABS(COALESCE(ss.sentiment, 0)) >= 0.3)"
        )
        # News-quality bypass — high-confidence event types with strong
        # sentiment and factual=1 can fire even without extreme model_p.
        # The ML model returns ~0.42 for everything news-related (the
        # training distribution doesn't separate signal from noise on news);
        # the LLM event classifier + composite scorer are the real edge here.
        # Captures earnings beats, M&A, FDA, analyst moves — the trades
        # like DELL +30% that previously slipped through unfiltered.
        # 2026-05-29: Audit found SEC EDGAR M&A + activist signals (avg
        # composite 6.27-6.88) NEVER reached the trader at the 7.5 bar.
        # Historical signal_outcomes show these events deliver 40-50% 5d
        # returns when they hit (e.g., RGS +49%, BGDE +47%, JCSE +47%).
        # Tiered the threshold by event type: high-conviction filings
        # bypass at 6.0, medium news at 7.0, weaker analyst moves at 7.5.
        news_bypass_clause = (
            "OR ("
            "  COALESCE(ss.factual, 0) = 1 "
            "  AND ABS(COALESCE(ss.sentiment, 0)) >= 0.5 "
            "  AND ("
            "    (ss.composite_score >= 6.0 AND ss.event_type IN ("
            "      'm_a_announcement','m_a_confirmed','activist_position',"
            "      'fda_approval','fda_rejection','insider_buying_cluster',"
            "      'short_squeeze_setup','spinoff_announcement'))"
            "    OR (ss.composite_score >= 7.0 AND ss.event_type IN ("
            "      'earnings_beat','earnings_miss','guidance_raise','guidance_cut',"
            "      'buyback_announcement','contract_win_major','partnership_major'))"
            "    OR (ss.composite_score >= 7.5 AND ss.event_type IN ("
            "      'analyst_upgrade','analyst_downgrade','dividend_cut'))"
            "  )"
            ")"
        )

        with get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT ss.id AS score_id, ss.ticker AS ticker,
                       ss.composite_score,
                       {p_expr} AS model_p,
                       ss.model_p_1d, ss.model_p_5d, ss.model_p_20d,
                       ss.sentiment, ss.sentiment_magnitude,
                       ss.factual, ss.corroboration_count,
                       ss.source_weight, ss.scored_at, ss.signal_class,
                       ss.event_type, rs.source AS signal_source
                FROM signal_scores ss
                JOIN raw_signals rs ON rs.id = ss.signal_id
                LEFT JOIN bot_decisions bd ON bd.score_id = ss.id
                WHERE ss.composite_score >= ?
                  AND COALESCE(ss.source_weight, 0) >= ?
                  AND ss.scored_at >= ?
                  AND bd.id IS NULL
                  AND (
                       (
                         {p_expr} IS NOT NULL
                         AND ({p_expr} >= ? OR {p_expr} <= ?)
                         {require_factual_clause}
                         {anti_pump_clause}
                         {corroboration_clause}
                         {blocked_event_clause}
                       )
                       {pa_bypass_clause}
                       {news_bypass_clause}
                  )
                ORDER BY
                  CASE WHEN rs.source LIKE 'price_action_%' THEN 1 ELSE 0 END DESC,
                  CASE WHEN {p_expr} IS NOT NULL AND {p_expr} >= 0.5 THEN {p_expr}
                       WHEN {p_expr} IS NOT NULL THEN 1.0 - {p_expr}
                       ELSE 0.5 END DESC,
                  ss.scored_at DESC
                LIMIT 50
                """,
                (
                    self.cfg.composite_threshold,
                    self.cfg.min_source_weight,
                    cutoff_iso,
                    self.cfg.direction_p_buy_min,
                    self.cfg.direction_p_sell_max,
                    *blocked_event_params,
                ),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            symbol = _normalize_ticker(d["ticker"])
            if not symbol or symbol in exclude_symbols:
                continue
            d["symbol"] = symbol
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Step 4: process one candidate (dispatch stock vs options)
    # ------------------------------------------------------------------
    def _process_candidate(self, cand: dict, *, account, market_open: bool,
                            positions=None) -> None:
        symbol = cand["symbol"]
        score_id = cand["score_id"]
        p = float(cand["model_p"]) if cand.get("model_p") is not None else 0.50
        composite = float(cand["composite_score"])

        # Direction from model probability + sentiment confirmation.
        # Pass composite/event_type/factual so the news-bypass branch can fire.
        direction = self._pick_direction(
            p, cand.get("sentiment"), cand.get("signal_source"), symbol,
            composite=cand.get("composite_score"),
            event_type=cand.get("event_type"),
            factual=cand.get("factual"),
        )
        if direction is None:
            self._persist_decision(
                cand, gate_passed=False, gate_reason=f"p={p:.3f} neither buy/sell zone",
                risk_passed=False, risk_reason="n/a",
                outcome="gate_blocked", outcome_detail="no_direction",
            )
            return

        # OPEN UNIVERSE: for ANY US stock candidate (no '/' = not crypto),
        # try options first if enabled. If the spread can't be built
        # (no chain / illiquid / wide spread), the options path will signal
        # fallback via return value False, and we route to the stock path.
        is_crypto_sym = "/" in symbol
        if (self.cfg.options_enabled
                and self.options is not None
                and not is_crypto_sym):
            routed = self._process_option_candidate(
                cand, direction=direction, account=account, market_open=market_open,
                positions=positions,
            )
            # If options path successfully placed or risk-blocked, we're done.
            # If it returned the special FALLBACK_TO_STOCK sentinel, route to stock.
            if routed != "FALLBACK_TO_STOCK":
                return

        # Stock path (the legacy single-leg bracket flow below)
        self._process_stock_candidate(
            cand, direction=direction, account=account, market_open=market_open,
            positions=positions,
        )

    # ------------------------------------------------------------------
    # OPTIONS PATH — build a vertical debit spread + submit multi-leg
    # ------------------------------------------------------------------
    def _process_option_candidate(self, cand: dict, *, direction, account, market_open: bool,
                                   positions=None) -> None:
        """Build + size + submit a vertical debit spread for one signal.

        Persists to ``bot_decisions`` with outcome details so the audit trail
        is unified across stock and options paths.
        """
        symbol = cand["symbol"]
        p = float(cand["model_p"]) if cand.get("model_p") is not None else 0.50
        score_id = cand["score_id"]
        # Same PA bypass applies to options path
        is_pa_signal = (cand.get("signal_source") or "").startswith("price_action_")

        # Options markets only when stock market is open (Alpaca options follow
        # equity hours, no pre/post). Skip if closed.
        if not market_open:
            self._persist_decision(
                cand, gate_passed=True, gate_reason="market_closed (options)",
                risk_passed=False, risk_reason="market_closed",
                outcome="closed_market",
            )
            return

        # Build the spread + size it. Honor the equity override (paper £5k mode).
        # CRITICAL: PA signals have model_p ≈ 0.42 (ML isn't trained on raw
        # price-action setups). Pass the SAME synthetic high-conviction p
        # we use for sizing/risk — otherwise Kelly returns negative, every
        # PA-driven spread is "unbuildable", and options NEVER fire.
        builder_p = (0.70 if direction == "buy" else 0.30) if is_pa_signal else p
        eff_equity = self._effective_equity(account)
        # Pass composite + |sentiment| so the sizer can trigger high-conviction
        # 8% cap on strong PA signals (where model_p stays at 0.42 and the
        # ML high-conviction trigger never fires).
        _comp = cand.get("composite_score")
        _sent = cand.get("sentiment")
        try:
            spec, sizing = build_vertical_debit_spread(
                underlying=symbol, direction=direction, model_p=builder_p,
                account_equity_usd=eff_equity,
                options_client=self.options,
                target_dte=self.cfg.options_target_dte,
                min_dte=self.cfg.options_min_dte,
                max_dte=self.cfg.options_max_dte,
                composite_score=float(_comp) if _comp is not None else None,
                abs_sentiment=abs(float(_sent)) if _sent is not None else None,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Options spread build failed for %s: %s", symbol, exc)
            self._persist_decision(
                cand, gate_passed=True, gate_reason="ok",
                risk_passed=False, risk_reason=f"build_exception:{exc}",
                outcome="execution_failed",
            )
            return

        if spec is None or not sizing.tradeable:
            # OPEN UNIVERSE fallback: if the underlying just doesn't have a
            # viable spread (no chain, wide bid-ask, low OI, can't afford
            # 1 contract), signal the caller to route to stock path instead
            # of writing a permanent gate_block decision row.
            reason = (sizing.reason or "")
            fallback_triggers = (
                "no underlying price",
                "no call contracts",
                "no put contracts",
                "no tradable contracts",
                "no quote",
                "spread too wide",
                "OI too low",
                "no liquid strike pair",
                "can't afford even 1 contract",
            )
            if any(t in reason for t in fallback_triggers):
                log.info("[%s] options unbuildable (%s) — falling through to stock path",
                         symbol, reason[:80])
                return "FALLBACK_TO_STOCK"
            self._persist_decision(
                cand, gate_passed=False,
                gate_reason=f"options_unbuildable: {sizing.reason}",
                risk_passed=False, risk_reason="n/a",
                outcome="gate_blocked", outcome_detail=sizing.reason,
            )
            log.info("[%s OPT] skip: %s", symbol, sizing.reason)
            return

        # Risk manager — reuse same caps as stock path. PA signals get the
        # same synthetic-p substitution as the stock path.
        risk_p = (0.70 if direction == "buy" else 0.30) if is_pa_signal else p
        proposal = TradeProposal(
            ticker=symbol, direction=direction, size_pct=sizing.size_pct,
            calibrated_p=risk_p, sector=SECTOR_MAP.get(symbol.upper()),
            kind="option",
        )
        # Compute Alpaca's TRUE current gross from positions list (avoids
        # stale bot_orders rows). Pass spread debit as the notional —
        # max-loss is the right measure for options exposure.
        # Compute gross properly: stocks/crypto at |market_value|, but options
        # at their spread debit (max loss) — NOT summed legs (which double-
        # counts the spread). Each option contract symbol is 15+ chars.
        if positions is None:
            alpaca_gross = None
        else:
            stock_crypto_gross = sum(
                abs(float(pp.market_value)) for pp in positions
                if len(pp.symbol) <= 9  # stock or crypto pair
            )
            with get_connection() as _cg_conn:
                _opt_row = _cg_conn.execute(
                    "SELECT COALESCE(SUM(total_debit_usd), 0) FROM bot_option_spreads "
                    "WHERE closed_at IS NULL AND status='filled'"
                ).fetchone()
                option_gross = float(_opt_row[0]) if _opt_row else 0.0
            alpaca_gross = stock_crypto_gross + option_gross
        decision = self.risk.evaluate(
            proposal,
            account_equity_usd=self._effective_equity(account),
            current_gross_usd=alpaca_gross,
            current_notional_usd=sizing.total_debit_usd,
        )
        if not decision.allowed:
            self._persist_decision(
                cand, gate_passed=True, gate_reason="ok",
                risk_passed=False, risk_reason=decision.reason,
                risk_blocking_rule=decision.blocking_rule,
                outcome="risk_blocked", outcome_detail=decision.blocking_rule,
            )
            log.info("[%s OPT] RISK BLOCKED [%s]: %s",
                     symbol, decision.blocking_rule, decision.reason)
            return

        # Dry-run mode
        if self.cfg.dry_run:
            self._persist_option_decision(
                cand, spec=spec, sizing=sizing, direction=direction,
                outcome="dry_run", alpaca_order_id=None,
            )
            log.info(
                "[DRY-RUN OPT] %s %s %s  long=%s short=%s  exp=%s  contracts=%d  "
                "debit=$%.2f/spread  total_risk=$%.0f  max_gain=$%.0f  R:R=%.2f",
                direction.upper(), spec.strategy, symbol,
                spec.long_contract.strike_price, spec.short_contract.strike_price,
                spec.long_contract.expiration_date, sizing.contracts,
                spec.debit_per_spread, sizing.total_debit_usd, sizing.total_max_gain_usd,
                spec.reward_risk_ratio,
            )
            return

        # No per-time guard on options — the risk manager's per-ticker total
        # exposure cap (12% for options) handles stacking. High-conviction
        # signals can scale up to that ceiling, then naturally stop.

        # Submit the multi-leg order
        try:
            import time as _t_opt_open
            legs = spec_to_legs(spec)
            mleg = self.options.submit_multi_leg(
                legs=legs, qty=sizing.contracts,
                limit_price=spec.debit_per_spread,
                client_order_id=f"mr-opt-{score_id}-{_t_opt_open.time_ns()}",
            )
        except AlpacaError as exc:
            self._persist_option_decision(
                cand, spec=spec, sizing=sizing, direction=direction,
                outcome="execution_failed", outcome_detail=str(exc)[:200],
            )
            log.error("[%s OPT] submit failed: %s", symbol, exc)
            return

        decision_id = self._persist_option_decision(
            cand, spec=spec, sizing=sizing, direction=direction,
            outcome="placed", outcome_detail="multi_leg_submitted",
            alpaca_order_id=mleg.id,
        )
        self._persist_option_spread(
            decision_id=decision_id, spec=spec, sizing=sizing, mleg=mleg,
        )
        log.info(
            "OPTIONS SUBMITTED %s %s %s  K=%s/%s exp=%s  qty=%d  "
            "debit=$%.2f net=$%.0f  max_gain=$%.0f  order_id=%s",
            direction.upper(), spec.strategy, symbol,
            spec.long_contract.strike_price, spec.short_contract.strike_price,
            spec.long_contract.expiration_date, sizing.contracts,
            spec.debit_per_spread, sizing.total_debit_usd, sizing.total_max_gain_usd,
            mleg.id,
        )
        # NOTE: No Telegram notification at submit time. We only notify
        # when Alpaca confirms a FILL, which happens in
        # _reconcile_option_spreads. Submit ≠ fill — many submits get
        # rejected by Alpaca (bad quote, halted underlying, etc.) and would
        # send false-positive notifications.

    def _persist_option_decision(
        self, cand: dict, *, spec, sizing, direction, outcome: str,
        outcome_detail: Optional[str] = None, alpaca_order_id: Optional[str] = None,
    ) -> int:
        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO bot_option_decisions
                    (score_id, underlying, direction, model_p, strategy,
                     long_strike, short_strike, expiration_date, width_usd,
                     debit_per_spread, max_loss_per_spread, max_gain_per_spread,
                     contracts, total_debit_usd, kelly_raw, size_pct,
                     gate_passed, gate_reason, risk_passed, risk_reason,
                     outcome, outcome_detail, alpaca_order_id, decided_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'ok', 1, 'ok', ?, ?, ?, ?)
                """,
                (cand["score_id"], cand["symbol"], direction, cand["model_p"],
                 spec.strategy if spec else "unknown",
                 spec.long_contract.strike_price if spec else None,
                 spec.short_contract.strike_price if spec else None,
                 spec.long_contract.expiration_date if spec else None,
                 spec.width if spec else None,
                 spec.debit_per_spread if spec else None,
                 spec.max_loss_per_spread if spec else None,
                 spec.max_gain_per_spread if spec else None,
                 sizing.contracts, sizing.total_debit_usd,
                 sizing.kelly_raw, sizing.size_pct,
                 outcome, outcome_detail, alpaca_order_id, utc_now()),
            )
            decision_id = cur.lastrowid
            # INSERT OR IGNORE returns 0 when row already existed — look up
            # the existing decision id so the FK on bot_option_spreads holds.
            if not decision_id:
                row = conn.execute(
                    "SELECT id FROM bot_option_decisions WHERE score_id=?",
                    (cand["score_id"],),
                ).fetchone()
                decision_id = row[0] if row else None
            return decision_id

    def _persist_option_spread(self, *, decision_id: int, spec, sizing, mleg) -> None:
        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO bot_option_spreads
                    (decision_id, alpaca_order_id, client_order_id, underlying,
                     strategy, direction, long_strike, short_strike,
                     expiration_date, contracts, entry_debit_usd, total_debit_usd,
                     max_loss_usd, max_gain_usd, status, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (decision_id, mleg.id, mleg.client_order_id, spec.underlying,
                 spec.strategy, spec.direction,
                 spec.long_contract.strike_price, spec.short_contract.strike_price,
                 spec.long_contract.expiration_date, sizing.contracts,
                 spec.debit_per_spread, sizing.total_debit_usd,
                 sizing.total_max_loss_usd, sizing.total_max_gain_usd,
                 mleg.status, mleg.submitted_at or utc_now()),
            )
            spread_id = cur.lastrowid
            for role, leg_opt in (("long", spec.long_contract),
                                   ("short", spec.short_contract)):
                conn.execute(
                    """
                    INSERT INTO bot_option_legs
                        (spread_id, role, contract_symbol, option_type,
                         strike, expiration_date, side, ratio_qty)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (spread_id, role, leg_opt.symbol, leg_opt.type,
                     leg_opt.strike_price, leg_opt.expiration_date,
                     "buy" if role == "long" else "sell"),
                )

    def _process_stock_candidate(self, cand: dict, *, direction, account, market_open: bool,
                                  positions=None) -> None:
        """Original stock-bracket path. Dispatch upstream has already done
        the direction pick and routing decision."""
        symbol = cand["symbol"]
        p = float(cand["model_p"]) if cand.get("model_p") is not None else 0.50
        score_id = cand["score_id"]
        is_pa_signal = (cand.get("signal_source") or "").startswith("price_action_")

        # Market hours
        is_crypto = symbol in self.cfg.crypto_tickers or "/" in symbol
        if not is_crypto and not market_open and not self.cfg.allow_after_hours:
            self._persist_decision(
                cand, gate_passed=True, gate_reason="market_closed",
                risk_passed=False, risk_reason="market_closed",
                outcome="closed_market",
            )
            return

        # LATE-DAY GUARD: don't open new stock positions in the last 30 minutes.
        # EOD flatten fires at 5min-to-close which kills our P&L if we just
        # opened — TPs need time to mature. Crypto is exempt (24/7).
        if not is_crypto:
            mtc = self._minutes_until_close()
            if mtc is not None and mtc <= 30:
                self._persist_decision(
                    cand, gate_passed=True, gate_reason=f"late_day ({mtc}min to close)",
                    risk_passed=False, risk_reason=f"late_day_no_new_opens (mtc={mtc})",
                    outcome="closed_market",
                )
                return

        # PDT guard — block new stock positions if we're at the day-trade limit
        if self._pdt_blocked(symbol, account):
            self._persist_decision(
                cand, gate_passed=True, gate_reason="pdt_limit",
                risk_passed=False, risk_reason="pdt_day_trade_limit_reached",
                outcome="risk_blocked", outcome_detail="pdt_limit",
            )
            log.info("[%s] PDT BLOCKED — day-trade limit reached for the rolling 5-day window", symbol)
            return

        # Selective conformal gate — but bypass for price-action signals
        # (the bot's price-action scanner already enforces its own confluence
        # rules: volume confirmation, indicator extremes, etc. The conformal
        # gate is meant for ML-prediction signals, not direct observations.)
        is_pa_signal = (cand.get("signal_source") or "").startswith("price_action_")
        if is_pa_signal:
            from ..ml.selective_gate import GateDecision
            gate = GateDecision(
                trade=True, reason="price_action_bypass",
                p_calibrated=p, interval_width=None, base_disagreement=None,
            )
        else:
            gate = gate_decide(p_calibrated=p)
            if not gate.trade:
                self._persist_decision(
                    cand, gate_passed=False, gate_reason=gate.reason,
                    risk_passed=False, risk_reason="n/a",
                    outcome="gate_blocked",
                )
                log.info("[%s] gate blocked: %s", symbol, gate.reason)
                return

        # ATR + entry quote
        atr = get_atr(symbol)
        if atr is None:
            self._persist_decision(
                cand, gate_passed=True, gate_reason=gate.reason,
                risk_passed=False, risk_reason="atr_unavailable",
                outcome="no_quote", outcome_detail="atr_unavailable",
            )
            log.warning("[%s] no ATR — skipping", symbol)
            return

        quote = self.alpaca.get_latest_quote(symbol)
        if quote is None:
            last = self.alpaca.get_latest_trade(symbol)
            if last is None:
                self._persist_decision(
                    cand, gate_passed=True, gate_reason=gate.reason,
                    risk_passed=False, risk_reason="no_quote",
                    outcome="no_quote",
                )
                log.warning("[%s] no quote/trade — skipping", symbol)
                return
            entry = last
        else:
            bid, ask = quote
            entry = ask if direction == "buy" else bid

        # Sizing — base Kelly size, then apply confluence + macro regime
        # + learning multipliers.
        from ..signals.confluence import get_confluence_multiplier
        conf_mult, conf_reasons = get_confluence_multiplier(
            symbol, direction, return_reasons=True,
        )
        regime_mult = (self._current_regime.size_multiplier
                       if self._current_regime else 1.0)
        # LEARNING MULT — adapts position size to historical win rate on this
        # (ticker, event_type) combination. Defaults to 1.0× until ≥3 trades.
        learn_mult, learn_reason = self._learning_multiplier(
            symbol, cand.get("event_type"),
        )
        total_mult = conf_mult * regime_mult * learn_mult
        eff_equity = self._effective_equity(account)
        adjusted_equity = eff_equity * total_mult
        # Crypto allows fractional qty, lower min_qty floor too
        is_crypto_sym = "/" in symbol
        # PA signals: use the synthetic high-conviction p for sizing too,
        # otherwise Kelly produces tiny sizes ($135) because PA signals'
        # model_p is centered around 0.42-0.45.
        sizing_p = (0.70 if direction == "buy" else 0.30) if is_pa_signal else p
        sized = size_trade(
            direction=direction, entry_price=entry, atr=atr,
            calibrated_p=sizing_p, account_equity_usd=adjusted_equity,
            sl_atr_mult=self.cfg.stock_sl_atr_mult,
            tp_atr_mult=self.cfg.stock_tp_atr_mult,
            allow_fractional=is_crypto_sym,
            min_qty=(0.0001 if is_crypto_sym else 1.0),
        )
        if conf_reasons or regime_mult != 1.0 or learn_mult != 1.0:
            log.info(
                "[%s] sizing mults: confluence=%.2fx (%s) regime=%.2fx learn=%.2fx (%s) → total=%.2fx",
                symbol, conf_mult, ",".join(conf_reasons) or "none",
                regime_mult, learn_mult, learn_reason or "neutral", total_mult,
            )
        if not sized.tradeable:
            self._persist_decision(
                cand, gate_passed=True, gate_reason=gate.reason,
                risk_passed=False, risk_reason=sized.reason,
                outcome="gate_blocked", outcome_detail=f"sizing:{sized.reason}",
                sized=sized,
            )
            log.info("[%s] sizing rejected: %s", symbol, sized.reason)
            return

        # Risk manager (7 hard rules, fed by Alpaca state). For price-action
        # signals we substitute calibrated_p with a synthetic value derived
        # from sentiment so we don't trip the risk manager's min_p rule —
        # the price-action scanner has its own edge proof (volume, indicator
        # extremes) that the risk manager's prob-based gate doesn't see.
        if is_pa_signal:
            sentiment = cand.get("sentiment") or 0.0
            synth_p = 0.70 if direction == "buy" else 0.30
            risk_p = synth_p
        else:
            risk_p = p

        proposal = TradeProposal(
            ticker=symbol, direction=direction, size_pct=sized.size_pct,
            calibrated_p=risk_p, sector=SECTOR_MAP.get(symbol.upper()),
            kind=("crypto" if "/" in symbol else "stock"),
        )
        # Pass Alpaca's TRUE current state to risk manager — avoids stale
        # bot_orders rows. Stock/crypto at |market_value|; options at spread
        # debit (NOT both legs).
        if positions is None:
            alpaca_gross = None
            alpaca_crypto = None
            alpaca_ticker = None
        else:
            stock_crypto_gross = sum(
                abs(float(pp.market_value)) for pp in positions
                if len(pp.symbol) <= 9
            )
            alpaca_crypto = sum(
                abs(float(pp.market_value)) for pp in positions
                if pp.symbol.upper().endswith("USD") and len(pp.symbol) <= 9
            )
            # Exposure on THIS ticker only (normalize crypto BTC/USD ↔ BTCUSD)
            sym_norm = symbol.replace("/", "").upper()
            alpaca_ticker = sum(
                abs(float(pp.market_value)) for pp in positions
                if pp.symbol.upper() == sym_norm or pp.symbol.upper() == symbol.upper()
            )
            with get_connection() as _cg_conn:
                _opt_row = _cg_conn.execute(
                    "SELECT COALESCE(SUM(total_debit_usd), 0) FROM bot_option_spreads "
                    "WHERE closed_at IS NULL AND status='filled'"
                ).fetchone()
                option_gross = float(_opt_row[0]) if _opt_row else 0.0
            alpaca_gross = stock_crypto_gross + option_gross
        decision = self.risk.evaluate(
            proposal,
            account_equity_usd=self._effective_equity(account),
            current_gross_usd=alpaca_gross,
            current_notional_usd=sized.notional_usd,
            current_crypto_usd=alpaca_crypto,
            current_ticker_usd=alpaca_ticker,
        )
        if not decision.allowed:
            self._persist_decision(
                cand, gate_passed=True, gate_reason=gate.reason,
                risk_passed=False, risk_reason=decision.reason,
                risk_blocking_rule=decision.blocking_rule,
                outcome="risk_blocked", outcome_detail=decision.blocking_rule,
                sized=sized,
            )
            log.info("[%s] RISK BLOCKED [%s]: %s",
                     symbol, decision.blocking_rule, decision.reason)
            return

        # Submit (or dry-run)
        if self.cfg.dry_run:
            self._persist_decision(
                cand, gate_passed=True, gate_reason=gate.reason,
                risk_passed=True, risk_reason="ok",
                outcome="placed", outcome_detail="dry_run",
                sized=sized,
            )
            log.info(
                "[DRY-RUN] %s %s qty=%.0f @~%.2f  SL=%.2f TP=%.2f  size=%.2f%% notional=$%.0f",
                direction.upper(), symbol, sized.qty, entry,
                sized.stop_loss, sized.take_profit, sized.size_pct, sized.notional_usd,
            )
            return

        # PRE-SUBMIT PRICE SANITY: refetch latest trade price and confirm
        # stops are on the correct side of current price. Catches the stale-
        # quote failure mode where IEX feed lagged 5%+ and our bracket would
        # be rejected as `stop_price > base_price` (the JPM-at-$315-vs-$297 case).
        is_crypto_check = "/" in symbol
        if not is_crypto_check:  # crypto path uses simple orders, no stop validation
            latest = self.alpaca.get_latest_trade(symbol)
            if latest is not None and latest > 0:
                if direction == "buy" and (sized.stop_loss >= latest or sized.take_profit <= latest):
                    log.warning(
                        "[%s] STALE QUOTE: cached $%.2f vs fresh $%.2f, "
                        "SL=$%.2f TP=$%.2f would be invalid for buy — skipping",
                        symbol, entry, latest, sized.stop_loss, sized.take_profit,
                    )
                    self._persist_decision(
                        cand, gate_passed=True, gate_reason=gate.reason,
                        risk_passed=False, risk_reason="stale_quote_invalid_bracket",
                        outcome="no_quote",
                        outcome_detail=f"latest={latest:.2f} SL={sized.stop_loss:.2f} TP={sized.take_profit:.2f}",
                        sized=sized,
                    )
                    return
                if direction == "sell" and (sized.stop_loss <= latest or sized.take_profit >= latest):
                    log.warning(
                        "[%s] STALE QUOTE: cached $%.2f vs fresh $%.2f, "
                        "SL=$%.2f TP=$%.2f would be invalid for sell — skipping",
                        symbol, entry, latest, sized.stop_loss, sized.take_profit,
                    )
                    self._persist_decision(
                        cand, gate_passed=True, gate_reason=gate.reason,
                        risk_passed=False, risk_reason="stale_quote_invalid_bracket",
                        outcome="no_quote",
                        outcome_detail=f"latest={latest:.2f} SL={sized.stop_loss:.2f} TP={sized.take_profit:.2f}",
                        sized=sized,
                    )
                    return

        # DOUBLE-OPEN GUARD: crypto market orders may take 5-30s to show up
        # in get_positions(). If the bot iterates faster than fill propagation,
        # it sees "no position" and submits AGAIN for the same ticker.
        # Track recent submits in-memory; reject if <90s since last submit
        # for this ticker.
        import time as _t
        if not hasattr(self, "_recent_submits"):
            self._recent_submits = {}
        sym_upper = symbol.upper()
        last_submit = self._recent_submits.get(sym_upper, 0)
        now_ts = _t.time()
        if now_ts - last_submit < 90:
            log.warning(
                "[%s] DOUBLE-OPEN GUARD: %.0fs since last submit (<90s) — skipping",
                symbol, now_ts - last_submit,
            )
            self._persist_decision(
                cand, gate_passed=True, gate_reason=gate.reason,
                risk_passed=False, risk_reason="double_open_guard",
                outcome="risk_blocked",
                outcome_detail=f"submitted_{int(now_ts - last_submit)}s_ago",
                sized=sized,
            )
            return
        self._recent_submits[sym_upper] = now_ts

        # IDEMPOTENCY FIX: persist a 'pending_submit' decision row BEFORE
        # calling Alpaca. On a 5xx/timeout the row blocks duplicate retries
        # of the same score_id (UNIQUE constraint).
        # Include a timestamp suffix in client_order_id so retries don't
        # collide with previous attempts at Alpaca (which would 422).
        import time as _t_mod
        # Nanosecond resolution + 7-digit slice so retries within the same
        # second still get distinct IDs (the previous per-second suffix
        # collided on retry storms and Alpaca 422'd them).
        client_order_id = f"mr-s{score_id}-{_t_mod.time_ns()}"
        self._persist_decision(
            cand, gate_passed=True, gate_reason=gate.reason,
            risk_passed=True, risk_reason="ok",
            outcome="pending_submit", outcome_detail=client_order_id,
            sized=sized,
        )

        # Crypto branch: Alpaca does NOT support bracket orders on crypto.
        # Submit a simple market order; stop/TP enforcement is handled by
        # the crypto-position poller (the next loop iteration will close
        # at TP/SL based on quote monitoring).
        is_crypto_order = "/" in symbol
        try:
            if is_crypto_order:
                # Crypto supports fractional quantities — convert to notional-sized fractional qty
                # so the trade isn't rejected by the qty<1 floor.
                # sized.qty was integer-floored for stocks; for crypto, recompute as notional/price.
                fractional_qty = round(sized.notional_usd / entry, 6) if entry > 0 else sized.qty
                if fractional_qty <= 0:
                    raise AlpacaError(f"crypto fractional qty {fractional_qty} <= 0")
                simple = self.alpaca.submit_simple_order(
                    symbol=symbol, side=direction, qty=fractional_qty,
                    order_type="market", time_in_force="gtc",
                    client_order_id=client_order_id,
                )
                placed_id = simple.id
                placed_qty = fractional_qty
                placed_kind = "simple_market_crypto"
            else:
                # Extended-hours flag: True when market is closed but
                # allow_after_hours is enabled. Alpaca requires limit_price
                # for extended-hours fills (no market orders allowed).
                ext_hours = (not market_open) and self.cfg.allow_after_hours
                bracket = self.alpaca.submit_bracket_order(
                    symbol=symbol, side=direction, qty=sized.qty,
                    take_profit=sized.take_profit, stop_loss=sized.stop_loss,
                    time_in_force="day",
                    client_order_id=client_order_id,
                    extended_hours=ext_hours,
                    limit_price=(entry if ext_hours else None),
                )
                placed_id = bracket.parent.id
                placed_qty = sized.qty
                placed_kind = ("bracket_submitted_extended"
                               if ext_hours else "bracket_submitted")
        except AlpacaError as exc:
            with get_connection() as conn:
                conn.execute(
                    "UPDATE bot_decisions SET outcome=?, outcome_detail=? WHERE score_id=?",
                    ("execution_failed", str(exc)[:200], cand["score_id"]),
                )
            log.error("[%s] Alpaca submit failed (decision row kept to block retry): %s",
                      symbol, exc)
            return

        # Success — flip the row from pending_submit to placed.
        with get_connection() as conn:
            conn.execute(
                "UPDATE bot_decisions SET outcome=?, outcome_detail=?, alpaca_order_id=? "
                "WHERE score_id=?",
                ("placed", placed_kind, placed_id, cand["score_id"]),
            )
            row = conn.execute(
                "SELECT id FROM bot_decisions WHERE score_id=?",
                (cand["score_id"],),
            ).fetchone()
            decision_id = row[0] if row else None
        if decision_id is not None and not is_crypto_order:
            self._persist_order(bracket, sized=sized, direction=direction,
                                symbol=symbol, decision_id=decision_id)
        log.info(
            "PLACED %s %s %s qty=%g @~%.2f  SL=%.2f TP=%.2f  "
            "size=%.2f%% notional=$%.0f  order_id=%s",
            "CRYPTO" if is_crypto_order else "STOCK",
            direction.upper(), symbol, placed_qty, entry,
            sized.stop_loss, sized.take_profit,
            sized.size_pct, sized.notional_usd, placed_id,
        )
        # NOTE: No Telegram notification at submit time. Stock/crypto fills
        # are confirmed via _reconcile_orders, which sends the FILLED alert
        # only on confirmed state transitions. Eliminates false-positive
        # notifications on rejected/canceled submits.

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _persist_decision(
        self,
        cand: dict,
        *,
        gate_passed: bool,
        gate_reason: str,
        risk_passed: bool,
        risk_reason: str,
        outcome: str,
        outcome_detail: Optional[str] = None,
        sized: Optional[SizingResult] = None,
        risk_blocking_rule: Optional[str] = None,
        alpaca_order_id: Optional[str] = None,
    ) -> int:
        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO bot_decisions
                    (score_id, ticker, direction, model_p, composite_score,
                     gate_passed, gate_reason, risk_passed, risk_reason,
                     risk_blocking_rule, account_equity, size_pct, notional_usd,
                     qty, entry_estimate, stop_loss, take_profit, atr,
                     outcome, outcome_detail, alpaca_order_id, decided_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (cand["score_id"], cand["symbol"],
                 self._pick_direction(cand["model_p"], cand.get("sentiment")) or "buy",
                 cand["model_p"], cand["composite_score"],
                 int(gate_passed), gate_reason, int(risk_passed), risk_reason,
                 risk_blocking_rule,
                 None, sized.size_pct if sized else None,
                 sized.notional_usd if sized else None,
                 sized.qty if sized else None,
                 sized.entry_estimate if sized else None,
                 sized.stop_loss if sized else None,
                 sized.take_profit if sized else None,
                 sized.atr if sized else None,
                 outcome, outcome_detail, alpaca_order_id, utc_now()),
            )
            return cur.lastrowid

    def _persist_order(
        self,
        bracket: BracketOrder,
        *,
        sized: SizingResult,
        direction: str,
        symbol: str,
        decision_id: int,
    ) -> None:
        p = bracket.parent
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO bot_orders
                    (decision_id, alpaca_order_id, client_order_id, ticker,
                     direction, order_class, qty, submitted_price,
                     stop_loss, take_profit,
                     status, filled_qty, filled_avg_price, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (decision_id, p.id, p.client_order_id, symbol, direction,
                 "bracket", sized.qty, p.limit_price or sized.entry_estimate,
                 sized.stop_loss, sized.take_profit,
                 p.status, p.filled_qty, p.filled_avg_price, p.submitted_at or utc_now()),
            )

    # ------------------------------------------------------------------
    # Direction picker — symmetric "buy" / "sell" zones around 0.5
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # OPTIONS EXIT POLLER — closes open spreads at TP/SL/time-stop.
    # Without this, spreads sit until expiration (max loss or full gain
    # both equally likely). With this, we lock 50% of max gain or cut
    # 50% of max loss, whichever fires first. Time-stop at 1-DTE avoids
    # pin risk + assignment.
    # ------------------------------------------------------------------
    def _poll_crypto_exits(self, positions) -> None:
        """Poll open positions and close at SL/TP from bot_decisions.

        Covers BOTH:
        - Crypto positions (no Alpaca-side brackets available — broker limit)
        - Stock positions whose Alpaca brackets were cancelled (e.g. by EOD
          flatten or cancel_all_orders) — backfilled SL/TP in bot_decisions

        Skips option contract legs (managed by _poll_option_exits).
        """
        # Filter: include crypto + stocks, EXCLUDE option contracts (long symbols)
        eligible = [p for p in positions if len(p.symbol) <= 15]
        if not eligible:
            return
        crypto_positions = eligible  # variable kept for downstream code compat
        # Build symbol-normalized lookup (Alpaca returns 'BATUSD', we stored 'BAT/USD')
        def _norm(s: str) -> str:
            return s.upper().replace("/", "")
        with get_connection() as conn:
            sl_tp_rows = conn.execute(
                """
                SELECT ticker, direction, stop_loss, take_profit, score_id, qty
                FROM bot_decisions
                WHERE outcome='placed' AND ticker LIKE '%/%'
                  AND stop_loss IS NOT NULL AND take_profit IS NOT NULL
                ORDER BY id DESC
                """
            ).fetchall()
        # Index by normalized symbol — take the MOST RECENT decision per ticker
        sl_tp_by_sym: dict[str, dict] = {}
        for r in sl_tp_rows:
            key = _norm(r["ticker"])
            if key not in sl_tp_by_sym:
                sl_tp_by_sym[key] = dict(r)

        for p in crypto_positions:
            key = _norm(p.symbol)
            tp_sl = sl_tp_by_sym.get(key)
            if not tp_sl:
                continue  # no recorded SL/TP for this position
            # Pair form Alpaca expects (BAT/USD not BATUSD)
            pair = tp_sl["ticker"]
            current_price = float(p.current_price) if p.current_price else None
            if not current_price:
                latest = self.alpaca.get_latest_crypto_trade(pair)
                if not latest:
                    continue
                current_price = float(latest)
            sl = float(tp_sl["stop_loss"])
            tp = float(tp_sl["take_profit"])
            direction = tp_sl["direction"]
            entry = float(p.avg_entry_price)

            # TRAILING STOP for winners:
            # Once price has moved >= TRAIL_ARM_PCT in our favor, ratchet the
            # stop to lock in profit. Trail stop = max(original_SL, peak * (1 - TRAIL_GIVEBACK_PCT))
            # Lets winners run beyond the fixed TP and exits when momentum reverses.
            TRAIL_ARM_PCT = 0.02   # arm trail once up 2% from entry
            TRAIL_GIVEBACK_PCT = 0.020  # exit if drops 2.0% from peak — sized for crypto noise band (1-3% intraday)

            # Track peak per ticker (in-memory cache — survives within bot lifetime)
            if not hasattr(self, "_crypto_peaks"):
                self._crypto_peaks = {}
            peak_key = (pair, direction)
            prev_peak = self._crypto_peaks.get(peak_key, entry)

            # Ratchet peak (favorable direction)
            if direction == "buy":
                peak = max(prev_peak, current_price)
                gain_pct = (peak - entry) / entry
                trail_armed = gain_pct >= TRAIL_ARM_PCT
                effective_sl = max(sl, peak * (1 - TRAIL_GIVEBACK_PCT)) if trail_armed else sl
            else:  # short
                peak = min(prev_peak, current_price)
                gain_pct = (entry - peak) / entry
                trail_armed = gain_pct >= TRAIL_ARM_PCT
                effective_sl = min(sl, peak * (1 + TRAIL_GIVEBACK_PCT)) if trail_armed else sl
            self._crypto_peaks[peak_key] = peak

            exit_reason = None
            if direction == "buy":
                # Take profit hit (fixed TP) — early lock
                if current_price >= tp:
                    exit_reason = "take_profit"
                # Trail stop hit (ratcheted up from peak)
                elif current_price <= effective_sl:
                    exit_reason = "trail_stop" if trail_armed else "stop_loss"
            else:  # sell / short
                if current_price <= tp:
                    exit_reason = "take_profit"
                elif current_price >= effective_sl:
                    exit_reason = "trail_stop" if trail_armed else "stop_loss"
            if not exit_reason:
                if trail_armed:
                    log.debug("  trail tracking %s: peak=$%.4f effSL=$%.4f current=$%.4f",
                              pair, peak, effective_sl, current_price)
                continue
            # Close at market
            try:
                close_side = "sell" if direction == "buy" else "buy"
                qty = abs(float(p.qty))
                log.info(
                    "CRYPTO EXIT [%s] %s %s qty=%g @ \$%.4f (entry SL=\$%.4f TP=\$%.4f)",
                    exit_reason, pair, close_side, qty, current_price, sl, tp,
                )
                if self.cfg.dry_run:
                    continue
                import time as _t_exit
                order = self.alpaca.submit_simple_order(
                    symbol=pair, side=close_side, qty=qty,
                    order_type="market", time_in_force="gtc",
                    client_order_id=f"mr-cx-{tp_sl['score_id']}-{_t_exit.time_ns()}",
                )
                # Estimate realized P&L (will be exact after fill reconciliation)
                entry = float(p.avg_entry_price)
                if direction == "buy":
                    pnl = (current_price - entry) * qty
                else:
                    pnl = (entry - current_price) * qty
                with get_connection() as conn:
                    self._update_daily_pnl(conn, pnl)
                try:
                    from ..notifications.realtime import TradeAlert, notify_trade
                    notify_trade(TradeAlert(
                        kind="FILLED", symbol=pair, pnl_usd=pnl,
                        extra=f"CRYPTO {exit_reason}: entry=${entry:.4f}→${current_price:.4f}",
                    ))
                except Exception:  # noqa: BLE001
                    pass
            except AlpacaError as exc:
                log.error("CRYPTO EXIT submit failed for %s: %s", pair, exc)

    def _check_crypto_flips(self, positions) -> None:
        """Close held crypto when a strong opposite-direction PA signal arrives
        AND the position is meaningfully losing.

        Guards (all must hold):
        - Position unrealized loss >= 1.5% (only cut clear losers, not chop)
        - Opposite signal composite >= 7.5 AND |sentiment| >= 0.7 (strong)
        - No SAME-direction PA signal in the same 5min window (avoid mixed reads)
        - 10-minute cooldown per pair (no thrashing)
        """
        if not positions:
            return
        crypto = [p for p in positions
                  if p.symbol.upper().endswith("USD") and len(p.symbol) <= 9]
        if not crypto:
            return

        import time as _t_flip
        now_ts = _t_flip.time()
        if not hasattr(self, "_recent_flips"):
            self._recent_flips = {}

        for p in crypto:
            slashed = _normalize_ticker(p.symbol.upper())
            if not slashed or "/" not in slashed:
                continue
            if now_ts - self._recent_flips.get(slashed, 0) < 600:
                continue

            # Only flip CLEAR losers — skip flat or winning positions
            try:
                mv = float(p.market_value)
                upnl = float(p.unrealized_pl)
                loss_pct = (upnl / mv) if mv > 0 else 0.0
            except Exception:  # noqa: BLE001
                continue
            if loss_pct > -0.015:
                continue

            held_dir = "buy" if float(p.qty) >= 0 else "sell"
            sent_sign = -1.0 if held_dir == "buy" else 1.0

            with get_connection() as conn:
                # Strong opposite signal in last 5 min
                opp = conn.execute(
                    """
                    SELECT ss.composite_score, ss.sentiment, rs.source
                    FROM signal_scores ss
                    JOIN raw_signals rs ON rs.id = ss.signal_id
                    WHERE ss.ticker = ?
                      AND rs.source LIKE 'price_action_%'
                      AND ss.composite_score >= 7.5
                      AND ss.scored_at > datetime('now','-5 minutes')
                      AND ((? > 0 AND ss.sentiment >= 0.7)
                           OR (? < 0 AND ss.sentiment <= -0.7))
                    ORDER BY ss.scored_at DESC LIMIT 1
                    """,
                    (slashed, sent_sign, sent_sign),
                ).fetchone()
                if not opp:
                    continue
                # Reject if same-direction PA signal also exists in window
                same = conn.execute(
                    """
                    SELECT 1 FROM signal_scores ss
                    JOIN raw_signals rs ON rs.id = ss.signal_id
                    WHERE ss.ticker = ?
                      AND rs.source LIKE 'price_action_%'
                      AND ss.composite_score >= 6.5
                      AND ss.scored_at > datetime('now','-5 minutes')
                      AND ((? > 0 AND ss.sentiment <= -0.3)
                           OR (? < 0 AND ss.sentiment >= 0.3))
                    LIMIT 1
                    """,
                    (slashed, sent_sign, sent_sign),
                ).fetchone()
                if same:
                    continue

            log.warning(
                "🔄 CRYPTO FLIP: %s held %s loss=%.1f%%, strong opposite "
                "(composite=%.1f sentiment=%+.2f from %s) — cutting loser",
                slashed, held_dir, loss_pct*100, opp[0], opp[1], opp[2],
            )
            try:
                close_side = "sell" if held_dir == "buy" else "buy"
                self.alpaca.submit_simple_order(
                    symbol=slashed, side=close_side, qty=abs(float(p.qty)),
                    order_type="market", time_in_force="gtc",
                    client_order_id=f"mr-flip-{slashed.replace('/','')[:6]}-{_t_flip.time_ns()}",
                )
                self._recent_flips[slashed] = now_ts
            except AlpacaError as exc:
                log.error("CRYPTO FLIP close failed for %s: %s", slashed, exc)

    def _poll_option_exits(self) -> None:
        if self.options is None:
            return
        with get_connection() as conn:
            open_spreads = conn.execute(
                """
                SELECT s.id, s.alpaca_order_id, s.underlying, s.direction,
                       s.strategy, s.long_strike, s.short_strike,
                       s.expiration_date, s.contracts, s.entry_debit_usd,
                       s.total_debit_usd, s.max_gain_usd, s.submitted_at
                FROM bot_option_spreads s
                WHERE s.closed_at IS NULL AND s.status IN ('filled','accepted','new')
                """
            ).fetchall()
            if not open_spreads:
                return
            leg_rows_by_spread: dict[int, list[dict]] = {}
            for sp in open_spreads:
                legs = conn.execute(
                    "SELECT contract_symbol, role, side FROM bot_option_legs WHERE spread_id=?",
                    (sp["id"],),
                ).fetchall()
                leg_rows_by_spread[sp["id"]] = [dict(L) for L in legs]

        from datetime import datetime as _dt, date as _date
        today = _date.today()

        for sp in open_spreads:
            try:
                legs = leg_rows_by_spread.get(sp["id"], [])
                if not legs:
                    continue

                # Mark-to-market: get current quotes for both legs, compute
                # current spread debit (= long_mid - short_mid). Compare to
                # entry debit. Profit = (entry - current) * contracts * 100
                # for the BUYER side... wait, for a debit spread you PAID
                # entry_debit and want the spread to widen (long leg gains
                # faster than short leg). To CLOSE, you sell-to-close the
                # long + buy-to-close the short, netting current_debit credit.
                # PnL = (current_debit - entry_debit) * contracts * 100.
                quotes = self.options.get_snapshots(
                    [L["contract_symbol"] for L in legs]
                )
                long_q = next((quotes.get(L["contract_symbol"]) for L in legs
                               if L["role"] == "long"), None)
                short_q = next((quotes.get(L["contract_symbol"]) for L in legs
                                if L["role"] == "short"), None)
                if not long_q or not short_q or long_q.mid <= 0 or short_q.mid <= 0:
                    continue

                current_debit = max(long_q.mid - short_q.mid, 0.01)
                entry_debit = float(sp["entry_debit_usd"])
                contracts = int(sp["contracts"])
                pnl_pct_of_debit = (current_debit - entry_debit) / entry_debit
                # max_gain per spread = (width - debit). TP at X% of max gain:
                # pnl_per_spread >= X * max_gain_per_spread
                # In pct-of-debit space: threshold = X * (max_gain/entry_debit)
                # The previous "(max_gain/entry - 1)" formula was BROKEN — it
                # produced negative thresholds (TP firing on ANY tiny gain or
                # even small losses) when max_gain < entry_debit (low-R:R
                # spreads). Fixed: drop the "- 1" — pure ratio of max gain.
                max_gain_per_spread = float(sp["max_gain_usd"]) / contracts / 100.0
                base_tp_pct = self.cfg.options_take_profit_pct * (
                    max_gain_per_spread / entry_debit
                )

                # ADAPTIVE MILESTONE LOCKING: track peak fraction-of-max-gain
                # captured. If peak crossed 50%, exit when current pulls back
                # to 80% of peak (lock most of the gain). If 75%, lock 90%.
                # Prevents AAPL-class give-back ($1687 → $1498) we saw today.
                # Peak is in-memory per spread; restart loses peak (conservative).
                if not hasattr(self, "_opt_peak_pnl"):
                    self._opt_peak_pnl = {}
                sp_id = sp["id"]
                # pnl_per_max = fraction of max-gain captured (0.0 = none, 1.0 = full)
                pnl_per_max = (current_debit - entry_debit) / max(max_gain_per_spread, 0.01)
                prev_peak = self._opt_peak_pnl.get(sp_id, 0.0)
                if pnl_per_max > prev_peak:
                    self._opt_peak_pnl[sp_id] = pnl_per_max
                    prev_peak = pnl_per_max
                # Compute the adaptive give-back exit level (in fraction-of-max units)
                if prev_peak >= 0.75:
                    adaptive_lock_level = 0.90 * prev_peak
                elif prev_peak >= 0.50:
                    adaptive_lock_level = 0.80 * prev_peak
                else:
                    adaptive_lock_level = 0.0  # not armed yet — only base TP applies

                tp_threshold_pct = base_tp_pct
                sl_threshold_pct = -self.cfg.options_stop_loss_pct  # e.g., -0.50 = lost half the debit

                # Time-based stop: close if expiry is <= time_stop_dte days
                exp_dt = _dt.strptime(sp["expiration_date"], "%Y-%m-%d").date()
                dte_remaining = (exp_dt - today).days
                # Max-hold stop: never hold > max_hold_hours
                hours_held = None
                try:
                    sub_dt = _dt.fromisoformat(sp["submitted_at"].replace("Z", "+00:00"))
                    hours_held = (_dt.now(sub_dt.tzinfo) - sub_dt).total_seconds() / 3600
                except Exception:  # noqa: BLE001
                    hours_held = None

                exit_reason = None
                # Adaptive trail: if peak was >= 50% of max gain and we've
                # pulled back to the lock level, exit and bank the win.
                if (adaptive_lock_level > 0
                        and pnl_per_max <= adaptive_lock_level
                        and prev_peak >= 0.50):
                    exit_reason = "trail_lock"
                elif pnl_pct_of_debit >= tp_threshold_pct:
                    exit_reason = "take_profit"
                elif pnl_pct_of_debit <= sl_threshold_pct:
                    exit_reason = "stop_loss"
                elif dte_remaining <= self.cfg.options_time_stop_dte:
                    exit_reason = "time_stop"
                elif hours_held is not None and hours_held >= self.cfg.options_max_hold_hours:
                    exit_reason = "max_hold"

                if exit_reason is None:
                    continue

                log.info(
                    "OPTIONS EXIT [%s] %s: %s/%s K=%g/%g  entry=$%.2f → current=$%.2f "
                    "(%.1f%%)  DTE=%d  reason=%s",
                    exit_reason, sp["underlying"], sp["strategy"], sp["direction"],
                    sp["long_strike"], sp["short_strike"],
                    entry_debit, current_debit, pnl_pct_of_debit * 100,
                    dte_remaining, exit_reason,
                )

                # Submit closing multi-leg order
                from .options.alpaca_options import OptionLeg
                close_legs = [
                    OptionLeg(
                        symbol=L["contract_symbol"],
                        side=("sell" if L["side"] == "buy" else "buy"),
                        position_intent=("sell_to_close" if L["side"] == "buy" else "buy_to_close"),
                        ratio_qty=1,
                    )
                    for L in legs
                ]
                if self.cfg.dry_run:
                    log.info("[DRY-RUN OPT EXIT] %s contracts=%d limit=$%.2f",
                             sp["underlying"], contracts, current_debit)
                    realized = (current_debit - entry_debit) * contracts * 100
                    with get_connection() as conn:
                        conn.execute(
                            "UPDATE bot_option_spreads SET closed_at=?, "
                            "exit_credit_usd=?, realized_pnl_usd=?, pnl_pct=?, exit_reason=? "
                            "WHERE id=?",
                            (utc_now(), current_debit, realized,
                             pnl_pct_of_debit * 100, exit_reason, sp["id"]),
                        )
                    continue

                try:
                    import time as _t_optx
                    mleg = self.options.submit_multi_leg(
                        legs=close_legs, qty=contracts,
                        limit_price=current_debit,
                        client_order_id=f"mr-ox-{sp['id']}-{_t_optx.time_ns()}",
                    )
                except AlpacaError as exc:
                    log.error("OPTIONS EXIT submit failed for spread %d: %s",
                              sp["id"], exc)
                    continue

                realized = (current_debit - entry_debit) * contracts * 100
                with get_connection() as conn:
                    conn.execute(
                        "UPDATE bot_option_spreads SET closed_at=?, "
                        "exit_credit_usd=?, realized_pnl_usd=?, pnl_pct=?, exit_reason=? "
                        "WHERE id=?",
                        (utc_now(), current_debit, realized,
                         pnl_pct_of_debit * 100, exit_reason, sp["id"]),
                    )
                    # Also book to daily P&L
                    self._update_daily_pnl(conn, realized)

                try:
                    from ..notifications.realtime import TradeAlert, notify_trade
                    notify_trade(TradeAlert(
                        kind="FILLED", symbol=sp["underlying"],
                        pnl_usd=realized,
                        extra=f"OPTIONS {exit_reason}: {sp['strategy']} "
                              f"entry=${entry_debit:.2f}→${current_debit:.2f} ({pnl_pct_of_debit*100:+.1f}%)",
                    ))
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001
                log.exception("Options exit check failed for spread %d: %s",
                              sp["id"], exc)

    # ------------------------------------------------------------------
    # Day-trading helpers: effective equity, EOD flatten, PDT guard
    # ------------------------------------------------------------------
    def _effective_equity(self, account) -> float:
        """Account equity used for sizing. Honors LIVE_OVERRIDE_EQUITY_USD
        so paper accounts (start $100k) size for the user's real capital."""
        if self.cfg.override_equity_usd > 0:
            return min(float(account.equity), self.cfg.override_equity_usd)
        return float(account.equity)

    def _minutes_until_close(self) -> Optional[int]:
        """Minutes until next market close. None if clock fetch fails."""
        try:
            clock = self.alpaca.get_market_clock()
        except AlpacaError:
            return None
        if not clock.get("is_open"):
            return None
        nxt = clock.get("next_close")
        if not nxt:
            return None
        try:
            from datetime import datetime as _dt
            close_dt = _dt.fromisoformat(nxt.replace("Z", "+00:00"))
            now = _dt.now(close_dt.tzinfo)
            return max(int((close_dt - now).total_seconds() // 60), 0)
        except Exception:  # noqa: BLE001
            return None

    def _daily_profit_take_if_due(self, account, positions) -> bool:
        """Two modes (both check intraday = realized + unrealized P&L):

        1. HARD CAP: ``daily_profit_take_usd`` > 0 → flatten when crossed
        2. TRAILING: once intraday >= ``daily_tp_arm_at_usd``, track the peak;
           flatten when intraday drops ``daily_tp_giveback_usd`` below peak.

        Trailing is strategically optimal — locks target+ when hit but lets
        winners run as long as they keep climbing. Hard cap caps upside.

        Returns True if we just fired (or already fired today).
        """
        from datetime import date as _date
        today = _date.today()
        # Already fired today — but if positions remain, RETRY the flatten
        # (closes can fail silently for crypto/options; we must keep trying)
        if getattr(self, "_daily_tp_fired_on", None) == today:
            if positions:
                log.warning(
                    "🎯 DAILY TP already fired but %d positions remain — retrying close",
                    len(positions),
                )
                intraday = float(account.equity - account.last_equity)
                self._fire_daily_flatten(positions, intraday, "retry")
            return True

        intraday = float(account.equity - account.last_equity)

        # Mode 1: hard cap (if enabled)
        if self.cfg.daily_profit_take_usd > 0:
            if intraday >= self.cfg.daily_profit_take_usd:
                return self._fire_daily_flatten(positions, intraday, "hard_cap")

        # Mode 2: trailing TP (only armed once we cross arm threshold)
        if self.cfg.daily_tp_arm_at_usd > 0:
            peak = getattr(self, "_daily_tp_peak", None)
            armed = getattr(self, "_daily_tp_armed_on", None) == today

            if not armed and intraday >= self.cfg.daily_tp_arm_at_usd:
                self._daily_tp_armed_on = today
                self._daily_tp_peak = intraday
                armed = True
                peak = intraday
                log.warning("🎯 TRAILING TP ARMED at $%.2f — will lock if drops $%.2f from peak",
                            intraday, self.cfg.daily_tp_giveback_usd)
                try:
                    from ..notifications.realtime import TradeAlert, notify_trade
                    notify_trade(TradeAlert(
                        kind="PNL_DAY", symbol="ALL", pnl_usd=intraday,
                        extra=f"🎯 Trail armed — peak ${intraday:.0f}, will lock if drops ${self.cfg.daily_tp_giveback_usd:.0f}",
                    ))
                except Exception:  # noqa: BLE001
                    pass

            if armed:
                if intraday > (peak or 0):
                    self._daily_tp_peak = intraday  # ratchet up
                    peak = intraday
                # Trailing stop with FLOOR at arm price.
                # = max(arm_price, peak - giveback)
                # Effect: minimum lock = arm (= £150 target floor).
                # On big winners, ratchets up to capture the gain.
                # Strictly better than pure trail for "guaranteed target" goal.
                trail_stop = max(self.cfg.daily_tp_arm_at_usd,
                                 (peak or 0) - self.cfg.daily_tp_giveback_usd)
                if intraday <= trail_stop:
                    return self._fire_daily_flatten(
                        positions, intraday,
                        f"trail_peak_${peak:.0f}_stop_${trail_stop:.0f}"
                    )
        return False

    def _fire_daily_flatten(self, positions, intraday_pnl: float, mode: str) -> bool:
        """Flatten all positions + halt for day.

        VERIFICATION: re-fetches positions after close attempts, logs anything
        that didn't actually close. The 'fired' flag is set immediately to block
        new opens, but if any position remains we retry on subsequent iterations.

        Per-asset close strategy:
        - Stocks: close_position (Alpaca handles via market sell)
        - Crypto: close_position with /USD pair format
        - Option contract legs: submit_simple_order (close_position can 403 on isolated legs)
        """
        from datetime import date as _date
        import time as _time
        log.warning(
            "🎯 DAILY TP FIRED [%s]: intraday $%.2f. Flattening %d positions + halting for day.",
            mode, intraday_pnl, len(positions),
        )
        self._daily_tp_fired_on = _date.today()  # set IMMEDIATELY so new opens are blocked
        self._persist_daily_tp_state()  # persist to DB so restarts respect it

        # 1. Cancel all open orders FIRST (frees bracket-child qty)
        try:
            n = self.alpaca.cancel_all_orders()
            log.info("  cancelled %d open orders", n or 0)
        except AlpacaError as exc:
            log.warning("  cancel_all_orders failed: %s", exc)
        _time.sleep(2)

        # 2. Close each position with the right method per asset class
        attempted, succeeded = 0, 0
        for p in positions:
            attempted += 1
            sym = p.symbol
            is_option = len(sym) > 15
            # Crypto: Alpaca expects BAT/USD form, not BATUSD
            is_crypto = sym.endswith("USD") and "/" not in sym and len(sym) > 4 and sym != "PYUSD"
            close_sym = sym[:-3] + "/USD" if is_crypto else sym
            try:
                qty = abs(float(p.qty))
                side = "sell" if float(p.qty) > 0 else "buy"
                if is_option:
                    # Option legs need explicit simple_order (close_position 403s on naked legs)
                    self.alpaca.submit_simple_order(
                        symbol=sym, side=side, qty=qty,
                        order_type="market", time_in_force="day",
                        client_order_id=f"tpfire-{sym[:10]}-{_time.time_ns()}",
                    )
                elif is_crypto:
                    # Crypto close_position via DELETE endpoint accepts but
                    # often doesn't fill on Alpaca paper. Use explicit
                    # market+IOC submit for reliable fills.
                    self.alpaca.submit_simple_order(
                        symbol=close_sym, side=side, qty=qty,
                        order_type="market", time_in_force="ioc",
                        client_order_id=f"tpfirec-{close_sym.replace('/','')}-{_time.time_ns()}",
                    )
                else:
                    # Stocks: standard close_position works fine
                    self.alpaca.close_position(close_sym)
                succeeded += 1
                log.info("  ✓ submitted close for %s qty=%g u_pnl=$%.2f", sym, p.qty, p.unrealized_pl)
            except AlpacaError as exc:
                log.error("  ✗ close(%s) FAILED: %s — will retry next iteration", sym, exc)

        # 3. VERIFY — wait 3s, re-fetch positions, log anything still open
        _time.sleep(3)
        try:
            remaining = self.alpaca.get_positions()
        except AlpacaError:
            remaining = positions  # assume worst, retry next iteration

        if remaining:
            log.warning(
                "🎯 DAILY TP partial flatten: %d/%d closed, %d STILL OPEN — will retry next iteration",
                succeeded, attempted, len(remaining),
            )
            for p in remaining:
                log.warning("    still open: %s qty=%g", p.symbol, p.qty)
        else:
            log.warning("🎯 DAILY TP CONFIRMED FLATTEN: 0 positions remain")

        # Book realized P&L for actually-closed positions
        closed_syms = {p.symbol for p in positions} - {p.symbol for p in remaining}
        total_realized = 0.0
        for p in positions:
            if p.symbol in closed_syms:
                pnl = float(p.unrealized_pl)
                total_realized += pnl
                with get_connection() as conn:
                    conn.execute(
                        "UPDATE bot_orders SET realized_pnl_usd=?, exit_reason='daily_profit_take', "
                        "canceled_at=? WHERE ticker=? AND status='filled' AND realized_pnl_usd IS NULL",
                        (pnl, utc_now(), p.symbol),
                    )
                    self._update_daily_pnl(conn, pnl)

        # Telegram alert — ONLY on first fire of the day, NOT on retries.
        # Retries spam the user with identical messages.
        if mode != "retry" and not getattr(self, "_daily_tp_notified", False):
            self._daily_tp_notified = True
            try:
                from ..notifications.realtime import TradeAlert, notify_trade
                note = f"🎯 DAILY TP — closed {len(closed_syms)}/{attempted}, locked +${intraday_pnl:.2f}"
                if remaining:
                    note += f" ({len(remaining)} retrying)"
                notify_trade(TradeAlert(
                    kind="PNL_DAY", symbol="ALL", pnl_usd=intraday_pnl, extra=note,
                ))
            except Exception:  # noqa: BLE001
                pass
        try:
            from ..notifications.realtime import TradeAlert, notify_trade
            notify_trade(TradeAlert(
                kind="PNL_DAY", symbol="ALL", pnl_usd=intraday,
                extra=f"🎯 DAILY TP HIT — flattened {closed} positions, locked +${intraday:.2f}",
            ))
        except Exception:  # noqa: BLE001
            pass
        return True

    def _eod_flatten_if_due(self, positions) -> None:
        """Close all stock positions + day-trader option spreads
        ``eod_flatten_minutes_before_close`` before market close.

        Day-trader rule: no overnight stock holds, no overnight day-trader
        options (DTE ≤ 2 = a day-trader spread). Legacy 5+ DTE spreads
        carried over from earlier sessions ride to their own natural exits.
        Crypto is unaffected (24/7 market, different path).

        Records realized P&L into ``bot_orders`` + ``bot_daily_pnl`` so the
        dashboard + reporting see the closure. Uses unrealized_pl at flatten
        time as the realized P&L (approximation — actual fill price may slip
        a few cents below this on the market-sell, reconciliation catches it
        on the next loop).

        DEDUP: only fires ONCE per trading day. Tracks last-fired date in
        the instance so the 30s loop doesn't submit dup close orders every
        iteration for 5min (which was filling the Alpaca order history with
        ~10 duplicate cancels per ticker)."""
        from datetime import date as _date
        today = _date.today()
        if getattr(self, "_eod_flatten_fired_on", None) == today:
            return  # already fired today
        mtc = self._minutes_until_close()
        if mtc is None or mtc > self.cfg.eod_flatten_minutes_before_close:
            return
        if not positions:
            return
        self._eod_flatten_fired_on = today  # mark BEFORE submit to prevent races
        log.warning("EOD FLATTEN: %d minutes to close — closing %d stock positions",
                    mtc, len(positions))
        # FIX (2026-05-27): cancel ALL open orders FIRST. Otherwise the bracket
        # TP/SL child orders hold the position qty hostage and close_position
        # fails. This was the bug that left XLE/XOM/QCOM stuck overnight.
        try:
            n = self.alpaca.cancel_all_orders()
            log.info("EOD: pre-flatten cancelled %d open orders", n or 0)
        except AlpacaError as exc:
            log.warning("EOD pre-cancel failed (continuing): %s", exc)
        # Give Alpaca a beat to process the cancels before submitting closes
        import time as _time
        _time.sleep(2)
        total_pnl = 0.0
        closed_count = 0
        for p in positions:
            sym = p.symbol.upper()
            # Skip crypto (24/7, different path) and option contracts
            # (closed separately via _eod_flatten_options below — closing
            # individual legs via close_position can leave naked legs)
            if "/" in sym or len(sym) > 9:
                continue
            try:
                self.alpaca.close_position(sym)
                pnl = float(p.unrealized_pl)
                total_pnl += pnl
                closed_count += 1
                log.info("  EOD closed %s qty=%.2f u_pnl=$%.2f", sym, p.qty, pnl)
                # Record realized P&L into bot_orders (matched by ticker)
                # + book to daily P&L. Idempotent — won't double-count because
                # the reconcile loop only fills realized_pnl_usd once.
                try:
                    with get_connection() as conn:
                        conn.execute(
                            """
                            UPDATE bot_orders
                            SET realized_pnl_usd = ?,
                                pnl_pct = ?,
                                exit_reason = 'eod_flatten',
                                canceled_at = ?
                            WHERE ticker = ?
                              AND status = 'filled'
                              AND realized_pnl_usd IS NULL
                            """,
                            (pnl,
                             pnl / float(p.market_value) if p.market_value else 0,
                             utc_now(), sym),
                        )
                        self._update_daily_pnl(conn, pnl)
                except Exception as exc:  # noqa: BLE001
                    log.warning("  EOD P&L record(%s) failed: %s", sym, exc)
            except AlpacaError as exc:
                log.warning("  EOD close(%s) failed: %s", sym, exc)
        # Day-trader option spreads (DTE ≤ options_max_dte): close them
        # before market close. Legacy 5+ DTE spreads ride their own exits.
        if self.cfg.options_eod_flatten and self.options is not None:
            self._eod_flatten_day_trader_options()

        log.info("EOD FLATTEN done: %d positions closed, realized P&L $%.2f",
                 closed_count, total_pnl)
        try:
            from ..notifications.realtime import TradeAlert, notify_trade
            notify_trade(TradeAlert(
                kind="PNL_DAY", symbol="ALL",
                pnl_usd=total_pnl,
                extra=f"EOD flatten — {closed_count} positions closed",
            ))
        except Exception:  # noqa: BLE001
            pass

    def _eod_flatten_day_trader_options(self) -> None:
        """Close all day-trader option spreads (DTE ≤ options_max_dte).

        Walks bot_option_spreads where status='filled' AND closed_at IS NULL
        AND expiration_date is within the day-trader DTE window. Submits
        a closing multi-leg order for each. Legacy long-DTE spreads
        (carried from earlier sessions) are left to their own exit logic.
        """
        try:
            from datetime import datetime as _dt, date as _date
            today = _date.today()
            cutoff_date = (today + timedelta(days=self.cfg.options_max_dte))
            with get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT id, underlying, contracts, entry_debit_usd, expiration_date
                    FROM bot_option_spreads
                    WHERE status='filled' AND closed_at IS NULL
                      AND expiration_date <= ?
                    """,
                    (cutoff_date.isoformat(),),
                ).fetchall()
                # For each, fetch legs
                leg_rows: dict[int, list[dict]] = {}
                for r in rows:
                    legs = conn.execute(
                        "SELECT contract_symbol, role, side FROM bot_option_legs "
                        "WHERE spread_id=?",
                        (r["id"],),
                    ).fetchall()
                    leg_rows[r["id"]] = [dict(L) for L in legs]
            if not rows:
                log.info("  EOD options flatten: 0 day-trader spreads to close")
                return
            from .options.alpaca_options import OptionLeg
            for sp in rows:
                sp_d = dict(sp)
                legs = leg_rows.get(sp_d["id"], [])
                if not legs:
                    continue
                close_legs = [
                    OptionLeg(
                        symbol=L["contract_symbol"],
                        side=("sell" if L["side"] == "buy" else "buy"),
                        position_intent=("sell_to_close" if L["side"] == "buy"
                                         else "buy_to_close"),
                        ratio_qty=1,
                    )
                    for L in legs
                ]
                try:
                    import time as _t_eod_opt
                    self.options.submit_multi_leg(
                        legs=close_legs, qty=int(sp_d["contracts"]),
                        limit_price=None,  # market close at EOD
                        client_order_id=f"mr-eod-opt-{sp_d['id']}-{_t_eod_opt.time_ns()}",
                    )
                    with get_connection() as conn:
                        conn.execute(
                            "UPDATE bot_option_spreads SET closed_at=?, "
                            "exit_reason='eod_flatten_day_trader' WHERE id=?",
                            (utc_now(), sp_d["id"]),
                        )
                    log.info("  EOD closed day-trader OPT spread %s (exp %s)",
                             sp_d["underlying"], sp_d["expiration_date"])
                except AlpacaError as exc:
                    log.warning("  EOD options close failed for %s: %s",
                                sp_d["underlying"], exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("_eod_flatten_day_trader_options error: %s", exc)

    def _learning_multiplier(self, ticker: str,
                              event_type: Optional[str]) -> tuple[float, str]:
        """Adaptive sizing multiplier from rolling 30-day win rate.

        Computes win rate for (ticker, event_type) over the last 30 days
        across both bot_orders (stocks/crypto) and bot_option_spreads
        (options). Below 3 historical trades: returns 1.0× (baseline).
        Above 3: scales 0.5×–1.6× based on win rate band.
        """
        if not event_type:
            return 1.0, ""
        try:
            with get_connection() as conn:
                # Stocks + crypto path: join bot_orders → bot_decisions → signal_scores
                stock_row = conn.execute(
                    """
                    SELECT
                      COUNT(*) AS n,
                      SUM(CASE WHEN bo.realized_pnl_usd > 0 THEN 1 ELSE 0 END) AS wins
                    FROM bot_orders bo
                    JOIN bot_decisions bd ON bd.alpaca_order_id = bo.alpaca_order_id
                    JOIN signal_scores ss ON ss.id = bd.score_id
                    WHERE UPPER(bo.ticker) IN (?, ?)
                      AND ss.event_type = ?
                      AND bo.realized_pnl_usd IS NOT NULL
                      AND bo.filled_at > datetime('now','-30 days')
                    """,
                    (ticker.upper(),
                     ticker.replace("/", "").upper(),  # crypto BTC/USD vs BTCUSD
                     event_type),
                ).fetchone()
                # Options path: bot_option_spreads → bot_option_decisions → signal_scores
                opt_row = conn.execute(
                    """
                    SELECT
                      COUNT(*) AS n,
                      SUM(CASE WHEN sp.realized_pnl_usd > 0 THEN 1 ELSE 0 END) AS wins
                    FROM bot_option_spreads sp
                    JOIN bot_option_decisions od ON od.id = sp.decision_id
                    JOIN signal_scores ss ON ss.id = od.score_id
                    WHERE UPPER(sp.underlying) = ?
                      AND ss.event_type = ?
                      AND sp.realized_pnl_usd IS NOT NULL
                      AND sp.closed_at > datetime('now','-30 days')
                    """,
                    (ticker.upper(), event_type),
                ).fetchone()
            n = int((stock_row[0] or 0) + (opt_row[0] or 0))
            wins = int((stock_row[1] or 0) + (opt_row[1] or 0))
            if n < 3:
                return 1.0, ""
            wr = wins / n
            if wr < 0.30:
                return 0.5, f"learn_wr={wr:.0%}_n={n}_LOW"
            if wr < 0.60:
                return 1.0, f"learn_wr={wr:.0%}_n={n}_AVG"
            if wr < 0.80:
                return 1.3, f"learn_wr={wr:.0%}_n={n}_GOOD"
            return 1.6, f"learn_wr={wr:.0%}_n={n}_EXCELLENT"
        except Exception as exc:  # noqa: BLE001
            log.debug("learning_multiplier query failed: %s", exc)
            return 1.0, ""

    def _pdt_day_trades_remaining(self, account) -> int:
        """Day-trades available before PDT lockout. Returns 999 if account
        is above the PDT threshold ($25k) or PDT enforcement is off."""
        if not self.cfg.pdt_enforce:
            return 999
        if self._effective_equity(account) >= self.cfg.pdt_account_equity_threshold:
            return 999
        # Alpaca tracks 5-day rolling day trade count
        used = int(getattr(account, "daytrade_count", 0) or 0)
        cap = self.cfg.pdt_day_trade_limit_per_5d - self.cfg.pdt_safety_margin
        return max(cap - used, 0)

    def _pdt_blocked(self, symbol: str, account) -> bool:
        """True if opening a stock position in ``symbol`` would risk PDT lockout.
        Crypto is exempt (no PDT rule). Options held overnight are exempt."""
        if "/" in symbol:  # crypto
            return False
        return self._pdt_day_trades_remaining(account) <= 0

    def _pick_direction(self, p: float, sentiment: Optional[float],
                        signal_source: Optional[str] = None,
                        symbol: Optional[str] = None,
                        composite: Optional[float] = None,
                        event_type: Optional[str] = None,
                        factual: Optional[int] = None) -> Optional[str]:
        """Direction picker. Three paths:

        1. PRICE-ACTION: sentiment sign is direction (PA scanner sets it).
        2. NEWS-QUALITY BYPASS: composite≥7.5 + factual + high-confidence
           event_type → use sentiment sign as direction (same as PA logic).
           Captures earnings beats / M&A / FDA moves where model_p is
           uninformative (always ~0.42 on news).
        3. ML/NEWS DEFAULT: require extreme calibrated probability.

        Crypto (symbol contains '/') bypasses macro regime direction veto."""
        regime = self._current_regime
        is_crypto = symbol is not None and "/" in symbol

        # PRICE-ACTION PATH
        is_pa = signal_source and signal_source.startswith("price_action_")
        if is_pa and sentiment is not None:
            s = float(sentiment)
            if s >= 0.3:
                if regime is not None and not regime.allow_longs and not is_crypto:
                    return None
                return "buy"
            if s <= -0.3:
                if regime is not None and not regime.allow_shorts and not is_crypto:
                    return None
                return "sell"
            return None

        # NEWS-QUALITY BYPASS PATH — matches SQL news_bypass_clause exactly.
        # Tiered thresholds: SEC events bypass at 6.0, news at 7.0, analyst at 7.5.
        SEC_HIGH_ALPHA = {
            "m_a_announcement", "m_a_confirmed", "activist_position",
            "fda_approval", "fda_rejection", "insider_buying_cluster",
            "short_squeeze_setup", "spinoff_announcement",
        }
        NEWS_MEDIUM_ALPHA = {
            "earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut",
            "buyback_announcement", "contract_win_major", "partnership_major",
        }
        NEWS_LOWER_ALPHA = {
            "analyst_upgrade", "analyst_downgrade", "dividend_cut",
        }
        news_bypass_qualifies = (
            factual == 1
            and sentiment is not None and abs(float(sentiment)) >= 0.5
            and composite is not None
            and (
                (composite >= 6.0 and event_type in SEC_HIGH_ALPHA)
                or (composite >= 7.0 and event_type in NEWS_MEDIUM_ALPHA)
                or (composite >= 7.5 and event_type in NEWS_LOWER_ALPHA)
            )
        )
        if news_bypass_qualifies:
            s = float(sentiment)
            if s > 0:
                if regime is not None and not regime.allow_longs and not is_crypto:
                    return None
                return "buy"
            if s < 0:
                if regime is not None and not regime.allow_shorts and not is_crypto:
                    return None
                return "sell"

        # ML/NEWS DEFAULT PATH: require extreme calibrated probability.
        if p is None:
            return None
        if p >= self.cfg.direction_p_buy_min:
            if sentiment is not None and float(sentiment) < -0.4:
                return None
            if regime is not None and not regime.allow_longs and not is_crypto:
                return None
            return "buy"
        if p <= self.cfg.direction_p_sell_max:
            if sentiment is not None and float(sentiment) > 0.4:
                return None
            if regime is not None and not regime.allow_shorts and not is_crypto:
                return None
            return "sell"
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_ticker(raw: str) -> Optional[str]:
    """Normalize a DB ticker shape to Alpaca's expected form.

    - ``AAPL_US_EQ`` → ``AAPL`` (T212-style suffix)
    - ``BTCUSD`` → ``BTC/USD`` (crypto shorthand)
    - ``BTC/USD`` → ``BTC/USD`` (already correct — DO NOT double-slash!)
    """
    if not raw:
        return None
    t = raw.strip().upper()
    # Already formatted as crypto pair — pass through unchanged
    if "/" in t:
        return t
    # T212-style suffix path: AAPL_US_EQ → AAPL
    if "_" in t:
        parts = t.split("_")
        if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
            t = "_".join(parts[:-2])
        else:
            t = parts[0]
    # Crypto shorthand without slash: BTCUSD → BTC/USD
    if t.endswith("USD") and len(t) in (6, 7) and t not in {"PYUSD"}:
        return f"{t[:-3]}/USD"
    return t


def _us_eastern_date():
    """Return today's date in US/Eastern (NYSE) — for daily-PnL keying.

    Uses ``zoneinfo`` (stdlib) which respects DST transitions. The previous
    hardcoded UTC-5 offset broke twice a year and would silently misattribute
    daily P&L across the spring-forward / fall-back boundary.
    """
    try:
        from zoneinfo import ZoneInfo  # stdlib since Python 3.9
        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:  # noqa: BLE001 — fallback path if tzdata missing
        return (datetime.now(timezone.utc) - timedelta(hours=5)).date()
