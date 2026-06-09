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
import os
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


# ── News-catalyst bypass event types, by alpha tier (SINGLE SOURCE OF TRUTH) ──
# These MUST be tokens the LLM classifier actually emits (see llm/prompt.py) — a
# mismatch silently disables a whole catalyst class (the 2026-06-05 audit bug,
# where 'buyback_announcement'/'contract_win_major'/etc. never matched the real
# 'buyback'/'contract_award'). Consumed by: the candidate SQL filter, the stock +
# options gates, and _pick_direction. EDIT HERE ONLY — do not re-inline copies.
SEC_HIGH_ALPHA_EVENTS = (
    "m_a_announcement", "activist_position", "fda_approval", "fda_rejection",
    "insider_buy", "stock_split", "short_seller_report", "clinical_trial_result",
)
NEWS_MEDIUM_ALPHA_EVENTS = (
    "earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut",
    "buyback", "contract_award", "product_launch",
)
NEWS_LOWER_ALPHA_EVENTS = (
    "analyst_upgrade", "analyst_downgrade", "dividend",
)


def _sql_event_in(events: tuple) -> str:
    """Render event types as a SQL IN-list literal: 'a','b','c'."""
    return ",".join("'" + e + "'" for e in events)


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

    # Hard-block list of signal SOURCES excluded from TRADING (prefix-matched
    # via rs.source LIKE 'source%'). 2026-05-31 audit found stocktwits_trending
    # at -1.96% avg return (t=-16.5) — noise, not edge — so it stays blocked.
    # alpaca_news (the market-wide catalyst firehose) IS allowed to trade as of
    # 2026-06-04, but only via the min_stock_price liquidity floor below, so the
    # bot auto-trades catalysts it can realistically fill and skips the sub-$
    # micro-cap pump zone. Thin names are still scored + Telegram-alerted.
    blocked_sources: tuple[str, ...] = ("stocktwits",)

    # Liquidity floor for STOCK trades (USD): skip names below this price, where
    # fills are unreliable and paper P&L is fiction. The main guard now that the
    # market-wide news firehose can surface thin micro-caps. Crypto is exempt.
    min_stock_price: float = 5.0

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
            min_stock_price=_f("LIVE_MIN_STOCK_PRICE", 5.0),
            eod_flatten_minutes_before_close=_i("LIVE_EOD_FLATTEN_MIN", 5),
            pdt_enforce=_b("LIVE_PDT_ENFORCE", True),
            pdt_day_trade_limit_per_5d=_i("LIVE_PDT_LIMIT", 3),
            pdt_safety_margin=_i("LIVE_PDT_SAFETY_MARGIN", 1),
            override_equity_usd=_f("LIVE_OVERRIDE_EQUITY_USD", 0.0),
            use_macro_regime=_b("LIVE_USE_MACRO_REGIME", True),
            daily_profit_take_usd=_f("LIVE_DAILY_TP_USD", 0.0),
            daily_tp_arm_at_usd=_f("LIVE_DAILY_TP_ARM_USD", 190.0),
            daily_tp_giveback_usd=_f("LIVE_DAILY_TP_GIVEBACK_USD", 40.0),
            blocked_sources=tuple(
                s.strip() for s in os.getenv(
                    "LIVE_BLOCKED_SOURCES", "stocktwits"
                ).split(",")
                if s.strip()
            ),
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
        # Silent-halt alerting: map global-halt rule -> Eastern date last alerted,
        # so a persistent halt (loop runs every 30s) alerts ONCE per rule per day
        # instead of spamming. Only account-wide halts alert (not per-trade blocks).
        self._halt_alerted: dict[str, str] = {}
        # SELECTIVE loss-stop halt flag (Eastern/local date). SEPARATE from
        # _daily_tp_fired_on so the loss-stop can keep winners open without the
        # profit-take retry flattening them. Reset on day rollover in run_once.
        self._daily_loss_halt_on = None
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

    # Account-wide halts that silently stop ALL trading. Per-trade blocks
    # (per-ticker cap, min_p, sector, gross) are NOT in this set — they're
    # normal and would spam. Only these three mean "the bot has stopped".
    _GLOBAL_HALT_RULES = {"emergency_stop", "daily_loss_cap", "monthly_drawdown"}

    def _maybe_alert_halt(self, blocking_rule: Optional[str], reason: str) -> None:
        """Telegram alert the FIRST time an account-wide halt blocks a trade
        today. Without this, the bot can sit halted (loss cap hit, drawdown,
        emergency stop) all day and the user never knows trading stopped."""
        if not blocking_rule or blocking_rule not in self._GLOBAL_HALT_RULES:
            return
        today = _us_eastern_date().isoformat()
        if self._halt_alerted.get(blocking_rule) == today:
            return  # already alerted for this rule today
        self._halt_alerted[blocking_rule] = today
        labels = {
            "emergency_stop": "🛑 EMERGENCY STOP",
            "daily_loss_cap": "🛑 DAILY LOSS CAP HIT",
            "monthly_drawdown": "🛑 MONTHLY DRAWDOWN HALT",
        }
        title = labels.get(blocking_rule, f"🛑 HALT [{blocking_rule}]")
        body = (
            f"<b>{title}</b>\n"
            f"All trading is now halted for the rest of today.\n"
            f"<i>{reason}</i>\n"
            f"Existing positions keep their SL/TP. The bot resumes "
            f"automatically next session unless the cap is still breached."
        )
        try:
            from ..notifications.realtime import _send_telegram_simple
            _send_telegram_simple(body)
            log.warning("[HALT ALERT] %s — %s", blocking_rule, reason)
        except Exception as exc:  # noqa: BLE001
            log.error("[HALT ALERT] failed to send: %s", exc)

    def _add_pending_exposure(self, symbol: str, notional: float, is_crypto: bool) -> None:
        """Record a just-submitted order's notional so later candidates in the
        SAME loop iteration see it in their risk checks (the positions snapshot
        is loop-start-stale). Fail-safe: no-ops if the per-loop dict is absent."""
        p = getattr(self, "_loop_pending", None)
        if p is None:
            return
        try:
            n = float(notional or 0)
            p["gross"] = p.get("gross", 0.0) + n
            if is_crypto:
                p["crypto"] = p.get("crypto", 0.0) + n
            key = symbol.replace("/", "").upper()
            p["ticker"][key] = p["ticker"].get(key, 0.0) + n
        except Exception:  # noqa: BLE001 — never let bookkeeping break a submit
            pass

    def _load_daily_tp_state(self) -> None:
        """Restore _daily_tp_fired_on / peak from DB so restarts don't reopen
        after we've already locked the day. Uses bot_daily_pnl.tp_fired column
        (added via migration below)."""
        from datetime import date as _date
        try:
            with get_connection() as conn:
                # Ensure columns exist (idempotent — silent on already-exist)
                for col, ctype in [("tp_fired", "INTEGER"), ("tp_peak_usd", "REAL"),
                                   ("loss_halt_fired", "INTEGER")]:
                    try:
                        conn.execute(f"ALTER TABLE bot_daily_pnl ADD COLUMN {col} {ctype}")
                    except Exception:  # noqa: BLE001
                        pass
                row = conn.execute(
                    "SELECT tp_fired, tp_peak_usd, loss_halt_fired FROM bot_daily_pnl WHERE trading_date=?",
                    (_us_eastern_date().isoformat(),),
                ).fetchone()
            if row and row[0]:
                self._daily_tp_fired_on = _date.today()
                self._daily_tp_armed_on = _date.today()
                self._daily_tp_peak = row[1] or 0.0
                log.warning("🎯 Loaded TP state from DB: ALREADY FIRED today, peak was $%.2f",
                            self._daily_tp_peak or 0)
            # Restore the loss-halt flag too, so a restart during a halted day
            # keeps new entries blocked (don't re-expose the account).
            if row and len(row) > 2 and row[2]:
                self._daily_loss_halt_on = _date.today()
                log.warning("🛑 Loaded loss-halt state from DB: ALREADY HALTED today "
                            "— new entries stay blocked until tomorrow.")
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

    def _persist_loss_halt_state(self) -> None:
        """Persist the daily loss-halt flag so a restart during a halted day
        still blocks new entries (mirrors _persist_daily_tp_state). The
        equity-delta daily cap (risk Rule 2) is the backstop; this is the fast
        path so we don't re-expose for a loop or two until reconcile re-syncs."""
        try:
            with get_connection() as conn:
                conn.execute(
                    """INSERT INTO bot_daily_pnl
                       (trading_date, realized_pnl_usd, trades_count, wins, losses,
                        largest_win, largest_loss, updated_at, loss_halt_fired)
                       VALUES (?, 0, 0, 0, 0, 0, 0, ?, 1)
                       ON CONFLICT(trading_date) DO UPDATE SET
                         loss_halt_fired = 1,
                         updated_at = excluded.updated_at""",
                    (_us_eastern_date().isoformat(), utc_now()),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not persist loss-halt state: %s", exc)

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
                                # OPTIONS_OPENED — entry confirmation,
                                # no P/L yet. The OPTIONS_FILLED kind is
                                # reserved for spread EXITS with realized P/L.
                                notify_trade(TradeAlert(
                                    kind="OPTIONS_OPENED",
                                    symbol=sp["underlying"],
                                    direction=sp["direction"],
                                    qty=sp["contracts"],
                                    notional_usd=sp["total_debit_usd"],
                                    extra=(
                                        f"spread filled · debit ${sp['total_debit_usd']:.0f} · "
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

        # Step 2-ter: backfill realized P&L for any CLOSED option spread that
        # never got it booked — primarily spreads flattened by the daily
        # loss-stop (_fire_daily_flatten closes legs but historically never
        # touched bot_option_spreads, leaving realized_pnl_usd NULL → the
        # learning loop is blind to those outcomes). Reads the REAL close
        # fills from Alpaca (idempotent; guarded on realized_pnl_usd IS NULL).
        try:
            self._reconcile_option_spread_pnl()
        except Exception as exc:  # noqa: BLE001
            log.exception("Option spread P&L reconcile failed: %s", exc)

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
        # Track symbols closed this loop so the flip check below doesn't double-close
        # (and double-book P&L) a position the SL/TP poller just closed.
        self._closed_this_loop = set()
        try:
            self._poll_crypto_exits(positions)
        except Exception as exc:  # noqa: BLE001
            log.exception("Crypto exit poll failed: %s", exc)

        # Step 2b.65: reconcile EXTENDED-HOURS entry FILLS — only count orders that
        # actually filled. Unfilled limit orders are retried (if high-priority) or
        # marked 'unfilled', so the bot never treats an unfilled order as a position.
        try:
            self._reconcile_stock_fills(positions)
        except Exception as exc:  # noqa: BLE001
            log.exception("Stock fill reconcile failed: %s", exc)

        # Step 2b.7: poll & exit EXTENDED-HOURS stock positions at SL/TP. These were
        # entered as simple limit orders (no server-side bracket), so this poller is
        # their only stop/TP. Regular-hours bracket stocks are untouched (filtered by
        # outcome_detail='stock_polled').
        try:
            self._poll_stock_exits(positions)
        except Exception as exc:  # noqa: BLE001
            log.exception("Stock (ext-hours) exit poll failed: %s", exc)

        # Step 2b.6: crypto reversal flip — close held crypto when strong
        # opposite-direction price-action signal arrives. Frees the symbol so
        # next iteration can re-enter the new direction.
        try:
            self._check_crypto_flips(positions)
        except Exception as exc:  # noqa: BLE001
            log.exception("Crypto flip check failed: %s", exc)

        # Day rollover: clear the per-day TP/loss notification flag so the first
        # fire of a NEW day actually alerts. The flag is set on fire and never
        # otherwise reset, and the trader is a long-lived process — without this,
        # day-2+ loss-stop/TP Telegram alerts are silently suppressed.
        from datetime import date as _roll_date
        if getattr(self, "_daily_tp_fired_on", None) != _roll_date.today():
            self._daily_tp_notified = False
        # Reset the SELECTIVE loss-stop halt on day rollover too. It's a separate
        # flag (see _daily_loss_stop_if_due) and, like _daily_tp_fired_on, is set
        # on trigger and never otherwise cleared — without this, a loss-halt on
        # one day would silently block all new entries forever after.
        if getattr(self, "_daily_loss_halt_on", None) != _roll_date.today():
            self._daily_loss_halt_on = None

        # Step 2c: EOD flatten — close stock positions before market close
        try:
            self._eod_flatten_if_due(positions)
        except Exception as exc:  # noqa: BLE001
            log.exception("EOD flatten failed: %s", exc)

        # Step 2c2: Daily LOSS stop — hard intraday kill-switch. The risk gate's
        # daily_loss_cap only blocks NEW trades and counts realized P&L only, so
        # an underwater book can blow past the cap before EOD (2026-05-29 lost
        # -$1,953 ≈ 4x the $500 cap). Flatten + halt the moment intraday
        # (realized + unrealized) breaches the cap.
        try:
            if self._daily_loss_stop_if_due(account, positions):
                log.error("Daily LOSS stop fired. Flattened + halted for the day.")
                return
        except Exception as exc:  # noqa: BLE001
            log.exception("Daily loss-stop check failed: %s", exc)

        # Step 2d: Daily profit-take — if intraday P&L exceeds cap, flatten + halt
        try:
            if self._daily_profit_take_if_due(account, positions):
                log.warning("Daily profit-take fired. Skipping rest of iteration.")
                return
        except Exception as exc:  # noqa: BLE001
            log.exception("Daily profit-take check failed: %s", exc)

        # Step 3: market clock — skip stock trading when closed (unless after-hours allowed)
        market_open = self.alpaca.is_market_open()

        # Step 3a: WEEKEND→WEEKDAY crypto trim. When the market is open, crypto
        # reverts to its weekday ceiling so stocks+options have full budget.
        # Trim any weekend-built crypto excess back down once per open.
        try:
            if market_open:
                self._trim_crypto_to_weekday_cap(positions)
        except Exception as exc:  # noqa: BLE001
            log.exception("Crypto weekday-trim failed: %s", exc)

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

        # If daily profit-take fired OR the selective loss-stop halted today, do
        # not open new trades. Two SEPARATE flags by design: the loss-stop keeps
        # winners open (so it must NOT set _daily_tp_fired_on, which would make
        # the profit-take retry flatten them) but still halts new entries here.
        from datetime import date as _date
        _today = _date.today()
        if getattr(self, "_daily_tp_fired_on", None) == _today:
            log.debug("Daily TP fired earlier today — blocking new opens")
            return
        if getattr(self, "_daily_loss_halt_on", None) == _today:
            log.debug("Daily LOSS stop halted earlier today — blocking new opens "
                      "(kept winners ride their own stops)")
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

        # Within-loop stacking guard: reset per-iteration pending exposure.
        # The `positions` snapshot is taken once at loop start, so without this
        # each candidate is blind to positions opened EARLIER in the same loop
        # and the crypto/gross/per-ticker caps can be breached in one iteration
        # (7 crypto = $3,100 vs the $2,000 cap on 2026-05-28). Each successful
        # submit adds to this; later candidates' risk checks include it.
        self._loop_pending = {"gross": 0.0, "crypto": 0.0, "ticker": {}}

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
                    # ENTRY fill confirmation — no P/L yet. Skip the
                    # notification: bot's PA/options PLACED messages
                    # already announce intent. We send Telegram ONLY on
                    # exits (where P/L is known) so the user's phone
                    # isn't double-pinged for the same trade.
                    pass

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
        # Telegram on every exit fill (silent fail)
        try:
            from ..notifications.realtime import TradeAlert, notify_trade
            headline = self._headline_for_order(
                alpaca_order_id=parent.get("alpaca_order_id"))
            extra = f"{exit_reason} · ${entry:.2f}→${exit_:.2f} ({pct*100:+.1f}%)"
            if headline:
                extra = f"{extra}\n💬 {headline}"
            notify_trade(TradeAlert(
                kind="FILLED", symbol=row["ticker"], pnl_usd=pnl,
                notional_usd=float(abs(qty) * entry),
                qty=float(abs(qty)),
                direction=parent["direction"],
                extra=extra,
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

        # Drop negative-EV sources entirely (prefix match). This is a TOP-LEVEL
        # exclusion — applied no matter which bypass arm a signal would hit —
        # so a blocked source can never reach a trade decision. NULL sources
        # are kept (only an explicit prefix match is excluded).
        if self.cfg.blocked_sources:
            src_clauses = " AND ".join(
                ["(rs.source IS NULL OR rs.source NOT LIKE ?)"] * len(self.cfg.blocked_sources)
            )
            blocked_source_clause = f"AND ({src_clauses})"
            blocked_source_params = tuple(f"{s}%" for s in self.cfg.blocked_sources)
        else:
            blocked_source_clause = ""
            blocked_source_params = ()

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
        # High-conviction only: the 2026-05-28 winning day's price-action
        # trades were composite 6.5-8.5 with |sentiment| 0.6. Raised the
        # sentiment floor 0.3 -> 0.5 so only strong-conviction breakouts fire
        # (user: "rather 3 high-conviction trades than a million weak ones").
        pa_bypass_clause = (
            "OR (rs.source LIKE 'price_action_%' "
            "    AND ss.composite_score >= 6.5 "
            "    AND ABS(COALESCE(ss.sentiment, 0)) >= 0.5)"
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
            f"    (ss.composite_score >= 6.0 AND ss.event_type IN ({_sql_event_in(SEC_HIGH_ALPHA_EVENTS)}))"
            f"    OR (ss.composite_score >= 7.0 AND ss.event_type IN ({_sql_event_in(NEWS_MEDIUM_ALPHA_EVENTS)}))"
            f"    OR (ss.composite_score >= 7.5 AND ss.event_type IN ({_sql_event_in(NEWS_LOWER_ALPHA_EVENTS)}))"
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
                  {blocked_source_clause}
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
                    *blocked_source_params,
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
    @staticmethod
    def _distinct_position_count(positions) -> int:
        """Count distinct underlying names open now (an option spread's legs
        collapse to one underlying). Feeds the risk manager's concentration cap
        so the book can't pile into many correlated bets at once."""
        import re
        names = set()
        for p in (positions or []):
            m = re.match(r"^[A-Z]+", (getattr(p, "symbol", "") or "").upper())
            if m:
                names.add(m.group())
        return len(names)

    def _is_news_catalyst(self, cand: dict) -> bool:
        """True if this candidate qualifies as a NEWS CATALYST (the news-bypass
        tiers, using the shared event-tier constants). Options are RESERVED for
        catalysts as of 2026-06-05 — price-action / model-only signals route to
        stock instead, because PA-momentum-into-options drew the day's loss
        (bullish short-dated spreads taken at the open, market reversed)."""
        try:
            if int(cand.get("factual") or 0) != 1:
                return False
            if abs(float(cand.get("sentiment") or 0)) < 0.5:
                return False
            ev = cand.get("event_type")
            comp = float(cand.get("composite_score") or 0)
            return (
                (comp >= 6.0 and ev in SEC_HIGH_ALPHA_EVENTS)
                or (comp >= 7.0 and ev in NEWS_MEDIUM_ALPHA_EVENTS)
                or (comp >= 7.5 and ev in NEWS_LOWER_ALPHA_EVENTS)
            )
        except Exception:  # noqa: BLE001
            return False

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

        # Options are RESERVED for NEWS CATALYSTS (2026-06-05): only a qualifying
        # catalyst routes to the options spread path; price-action / model-only
        # signals trade as stock. (PA-momentum-into-options drew the day's loss.)
        # A catalyst whose spread is unbuildable falls back to stock.
        is_crypto_sym = "/" in symbol

        # CATALYST-ONLY mode (LIVE_CATALYST_ONLY=1): the news-catalyst radar IS the
        # edge. Reject price-action-scanner and model-only entries so the bot acts
        # ONLY on qualifying news catalysts. Crypto trades its own path, untouched.
        if (os.getenv("LIVE_CATALYST_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
                and not is_crypto_sym
                and not self._is_news_catalyst(cand)):
            self._persist_decision(
                cand, gate_passed=False,
                gate_reason="catalyst_only: not a qualifying news catalyst",
                risk_passed=False, risk_reason="n/a",
                outcome="gate_blocked", outcome_detail="catalyst_only",
            )
            return

        if (self.cfg.options_enabled
                and self.options is not None
                and not is_crypto_sym
                and self._is_news_catalyst(cand)):
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
        # News-catalyst bypass — SAME logic as the stock path. Without this,
        # catalyst-driven OPTIONS trades carry model_p ~0.5 and die at the risk
        # manager's min_p rule (2026-06-05 audit: stocks had the fix, options didn't).
        _ev_o = cand.get("event_type")
        _comp_o = float(cand.get("composite_score") or 0)
        is_news_bypass = (
            int(cand.get("factual") or 0) == 1
            and abs(float(cand.get("sentiment") or 0)) >= 0.5
            and (
                (_comp_o >= 6.0 and _ev_o in SEC_HIGH_ALPHA_EVENTS)
                or (_comp_o >= 7.0 and _ev_o in NEWS_MEDIUM_ALPHA_EVENTS)
                or (_comp_o >= 7.5 and _ev_o in NEWS_LOWER_ALPHA_EVENTS)
            )
        )

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
        # PA-driven spread is "unbuildable", and options NEVER fire. News
        # catalysts get the same treatment (model_p ~0.5; the edge is the event).
        builder_p = (0.70 if direction == "buy" else 0.30) if (is_pa_signal or is_news_bypass) else p
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

        # Risk manager — reuse same caps as stock path. PA + news-catalyst
        # signals get the same synthetic-p substitution as the stock path.
        risk_p = (0.70 if direction == "buy" else 0.30) if (is_pa_signal or is_news_bypass) else p
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
        # Stacking guard: include gross from orders already submitted this loop.
        _pend = getattr(self, "_loop_pending", None)
        if _pend and alpaca_gross is not None:
            alpaca_gross += _pend.get("gross", 0.0)
        decision = self.risk.evaluate(
            proposal,
            account_equity_usd=self._effective_equity(account),
            current_gross_usd=alpaca_gross,
            current_notional_usd=sizing.total_debit_usd,
            current_position_count=(self._distinct_position_count(positions)
                                    if positions is not None else None),
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
            self._maybe_alert_halt(decision.blocking_rule, decision.reason)
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
        # Stacking guard: this spread's debit isn't 'filled' in the DB yet, so
        # later candidates this loop would miss it on the gross cap. Record it.
        self._add_pending_exposure(symbol, sizing.total_debit_usd, False)
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

        # CRYPTO-NO-SHORT GUARD: Alpaca spot crypto cannot be shorted. A
        # bearish PA signal produces direction='sell', which bypasses the
        # macro-regime short veto for crypto and reaches submit — where
        # Alpaca returns 403 ("cannot short"). This was burning ~32 crypto
        # signals/day into execution_failed rows (and the generic 403
        # message masked the real cause). Short-circuit BEFORE sizing/submit.
        if "/" in symbol and direction == "sell":
            self._persist_decision(
                cand, gate_passed=False, gate_reason="crypto_no_short",
                risk_passed=False, risk_reason="alpaca_spot_crypto_no_short",
                outcome="gate_blocked", outcome_detail="crypto_short_not_supported",
            )
            return

        # Market hours
        is_crypto = symbol in self.cfg.crypto_tickers or "/" in symbol
        if not is_crypto and not market_open and not self.cfg.allow_after_hours:
            self._persist_decision(
                cand, gate_passed=True, gate_reason="market_closed",
                risk_passed=False, risk_reason="market_closed",
                outcome="closed_market",
            )
            return

        # WEEKEND CRYPTO CONVICTION GATE: when the US market is closed and
        # crypto gets the higher capital ceiling, only fire on the STRONGEST
        # setups (composite >= 7.5 AND |sentiment| >= 0.5) instead of the
        # weekday 6.5/0.3 bar. More capital deployed → demand more conviction.
        if is_crypto and not market_open:
            _comp = float(cand.get("composite_score") or 0.0)
            _sent = abs(float(cand.get("sentiment") or 0.0))
            if _comp < 7.5 or _sent < 0.5:
                self._persist_decision(
                    cand, gate_passed=False,
                    gate_reason=f"weekend_low_conviction (comp={_comp:.1f}, |sent|={_sent:.1f})",
                    risk_passed=False, risk_reason="weekend_conviction_bar",
                    outcome="gate_blocked",
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
        # News-quality CATALYST bypass — mirrors _pick_direction's news_bypass.
        # A high-conviction catalyst (factual + |sentiment|>=0.5 + composite tier
        # + high-alpha event type) trades on the SIGNAL, not model_p (~0.5 and
        # uninformative on news). Without this the direction is picked correctly
        # but gate_decide(p=0.5) blocks it at "p < min" — the exact reason
        # earnings/M&A/FDA catalysts NEVER traded despite scoring 'strong'.
        _ev = cand.get("event_type")
        _comp_b = float(cand.get("composite_score") or 0)
        is_news_bypass = (
            int(cand.get("factual") or 0) == 1
            and abs(float(cand.get("sentiment") or 0)) >= 0.5
            and (
                (_comp_b >= 6.0 and _ev in SEC_HIGH_ALPHA_EVENTS)
                or (_comp_b >= 7.0 and _ev in NEWS_MEDIUM_ALPHA_EVENTS)
                or (_comp_b >= 7.5 and _ev in NEWS_LOWER_ALPHA_EVENTS)
            )
        )
        if is_pa_signal or is_news_bypass:
            from ..ml.selective_gate import GateDecision
            gate = GateDecision(
                trade=True,
                reason="price_action_bypass" if is_pa_signal else "news_catalyst_bypass",
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
        # EARNINGS PROXIMITY MULT — pre-earnings drift is documented.
        # Boost sizing 1.3× if ticker reports earnings within 2 days.
        # Skip crypto (no earnings).
        earnings_mult, earnings_reason = 1.0, ""
        if "/" not in symbol:
            try:
                from ..ingestors.earnings_calendar import days_until_earnings
                dte_to_earnings = days_until_earnings(symbol)
                if dte_to_earnings is not None and dte_to_earnings <= 2:
                    earnings_mult = 1.3
                    earnings_reason = f"earnings_in_{dte_to_earnings}d"
            except Exception:  # noqa: BLE001
                pass
        # Bound the multiplier stack. Four weakly-independent multipliers (confluence,
        # regime, learning, earnings) multiplied together can compound to ~4x — over-Kelly,
        # which defeats the fractional-Kelly variance control — or shrink to ~0.13x. Clamp
        # the product to a sane band so conviction scales size modestly without the stack
        # ever blowing past a safe ceiling. The hard per-position cap (6%) remains the final
        # ceiling; env-tunable bounds.
        _raw_mult = conf_mult * regime_mult * learn_mult * earnings_mult
        _MULT_FLOOR = float(os.getenv("LIVE_SIZE_MULT_FLOOR", "0.4"))
        _MULT_CEIL = float(os.getenv("LIVE_SIZE_MULT_CEIL", "1.5"))
        total_mult = max(_MULT_FLOOR, min(_MULT_CEIL, _raw_mult))
        eff_equity = self._effective_equity(account)
        adjusted_equity = eff_equity * total_mult
        # Crypto allows fractional qty, lower min_qty floor too
        is_crypto_sym = "/" in symbol
        # PA signals: use the synthetic high-conviction p for sizing too,
        # otherwise Kelly produces tiny sizes ($135) because PA signals'
        # model_p is centered around 0.42-0.45.
        # CRITICAL: p in the Kelly formula = probability THIS TRADE wins
        # (not probability of UP). For shorts, our confidence is 0.70 in
        # the bet (= 0.30 confidence in UP). Using 0.30 here would give
        # negative kelly_raw and block every PA short signal. Pass 0.70
        # for BOTH directions when PA-confident.
        if is_pa_signal or is_news_bypass:
            sizing_p = 0.70
        else:
            # For ML/news signals: if direction is sell, the bet wins
            # when price goes DOWN. p (model_p_5d) represents P(up), so
            # P(bet wins) = 1 - p for shorts.
            sizing_p = p if direction == "buy" else (1.0 - p)

        # SIZE-TO-FIT: compute remaining budget headroom BEFORE sizing so a
        # high-conviction signal that doesn't fit at the full 6% still gets
        # whatever the remaining crypto / gross / per-ticker budget allows.
        # No capital sits idle. Headroom = None means "no positions snapshot",
        # so fall back to unconstrained (risk gate still backstops).
        max_notional = None
        if positions is not None:
            cur_crypto = sum(abs(float(pp.market_value)) for pp in positions
                             if pp.symbol.upper().endswith("USD") and len(pp.symbol) <= 9)
            cur_stock_crypto = sum(abs(float(pp.market_value)) for pp in positions
                                   if len(pp.symbol) <= 9)
            with get_connection() as _hc:
                _r = _hc.execute("SELECT COALESCE(SUM(total_debit_usd),0) "
                                 "FROM bot_option_spreads WHERE closed_at IS NULL "
                                 "AND status='filled'").fetchone()
                cur_opt = float(_r[0]) if _r else 0.0
            cur_gross = cur_stock_crypto + cur_opt
            from ..config import CONFIG as _CFG
            headrooms = []
            # Gross budget (non-option proposals share gross minus options reserve)
            non_opt_cap = _CFG.risk_max_gross_exposure_usd - _CFG.risk_options_reserve_usd
            headrooms.append(non_opt_cap - cur_gross)
            # Crypto ceiling (weekend vs weekday)
            if is_crypto_sym:
                crypto_cap = (_CFG.risk_max_crypto_exposure_usd if market_open
                              else _CFG.risk_max_crypto_exposure_weekend_usd)
                headrooms.append(crypto_cap - cur_crypto)
            max_notional = max(min(headrooms), 0.0)

        sized = size_trade(
            direction=direction, entry_price=entry, atr=atr,
            calibrated_p=sizing_p, account_equity_usd=adjusted_equity,
            sl_atr_mult=self.cfg.stock_sl_atr_mult,
            tp_atr_mult=self.cfg.stock_tp_atr_mult,
            allow_fractional=is_crypto_sym,
            min_qty=(0.0001 if is_crypto_sym else 1.0),
            min_entry_price=(0.0 if is_crypto_sym else self.cfg.min_stock_price),
            # Hard per-ticker cap binds on TRUE equity so multipliers can't
            # inflate a position past 6% and get it rejected by the risk gate.
            true_equity_usd=eff_equity,
            # Size-to-fit remaining budget so no capital sits idle.
            max_notional_usd=max_notional,
        )
        if conf_reasons or regime_mult != 1.0 or learn_mult != 1.0 or earnings_mult != 1.0:
            log.info(
                "[%s] sizing mults: confluence=%.2fx (%s) regime=%.2fx learn=%.2fx (%s) "
                "earnings=%.2fx (%s) → total=%.2fx",
                symbol, conf_mult, ",".join(conf_reasons) or "none",
                regime_mult, learn_mult, learn_reason or "neutral",
                earnings_mult, earnings_reason or "none", total_mult,
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

        # Risk manager (7 hard rules, fed by Alpaca state). For price-action AND
        # news-catalyst signals we substitute calibrated_p with a synthetic
        # conviction value so we don't trip the risk manager's min_p rule. These
        # names carry model_p ~0.50 (the ML model has no opinion on out-of-universe
        # catalyst tickers); their edge is the EVENT — factual news + sentiment +
        # event-type — already proven by the news-bypass gate, not model_p. The
        # other six risk rules (exposure, sector, daily-loss, …) still apply, and
        # this mirrors the sizing path, which already gives both a synthetic 0.70.
        if is_pa_signal or is_news_bypass:
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
        # Fold in same-loop pending exposure (stacking guard) so this candidate
        # sees orders already submitted earlier THIS iteration.
        _pend = getattr(self, "_loop_pending", None)
        if _pend and alpaca_gross is not None:
            alpaca_gross += _pend.get("gross", 0.0)
            if alpaca_crypto is not None:
                alpaca_crypto += _pend.get("crypto", 0.0)
            if alpaca_ticker is not None:
                alpaca_ticker += _pend.get("ticker", {}).get(
                    symbol.replace("/", "").upper(), 0.0)
        decision = self.risk.evaluate(
            proposal,
            account_equity_usd=self._effective_equity(account),
            current_gross_usd=alpaca_gross,
            current_notional_usd=sized.notional_usd,
            current_crypto_usd=alpaca_crypto,
            current_ticker_usd=alpaca_ticker,
            current_position_count=(self._distinct_position_count(positions)
                                    if positions is not None else None),
            market_open=market_open,
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
            self._maybe_alert_halt(decision.blocking_rule, decision.reason)
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

        # DOUBLE-OPEN / SAME-TICKER RE-ENTRY GUARD: an order can take time to
        # FILL and show up in get_positions(). The per-ticker cap reads filled
        # positions, so until a submit fills it's invisible to the cap — and a
        # second order on the same name slips through, stacking past the cap
        # (observed 2026-06-01: MARA bought twice 3min apart = $743 ≈ 2x the 6%
        # cap, because the 90s window expired before the first fill propagated).
        # Widened to 300s so the prior fill is reflected in the snapshot before
        # the same ticker can be re-entered, after which the per-ticker cap
        # correctly blocks further adds.
        import time as _t
        if not hasattr(self, "_recent_submits"):
            self._recent_submits = {}
        sym_upper = symbol.upper()
        last_submit = self._recent_submits.get(sym_upper, 0)
        now_ts = _t.time()
        if now_ts - last_submit < 300:
            log.warning(
                "[%s] RE-ENTRY GUARD: %.0fs since last submit (<300s) — skipping",
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
        pending_id = self._persist_decision(
            cand, gate_passed=True, gate_reason=gate.reason,
            risk_passed=True, risk_reason="ok",
            outcome="pending_submit", outcome_detail=client_order_id,
            sized=sized,
        )
        # SAFETY: never submit an order we couldn't record. A missing decision
        # row means no SL/TP tracking → an unmanaged position. If persist
        # failed (0 / None), abort this candidate rather than open a naked
        # position. (Caused 2 unmanaged crypto positions on 2026-05-30.)
        if not pending_id:
            log.error("[%s] pending_submit persist failed (no row) — ABORTING "
                      "submit to avoid an unmanaged position. score_id=%s",
                      symbol, score_id)
            return

        # Crypto branch: Alpaca does NOT support bracket orders on crypto.
        # Submit a simple market order; stop/TP enforcement is handled by
        # the crypto-position poller (the next loop iteration will close
        # at TP/SL based on quote monitoring).
        is_crypto_order = "/" in symbol
        try:
            entry_outcome = "placed"  # ext-hours entries flip to 'pending_fill' below (only 'placed' once the limit fills)
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
                ext_hours = (not market_open) and self.cfg.allow_after_hours
                if ext_hours:
                    # EXTENDED HOURS: Alpaca forbids bracket / stop / market orders
                    # outside regular hours — only simple LIMIT DAY orders with
                    # extended_hours=True. So enter with a marketable limit (priced
                    # slightly THROUGH the quote to fill in thin pre/after-hours
                    # liquidity) and manage SL/TP via _poll_stock_exits (there is no
                    # server-side bracket to protect this position — the poller is it).
                    buf = 1.003 if direction == "buy" else 0.997
                    limit_px = round(float(entry) * buf, 2)
                    simple = self.alpaca.submit_simple_order(
                        symbol=symbol, side=direction, qty=sized.qty,
                        order_type="limit", limit_price=limit_px,
                        time_in_force="day", extended_hours=True,
                        client_order_id=client_order_id,
                    )
                    placed_id = simple.id
                    placed_qty = sized.qty
                    placed_kind = "stock_polled"
                    entry_outcome = "pending_fill"  # NOT a position until the ext-hours limit actually fills
                else:
                    # RTH catalyst entry is ALSO poll-managed (plain market order) so the
                    # trailing take-profit in _poll_stock_exits applies — no fixed-TP bracket
                    # that would cap a volatile "bang". Hard SL + trailing TP live in the
                    # poller (the only stop), so the bot must stay alive (always-on).
                    simple = self.alpaca.submit_simple_order(
                        symbol=symbol, side=direction, qty=sized.qty,
                        order_type="market", time_in_force="day",
                        client_order_id=client_order_id,
                    )
                    placed_id = simple.id
                    placed_qty = sized.qty
                    placed_kind = "stock_polled"
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
                (entry_outcome, placed_kind, placed_id, cand["score_id"]),
            )
            row = conn.execute(
                "SELECT id FROM bot_decisions WHERE score_id=?",
                (cand["score_id"],),
            ).fetchone()
            decision_id = row[0] if row else None
        # Stacking guard: record this submit's notional for later candidates
        # this loop (the positions snapshot won't reflect it yet).
        self._add_pending_exposure(symbol, sized.notional_usd, is_crypto_order)
        if decision_id is not None and not is_crypto_order:
            # Poll-managed stock entries are plain orders (no server-side bracket),
            # so persist the bot_orders row from the simple order — not a 'bracket'
            # (which no longer exists here; referencing it raised NameError on every
            # stock entry and silently skipped bot_orders persistence).
            self._persist_stock_order(simple, sized=sized, direction=direction,
                                      symbol=symbol, decision_id=decision_id,
                                      order_class=placed_kind)
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
        # CRITICAL: bot_decisions.model_p + composite_score are NOT NULL.
        # PA signals (and some news signals) arrive with model_p=None before
        # the ML predict job fills it. Passing None made INSERT OR IGNORE
        # SILENTLY skip the row — so a placed crypto position got NO decision
        # row, hence NO SL/TP for _poll_crypto_exits → unmanaged position.
        # Coerce to safe defaults (0.5 = neutral, matching the rest of the
        # candidate-processing code which does the same).
        safe_model_p = cand.get("model_p")
        safe_model_p = float(safe_model_p) if safe_model_p is not None else 0.5
        safe_composite = cand.get("composite_score")
        safe_composite = float(safe_composite) if safe_composite is not None else 0.0
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
                 self._pick_direction(
                     safe_model_p, cand.get("sentiment"),
                     cand.get("signal_source"), cand.get("symbol"),
                     composite=safe_composite, event_type=cand.get("event_type"),
                     factual=cand.get("factual")) or "buy",
                 safe_model_p, safe_composite,
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
            if cur.rowcount == 0:
                # Row was IGNORED (existing score_id) — that's fine for the
                # idempotency case. But warn if we somehow lost a brand-new
                # row so this silent-skip class of bug surfaces in logs.
                existing = conn.execute(
                    "SELECT id FROM bot_decisions WHERE score_id=?",
                    (cand["score_id"],),
                ).fetchone()
                if not existing:
                    log.error("[%s] _persist_decision INSERT skipped AND no existing "
                              "row for score_id=%s — decision NOT recorded!",
                              cand.get("symbol"), cand["score_id"])
                return existing[0] if existing else 0
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

    def _persist_stock_order(self, order, *, sized: SizingResult, direction: str,
                             symbol: str, decision_id: int,
                             order_class: str = "stock_polled") -> None:
        """Persist a bot_orders row for a poll-managed stock entry — a plain order
        with no server-side bracket. Mirrors _persist_order but reads the simple
        Order directly (it has no .parent). Keeps the audit trail / reconciliation /
        exposure math (which read bot_orders) intact for stock trades."""
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
                (decision_id, getattr(order, "id", None),
                 getattr(order, "client_order_id", None), symbol, direction,
                 order_class, sized.qty, sized.entry_estimate,
                 sized.stop_loss, sized.take_profit,
                 getattr(order, "status", None), getattr(order, "filled_qty", 0),
                 getattr(order, "filled_avg_price", None),
                 getattr(order, "submitted_at", None) or utc_now()),
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
    def _book_crypto_exit_pnl(self, conn, pair: str, qty: float,
                              entry: float, pnl: float) -> None:
        """Mark the crypto position's bot_orders row CLOSED with realized P/L.

        Without this, crypto exits booked P/L to bot_daily_pnl but left the
        bot_orders row status='filled', realized_pnl_usd=NULL forever — a
        phantom 'open' position that inflated the risk manager's DB-based
        exposure math (10 phantom rows = $4,561 vs $1,374 real) and slowly
        starved trading. Matches both ticker forms (BTC/USD and BTCUSD).
        """
        try:
            norm = pair.replace("/", "").upper()
            conn.execute(
                """
                UPDATE bot_orders
                SET realized_pnl_usd = ?,
                    pnl_pct = ?,
                    exit_reason = 'crypto_exit',
                    canceled_at = ?
                WHERE UPPER(ticker) IN (?, ?)
                  AND status = 'filled'
                  AND realized_pnl_usd IS NULL
                """,
                (pnl,
                 (pnl / (abs(qty) * entry)) if (qty and entry) else 0.0,
                 utc_now(), pair.upper(), norm),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("_book_crypto_exit_pnl(%s) failed: %s", pair, exc)

    def _reconcile_stock_fills(self, positions) -> None:
        """Only count ext-hours entries that ACTUALLY FILLED, and chase unfilled
        high-priority catalysts only while it's still worth it.

        Ext-hours entries are LIMIT orders that may not fill in thin liquidity, so
        they're recorded as 'pending_fill' (NOT 'placed') — an unfilled order is
        never treated as a position. Each loop, for every pending_fill / placed
        stock_polled decision:
          - position now exists             -> promote 'pending_fill' -> 'placed'
          - order canceled/expired/rejected -> 'unfilled'
          - order still working             -> CHASE the fill (cancel + re-submit a
            fresh marketable order) ONLY while BOTH hold:
              * RELEVANT  — catalyst still fresh (age <= LIVE_STOCK_FILL_RELEVANCE_MIN
                            min); news alpha decays fast, so a stale signal is dropped.
              * PROFITABLE — price hasn't run so far there's no edge left: still
                            >= LIVE_STOCK_FILL_MIN_UPSIDE_PCT room to the take-profit.
            Else -> cancel + 'unfilled' (never chase a runaway with no profit left).
        Standard cancel-replace repricing bounded by a slippage/edge budget; a short
        per-symbol cooldown avoids hammering a stuck (e.g. halted) order every loop.
        """
        import time as _t_fill
        from datetime import datetime as _dt, timezone as _tz
        open_syms = {p.symbol.upper() for p in positions
                     if "/" not in p.symbol and len(p.symbol) <= 9}
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, score_id, ticker, direction, qty, composite_score,
                       take_profit, decided_at, alpaca_order_id, outcome
                FROM bot_decisions
                WHERE outcome IN ('pending_fill','placed')
                  AND outcome_detail='stock_polled' AND alpaca_order_id IS NOT NULL
                """
            ).fetchall()
        if not rows:
            return
        if not hasattr(self, "_stock_fill_retry_ts"):
            self._stock_fill_retry_ts = {}
        market_open = self.alpaca.is_market_open()
        RELEVANCE_MIN = float(os.getenv("LIVE_STOCK_FILL_RELEVANCE_MIN", "20"))
        MIN_UPSIDE = float(os.getenv("LIVE_STOCK_FILL_MIN_UPSIDE_PCT", "0.03"))
        RETRY_COMPOSITE = float(os.getenv("LIVE_STOCK_FILL_RETRY_COMPOSITE", "7.0"))
        RETRY_COOLDOWN_S = 45.0
        now = _dt.now(_tz.utc)
        now_ts = _t_fill.time()

        def _mark(did, outcome, detail):
            with get_connection() as conn:
                conn.execute("UPDATE bot_decisions SET outcome=?, outcome_detail=? WHERE id=?",
                             (outcome, detail, did))

        for r in rows:
            sym = r["ticker"].upper()
            score_id = r["score_id"]
            if sym in open_syms:
                if r["outcome"] == "pending_fill":
                    _mark(r["id"], "placed", "stock_polled")  # it filled — now a real position
                    self._stock_fill_retry_ts.pop(score_id, None)
                    log.info("[fill] %s filled — promoted to placed (now exit-managed)", sym)
                continue  # placed + open position = normal
            if sym in getattr(self, "_stock_exit_attempts", {}):
                continue  # an exit we fired — exit poller cleanup owns this
            try:
                o = self.alpaca.get_order(r["alpaca_order_id"])
            except AlpacaError:
                continue
            st = (o.status or "").lower()
            if st == "filled":
                continue  # position will sync next snapshot, then promote
            if st in ("canceled", "cancelled", "expired", "rejected",
                      "done_for_day", "suspended", "stopped"):
                _mark(r["id"], "unfilled", "ext_no_fill")
                self._stock_fill_retry_ts.pop(score_id, None)
                log.info("[fill-reconcile] %s order %s — marked unfilled", sym, st)
                continue
            if st not in ("new", "accepted", "partially_filled", "pending_new",
                          "accepted_for_bidding", "held", "pending_replace"):
                continue  # unknown status — recheck next loop

            # Still working. Keep chasing ONLY while relevant AND profitable.
            comp = float(r["composite_score"] or 0)
            direction = r["direction"]
            qty = abs(float(r["qty"] or 0))
            tp = float(r["take_profit"] or 0)
            cur_raw = self.alpaca.get_latest_trade(sym)
            cur = float(cur_raw) if cur_raw else 0.0

            relevant = True
            try:
                age_min = (now - _dt.strptime(r["decided_at"], "%Y-%m-%dT%H:%M:%SZ")
                           .replace(tzinfo=_tz.utc)).total_seconds() / 60.0
                relevant = age_min <= RELEVANCE_MIN
            except Exception:  # noqa: BLE001
                relevant = True
            profitable = True
            if tp > 0 and cur > 0:
                room = (tp - cur) / cur if direction == "buy" else (cur - tp) / cur
                profitable = room >= MIN_UPSIDE

            keep_chasing = (comp >= RETRY_COMPOSITE and relevant and profitable
                            and qty > 0 and cur > 0)

            if not keep_chasing:
                try:
                    self.alpaca.cancel_order(r["alpaca_order_id"])
                except AlpacaError:
                    pass
                _mark(r["id"], "unfilled", "ext_no_fill")
                self._stock_fill_retry_ts.pop(score_id, None)
                log.info("[fill-reconcile] %s stop chasing (comp=%.1f relevant=%s profitable=%s) — unfilled",
                         sym, comp, relevant, profitable)
                continue

            # Cooldown: leave the working order alone for a bit before re-pricing
            # (avoid cancel-replace churn / hammering a halted name every loop).
            if now_ts - self._stock_fill_retry_ts.get(score_id, 0.0) < RETRY_COOLDOWN_S:
                continue
            # Re-submit only the UNFILLED REMAINDER. A partially-filled order already
            # has live shares; re-submitting the full qty would double-buy (over-fill
            # and possibly breach the position cap).
            filled = float(getattr(o, "filled_qty", 0) or 0)
            remainder = qty - filled
            if remainder < 1:
                continue  # effectively filled — let it promote on the next snapshot
            try:
                self.alpaca.cancel_order(r["alpaca_order_id"])
            except AlpacaError:
                pass
            coid = f"mr-refill-{score_id}-{int(now_ts * 1000)}"
            try:
                if market_open:
                    new = self.alpaca.submit_simple_order(
                        symbol=sym, side=direction, qty=remainder,
                        order_type="market", time_in_force="day", client_order_id=coid)
                else:
                    buf = 1.005 if direction == "buy" else 0.995
                    new = self.alpaca.submit_simple_order(
                        symbol=sym, side=direction, qty=remainder,
                        order_type="limit", limit_price=round(cur * buf, 2),
                        time_in_force="day", extended_hours=True, client_order_id=coid)
                self._stock_fill_retry_ts[score_id] = now_ts
                with get_connection() as conn:
                    conn.execute("UPDATE bot_decisions SET alpaca_order_id=? WHERE id=?",
                                 (new.id, r["id"]))
                log.info("[fill-retry] %s comp=%.1f (relevant+profitable) — re-submitted to chase fill",
                         sym, comp)
            except AlpacaError as exc:
                log.warning("[fill-retry] %s re-submit failed: %s", sym, exc)
                _mark(r["id"], "unfilled", "ext_no_fill")

    def _poll_stock_exits(self, positions) -> None:
        """Poll EXTENDED-HOURS stock positions and close them at their recorded SL/TP.

        These were entered by _process_stock_candidate as simple LIMIT orders
        (outcome_detail='stock_polled') because Alpaca forbids bracket/stop/market
        orders outside regular hours — so they have NO server-side bracket and THIS
        poller is their only stop/TP protection. Regular-hours bracket stocks are
        protected by Alpaca server-side and are NOT touched here (the query filters
        on the marker). Closes use a market order in regular hours, or a marketable
        limit with extended_hours=True when the market is closed. P&L is an estimate
        (reconciliation/EOD corrects). A 120s per-ticker cooldown stops duplicate
        close orders stacking while a thin-liquidity limit close is still working.
        """
        import time as _t_sx
        stock_positions = [p for p in positions
                           if "/" not in p.symbol and len(p.symbol) <= 9]
        open_syms = {p.symbol.upper() for p in stock_positions}

        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT ticker, direction, stop_loss, take_profit, score_id
                FROM bot_decisions
                WHERE outcome='placed' AND outcome_detail='stock_polled'
                  AND ticker NOT LIKE '%/%'
                  AND stop_loss IS NOT NULL AND take_profit IS NOT NULL
                ORDER BY id DESC
                """
            ).fetchall()
        sl_tp_by_sym: dict[str, dict] = {}
        for r in rows:
            sl_tp_by_sym.setdefault(r["ticker"].upper(), dict(r))

        # Cleanup: any ticker we submitted a close for that is no longer an open
        # position has exited — mark its decision closed so we don't re-manage a
        # phantom (the bug that bit the option spreads). Also fires when all stocks
        # are flat (open_syms empty).
        if hasattr(self, "_stock_exit_attempts"):
            for sym in list(self._stock_exit_attempts.keys()):
                if sym not in open_syms:
                    td = sl_tp_by_sym.get(sym)
                    if td:
                        with get_connection() as conn:
                            conn.execute(
                                "UPDATE bot_decisions SET outcome='closed', "
                                "outcome_detail='stock_ext_exited' WHERE score_id=?",
                                (td["score_id"],),
                            )
                    self._stock_exit_attempts.pop(sym, None)
        # Drop trailing-peak state for any ticker no longer open, so a later re-entry
        # on the same name starts its peak fresh from entry (not a stale prior high).
        if hasattr(self, "_stock_peaks"):
            for pk in list(self._stock_peaks.keys()):
                if pk[0] not in open_syms:
                    self._stock_peaks.pop(pk, None)
        # Cancel + forget any server-side trailing_stop whose position is gone (it
        # already fired, or the position closed another way — cancel is a harmless
        # no-op if already filled/canceled; this prevents an orphan stop lingering).
        if hasattr(self, "_stock_trailstops"):
            for s in list(self._stock_trailstops.keys()):
                if s not in open_syms:
                    try:
                        self.alpaca.cancel_order(self._stock_trailstops[s])
                    except AlpacaError:
                        pass
                    # A position exited by the server-side trailing_stop is never in
                    # _stock_exit_attempts, so mark its decision closed here too —
                    # otherwise the 'placed' row lingers as a phantom (the exact bug
                    # the option-spread / fill reconcilers were built to prevent).
                    td = sl_tp_by_sym.get(s)
                    if td:
                        with get_connection() as conn:
                            conn.execute(
                                "UPDATE bot_decisions SET outcome='closed', "
                                "outcome_detail='stock_server_stop_exited' WHERE score_id=?",
                                (td["score_id"],),
                            )
                    self._stock_trailstops.pop(s, None)

        if not sl_tp_by_sym or not stock_positions:
            return

        market_open = self.alpaca.is_market_open()
        if not hasattr(self, "_stock_exit_attempts"):
            self._stock_exit_attempts = {}
        now_ts = _t_sx.time()

        for p in stock_positions:
            sym = p.symbol.upper()
            tp_sl = sl_tp_by_sym.get(sym)
            if not tp_sl:
                continue
            current_price = float(p.current_price) if p.current_price else None
            if not current_price:
                latest = self.alpaca.get_latest_trade(p.symbol)
                if not latest:
                    continue
                current_price = float(latest)
            sl = float(tp_sl["stop_loss"])
            direction = tp_sl["direction"]
            entry = float(p.avg_entry_price)

            # SERVER-SIDE protection (regular hours): arm a native trailing_stop ONCE,
            # broker-side, so a crash or a sleeping Mac can't leave a catalyst "bang"
            # unmanaged. It rides the run and exits on the same giveback % the poller
            # uses — but survives the bot dying. Once armed, the broker order OWNS the
            # exit and the poller hands off (no double-close). Extended hours can't use
            # server stops, so the poller keeps managing those itself (below).
            if not hasattr(self, "_stock_trailstops"):
                self._stock_trailstops = {}
            if not hasattr(self, "_stock_trailstop_fail_ts"):
                self._stock_trailstop_fail_ts = {}
            # After an arm failure (e.g. the name is HALTED -> 403 on POST /orders), back
            # off instead of retrying every 30s loop; the poller manages it meanwhile and
            # we re-attempt once the cooldown elapses (e.g. the halt clears).
            _ARM_RETRY_COOLDOWN_S = 300
            _arm_cooled = (now_ts - self._stock_trailstop_fail_ts.get(sym, 0.0)) < _ARM_RETRY_COOLDOWN_S
            if market_open and sym not in self._stock_trailstops and not _arm_cooled:
                existing_id = None
                try:  # adopt an existing trailing_stop (e.g. after a bot restart) — no dupes
                    for o in self.alpaca.list_orders(status="open", limit=200):
                        if o.symbol.upper() == sym and "trailing" in (o.order_type or "").lower():
                            existing_id = o.id
                            break
                except AlpacaError:
                    pass
                if existing_id:
                    self._stock_trailstops[sym] = existing_id
                else:
                    try:
                        giveback = float(os.getenv("LIVE_STOCK_TRAIL_GIVEBACK_PCT", "0.15"))
                        ts = self.alpaca.submit_trailing_stop_order(
                            symbol=p.symbol,
                            side=("sell" if direction == "buy" else "buy"),
                            qty=abs(float(p.qty)),
                            trail_percent=round(giveback * 100, 2),
                            time_in_force="gtc",
                            client_order_id=f"mr-ts-{tp_sl['score_id']}-{int(now_ts)}",
                        )
                        self._stock_trailstops[sym] = ts.id
                        self._stock_trailstop_fail_ts.pop(sym, None)
                        log.info("[server-stop] %s armed trailing_stop %.1f%% broker-side — "
                                 "poller hands off RTH exit", sym, giveback * 100)
                    except AlpacaError as exc:
                        self._stock_trailstop_fail_ts[sym] = now_ts
                        log.warning("[server-stop] %s arm failed — poller manages, backing off %ds "
                                    "(likely halted): %s", sym, _ARM_RETRY_COOLDOWN_S, exc)
            if market_open and sym in self._stock_trailstops:
                continue  # broker-side trailing_stop owns this RTH exit

            # FLEXIBLE / TRAILING TAKE-PROFIT — catalyst "bangs" are volatile, so RIDE the
            # run instead of capping it with a fixed TP. Ratchet the peak; once up
            # TRAIL_ARM_PCT from entry the trail is active, and we exit when price gives back
            # TRAIL_GIVEBACK_PCT from that peak (banking the gain). Before the trail rises
            # above entry, the hard ATR stop_loss (from sizing) caps the downside. Both are
            # env-tunable so they can be tightened/loosened for how volatile the names run.
            TRAIL_ARM_PCT = float(os.getenv("LIVE_STOCK_TRAIL_ARM_PCT", "0.05"))
            TRAIL_GIVEBACK_PCT = float(os.getenv("LIVE_STOCK_TRAIL_GIVEBACK_PCT", "0.15"))
            if not hasattr(self, "_stock_peaks"):
                self._stock_peaks = {}
            peak_key = (sym, direction)
            prev_peak = self._stock_peaks.get(peak_key, entry)

            exit_reason = None
            if direction == "buy":
                peak = max(prev_peak, current_price)
                gain_pct = (peak - entry) / entry if entry else 0.0
                trail_armed = gain_pct >= TRAIL_ARM_PCT
                effective_sl = max(sl, peak * (1 - TRAIL_GIVEBACK_PCT)) if trail_armed else sl
                if current_price <= effective_sl:
                    exit_reason = "trail_take_profit" if effective_sl > entry else "stop_loss"
            else:  # short
                peak = min(prev_peak, current_price)
                gain_pct = (entry - peak) / entry if entry else 0.0
                trail_armed = gain_pct >= TRAIL_ARM_PCT
                effective_sl = min(sl, peak * (1 + TRAIL_GIVEBACK_PCT)) if trail_armed else sl
                if current_price >= effective_sl:
                    exit_reason = "trail_take_profit" if effective_sl < entry else "stop_loss"
            self._stock_peaks[peak_key] = peak

            if not exit_reason:
                continue

            # Cooldown — don't stack duplicate closes while a prior one is working.
            if now_ts - self._stock_exit_attempts.get(sym, 0.0) < 120:
                continue
            if self.cfg.dry_run:
                continue

            try:
                close_side = "sell" if direction == "buy" else "buy"
                qty = abs(float(p.qty))
                coid = f"mr-sx-{tp_sl['score_id']}-{int(now_ts * 1000)}"
                if market_open:
                    self.alpaca.submit_simple_order(
                        symbol=p.symbol, side=close_side, qty=qty,
                        order_type="market", time_in_force="day",
                        client_order_id=coid,
                    )
                else:
                    cbuf = 0.997 if close_side == "sell" else 1.003
                    self.alpaca.submit_simple_order(
                        symbol=p.symbol, side=close_side, qty=qty,
                        order_type="limit", limit_price=round(current_price * cbuf, 2),
                        time_in_force="day", extended_hours=True,
                        client_order_id=coid,
                    )
                self._stock_exit_attempts[sym] = now_ts
                entry = float(p.avg_entry_price)
                pnl = ((current_price - entry) if direction == "buy"
                       else (entry - current_price)) * qty
                log.info(
                    "STOCK EXIT [%s] %s %s qty=%g @ $%.2f (peak=$%.2f exitSL=$%.2f) pnl~$%.2f",
                    exit_reason, p.symbol, close_side, qty, current_price, peak, effective_sl, pnl,
                )
                with get_connection() as conn:
                    self._update_daily_pnl(conn, pnl)
                try:
                    from ..notifications.realtime import TradeAlert, notify_trade
                    notify_trade(TradeAlert(
                        kind="FILLED", symbol=p.symbol, pnl_usd=pnl,
                        notional_usd=abs(float(p.market_value)),
                        qty=qty, extra=f"ext_{exit_reason}",
                    ))
                except Exception:  # noqa: BLE001
                    pass
            except AlpacaError as exc:
                log.error("STOCK-EXT EXIT submit failed for %s: %s", p.symbol, exc)

    def _poll_crypto_exits(self, positions) -> None:
        """Poll open CRYPTO positions and close at SL/TP from bot_decisions.

        Crypto has no Alpaca-side bracket orders (broker limit), so this poller
        IS its only SL/TP/trailing exit. The SL/TP lookup and the price fetch
        below are crypto-specific (pair form + get_latest_crypto_trade), so this
        covers crypto only. Stocks rely on their Alpaca bracket, the EOD flatten,
        and the daily loss-stop; option legs are managed by _poll_option_exits.
        (Earlier docstring claimed stock-bracket recovery here — it never did;
        true stock re-protection would need separate stock price/close logic.)
        """
        # Filter: crypto pairs only (short symbols), EXCLUDE option contracts.
        # The SL/TP query below further restricts to ticker LIKE '%/%'.
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
                    self._book_crypto_exit_pnl(conn, pair, qty, entry, pnl)
                self._closed_this_loop.add(pair)  # guard the flip check from double-booking
                try:
                    from ..notifications.realtime import TradeAlert, notify_trade
                    headline = None
                    try:
                        # tp_sl has decision context with score_id we can use
                        with get_connection() as _hc:
                            _row = _hc.execute(
                                """SELECT rs.title FROM signal_scores ss
                                   JOIN raw_signals rs ON rs.id = ss.signal_id
                                   WHERE ss.id = ? LIMIT 1""",
                                (tp_sl.get("score_id"),),
                            ).fetchone()
                            if _row and _row[0]:
                                t = str(_row[0]).strip()
                                headline = t[:80] + ("…" if len(t) > 80 else "")
                    except Exception:  # noqa: BLE001
                        pass
                    extra = (f"crypto {exit_reason} · "
                             f"${entry:.4f}→${current_price:.4f}")
                    if headline:
                        extra = f"{extra}\n💬 {headline}"
                    notify_trade(TradeAlert(
                        kind="FILLED", symbol=pair, pnl_usd=pnl,
                        notional_usd=float(qty * entry),
                        qty=float(qty),
                        extra=extra,
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
                      AND ss.scored_at > strftime('%Y-%m-%dT%H:%M:%SZ', datetime('now','-5 minutes'))
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
                      AND ss.scored_at > strftime('%Y-%m-%dT%H:%M:%SZ', datetime('now','-5 minutes'))
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
            if slashed in getattr(self, "_closed_this_loop", set()):
                continue  # SL/TP poller already closed this symbol this loop — no double-book
            try:
                close_side = "sell" if held_dir == "buy" else "buy"
                self.alpaca.submit_simple_order(
                    symbol=slashed, side=close_side, qty=abs(float(p.qty)),
                    order_type="market", time_in_force="gtc",
                    client_order_id=f"mr-flip-{slashed.replace('/','')[:6]}-{_t_flip.time_ns()}",
                )
                self._recent_flips[slashed] = now_ts
                # Book realized P/L so the bot_orders row closes (no phantom).
                _flip_pnl = float(p.unrealized_pl)
                with get_connection() as _fc:
                    self._update_daily_pnl(_fc, _flip_pnl)
                    self._book_crypto_exit_pnl(_fc, slashed, abs(float(p.qty)),
                                               float(p.avg_entry_price), _flip_pnl)
            except AlpacaError as exc:
                log.error("CRYPTO FLIP close failed for %s: %s", slashed, exc)

    # ------------------------------------------------------------------
    # Option-spread realized P&L from REAL close fills
    # ------------------------------------------------------------------
    def _exit_credit_from_close_fills(
        self,
        *,
        spread_id: int,
        legs: list[dict],
        contracts: int,
        close_order_ids: Optional[list[str]] = None,
    ) -> Optional[float]:
        """Compute the net exit CREDIT (per spread) from the spread's REAL
        closing fills at Alpaca — never from intraday quotes.

        Quotes on these names are illiquid/wide and swing wildly (a quote
        mark moved -$422 → -$843 in minutes on 2026-06-01). The reliable
        number is the actual fill. Two resolution strategies, in order:

        1. **By the close order(s) we submitted.** Pass ``close_order_ids``
           (Alpaca order ids). For a multi-leg ('mleg') close, the parent
           ``filled_avg_price`` is the SIGNED net (negative = credit
           received to close); we return ``abs(filled_avg_price)``. We also
           verify against the per-leg fills (sum(sell fills) − sum(buy
           fills)). For per-leg simple orders (the daily-flatten path) we
           sum each order's signed fill.

        2. **By leg symbol + FIFO allocation** (fallback / reconcile). Pull
           FILL account activities, keep only CLOSE-side fills for each leg
           (opposite the entry side), and allocate ``contracts`` worth to
           this spread FIFO. Handles spreads that share identical legs
           (e.g. two INTC spreads on the same strikes) by consuming a
           shared per-call pool.

        Returns the per-spread net credit (long-leg sell px − short-leg buy
        px), or ``None`` if fills can't be resolved. Read-only on Alpaca.
        """
        long_leg = next((L for L in legs if L["role"] == "long"), None)
        short_leg = next((L for L in legs if L["role"] == "short"), None)
        if long_leg is None or short_leg is None or contracts <= 0:
            return None

        # --- Strategy 1: from the specific close order(s) we submitted -----
        if close_order_ids:
            try:
                long_px = short_px = None
                # If a single mleg order closed both legs, the parent's
                # signed net is authoritative; cross-check via legs.
                for oid in close_order_ids:
                    o = self.alpaca.get_order(oid)
                    if (o.status or "").lower() != "filled":
                        return None  # not done filling yet — caller retries later
                    o_legs = o.legs or []
                    if o_legs:
                        # multi-leg close: derive per-leg fills
                        for L in o_legs:
                            fp = L.get("filled_avg_price")
                            if fp is None:
                                return None
                            fp = float(fp)
                            if L.get("symbol") == long_leg["contract_symbol"]:
                                long_px = fp
                            elif L.get("symbol") == short_leg["contract_symbol"]:
                                short_px = fp
                    else:
                        # per-leg simple order (daily-flatten path)
                        fp = o.filled_avg_price
                        if fp is None:
                            return None
                        fp = float(fp)
                        if o.symbol == long_leg["contract_symbol"]:
                            long_px = fp
                        elif o.symbol == short_leg["contract_symbol"]:
                            short_px = fp
                if long_px is not None and short_px is not None:
                    return long_px - short_px
            except AlpacaError as exc:
                log.debug("exit-credit by order-id failed for spread %d: %s",
                          spread_id, exc)
            except Exception as exc:  # noqa: BLE001
                log.debug("exit-credit by order-id error for spread %d: %s",
                          spread_id, exc)

        # --- Strategy 2: leg-symbol FIFO over FILL activities --------------
        try:
            pool = self._closing_fill_pool()
        except Exception as exc:  # noqa: BLE001
            log.debug("could not build closing-fill pool: %s", exc)
            return None

        def _close_side(entry_side: str) -> str:
            return "sell" if entry_side == "buy" else "buy"

        def _take(symbol: str, side: str, need: int) -> Optional[float]:
            fills = pool.get((symbol, side), [])
            got = 0.0
            cost = 0.0
            for f in fills:
                if got >= need:
                    break
                take_q = min(f["qty_left"], need - got)
                if take_q <= 0:
                    continue
                cost += take_q * f["price"]
                got += take_q
                f["qty_left"] -= take_q
            if got < need - 1e-6:
                return None
            return cost / got if got > 0 else None

        long_px = _take(long_leg["contract_symbol"],
                        _close_side(long_leg["side"]), contracts)
        short_px = _take(short_leg["contract_symbol"],
                         _close_side(short_leg["side"]), contracts)
        if long_px is None or short_px is None:
            return None
        return long_px - short_px

    def _closing_fill_pool(self) -> dict:
        """Build a FIFO-consumable pool of option CLOSE fills, keyed by
        (contract_symbol, side). Cached per loop-iteration so multiple
        spreads sharing legs allocate from the SAME pool (no double-count).

        Only fills from recognised close orders are included
        (``tpfire-*`` daily-flatten leg sells, ``mr-ox-*`` poll-exit mleg,
        ``mr-eod-opt-*`` EOD mleg). Read-only on Alpaca.
        """
        from collections import defaultdict
        import time as _t_pool
        # Short-TTL cache so spreads sharing legs within the same loop burst
        # (reconcile → poll-exit → flatten) allocate from ONE pool without
        # double-counting, while staying fresh across 30s iterations.
        now = _t_pool.monotonic()
        cache = getattr(self, "_closing_fill_pool_cache", None)
        if cache is not None and (now - cache[0]) < 10.0:
            return cache[1]

        acts = self.alpaca.list_account_activities(activity_type="FILL",
                                                   page_size=100) or []
        orders = self.alpaca.list_orders(status="all", limit=500, nested=True)
        coid_by_oid = {o.id: (o.client_order_id or "") for o in orders}

        pool: dict = defaultdict(list)
        for a in acts:
            sym = a.get("symbol", "") or ""
            if len(sym) <= 15:  # OCC option symbols only
                continue
            coid = coid_by_oid.get(a.get("order_id", ""), "")
            if not coid.startswith(("tpfire-", "mr-ox-", "mr-eod-opt-")):
                continue
            raw_side = a.get("side", "") or ""
            side = "sell" if raw_side.startswith("sell") else "buy"
            try:
                qty = abs(float(a.get("qty", 0) or 0))
                price = float(a.get("price", 0) or 0)
            except (TypeError, ValueError):
                continue
            if qty <= 0 or price <= 0:
                continue
            pool[(sym, side)].append(
                {"qty_left": qty, "price": price,
                 "tt": a.get("transaction_time", "")})
        # Stable FIFO order by fill time
        for k in pool:
            pool[k].sort(key=lambda f: f["tt"])
        self._closing_fill_pool_cache = (now, pool)
        return pool

    def _book_spread_exit(
        self,
        *,
        spread_id: int,
        underlying: str,
        entry_debit: float,
        contracts: int,
        legs: list[dict],
        exit_reason: str,
        close_order_ids: Optional[list[str]] = None,
        set_closed_at: bool = True,
    ) -> Optional[float]:
        """Idempotently book realized P&L for one option spread from its
        REAL close fills, and update daily P&L.

        Guards on ``realized_pnl_usd IS NULL`` so it can never double-book
        (safe to call from multiple paths + the reconcile backstop). Never
        places orders. Returns the booked realized P&L, or ``None`` if it
        couldn't resolve fills (left for a later pass to retry).
        """
        try:
            exit_credit = self._exit_credit_from_close_fills(
                spread_id=spread_id, legs=legs, contracts=contracts,
                close_order_ids=close_order_ids,
            )
            if exit_credit is None:
                log.debug("spread %d (%s): close fills not resolvable yet — "
                          "deferring realized P&L booking",
                          spread_id, underlying)
                return None
            realized = (exit_credit - entry_debit) * contracts * 100
            pnl_pct = ((exit_credit - entry_debit) / entry_debit * 100
                       if entry_debit else 0.0)
            with get_connection() as conn:
                # Idempotent: only book if not already booked.
                cur = conn.execute(
                    "UPDATE bot_option_spreads "
                    "SET exit_credit_usd=?, realized_pnl_usd=?, pnl_pct=?, "
                    "    exit_reason=COALESCE(exit_reason, ?), "
                    "    closed_at=COALESCE(closed_at, ?) "
                    "WHERE id=? AND realized_pnl_usd IS NULL",
                    (exit_credit, realized, pnl_pct, exit_reason,
                     (utc_now() if set_closed_at else None), spread_id),
                )
                if cur.rowcount == 0:
                    return None  # already booked by another path — no dup
                # NOTE: do NOT touch bot_daily_pnl here. _reconcile_realized_pnl
                # owns the daily total (authoritative Alpaca equity-delta, set
                # every loop). Adding realized here would transiently double-
                # count (could make the daily_loss_cap gate spuriously block for
                # one iteration). This booking's job is only the per-spread
                # realized_pnl_usd that the learning loop reads.
            log.info(
                "OPT P&L booked [%s] spread %d %s: entry=$%.2f exit=$%.2f "
                "× %d → realized $%+.2f (%.1f%%)",
                exit_reason, spread_id, underlying, entry_debit, exit_credit,
                contracts, realized, pnl_pct,
            )
            return realized
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not book spread %d (%s) realized P&L: %s",
                        spread_id, underlying, exc)
            return None

    def _reconcile_option_spread_pnl(self) -> None:
        """Backstop: book realized P&L for any spread that has been CLOSED
        (closed_at set, or legs gone from Alpaca) but whose
        ``realized_pnl_usd`` is still NULL.

        This catches spreads flattened by the daily loss-stop
        (``_fire_daily_flatten`` sells legs but historically never touched
        ``bot_option_spreads``) and any spread closed by older code paths.
        Idempotent (``_book_spread_exit`` guards on NULL) and fully
        fail-safe. Read-only on Alpaca.
        """
        try:
            with get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT id, underlying, entry_debit_usd, contracts,
                           closed_at, status
                    FROM bot_option_spreads
                    WHERE status='filled'
                      AND realized_pnl_usd IS NULL
                      AND entry_debit_usd IS NOT NULL
                      AND contracts > 0
                    """
                ).fetchall()
                if not rows:
                    return
                legs_by_spread: dict[int, list[dict]] = {}
                for r in rows:
                    legs = conn.execute(
                        "SELECT contract_symbol, role, side "
                        "FROM bot_option_legs WHERE spread_id=?",
                        (r["id"],),
                    ).fetchall()
                    legs_by_spread[r["id"]] = [dict(L) for L in legs]

            # Which option contracts are still open at Alpaca? A spread whose
            # legs are gone has been closed even if closed_at wasn't set.
            try:
                positions = self.alpaca.get_positions()
                open_occ = {p.symbol for p in positions if len(p.symbol) > 15}
            except AlpacaError:
                open_occ = None  # unknown — only process rows with closed_at

            for r in rows:
                sp = dict(r)
                legs = legs_by_spread.get(sp["id"], [])
                if not legs:
                    continue
                legs_open = (open_occ is not None
                             and any(L["contract_symbol"] in open_occ
                                     for L in legs))
                # Skip spreads still open (no closed_at and legs still held).
                if sp["closed_at"] is None and (legs_open or open_occ is None):
                    continue
                self._book_spread_exit(
                    spread_id=sp["id"], underlying=sp["underlying"],
                    entry_debit=float(sp["entry_debit_usd"]),
                    contracts=int(sp["contracts"]), legs=legs,
                    exit_reason="reconciled_close_fill",
                    set_closed_at=(sp["closed_at"] is None),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("_reconcile_option_spread_pnl failed: %s", exc)

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
                WHERE s.closed_at IS NULL AND s.status = 'filled'
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

                # Book realized P&L from the REAL close fill, not the quote
                # mark. Option quotes on these names are illiquid/wide and
                # swing wildly minute-to-minute; the close order's actual
                # fill is authoritative. _book_spread_exit reads the fill
                # (mleg parent net / leg fills) and is idempotent. If the
                # close order hasn't filled yet (resolves None), fall back to
                # the quote mark so we still close the row + book *something*;
                # the reconcile backstop will correct it to the real fill
                # next iteration (it re-books only while realized is NULL —
                # so a quote fallback that *did* write a number is final;
                # therefore only fall back when the row would otherwise stay
                # open/unbooked).
                realized = self._book_spread_exit(
                    spread_id=sp["id"], underlying=sp["underlying"],
                    entry_debit=entry_debit, contracts=contracts, legs=legs,
                    exit_reason=exit_reason,
                    close_order_ids=[mleg.id] if getattr(mleg, "id", None) else None,
                )
                if realized is None:
                    # Fill not yet resolvable — defer booking. Mark closed so
                    # the poller stops re-submitting; the reconcile backstop
                    # books the real realized P&L once the fill lands (it acts
                    # only while realized_pnl_usd IS NULL).
                    realized = (current_debit - entry_debit) * contracts * 100
                    with get_connection() as conn:
                        conn.execute(
                            "UPDATE bot_option_spreads SET closed_at=?, "
                            "exit_reason=COALESCE(exit_reason, ?) "
                            "WHERE id=? AND closed_at IS NULL",
                            (utc_now(), exit_reason, sp["id"]),
                        )

                try:
                    from ..notifications.realtime import TradeAlert, notify_trade
                    headline = self._headline_for_order(spread_id=sp["id"])
                    extra = (f"{exit_reason} · "
                             f"${entry_debit:.2f}→${current_debit:.2f} "
                             f"({pnl_pct_of_debit*100:+.1f}%)")
                    if headline:
                        extra = f"{extra}\n💬 {headline}"
                    notify_trade(TradeAlert(
                        kind="OPTIONS_FILLED", symbol=sp["underlying"],
                        pnl_usd=realized,
                        notional_usd=float(entry_debit * contracts * 100),
                        qty=contracts,
                        extra=extra,
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

    def _daily_loss_stop_if_due(self, account, positions) -> bool:
        """SELECTIVE intraday loss de-risk (the loss-side mirror of the daily
        profit-take, but surgical). The risk gate's daily_loss_cap only BLOCKS
        NEW trades and counts REALIZED P&L only — so a deeply underwater book can
        blow far past the cap before EOD flatten (2026-05-29: -$1,953 ≈ 4x the
        $500 cap). When intraday P&L (realized + unrealized = equity -
        last_equity) breaches the cap, this:

          1. Closes ONLY the LOSING positions (unrealized_pl < 0). WINNERS
             (unrealized_pl >= 0) are KEPT — they ride their own per-position
             stops (stock brackets, option TP/SL, crypto trail), which run
             elsewhere. (Old behaviour flattened EVERYTHING, e.g. cutting
             winning stocks +$82 alongside losing options −$270 → realized −$370
             instead of ~−$270.)
          2. HALTS new entries for the rest of the day via ``_daily_loss_halt_on``
             (a SEPARATE flag from the profit-take's ``_daily_tp_fired_on`` —
             see below).
          3. ROLLING: on EVERY subsequent iteration while halted, re-closes any
             position that has SINCE gone unrealized_pl < 0 (a kept winner that
             reverses gets cut next tick). Greens are kept only while green.

        FLAG SEPARATION (critical): this sets ``_daily_loss_halt_on``, NOT
        ``_daily_tp_fired_on``. ``_daily_profit_take_if_due`` has a retry branch
        ``if _daily_tp_fired_on == today and positions: _fire_daily_flatten(ALL)``
        — if the loss-stop set that flag, the profit-take retry would flatten the
        KEPT WINNERS and defeat this feature. So we keep ``_daily_tp_fired_on``
        untouched and pass ``set_halt_flag=False`` to ``_fire_daily_flatten`` (so
        it closes only what we hand it and doesn't set the TP flag). The
        profit-take path is unchanged: it still flattens EVERYTHING and sets its
        own flag.

        ONLY active during the equities session. Alpaca's `last_equity` is the
        PRIOR equities close, so `equity - last_equity` overnight/over a weekend
        includes 24/7 crypto drift (the bot holds crypto when the market is shut)
        — NOT a real intraday loss. Firing then would liquidate the crypto book
        at an illiquid off-hours mark on benign noise. Crypto is protected by its
        own per-position SL/TP/trailing exits while the market is closed.

        Returns True if we closed something this tick (caller skips the rest of
        the loop); False otherwise (loop continues — note new entries are still
        blocked downstream by the ``_daily_loss_halt_on`` guard)."""
        from datetime import date as _date
        from ..config import CONFIG
        cap = abs(float(CONFIG.risk_daily_loss_cap_usd or 0))
        if cap <= 0:
            return False
        # Session gate — see docstring. On a clock-check failure, do NOT fire
        # (avoid the off-hours false-fire); EOD flatten + per-position stops remain.
        try:
            if not self.alpaca.is_market_open():
                return False
        except Exception:  # noqa: BLE001
            return False
        today = _date.today()
        # If the full profit-take flatten has already fired today, it owns the
        # book (flattens EVERYTHING + retries). Don't run a competing close.
        if getattr(self, "_daily_tp_fired_on", None) == today:
            return False

        try:
            intraday = float(account.equity - account.last_equity)
            already_halted = getattr(self, "_daily_loss_halt_on", None) == today

            # Trigger condition: either we just breached the cap, OR we're
            # already in the halted state (rolling re-check of kept winners).
            if intraday > -cap and not already_halted:
                return False

            # Always recompute the CURRENT losers from the live snapshot so a
            # winner that reversed since last tick gets cut now, and a position
            # that recovered to >=0 is left to ride its own stop.
            losers = [p for p in positions if float(p.unrealized_pl) < 0]

            if not already_halted:
                # FIRST trigger this day: arm the halt + alert once.
                self._daily_loss_halt_on = today
                self._persist_loss_halt_state()  # survive a restart — don't re-expose
                log.error(
                    "🛑 DAILY LOSS STOP: intraday $%.2f <= -$%.2f — SELECTIVE "
                    "de-risk: cutting %d loser(s), keeping %d winner(s) on their "
                    "own stops, halting new entries for the day.",
                    intraday, cap, len(losers), len(positions) - len(losers),
                )
                try:
                    self._maybe_alert_halt(
                        "daily_loss_cap",
                        f"intraday P&L ${intraday:.0f} <= -${cap:.0f} — cut "
                        f"{len(losers)} loser(s), kept "
                        f"{len(positions) - len(losers)} winner(s), halted new entries",
                    )
                except Exception:  # noqa: BLE001
                    pass

            if not losers:
                # Halted, but nothing currently red — keep all winners, just
                # hold the halt. (Returns False so the loop continues; the
                # _daily_loss_halt_on guard downstream blocks any new entry.)
                if already_halted:
                    log.info("🛑 LOSS HALT active: intraday $%.2f, 0 losers — "
                             "keeping %d winner(s), no close this tick.",
                             intraday, len(positions))
                return False

            log.warning(
                "🛑 LOSS-STOP de-risk: closing %d loser(s) "
                "(total u_pnl $%.2f), keeping %d winner(s).",
                len(losers), sum(float(p.unrealized_pl) for p in losers),
                len(positions) - len(losers),
            )
            return self._fire_daily_flatten(
                losers, intraday, f"loss_stop_${intraday:.0f}",
                set_halt_flag=False,
            )
        except Exception as exc:  # noqa: BLE001 — never crash the trade loop
            log.exception("Selective loss-stop failed (fail-safe, continuing): %s", exc)
            return False

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

    def _fire_daily_flatten(self, positions, intraday_pnl: float, mode: str,
                            set_halt_flag: bool = True) -> bool:
        """Flatten the given positions + (optionally) halt for the day.

        VERIFICATION: re-fetches positions after close attempts, logs anything
        that didn't actually close. The 'fired' flag is set immediately to block
        new opens, but if any position remains we retry on subsequent iterations.

        ``set_halt_flag`` (default True — profit-take behaviour is UNCHANGED):
        when True, sets ``_daily_tp_fired_on``+persists, so the profit-take retry
        branch owns leftover positions and new opens are blocked. The SELECTIVE
        loss-stop passes ``set_halt_flag=False`` and ONLY its losing positions,
        so it must NOT set ``_daily_tp_fired_on`` — otherwise the profit-take
        retry (``if _daily_tp_fired_on == today and positions: flatten ALL``)
        would grab the KEPT WINNERS and defeat the selective de-risk. The
        loss-stop sets its own ``_daily_loss_halt_on`` flag for the new-entry
        halt; per-position closes are re-driven each tick by the loss-stop itself.

        Per-asset close strategy:
        - Stocks: close_position (Alpaca handles via market sell)
        - Crypto: close_position with /USD pair format
        - Option contract legs: submit_simple_order (close_position can 403 on isolated legs)
        """
        from datetime import date as _date
        import time as _time
        log.warning(
            "🎯 DAILY %s [%s]: intraday $%.2f. Closing %d positions%s.",
            "TP FIRED" if intraday_pnl >= 0 else "LOSS STOP",
            mode, intraday_pnl, len(positions),
            " + halting for day" if set_halt_flag else " (selective de-risk)",
        )
        if set_halt_flag:
            self._daily_tp_fired_on = _date.today()  # set IMMEDIATELY so new opens are blocked
            self._persist_daily_tp_state()  # persist to DB so restarts respect it

        # 1. Cancel open orders FIRST (frees bracket-child qty so the close
        # market order isn't rejected for held qty).
        #   - Full flatten (set_halt_flag=True): cancel EVERYTHING.
        #   - Selective de-risk (set_halt_flag=False): cancel ONLY the orders
        #     tied to the positions we're closing. A blanket cancel here would
        #     strip the protective bracket (SL/TP) off the KEPT WINNERS, leaving
        #     them naked — the exact opposite of "keep the winners on their own
        #     stops". Match by symbol incl. bracket child legs.
        try:
            if set_halt_flag:
                n = self.alpaca.cancel_all_orders()
                log.info("  cancelled %d open orders", n or 0)
            else:
                close_syms = {p.symbol for p in positions}
                cancelled = 0
                for o in self.alpaca.list_orders(status="open", limit=200):
                    o_syms = {o.symbol} | {
                        (leg or {}).get("symbol") for leg in (o.legs or [])
                    }
                    if close_syms & o_syms:
                        try:
                            self.alpaca.cancel_order(o.id)
                            cancelled += 1
                        except AlpacaError as exc:
                            log.warning("  cancel_order(%s) failed: %s", o.id, exc)
                log.info("  selectively cancelled %d order(s) for %d closing "
                         "position(s); kept winners' brackets intact",
                         cancelled, len(close_syms))
        except AlpacaError as exc:
            log.warning("  order cancel step failed: %s", exc)
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
                    # AND an explicit position_intent or Alpaca rejects the close — this is
                    # what historically left the loss-stop unable to cut option losers (they
                    # ran until manual close). side=="sell" closes a long leg, "buy" a short.
                    self.alpaca.submit_simple_order(
                        symbol=sym, side=side, qty=qty,
                        order_type="market", time_in_force="day",
                        position_intent=("sell_to_close" if side == "sell" else "buy_to_close"),
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

        # In selective mode `remaining` is the WHOLE account (kept winners +
        # any loser that didn't close). Only the targeted symbols that are
        # still present are real failures-to-close; the rest are kept winners.
        if set_halt_flag:
            still_open = remaining
        else:
            target_syms = {p.symbol for p in positions}
            still_open = [p for p in remaining if p.symbol in target_syms]

        _label = "TP" if intraday_pnl >= 0 else "LOSS STOP"
        if still_open:
            log.warning(
                "🎯 DAILY %s partial flatten: %d/%d closed, %d STILL OPEN — will retry next iteration",
                _label, succeeded, attempted, len(still_open),
            )
            for p in still_open:
                log.warning("    still open: %s qty=%g", p.symbol, p.qty)
        else:
            log.warning("🎯 DAILY %s CONFIRMED FLATTEN: 0 targeted positions remain", _label)

        # Book realized P&L for actually-closed positions.
        # NOTE: option legs (OCC symbols, len>15) are NOT booked here —
        # their P&L is per-SPREAD (long+short net), not per-leg, and the
        # leg's unrealized_pl is a mark-based number. Booking it here would
        # both mis-record (no bot_orders row matches an OCC symbol) and
        # double-count daily P&L against the spread-level booking below.
        # Stocks/crypto are still booked via their unrealized_pl as before.
        closed_syms = {p.symbol for p in positions} - {p.symbol for p in remaining}
        total_realized = 0.0
        closed_option_legs = False
        for p in positions:
            if p.symbol in closed_syms:
                if len(p.symbol) > 15:  # option leg — handled per-spread below
                    closed_option_legs = True
                    continue
                pnl = float(p.unrealized_pl)
                total_realized += pnl
                with get_connection() as conn:
                    # Label the exit by what ACTUALLY happened, not always
                    # 'daily_profit_take'. A loss-stop flatten labeled as a
                    # profit-take corrupts the learning/audit data (can't tell a
                    # capped-loss day from a locked-profit day).
                    _flat_reason = "daily_loss_stop" if intraday_pnl < 0 else "daily_profit_take"
                    conn.execute(
                        "UPDATE bot_orders SET realized_pnl_usd=?, exit_reason=?, "
                        "canceled_at=? WHERE ticker=? AND status='filled' AND realized_pnl_usd IS NULL",
                        (pnl, _flat_reason, utc_now(), p.symbol),
                    )
                    self._update_daily_pnl(conn, pnl)

        # Book any option spreads whose legs were just flattened, from their
        # REAL close fills (the tpfire-* leg sells land in the fill pool).
        # _reconcile_option_spread_pnl is idempotent + fail-safe and matches
        # spreads by closed-at / legs-gone-from-Alpaca. This is what makes
        # the loss-stop path stop being blind to option outcomes.
        if closed_option_legs:
            self._reconcile_option_spread_pnl()

        # Telegram alert — ONLY on first fire of the day, NOT on retries.
        # Retries spam the user with identical messages. P&L-sign-aware so a
        # loss-stop fire (intraday_pnl < 0) reads as a loss halt, not a "TP hit".
        # Selective de-risk (set_halt_flag=False) is NOT alerted here — the
        # loss-stop's own _maybe_alert_halt("daily_loss_cap", …) owns that
        # message (once/day), and `remaining` here includes the KEPT WINNERS so
        # a "(N retrying)" note would be misleading. Also: it fires every tick a
        # kept winner reverses, which would spam.
        if (set_halt_flag and mode != "retry"
                and not getattr(self, "_daily_tp_notified", False)):
            self._daily_tp_notified = True
            try:
                from ..notifications.realtime import TradeAlert, notify_trade
                if intraday_pnl >= 0:
                    note = f"🎯 DAILY TP — closed {len(closed_syms)}/{attempted}, locked +${intraday_pnl:.2f}"
                else:
                    note = f"🛑 DAILY LOSS STOP — closed {len(closed_syms)}/{attempted}, halted at ${intraday_pnl:.2f}"
                if remaining:
                    note += f" ({len(remaining)} retrying)"
                notify_trade(TradeAlert(
                    kind="PNL_DAY", symbol="ALL", pnl_usd=intraday_pnl, extra=note,
                ))
            except Exception:  # noqa: BLE001
                pass
        return True

    def _trim_crypto_to_weekday_cap(self, positions) -> None:
        """When the US market opens, bring crypto exposure back to the weekday
        ceiling so stocks + options have their full budget. Trims the excess
        by partially selling the largest crypto positions first. Fires at most
        once per market-open transition (tracked via _crypto_trimmed_on).

        Honors the user's "squash crypto before Monday open" instruction so
        the multi-asset day-trader starts the session with room for all three.
        """
        from datetime import date as _date
        today = _date.today()
        if getattr(self, "_crypto_trimmed_on", None) == today:
            return
        if not positions:
            self._crypto_trimmed_on = today
            return
        from ..config import CONFIG as _CFG
        weekday_cap = _CFG.risk_max_crypto_exposure_usd
        crypto = [p for p in positions
                  if p.symbol.upper().endswith("USD") and len(p.symbol) <= 9]
        total = sum(abs(float(p.market_value)) for p in crypto)
        if total <= weekday_cap + 1.0:
            self._crypto_trimmed_on = today
            return
        excess = total - weekday_cap
        log.warning("🔻 Crypto weekday-trim: %d positions total $%.0f > weekday cap "
                    "$%.0f — trimming $%.0f for stock/option budget",
                    len(crypto), total, weekday_cap, excess)
        import time as _t_trim
        # Sell from largest positions first until excess is covered.
        for p in sorted(crypto, key=lambda x: abs(float(x.market_value)), reverse=True):
            if excess <= 1.0:
                break
            mv = abs(float(p.market_value))
            qty = abs(float(p.qty))
            sell_mv = min(mv, excess)
            sell_qty = round(sell_mv / mv * qty, 6) if mv > 0 else 0
            if sell_qty <= 0:
                continue
            slashed = _normalize_ticker(p.symbol.upper())
            try:
                self.alpaca.submit_simple_order(
                    symbol=slashed, side="sell", qty=sell_qty,
                    order_type="market", time_in_force="gtc",
                    client_order_id=f"mr-wktrim-{slashed.replace('/','')[:6]}-{_t_trim.time_ns()}",
                )
                log.info("  trimmed %s: sold %.6f (~$%.0f)", slashed, sell_qty, sell_mv)
                excess -= sell_mv
            except AlpacaError as exc:
                log.warning("  crypto trim failed for %s: %s", slashed, exc)
        self._crypto_trimmed_on = today

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
                # Per-position EOD close notification with P/L.
                try:
                    from ..notifications.realtime import TradeAlert, notify_trade
                    notify_trade(TradeAlert(
                        kind="FILLED", symbol=sym, pnl_usd=pnl,
                        notional_usd=abs(float(p.market_value)),
                        qty=abs(float(p.qty)),
                        extra="eod_flatten",
                    ))
                except Exception:  # noqa: BLE001
                    pass
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
                    # Multi-leg market orders aren't supported, and limit_price=None
                    # crashed _round_price → EOD flatten silently failed → spreads
                    # held overnight (2026-06-05 audit bug). Compute the current
                    # spread mark (long_mid - short_mid) and close at that limit,
                    # mirroring the normal exit path; fall back to the entry debit.
                    _close_limit = max(float(sp_d["entry_debit_usd"]), 0.01)
                    try:
                        _q = self.options.get_snapshots(
                            [L["contract_symbol"] for L in legs])
                        _lq = next((_q.get(L["contract_symbol"]) for L in legs
                                    if L["side"] == "buy"), None)
                        _sq = next((_q.get(L["contract_symbol"]) for L in legs
                                    if L["side"] == "sell"), None)
                        if _lq and _sq and _lq.mid > 0 and _sq.mid > 0:
                            _close_limit = max(_lq.mid - _sq.mid, 0.01)
                    except Exception:  # noqa: BLE001 — fall back to entry debit
                        pass
                    mleg = self.options.submit_multi_leg(
                        legs=close_legs, qty=int(sp_d["contracts"]),
                        limit_price=_close_limit,
                        client_order_id=f"mr-eod-opt-{sp_d['id']}-{_t_eod_opt.time_ns()}",
                    )
                    # Mark closed immediately so the poller stops; then book
                    # realized P&L from the REAL close fill (idempotent). If
                    # the market close hasn't filled this instant, the
                    # reconcile backstop books it next iteration (it acts only
                    # while realized_pnl_usd IS NULL, so no double-book).
                    with get_connection() as conn:
                        conn.execute(
                            "UPDATE bot_option_spreads SET closed_at=?, "
                            "exit_reason=COALESCE(exit_reason,'eod_flatten_day_trader') "
                            "WHERE id=? AND closed_at IS NULL",
                            (utc_now(), sp_d["id"]),
                        )
                    self._book_spread_exit(
                        spread_id=sp_d["id"], underlying=sp_d["underlying"],
                        entry_debit=float(sp_d["entry_debit_usd"]),
                        contracts=int(sp_d["contracts"]),
                        legs=legs, exit_reason="eod_flatten_day_trader",
                        close_order_ids=[mleg.id] if getattr(mleg, "id", None) else None,
                        set_closed_at=False,
                    )
                    log.info("  EOD closed day-trader OPT spread %s (exp %s)",
                             sp_d["underlying"], sp_d["expiration_date"])
                except AlpacaError as exc:
                    log.warning("  EOD options close failed for %s: %s",
                                sp_d["underlying"], exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("_eod_flatten_day_trader_options error: %s", exc)

    def _headline_for_order(self, alpaca_order_id: Optional[str] = None,
                             spread_id: Optional[int] = None) -> Optional[str]:
        """Look up the originating signal title for a trade so fill
        notifications can show WHY the bot opened the position.

        Pass either ``alpaca_order_id`` (for stock/crypto bracket orders)
        or ``spread_id`` (for option spread closes). Returns the news
        headline (truncated to 80 chars) or None.
        """
        try:
            with get_connection() as conn:
                if alpaca_order_id:
                    row = conn.execute(
                        """
                        SELECT rs.title
                        FROM bot_orders bo
                        JOIN bot_decisions bd ON bd.alpaca_order_id = bo.alpaca_order_id
                        JOIN signal_scores ss ON ss.id = bd.score_id
                        JOIN raw_signals rs ON rs.id = ss.signal_id
                        WHERE bo.alpaca_order_id = ? OR bd.alpaca_order_id = ?
                        LIMIT 1
                        """,
                        (alpaca_order_id, alpaca_order_id),
                    ).fetchone()
                elif spread_id is not None:
                    row = conn.execute(
                        """
                        SELECT rs.title
                        FROM bot_option_spreads sp
                        JOIN bot_option_decisions od ON od.id = sp.decision_id
                        JOIN signal_scores ss ON ss.id = od.score_id
                        JOIN raw_signals rs ON rs.id = ss.signal_id
                        WHERE sp.id = ?
                        LIMIT 1
                        """,
                        (spread_id,),
                    ).fetchone()
                else:
                    return None
            if not row or not row[0]:
                return None
            t = str(row[0]).strip()
            return t[:80] + ("…" if len(t) > 80 else "")
        except Exception:  # noqa: BLE001
            return None

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
                      AND bo.filled_at > strftime('%Y-%m-%dT%H:%M:%SZ', datetime('now','-30 days'))
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
                      AND sp.closed_at > strftime('%Y-%m-%dT%H:%M:%SZ', datetime('now','-30 days'))
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

        # NEWS-QUALITY BYPASS PATH — uses the shared event-tier constants
        # (single source of truth at module top; kept in sync with the SQL
        # candidate filter + the stock/options gates). SEC 6.0, news 7.0, analyst 7.5.
        news_bypass_qualifies = (
            factual == 1
            and sentiment is not None and abs(float(sentiment)) >= 0.5
            and composite is not None
            and (
                (composite >= 6.0 and event_type in SEC_HIGH_ALPHA_EVENTS)
                or (composite >= 7.0 and event_type in NEWS_MEDIUM_ALPHA_EVENTS)
                or (composite >= 7.5 and event_type in NEWS_LOWER_ALPHA_EVENTS)
            )
        )
        if news_bypass_qualifies:
            s = float(sentiment)
            if s > 0:
                if regime is not None and not regime.allow_longs and not is_crypto:
                    return None
                return "buy"
            if s < 0:
                # Option B: a qualified bearish CATALYST (factual + |sentiment|>=0.5
                # + high-alpha event type) is stock-specific bad news — let it SHORT
                # even in a bullish regime, where allow_shorts=False would otherwise
                # block it. This is how we catch catalyst-driven drops (e.g. AVGO -10%
                # on bad earnings) instead of sitting long-only through a whole bull
                # market. We still honour a PANIC halt (size_multiplier<=0); crypto
                # can't short (blocked downstream); blanket PA/ML shorts stay
                # regime-vetoed below (those fight the trend, this is bad news).
                if (regime is not None and regime.size_multiplier <= 0.0
                        and not is_crypto):
                    return None  # panic — halt everything
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
