"""Real-time notifier — Telegram + macOS native banner.

Fires alerts when STRONG BUY / STRONG SELL signals appear. Owner sees
them on their phone (Telegram) and Mac (banner) within seconds.  There
is NO auto-trade — owner manually executes in the T212 app.

Per-ticker cooldown + daily cap prevent spam.  Logs every send +
its latency (ingest → notify) for monitoring.

Public API:
    from market_radar.notifications.realtime import Notifier, AlertCandidate
    r = Notifier().notify_if_eligible(AlertCandidate(
        signal_id=123, ticker="AAPL", direction="buy", action="STRONG_BUY",
        p=0.78, title="...", source="sec_edgar",
        ingested_at="2026-05-14T15:00:00Z",
    ))
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from ..config import CONFIG
from ..storage import get_connection

log = logging.getLogger(__name__)
TELEGRAM_API = "https://api.telegram.org"


@dataclass(frozen=True)
class AlertCandidate:
    signal_id: int
    ticker: str
    direction: str          # 'buy' | 'sell'
    action: str             # 'STRONG_BUY' | 'STRONG_SELL'
    p: float
    title: str
    source: str
    ingested_at: str        # ISO 8601 Z


@dataclass(frozen=True)
class SendResult:
    eligible: bool
    sent: bool
    reason: str
    channels_sent: list[str]
    latency_seconds: Optional[float]


class Notifier:
    """Handles eligibility, delivery, and record-keeping for real-time alerts."""

    def notify_if_eligible(self, c: AlertCandidate) -> SendResult:
        # 1. Threshold check (caller may have pre-filtered, but verify)
        if c.direction == "buy" and c.p < CONFIG.notify_buy_threshold:
            return SendResult(False, False,
                              f"p={c.p:.3f} below buy threshold "
                              f"{CONFIG.notify_buy_threshold}", [], None)
        if c.direction == "sell" and c.p > CONFIG.notify_sell_threshold:
            return SendResult(False, False,
                              f"p={c.p:.3f} above sell threshold "
                              f"{CONFIG.notify_sell_threshold}", [], None)
        if c.direction not in ("buy", "sell"):
            return SendResult(False, False,
                              f"direction {c.direction!r} invalid", [], None)

        # 2. Per-ticker + per-direction cooldown
        if self._fired_recently(c.ticker, c.direction):
            return SendResult(False, False,
                              "cooldown active for this ticker+direction",
                              [], None)

        # 3. Daily cap
        if self._daily_count() >= CONFIG.notify_max_daily:
            return SendResult(False, False,
                              f"daily cap {CONFIG.notify_max_daily} hit",
                              [], None)

        # 4. Build + send
        msg_telegram = self._format_telegram(c)
        msg_macos = self._format_macos(c)
        channels: list[str] = []
        if CONFIG.telegram_bot_token and CONFIG.telegram_chat_id:
            if self._send_telegram(msg_telegram):
                channels.append("telegram")
        if CONFIG.notify_macos_banner:
            if self._send_macos(c.action, msg_macos):
                channels.append("macos")

        sent = len(channels) > 0
        latency = self._compute_latency(c.ingested_at)

        # 5. Record-keep — always, even on partial fail
        self._record(c, channels, latency)

        if sent:
            log.info("Notified %s %s p=%.3f via %s latency=%s",
                     c.action, c.ticker, c.p, channels,
                     f"{latency:.1f}s" if latency is not None else "?")
        else:
            log.warning(
                "Notification eligible but no channel delivered: %s",
                c.ticker,
            )
        return SendResult(True, sent,
                          "all channels failed" if not sent else "sent",
                          channels, latency)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    def _format_telegram(self, c: AlertCandidate) -> str:
        emoji = "🟢" if c.direction == "buy" else "🔴"
        title = self._html_escape((c.title or "")[:120])
        return (
            f"{emoji} <b>{c.action.replace('_', ' ')}</b>\n"
            f"<b>${c.ticker}</b> · p = {c.p * 100:.1f}%\n"
            f"<i>{title}</i>\n"
            f"source: {self._html_escape(c.source)}\n"
            f"→ open http://127.0.0.1:8765/ for details"
        )

    def _format_macos(self, c: AlertCandidate) -> str:
        sign = "▲" if c.direction == "buy" else "▼"
        return f"{sign} {c.ticker} {c.action.replace('_', ' ')} p={c.p * 100:.0f}%"

    @staticmethod
    def _html_escape(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------
    def _send_telegram(self, body: str) -> bool:
        try:
            url = f"{TELEGRAM_API}/bot{CONFIG.telegram_bot_token}/sendMessage"
            r = requests.post(url, json={
                "chat_id": CONFIG.telegram_chat_id,
                "text": body,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }, timeout=10)
            if r.status_code != 200:
                log.warning("Telegram send failed: %s %s",
                            r.status_code, r.text[:200])
                return False
            return True
        except requests.RequestException as exc:
            log.warning("Telegram send error: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001 — never crash on notify
            log.warning("Telegram unexpected error: %s", exc)
            return False

    def _send_macos(self, subtitle: str, body: str) -> bool:
        try:
            # Quote-safe escape for AppleScript string literals.
            def esc(s: str) -> str:
                return (s or "").replace("\\", "\\\\").replace('"', '\\"')
            script = (
                f'display notification "{esc(body)}" '
                f'with title "MARKET RADAR" '
                f'subtitle "{esc(subtitle.replace("_", " "))}" '
                f'sound name "Glass"'
            )
            subprocess.run(
                ["osascript", "-e", script], check=False, timeout=5,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("macOS banner failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def _fired_recently(self, ticker: str, direction: str) -> bool:
        cutoff = (datetime.now(timezone.utc) -
                  timedelta(minutes=CONFIG.notify_per_ticker_cooldown_min))
        cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT id FROM notifications_sent "
                    "WHERE ticker = ? AND direction = ? AND sent_at >= ? LIMIT 1",
                    (ticker.upper(), direction, cutoff_iso),
                ).fetchone()
            return row is not None
        except Exception as exc:  # noqa: BLE001
            log.warning("cooldown check failed: %s — treating as cooled down", exc)
            return False

    def _daily_count(self) -> int:
        try:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM notifications_sent "
                    "WHERE sent_at >= datetime('now','start of day')"
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception as exc:  # noqa: BLE001
            log.warning("daily count failed: %s — treating as 0", exc)
            return 0

    def _compute_latency(self, ingested_at: str) -> Optional[float]:
        if not ingested_at:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
            try:
                ts = datetime.strptime(ingested_at, fmt).replace(tzinfo=timezone.utc)
                return (datetime.now(timezone.utc) - ts).total_seconds()
            except ValueError:
                continue
        return None

    def _record(self, c: AlertCandidate, channels: list[str],
                latency: Optional[float]) -> None:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        channels_csv = ",".join(channels)
        try:
            with get_connection() as conn:
                # Legacy schema has score_id NOT NULL and channel NOT NULL.
                # Populate both legacy and new columns so the row is valid.
                conn.execute(
                    """INSERT INTO notifications_sent
                    (score_id, channel, sent_at,
                     signal_id, ticker, direction, action, p,
                     channels_sent, ingested_at, latency_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        int(c.signal_id), channels_csv or "none", now_iso,
                        int(c.signal_id),
                        c.ticker.upper(), c.direction, c.action, float(c.p),
                        channels_csv, c.ingested_at, latency,
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to record notification for %s: %s",
                        c.ticker, exc)


# ===========================================================================
# Bot trade alerts — added 2026-05-27 for the live execution layer.
# Lighter-weight than the AlertCandidate path (no DB cooldown bookkeeping),
# because trade events are inherently rate-limited by the trader loop's
# max_candidates_per_loop cap.
# ===========================================================================

@dataclass(frozen=True)
class TradeAlert:
    kind: str                  # 'PLACED' | 'RISK_BLOCKED' | 'FILLED' | 'PNL_DAY'
    symbol: str
    direction: str = ""
    qty: float = 0.0
    price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    notional_usd: float = 0.0
    pnl_usd: float = 0.0
    extra: str = ""


_TRADE_DEDUP: dict[str, datetime] = {}
_TRADE_DEDUP_TTL = timedelta(seconds=90)


def notify_trade(alert: TradeAlert) -> bool:
    """Fire a Telegram + macOS notification for a bot trade event.

    Dedup window: within ``_TRADE_DEDUP_TTL`` (90s) the same
    (kind, symbol, qty, price, pnl_usd) tuple is silently suppressed —
    protects against the reconciliation loop sending the same FILL twice
    in back-to-back iterations.

    Returns True if at least one channel delivered. Never raises.
    """
    try:
        key = "|".join([
            alert.kind, alert.symbol.upper(), alert.direction,
            f"{alert.qty:.2f}", f"{alert.price:.2f}",
            f"{alert.pnl_usd:.2f}",
        ])
        now = datetime.utcnow()
        last = _TRADE_DEDUP.get(key)
        if last and (now - last) < _TRADE_DEDUP_TTL:
            log.debug("notify_trade dedup hit: %s", key)
            return False
        _TRADE_DEDUP[key] = now
        if len(_TRADE_DEDUP) > 1000:
            cutoff = now - _TRADE_DEDUP_TTL
            for k, t in list(_TRADE_DEDUP.items()):
                if t < cutoff:
                    _TRADE_DEDUP.pop(k, None)

        body = _format_trade(alert)
        sent_any = False
        if CONFIG.telegram_bot_token and CONFIG.telegram_chat_id:
            sent_any |= _send_telegram_simple(body)
        if CONFIG.notify_macos_banner:
            sent_any |= _send_macos_simple(alert.kind, alert.symbol, body)
        return sent_any
    except Exception as exc:  # noqa: BLE001
        log.warning("notify_trade(%s %s) failed: %s",
                    alert.kind, alert.symbol, exc)
        return False


def _format_trade(a: TradeAlert) -> str:
    """Telegram format. FILL notifications LEAD with symbol + P/L so the
    user sees "AMZN +$100" at a glance — the user-requested format."""
    # Profit/loss-aware emojis — win vs loss visually distinct.
    is_profit = a.pnl_usd > 0
    is_loss = a.pnl_usd < 0
    emojis = {
        "PLACED":       "🟢" if a.direction == "buy" else "🔴",
        "OPTIONS_PLACED": "💎",
        "OPTIONS_SUBMITTED": "📨",
        "OPTIONS_OPENED": "💎",
        "RISK_BLOCKED": "🛑",
        "FILLED":       "✅" if is_profit else ("❌" if is_loss else "⚪"),
        "OPTIONS_FILLED": "✅" if is_profit else ("❌" if is_loss else "💎"),
        "PNL_DAY":      "📊" if a.pnl_usd >= 0 else "📉",
        "REGIME_HALT":  "⚠️",
    }
    emoji = emojis.get(a.kind, "•")
    if a.kind in ("PLACED", "OPTIONS_PLACED", "OPTIONS_SUBMITTED"):
        return (
            f"{emoji} <b>{a.kind} {a.direction.upper()} {a.symbol}</b>\n"
            f"qty: {a.qty:g}  @ ${a.price:.2f}\n"
            f"SL: ${a.stop_loss:.2f}  TP: ${a.take_profit:.2f}\n"
            f"notional: ${a.notional_usd:,.0f}\n"
            f"{a.extra}"
        ).strip()
    if a.kind == "OPTIONS_OPENED":
        # Entry confirmation for option spreads — no P/L yet.
        return (
            f"{emoji} <b>{a.symbol.upper()} OPEN {a.direction.upper()}</b>\n"
            f"{a.qty:g} contracts · {a.extra}"
        ).strip()
    if a.kind == "RISK_BLOCKED":
        return (
            f"{emoji} <b>RISK BLOCKED {a.symbol}</b>\n"
            f"{a.direction.upper()} blocked by risk manager\n"
            f"{a.extra}"
        ).strip()
    if a.kind in ("FILLED", "OPTIONS_FILLED"):
        sign = "+" if a.pnl_usd >= 0 else ""
        # P/L is the headline. Extra context goes below.
        # Format: "✅ AMZN  +$100.00" or "❌ TSLA  -$45.00"
        # Inline: "OPTIONS_FILLED" → "AMZN spread" to make it clear
        label = "spread" if a.kind == "OPTIONS_FILLED" else ""
        line1 = (
            f"{emoji} <b>{a.symbol.upper()} {label} "
            f"{sign}${a.pnl_usd:,.2f}</b>"
        ).replace("  ", " ").strip()
        details: list[str] = []
        if a.notional_usd:
            details.append(f"size ${a.notional_usd:,.0f}")
        if a.qty and a.qty > 0:
            details.append(f"qty {a.qty:g}")
        if a.extra:
            details.append(a.extra)
        line2 = " · ".join(details) if details else ""
        return (line1 + ("\n" + line2 if line2 else "")).strip()
    if a.kind == "PNL_DAY":
        sign = "+" if a.pnl_usd >= 0 else ""
        return (
            f"{emoji} <b>Daily P&L: {sign}${a.pnl_usd:,.2f}</b>\n"
            f"{a.extra}"
        ).strip()
    if a.kind == "REGIME_HALT":
        return f"{emoji} <b>REGIME HALT</b>\n{a.extra}"
    return f"{emoji} {a.kind} {a.symbol} {a.extra}"


def _send_telegram_simple(body: str) -> bool:
    try:
        url = f"{TELEGRAM_API}/bot{CONFIG.telegram_bot_token}/sendMessage"
        r = requests.post(url, json={
            "chat_id": CONFIG.telegram_chat_id,
            "text": body, "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=10)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _send_macos_simple(kind: str, symbol: str, body: str) -> bool:
    try:
        # Strip HTML for macOS banner
        plain = (body
                 .replace("<b>", "").replace("</b>", "")
                 .replace("<i>", "").replace("</i>", ""))
        def esc(s: str) -> str:
            return (s or "").replace("\\", "\\\\").replace('"', '\\"')
        script = (
            f'display notification "{esc(plain)}" '
            f'with title "MARKET RADAR · {esc(kind)}" '
            f'subtitle "{esc(symbol)}" '
            f'sound name "Glass"'
        )
        subprocess.run(["osascript", "-e", script], check=False, timeout=5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:  # noqa: BLE001
        return False
