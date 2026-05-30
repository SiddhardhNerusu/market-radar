"""Position sizing + stop/take-profit calculator.

Given a calibrated probability + entry price + recent volatility, return:

  - ``qty``         — share count to submit
  - ``size_pct``    — % of account equity allocated
  - ``stop_loss``   — protective stop price
  - ``take_profit`` — profit target price

Sizing rule: **fractional Kelly, capped**.

  f* = (p * b - q) / b
        where  b = TP_distance / SL_distance   (reward:risk in $ terms)
               q = 1 - p

We multiply f* by ``KELLY_FRACTION`` (default 0.25 — "quarter Kelly", the
empirical sweet spot that survives the volatile-edge regime) and cap by
``CONFIG.risk_max_position_pct``. Quarter-Kelly retains ~75% of full-Kelly
growth but cuts drawdown variance by roughly 4×.

Stops use ATR(14):
  - buy:  stop = entry - SL_ATR_MULT * ATR,  TP = entry + TP_ATR_MULT * ATR
  - sell: stop = entry + SL_ATR_MULT * ATR,  TP = entry - TP_ATR_MULT * ATR

Defaults (SL=1.5×, TP=2.5×) give a 1:1.67 R:R. With min calibrated p≥0.62
this has a positive expected value:
    EV = p * 2.5 - (1-p) * 1.5 = 0.62*2.5 - 0.38*1.5 = +0.98 ATR per trade.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional

from ..config import CONFIG

log = logging.getLogger("marketradar.execution.sizer")

Direction = Literal["buy", "sell"]

# Tunable constants — sane defaults validated against literature.
KELLY_FRACTION = 0.25            # quarter-Kelly
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.5
MIN_NOTIONAL_USD = 50.0          # below this, the round-down to whole shares is too lossy
MIN_QTY = 1.0                    # integer-share path; switch to fractional w/ Alpaca if needed


@dataclass(frozen=True)
class SizingResult:
    qty: float
    size_pct: float            # of equity, 0-100
    notional_usd: float
    entry_estimate: float
    stop_loss: float
    take_profit: float
    atr: float
    kelly_raw: float           # full-Kelly fraction (before quarter / cap)
    reason: str                # 'sized', 'kelly_negative', 'qty_too_small', …
    tradeable: bool


def size_trade(
    *,
    direction: Direction,
    entry_price: float,
    atr: float,
    calibrated_p: float,
    account_equity_usd: float,
    max_position_pct: Optional[float] = None,
    kelly_fraction: float = KELLY_FRACTION,
    sl_atr_mult: float = SL_ATR_MULT,
    tp_atr_mult: float = TP_ATR_MULT,
    min_qty: float = MIN_QTY,
    allow_fractional: bool = False,
    true_equity_usd: Optional[float] = None,
) -> SizingResult:
    """Return a sized, bracketed proposal — or a non-tradeable explanation.

    ``account_equity_usd`` may be multiplier-inflated (conviction × regime ×
    learning × earnings). ``true_equity_usd`` is the un-inflated equity used
    to enforce the HARD per-ticker cap: multipliers scale size TOWARD the cap
    but can never push the final notional past max_pct of TRUE equity. Without
    this clamp the sizer produced e.g. 7.8% positions that the risk manager
    then rejected ("Ticker exposure exceeds per-ticker cap 6.0%"). Falls back
    to account_equity_usd when not provided (legacy behavior).
    """
    max_pct = (
        max_position_pct if max_position_pct is not None
        else CONFIG.risk_max_position_pct
    )
    cap_equity = true_equity_usd if true_equity_usd is not None else account_equity_usd

    if entry_price <= 0:
        return _untradeable("entry_price <= 0", entry_price, atr, 0.0)
    if atr <= 0:
        return _untradeable("atr <= 0 — no volatility estimate",
                            entry_price, atr, 0.0)
    if account_equity_usd <= 0:
        return _untradeable("account_equity_usd <= 0", entry_price, atr, 0.0)

    # Stops + targets.
    if direction == "buy":
        stop_loss = entry_price - sl_atr_mult * atr
        take_profit = entry_price + tp_atr_mult * atr
    else:  # sell / short
        stop_loss = entry_price + sl_atr_mult * atr
        take_profit = entry_price - tp_atr_mult * atr

    sl_distance = abs(entry_price - stop_loss)
    tp_distance = abs(take_profit - entry_price)
    if sl_distance <= 0 or tp_distance <= 0:
        return _untradeable("degenerate stop/TP distance",
                            entry_price, atr, 0.0)

    b = tp_distance / sl_distance           # reward:risk
    p = float(calibrated_p)
    q = 1.0 - p
    kelly_raw = (p * b - q) / b

    if kelly_raw <= 0:
        return _untradeable(
            f"kelly_raw={kelly_raw:.3f} <= 0 (p={p:.2f}, b={b:.2f}) — "
            "edge does not survive R:R",
            entry_price, atr, kelly_raw,
        )

    size_pct = min(kelly_raw * kelly_fraction * 100.0, max_pct)
    notional = account_equity_usd * (size_pct / 100.0)
    # HARD CAP: never let multiplier-inflated equity push the position past
    # max_pct of TRUE equity (what the risk manager's per-ticker rule checks).
    # Multipliers scale size toward the cap, never beyond it.
    hard_cap_notional = cap_equity * (max_pct / 100.0)
    if notional > hard_cap_notional:
        notional = hard_cap_notional
        size_pct = (notional / cap_equity) * 100.0 if cap_equity > 0 else size_pct
    qty = notional / entry_price

    # Integer shares for stocks (bracket orders require whole shares).
    # Fractional permitted for crypto path (no bracket on crypto anyway).
    if not allow_fractional:
        qty = max(0.0, float(int(qty)))
    else:
        qty = max(0.0, round(qty, 6))  # 6 decimals = Alpaca crypto precision
    if qty < min_qty:
        return SizingResult(
            qty=0.0, size_pct=size_pct, notional_usd=notional,
            entry_estimate=entry_price, stop_loss=stop_loss,
            take_profit=take_profit, atr=atr, kelly_raw=kelly_raw,
            reason=(f"qty {qty:.2f} < min {min_qty:.1f} "
                    f"(notional ${notional:.0f}, entry ${entry_price:.2f})"),
            tradeable=False,
        )
    notional = qty * entry_price
    if notional < MIN_NOTIONAL_USD:
        return SizingResult(
            qty=qty, size_pct=size_pct, notional_usd=notional,
            entry_estimate=entry_price, stop_loss=stop_loss,
            take_profit=take_profit, atr=atr, kelly_raw=kelly_raw,
            reason=f"notional ${notional:.0f} < min ${MIN_NOTIONAL_USD:.0f}",
            tradeable=False,
        )

    # Cap actual_pct just under max_position_pct to avoid floating-point
    # equality bumps tripping the risk manager's strict >cap check.
    actual_pct = min(notional / account_equity_usd * 100.0, max_pct - 0.001)
    return SizingResult(
        qty=qty,
        size_pct=actual_pct,
        notional_usd=notional,
        entry_estimate=entry_price,
        stop_loss=round(stop_loss, 4),
        take_profit=round(take_profit, 4),
        atr=atr,
        kelly_raw=kelly_raw,
        reason="sized",
        tradeable=True,
    )


def _untradeable(reason: str, entry: float, atr: float,
                 kelly_raw: float) -> SizingResult:
    return SizingResult(
        qty=0.0, size_pct=0.0, notional_usd=0.0,
        entry_estimate=entry, stop_loss=0.0, take_profit=0.0,
        atr=atr, kelly_raw=kelly_raw, reason=reason, tradeable=False,
    )


# ---------------------------------------------------------------------------
# ATR helper — pulled from yfinance daily bars (no API key needed).
# Cached in-process across the trader loop. Refreshes once per trading day.
# ---------------------------------------------------------------------------

_ATR_CACHE: dict[str, tuple[datetime, float]] = {}
_ATR_TTL = timedelta(hours=6)


def get_atr(ticker: str, *, period: int = 14, lookback_days: int = 30) -> Optional[float]:
    """Return ATR(``period``) for ``ticker`` in USD per share.

    Routes to Alpaca's crypto bars endpoint when ``ticker`` contains '/'
    (yfinance doesn't support pair-format crypto symbols). Stocks use yfinance.

    Returns None if data can't be fetched. Cached for ``_ATR_TTL``.
    """
    now = datetime.utcnow()
    cached = _ATR_CACHE.get(ticker)
    if cached and (now - cached[0]) < _ATR_TTL:
        return cached[1]

    is_crypto = "/" in ticker
    if is_crypto:
        atr_val = _crypto_atr(ticker, period=period, lookback_days=lookback_days)
    else:
        atr_val = _stock_atr_yfinance(ticker, period=period, lookback_days=lookback_days)

    if atr_val is None or atr_val <= 0:
        return None
    _ATR_CACHE[ticker] = (now, atr_val)
    return atr_val


def _stock_atr_yfinance(ticker: str, *, period: int, lookback_days: int) -> Optional[float]:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        log.warning("yfinance not installed — ATR unavailable")
        return None
    try:
        end = datetime.utcnow().date() + timedelta(days=1)
        start = end - timedelta(days=lookback_days + 5)
        df = yf.download(
            ticker, start=start.isoformat(), end=end.isoformat(),
            progress=False, auto_adjust=False, threads=False,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("yfinance download(%s) failed: %s", ticker, exc)
        return None
    if df is None or df.empty or len(df) < period + 1:
        return None
    # yfinance occasionally returns MultiIndex columns — flatten defensively.
    def _to_list(col):
        s = df[col]
        try:
            return s.values.flatten().astype(float).tolist()
        except Exception:  # noqa: BLE001
            return [float(x) for x in s]
    return _atr_from_ohlc(_to_list("High"), _to_list("Low"), _to_list("Close"), period)


def _crypto_atr(pair: str, *, period: int, lookback_days: int) -> Optional[float]:
    """Crypto ATR via Alpaca daily bars (yfinance doesn't have pair-format)."""
    try:
        from .alpaca_client import AlpacaClient, AlpacaError
        c = AlpacaClient()
    except Exception as exc:  # noqa: BLE001
        log.debug("crypto ATR alpaca init failed: %s", exc)
        return None
    from datetime import datetime as _dt, timezone as _tz
    end = _dt.now(_tz.utc)
    start = end - timedelta(days=lookback_days + 5)
    params = {
        "symbols": pair, "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 100,
    }
    try:
        d = c._request("GET", "/v1beta3/crypto/us/bars",
                       params=params, base=c.data_base_url)
    except Exception as exc:  # noqa: BLE001
        log.warning("crypto ATR bars(%s) failed: %s", pair, exc)
        return None
    bars = (d or {}).get("bars", {}).get(pair) or []
    if len(bars) < period + 1:
        return None
    highs = [b["h"] for b in bars]
    lows = [b["l"] for b in bars]
    closes = [b["c"] for b in bars]
    return _atr_from_ohlc(highs, lows, closes, period)


def _atr_from_ohlc(highs, lows, closes, period: int) -> Optional[float]:
    """Pure-Python ATR computation. Works for stocks (pandas Series → tolist)
    or crypto (Alpaca bars → list directly)."""
    if len(highs) < period + 1:
        return None
    trs = []
    for i in range(-period, 0):
        h, l = highs[i], lows[i]
        pc = closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    avg = sum(trs) / period
    return float(avg) if avg > 0 else None
