"""TA-Lib-style indicator pack — RSI, MACD, Bollinger position, ATR, ADX,
VWAP distance, breakout flag.

All computed in pure Python from the OHLC + Volume series already cached
in ``MarketFeatureCache``. No native dependency on the TA-Lib C library.

These features are most informative for trending markets (per 2025
quantified-strategies research, win rate 65-73% in trends vs 45-55% in
chop). The model receives them all and the regularised HGB decides how
much to lean on each.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class TAFeatures:
    rsi_14:            Optional[float] = None
    macd_histogram:    Optional[float] = None
    macd_above_signal: Optional[float] = None
    bb_position_20d:   Optional[float] = None
    atr_pct:           Optional[float] = None
    adx_14:            Optional[float] = None
    vwap_distance_pct: Optional[float] = None
    breakout_20d:      Optional[float] = None


def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, period + 1):
        delta = closes[-i] - closes[-i - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-delta)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(values: list[float], period: int) -> Optional[list[float]]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def _macd(closes: list[float]) -> tuple[Optional[float], Optional[float]]:
    """Return (histogram, above_signal)."""
    if len(closes) < 35:
        return None, None
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    if ema12 is None or ema26 is None:
        return None, None
    # Align: ema12 is longer than ema26 by 14 entries (26-12)
    offset = len(ema12) - len(ema26)
    macd_line = [a - b for a, b in zip(ema12[offset:], ema26)]
    signal_line = _ema(macd_line, 9)
    if signal_line is None or len(signal_line) < 1:
        return None, None
    aligned_macd = macd_line[-len(signal_line):]
    hist = aligned_macd[-1] - signal_line[-1]
    above = 1.0 if aligned_macd[-1] > signal_line[-1] else 0.0
    return hist, above


def _bollinger_position(closes: list[float], period: int = 20) -> Optional[float]:
    if len(closes) < period:
        return None
    window = closes[-period:]
    mu = sum(window) / period
    var = sum((c - mu) ** 2 for c in window) / period
    sd = math.sqrt(var)
    if sd == 0:
        return 0.5
    upper = mu + 2 * sd
    lower = mu - 2 * sd
    cur = closes[-1]
    return (cur - lower) / (upper - lower)


def _atr_pct(highs: list[float], lows: list[float], closes: list[float],
             period: int = 14) -> Optional[float]:
    n = len(closes)
    if n < period + 1:
        return None
    trs: list[float] = []
    for i in range(n - period, n):
        if i <= 0:
            continue
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if not trs or closes[-1] <= 0:
        return None
    atr = sum(trs) / len(trs)
    return atr / closes[-1] * 100.0


def _adx(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> Optional[float]:
    n = len(closes)
    if n < 2 * period + 1:
        return None
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    trs: list[float] = []
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)

    def _smoothed(seq: list[float]) -> float:
        # Wilder's smoothing approx
        if len(seq) < period:
            return 0.0
        rolling = sum(seq[:period])
        for v in seq[period:]:
            rolling = rolling - rolling / period + v
        return rolling

    smoothed_tr = _smoothed(trs)
    if smoothed_tr == 0:
        return None
    smoothed_plus = _smoothed(plus_dm)
    smoothed_minus = _smoothed(minus_dm)
    plus_di = 100.0 * smoothed_plus / smoothed_tr
    minus_di = 100.0 * smoothed_minus / smoothed_tr
    if (plus_di + minus_di) == 0:
        return None
    dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx


def _vwap_distance(closes: list[float], vols: list[float]) -> Optional[float]:
    """Approximate VWAP over the entire window we have (typically 30 days)."""
    if len(closes) < 5 or len(vols) < 5:
        return None
    total_v = sum(vols)
    if total_v <= 0:
        return None
    vwap = sum(c * v for c, v in zip(closes, vols)) / total_v
    if vwap <= 0:
        return None
    return (closes[-1] - vwap) / vwap * 100.0


def _breakout_20d(closes: list[float]) -> Optional[float]:
    if len(closes) < 21:
        return None
    prior = closes[-21:-1]
    if not prior:
        return None
    hi = max(prior)
    lo = min(prior)
    cur = closes[-1]
    if cur > hi:
        return 1.0
    if cur < lo:
        return -1.0
    return 0.0


def features_from_cache(market_cache, ticker: str, when: date) -> TAFeatures:
    """Compute the TA pack for one (ticker, date) from the market cache.

    Falls back gracefully (None fields) if the cache lacks enough history.
    The model will see defaults via ``extract_features``.
    """
    out = TAFeatures()
    # Normalise to the same shape MarketFeatureCache uses
    sym = ticker.strip().upper()
    if "_" in sym:
        parts = sym.split("_")
        if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
            sym = "_".join(parts[:-2]).replace("_", "-")

    df = market_cache.by_ticker.get(sym) if market_cache else None
    if df is None or df.empty or "Close" not in df.columns:
        return out

    try:
        mask = df.index.date <= when
        positions = (mask.nonzero()[0] if hasattr(mask, "nonzero")
                     else [i for i, b in enumerate(mask) if b])
        if not len(positions):
            return out
        end = int(positions[-1]) + 1
        start = max(0, end - 60)
        window = df.iloc[start:end]
    except Exception:  # noqa: BLE001
        return out

    closes = [float(c) for c in window["Close"].tolist()]
    # Highs/Lows aren't in the cache today — approximate with daily close
    highs = closes
    lows = closes
    vols = ([float(v) for v in window["Volume"].tolist()]
            if "Volume" in window.columns else [1.0] * len(closes))

    if not closes:
        return out

    out.rsi_14 = _rsi(closes, 14)
    hist, above = _macd(closes)
    out.macd_histogram = hist
    out.macd_above_signal = above
    out.bb_position_20d = _bollinger_position(closes, 20)
    out.atr_pct = _atr_pct(highs, lows, closes, 14)
    out.adx_14 = _adx(highs, lows, closes, 14)
    out.vwap_distance_pct = _vwap_distance(closes, vols)
    out.breakout_20d = _breakout_20d(closes)
    return out


def attach_ta_features(rows: list[dict], market_cache) -> None:
    """In-place attach TA pack to each row."""
    from datetime import datetime
    for r in rows:
        ticker = (r.get("ticker") or "").upper()
        ts = r.get("price_at_flag_ts") or r.get("published_at") or r.get("scored_at")
        if not ticker or not isinstance(ts, str):
            continue
        try:
            d = datetime.strptime(ts[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        ta = features_from_cache(market_cache, ticker, d)
        r["rsi_14"]            = ta.rsi_14
        r["macd_histogram"]    = ta.macd_histogram
        r["macd_above_signal"] = ta.macd_above_signal
        r["bb_position_20d"]   = ta.bb_position_20d
        r["atr_pct"]           = ta.atr_pct
        r["adx_14"]            = ta.adx_14
        r["vwap_distance_pct"] = ta.vwap_distance_pct
        r["breakout_20d"]      = ta.breakout_20d
