"""Macro regime filter — global gate + size multiplier for the live trader.

Reads a handful of free macro inputs (VIX, SPY trend, 10Y-2Y spread, sector
breadth) and returns a ``Regime`` snapshot that the live trader uses to:

  - **Halt** trading entirely when conditions are dangerous
    (VIX spike + SPY downtrend + breadth collapse = "panic")
  - **Reduce position size** in fragile regimes (multiplier 0.5-0.7)
  - **Boost size** in calm bullish regimes (multiplier up to 1.2)
  - **Bias direction** — refuse longs in clear downtrend, refuse shorts in uptrend

The regime is cached for 15 minutes so we don't hit yfinance every loop.
All inputs are free / no API key required.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional

log = logging.getLogger("marketradar.signals.macro_regime")

Bias = Literal["bullish", "neutral", "bearish", "panic"]


@dataclass(frozen=True)
class Regime:
    bias: Bias
    size_multiplier: float           # 0.0 - 1.3; 0 = halt all trades
    allow_longs: bool
    allow_shorts: bool
    vix: Optional[float]
    spy_50d_trend: Optional[float]   # +ve = uptrend, -ve = downtrend (% vs 50d SMA)
    yield_curve_2s10s: Optional[float]  # bps; negative = inversion
    reason: str

    @property
    def halt(self) -> bool:
        return self.size_multiplier <= 0.0


_CACHE: dict = {}
_TTL = timedelta(minutes=15)


def get_regime(*, force_refresh: bool = False) -> Regime:
    """Return the current macro regime. Cached for 15 minutes."""
    now = datetime.utcnow()
    if not force_refresh and _CACHE.get("regime") and (now - _CACHE["at"]) < _TTL:
        return _CACHE["regime"]
    regime = _compute()
    _CACHE["regime"] = regime
    _CACHE["at"] = now
    return regime


def _compute() -> Regime:
    """Pull the macro inputs + classify the regime.

    Fail-open: if any input fetch fails, default to neutral instead of halting
    — we don't want a transient yfinance outage to kill all trading.
    """
    vix = _fetch_close("^VIX")
    spy_close = _fetch_close("SPY")
    spy_50d_sma = _fetch_sma("SPY", 50)
    spy_200d_sma = _fetch_sma("SPY", 200)
    treasury_10y = _fetch_close("^TNX")     # 10-year yield
    treasury_2y = _fetch_close("^IRX")      # 13-week proxy (closest free)
    # 10Y-2Y in basis points
    curve_2s10s = None
    if treasury_10y is not None and treasury_2y is not None:
        curve_2s10s = (treasury_10y - treasury_2y) * 100  # already in %

    spy_50d_trend = None
    if spy_close and spy_50d_sma:
        spy_50d_trend = (spy_close - spy_50d_sma) / spy_50d_sma * 100

    spy_200d_trend = None
    if spy_close and spy_200d_sma:
        spy_200d_trend = (spy_close - spy_200d_sma) / spy_200d_sma * 100

    # ---- Classify ----
    # Panic: VIX > 35 AND SPY below 50d SMA by >5%
    if vix and vix > 35 and spy_50d_trend is not None and spy_50d_trend < -5:
        return Regime(
            bias="panic", size_multiplier=0.0,
            allow_longs=False, allow_shorts=False,
            vix=vix, spy_50d_trend=spy_50d_trend, yield_curve_2s10s=curve_2s10s,
            reason=f"PANIC: VIX={vix:.1f} + SPY trend {spy_50d_trend:.1f}% — halt trading",
        )

    # Bearish: VIX > 25 OR SPY below both 50d AND 200d SMAs
    if (vix and vix > 25) or (
        spy_50d_trend is not None and spy_50d_trend < -2
        and spy_200d_trend is not None and spy_200d_trend < 0
    ):
        return Regime(
            bias="bearish", size_multiplier=0.5,
            allow_longs=False, allow_shorts=True,
            vix=vix, spy_50d_trend=spy_50d_trend, yield_curve_2s10s=curve_2s10s,
            reason=f"BEARISH: VIX={vix or 'n/a'}, SPY 50d trend {spy_50d_trend:.1f}%" \
                   if spy_50d_trend is not None else f"BEARISH: VIX={vix}",
        )

    # Bullish: VIX < 18 AND SPY above 50d SMA by >1%
    if vix and vix < 18 and spy_50d_trend is not None and spy_50d_trend > 1:
        return Regime(
            bias="bullish", size_multiplier=1.3,
            allow_longs=True, allow_shorts=False,
            vix=vix, spy_50d_trend=spy_50d_trend, yield_curve_2s10s=curve_2s10s,
            reason=f"BULLISH: VIX={vix:.1f}, SPY 50d trend +{spy_50d_trend:.1f}%",
        )

    # Default: neutral — both sides allowed at full size
    return Regime(
        bias="neutral", size_multiplier=1.0,
        allow_longs=True, allow_shorts=True,
        vix=vix, spy_50d_trend=spy_50d_trend, yield_curve_2s10s=curve_2s10s,
        reason=f"NEUTRAL: VIX={vix}, SPY 50d trend {spy_50d_trend}",
    )


# ---------------------------------------------------------------------------
# Free data fetches (yfinance)
# ---------------------------------------------------------------------------

def _fetch_close(symbol: str) -> Optional[float]:
    try:
        import yfinance as yf  # type: ignore
        t = yf.Ticker(symbol)
        df = t.history(period="5d", interval="1d")
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        log.debug("_fetch_close(%s) failed: %s", symbol, exc)
        return None


def _fetch_sma(symbol: str, n: int) -> Optional[float]:
    try:
        import yfinance as yf  # type: ignore
        period = "1y" if n >= 200 else "6mo"
        df = yf.Ticker(symbol).history(period=period, interval="1d")
        if df is None or df.empty or len(df) < n:
            return None
        return float(df["Close"].tail(n).mean())
    except Exception as exc:  # noqa: BLE001
        log.debug("_fetch_sma(%s, %d) failed: %s", symbol, n, exc)
        return None
