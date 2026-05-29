"""Currency detection + USD conversion for T212 positions.

T212 quotes some instruments in pence (GBp). Without conversion these
look like dollar amounts to the rest of the system, inflating gross
exposure by ~150x for those tickers.

This module:
  1. Detects the quote currency from the ticker shape.
  2. Fetches FX rates (cached for 6 hours) via yfinance.
  3. Converts any native price to true USD.

T212 ticker conventions observed in the live DB:
  AAPL_US_EQ        → USD (3-part, exchange='US')
  EQGBl_EQ          → GBp pence  (2-part, symbol ends lowercase 'l')
  EQGB_EQ           → GBP pounds (2-part, no 'l' suffix)
  TSLA_DE_EQ        → EUR        (3-part, exchange='DE')
  unrecognised      → None (caller fails closed)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# Cache: {(from_ccy, "USD"): (rate, fetched_at_epoch)}
_FX_CACHE: dict[tuple[str, str], tuple[float, float]] = {}
_FX_TTL_SECONDS = 6 * 3600

_LSE_EXCHANGES = {"LSE", "GB", "UK"}
_US_EXCHANGES = {"US", "NASDAQ", "NYSE"}
_EU_EXCHANGES = {"DE", "FR", "IT", "ES", "NL", "BE", "PT", "AT", "FI",
                 "IE", "EU", "LU"}
_FX_CHF = {"CH"}
_FX_CAD = {"CA", "TSX"}
_FX_JPY = {"JP", "TSE"}
_FX_AUD = {"AU", "ASX"}


def detect_quote_currency(ticker: str) -> Optional[str]:
    """Return 'USD', 'GBP', 'GBp' (pence), 'EUR', etc., or None.

    Case matters for the pence detection: T212 marks pence-denominated
    UK instruments with a *lowercase* 'l' at the end of the symbol
    portion (e.g. ``EQGBl_EQ``).
    """
    if not ticker:
        return None
    parts = ticker.split("_")
    if len(parts) < 2:
        return None
    # T212 convention: last two parts are EXCH and TYPE; everything
    # before is the symbol (may contain underscores, e.g. BRK_B).
    if len(parts) == 2:
        raw_sym = parts[0]
        exch = ""
    else:
        raw_sym = "_".join(parts[:-2])
        exch = parts[-2].upper()

    # Pence marker first: only when the symbol literally ends with a
    # lowercase 'l' (not 'll', not capital I).  Independent of the rest
    # of the ticker shape.
    if raw_sym.endswith("l") and not raw_sym.endswith("ll"):
        return "GBp"

    if len(parts) == 2:
        # 2-part tickers without the pence marker are pound-quoted UK
        # securities (observed in live T212 snapshots).
        return "GBP"
    if exch in _US_EXCHANGES:
        return "USD"
    if exch in _LSE_EXCHANGES:
        return "GBP"
    if exch in _EU_EXCHANGES:
        return "EUR"
    if exch in _FX_CHF:
        return "CHF"
    if exch in _FX_CAD:
        return "CAD"
    if exch in _FX_JPY:
        return "JPY"
    if exch in _FX_AUD:
        return "AUD"
    return None


def fx_rate_to_usd(from_ccy: Optional[str]) -> Optional[float]:
    """Return the multiplier converting ``from_ccy`` to USD.

    Special case: 'GBp' (pence) returns (GBP→USD) / 100.
    None on failure — callers must fail closed.
    """
    if not from_ccy:
        return None
    if from_ccy.upper() == "USD":
        return 1.0

    if from_ccy == "GBp":
        gbp_usd = fx_rate_to_usd("GBP")
        return None if gbp_usd is None else (gbp_usd / 100.0)

    key = (from_ccy.upper(), "USD")
    cached = _FX_CACHE.get(key)
    now = time.time()
    if cached and (now - cached[1]) < _FX_TTL_SECONDS:
        return cached[0]

    try:
        import yfinance as yf  # type: ignore
        symbol = f"{from_ccy.upper()}USD=X"
        df = yf.download(
            symbol, period="5d", interval="1d",
            progress=False, auto_adjust=False, threads=False,
        )
        if df is None or df.empty:
            log.warning("FX fetch for %s returned empty", symbol)
            return None
        close = df["Close"]
        # When yfinance returns a MultiIndex column (newer versions),
        # squeeze to a 1-D series.
        if hasattr(close, "to_numpy"):
            arr = close.to_numpy().ravel()
        else:
            arr = list(close)
        if len(arr) == 0:
            return None
        rate = float(arr[-1])
        if rate <= 0 or rate != rate:  # NaN guard
            return None
        _FX_CACHE[key] = (rate, now)
        log.info("FX cached: %s = %.6f", symbol, rate)
        return rate
    except Exception as exc:  # noqa: BLE001
        log.warning("FX fetch failed for %s: %s", from_ccy, exc)
        return None


def to_usd(price: Optional[float], ticker: str) -> Optional[float]:
    """Convert a native-currency price to USD.

    Returns None when conversion cannot be performed (unknown ccy,
    FX fetch failed).  Callers should fail closed (store 0 or skip).
    """
    if price is None:
        return None
    try:
        f = float(price)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    ccy = detect_quote_currency(ticker)
    if ccy is None:
        log.debug("No ccy detected for %s — failing closed", ticker)
        return None
    rate = fx_rate_to_usd(ccy)
    if rate is None:
        log.debug("No FX rate for %s (%s) — failing closed", ticker, ccy)
        return None
    return f * rate


def clear_fx_cache() -> None:
    """For tests — force a refresh on next call."""
    _FX_CACHE.clear()
