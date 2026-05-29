"""Free-tier price fetcher built on yfinance.

We snapshot the *current* mid/last price for a list of tickers in batches.
For history (1d/5d/20d), we query Yahoo's historical bars and pick the
closing price at the appropriate trading day.

Notes:
  - yfinance can be flaky. We retry once with exponential backoff and
    return None for tickers we can't price.
  - Non-US tickers (UK ".L", Frankfurt ".DE", etc.) require a yfinance
    suffix we don't currently map. v1 supports US only; non-US tickers
    are returned as None so the outcome row is still created and can be
    backfilled later.
  - We never crash the caller — every method returns either a value or
    None.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

try:
    import yfinance as yf  # type: ignore
    YF_AVAILABLE = True
except ImportError:  # pragma: no cover
    yf = None  # type: ignore
    YF_AVAILABLE = False

log = logging.getLogger(__name__)


def _normalize_ticker(ticker: str) -> Optional[str]:
    """Convert internal ticker shapes (e.g. T212's ``AAPL_US_EQ``) to a
    yfinance-friendly symbol. Returns None for shapes we can't normalize."""
    if not ticker:
        return None
    raw = ticker.strip().upper()
    # T212 shape: <SYM>_<EXCH>_<TYPE>
    if "_" in raw:
        # Strip the trailing _<EXCH>_<TYPE> chunks
        parts = raw.split("_")
        if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
            exch = parts[-2]
            sym = "_".join(parts[:-2])  # rejoin if symbol itself had underscores (e.g. BRK_B)
            if exch == "US":
                return sym.replace("_", "-")  # yfinance uses BRK-B
            # Non-US suffix mapping (partial)
            yf_suffix = {
                "LSE": ".L", "LSX": ".L",
                "XETRA": ".DE", "ETR": ".DE", "FRA": ".F",
                "EURONEXT": ".PA",
                "TSX": ".TO",
                "SWX": ".SW",
                "MIL": ".MI",
                "ASX": ".AX",
                "JSE": ".JO",
                "HKG": ".HK",
            }.get(exch)
            if yf_suffix:
                return f"{sym.replace('_', '-')}{yf_suffix}"
            return None
    return raw


@dataclass
class PriceSnapshot:
    ticker: str
    price: Optional[float]
    timestamp_iso: str
    source: str = "yfinance"


class PriceFetcher:
    """Free price fetcher with batching and retry."""

    def __init__(self, max_retries: int = 1, retry_backoff: float = 2.0) -> None:
        if not YF_AVAILABLE:
            log.warning(
                "yfinance not installed — PriceFetcher will always return None. "
                "Add `yfinance` to requirements.txt and pip install."
            )
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    # ------------------------------------------------------------------

    def fetch_latest(self, tickers: list[str]) -> dict[str, PriceSnapshot]:
        """Return ``{original_ticker: PriceSnapshot}`` for as many as possible."""
        out: dict[str, PriceSnapshot] = {}
        if not tickers or not YF_AVAILABLE:
            now = _utc_now_iso()
            for t in tickers:
                out[t] = PriceSnapshot(t, None, now)
            return out

        # Normalize and build (yf_symbol → list[original_ticker]) groups
        groups: dict[str, list[str]] = {}
        unmapped: list[str] = []
        for t in tickers:
            yf_sym = _normalize_ticker(t)
            if yf_sym is None:
                unmapped.append(t)
                continue
            groups.setdefault(yf_sym, []).append(t)

        now_iso = _utc_now_iso()
        for orig in unmapped:
            out[orig] = PriceSnapshot(orig, None, now_iso)

        if not groups:
            return out

        symbols = list(groups.keys())
        prices = self._batch_last_prices(symbols)
        for yf_sym, price in prices.items():
            for orig in groups.get(yf_sym, []):
                out[orig] = PriceSnapshot(orig, price, now_iso)

        # Tickers we batched for but didn't get a price back for
        for yf_sym, originals in groups.items():
            for orig in originals:
                if orig not in out:
                    out[orig] = PriceSnapshot(orig, None, now_iso)
        return out

    def fetch_closing_on_or_after(
        self,
        ticker: str,
        target_date: datetime,
    ) -> Optional[tuple[float, str]]:
        """Return (close, iso_ts) of the first trading day on/after ``target_date``.

        Used to look up 1d / 5d / 20d post-flag prices.
        """
        yf_sym = _normalize_ticker(ticker)
        if not yf_sym or not YF_AVAILABLE:
            return None

        try:
            # Fetch ~5 trading days starting from target_date
            from datetime import timedelta
            start = target_date.astimezone(timezone.utc)
            end = start + timedelta(days=10)  # gives us weekends + holidays room
            hist = yf.Ticker(yf_sym).history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("history fetch failed for %s: %s", yf_sym, exc)
            return None

        if hist is None or hist.empty:
            return None

        try:
            first_row = hist.iloc[0]
            close = float(first_row["Close"])
            ts = hist.index[0].to_pydatetime().astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            return close, ts
        except (KeyError, IndexError, ValueError):
            return None

    # ------------------------------------------------------------------

    def _batch_last_prices(self, symbols: list[str]) -> dict[str, Optional[float]]:
        """Hit yfinance once for a batch of symbols. Returns last close price."""
        if not symbols:
            return {}

        out: dict[str, Optional[float]] = {s: None for s in symbols}
        attempt = 0
        while attempt <= self.max_retries:
            attempt += 1
            try:
                # yf.download for batches is faster than per-ticker calls
                df = yf.download(
                    tickers=" ".join(symbols),
                    period="2d",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
                if df is None or df.empty:
                    raise RuntimeError("empty dataframe")

                if len(symbols) == 1:
                    sym = symbols[0]
                    try:
                        last = df["Close"].iloc[-1]
                        out[sym] = float(last) if last == last else None
                    except (KeyError, IndexError, ValueError):
                        pass
                else:
                    for sym in symbols:
                        try:
                            last = df[sym]["Close"].iloc[-1]
                            out[sym] = float(last) if last == last else None
                        except (KeyError, IndexError, ValueError, AttributeError):
                            continue
                return out
            except Exception as exc:  # noqa: BLE001
                log.debug("batch price fetch attempt %d failed: %s", attempt, exc)
                if attempt > self.max_retries:
                    break
                time.sleep(self.retry_backoff * attempt)
        return out


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
