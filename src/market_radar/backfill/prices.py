"""Bulk historical price loader for the backfill.

For backfill we need the closing price of every (ticker, filing_date) pair,
plus the closes at +1, +5, +20 trading days. Doing this one ticker at a
time would be slow. Instead, we fetch each ticker's *full daily history*
over the backfill window in big batches, hold it in memory, and look up
each filing against that.

For a 2-year backfill window across ~3,000 unique tickers, this is
manageable RAM (~150 MB) and ~5 minutes of yfinance fetching.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

try:
    import pandas as pd  # type: ignore
    import yfinance as yf  # type: ignore
    YF_AVAILABLE = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore
    yf = None  # type: ignore
    YF_AVAILABLE = False

log = logging.getLogger(__name__)


@dataclass
class PriceLookup:
    """Lookup result for one (ticker, filing_date) point."""
    price_at_flag: Optional[float]
    price_at_flag_ts: Optional[str]
    price_1d: Optional[float]
    price_5d: Optional[float]
    price_20d: Optional[float]
    return_1d_pct: Optional[float]
    return_5d_pct: Optional[float]
    return_20d_pct: Optional[float]


class HistoricalPriceCache:
    """In-memory cache of daily price history keyed by yfinance ticker."""

    def __init__(self, *, batch_size: int = 50, request_pause: float = 0.5):
        self.batch_size = batch_size
        self.request_pause = request_pause
        # ticker → pandas DataFrame (indexed by date)
        self._cache: dict[str, "pd.DataFrame"] = {}

    # ------------------------------------------------------------------

    def warm(self, tickers: list[str], *, start: date, end: date) -> None:
        """Fetch ``tickers`` daily history over [start, end] and cache it."""
        if not YF_AVAILABLE:
            log.error("yfinance not installed — price cache will be empty")
            return

        # Normalize: strip duplicates, sort for deterministic batching
        unique = sorted({t.strip().upper() for t in tickers if t})
        # Drop ones we already have
        to_fetch = [t for t in unique if t not in self._cache]
        log.info(
            "warming price cache: %d unique tickers (%d already cached), %s → %s",
            len(unique), len(unique) - len(to_fetch), start, end,
        )

        for i in range(0, len(to_fetch), self.batch_size):
            batch = to_fetch[i : i + self.batch_size]
            log.info(
                "  batch %d/%d (%d tickers)",
                i // self.batch_size + 1,
                (len(to_fetch) + self.batch_size - 1) // self.batch_size,
                len(batch),
            )
            self._fetch_batch(batch, start=start, end=end)
            time.sleep(self.request_pause)

        log.info("price cache now holds %d tickers", len(self._cache))

    def _fetch_batch(self, batch: list[str], *, start: date, end: date) -> None:
        try:
            df = yf.download(
                tickers=" ".join(batch),
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                group_by="ticker",
                auto_adjust=True,  # split/div-adjusted: anchor + offset on one basis (P1)
                progress=False,
                threads=True,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("batch fetch failed (%d tickers): %s", len(batch), exc)
            return

        if df is None or df.empty:
            log.warning("batch returned empty df (%d tickers)", len(batch))
            return

        if len(batch) == 1:
            sym = batch[0]
            sub = df.copy()
            if "Close" in sub.columns:
                self._cache[sym] = sub[["Close"]].dropna()
            return

        for sym in batch:
            try:
                sub = df[sym]
                if "Close" in sub.columns:
                    closes = sub[["Close"]].dropna()
                    if not closes.empty:
                        self._cache[sym] = closes
            except (KeyError, AttributeError):
                continue

    # ------------------------------------------------------------------

    def lookup(self, ticker: str, filed: date) -> PriceLookup:
        """Compute the outcome for one (ticker, filed-date) pair."""
        empty = PriceLookup(None, None, None, None, None, None, None, None)
        if not YF_AVAILABLE:
            return empty
        df = self._cache.get(ticker.upper())
        if df is None or df.empty:
            return empty

        # Find first trading day on or after filed
        try:
            mask = df.index.date >= filed
        except AttributeError:
            return empty

        idx_positions = mask.nonzero()[0] if hasattr(mask, "nonzero") else [
            i for i, b in enumerate(mask) if b
        ]
        if len(idx_positions) == 0:
            return empty
        anchor_pos = int(idx_positions[0])
        try:
            anchor_close = float(df["Close"].iloc[anchor_pos])
        except (KeyError, IndexError, ValueError, TypeError):
            return empty
        if not (anchor_close == anchor_close):  # NaN check
            return empty

        anchor_ts = df.index[anchor_pos].strftime("%Y-%m-%dT00:00:00Z")

        def _at(offset: int) -> Optional[float]:
            pos = anchor_pos + offset
            if pos >= len(df):
                return None
            try:
                v = float(df["Close"].iloc[pos])
            except (KeyError, IndexError, ValueError, TypeError):
                return None
            return v if v == v else None  # NaN guard

        def _ret(price: Optional[float]) -> Optional[float]:
            if price is None or anchor_close == 0:
                return None
            return (price - anchor_close) / anchor_close * 100.0

        p1 = _at(1)
        p5 = _at(5)
        p20 = _at(20)

        return PriceLookup(
            price_at_flag=anchor_close,
            price_at_flag_ts=anchor_ts,
            price_1d=p1,
            price_5d=p5,
            price_20d=p20,
            return_1d_pct=_ret(p1),
            return_5d_pct=_ret(p5),
            return_20d_pct=_ret(p20),
        )

    def has(self, ticker: str) -> bool:
        return ticker.upper() in self._cache
