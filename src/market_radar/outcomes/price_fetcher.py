"""Price fetcher for the outcome tracker.

Primary source is **Alpaca** (the same broker the bot trades through): it's
reliable, authenticated, and not rate-limited for our volume. yfinance is
kept ONLY as a fallback for symbols Alpaca can't price (non-US listings).

History: this used to be yfinance-only, which rate-limited so aggressively
that ~85% of 1d/5d/20d outcomes never resolved — the ML model was training
on almost no labeled data. Switching to Alpaca-first unblinds the learning
loop (the bot's whole edge depends on it).

Contract (unchanged): every method returns either a value or None and never
crashes the caller.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


def _alpaca_symbol(ticker: str) -> Optional[str]:
    """Map an internal ticker to an Alpaca symbol, or None if Alpaca can't
    price it (non-US listing). Crypto pairs ('BTC/USD') pass through; T212
    shapes are stripped; US dotted tickers use Alpaca's '.' convention
    (BRK.B, not yfinance's BRK-B)."""
    if not ticker:
        return None
    raw = ticker.strip().upper()
    if "/" in raw:
        return raw  # crypto pair, e.g. BTC/USD
    if "_" in raw:
        parts = raw.split("_")
        if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
            exch = parts[-2]
            sym = "_".join(parts[:-2])
            if exch == "US":
                return sym.replace("_", ".")  # Alpaca uses BRK.B
            return None  # non-US — let yfinance handle it
        return None
    return raw  # plain symbol — assume US equity


@dataclass
class PriceSnapshot:
    ticker: str
    price: Optional[float]
    timestamp_iso: str
    source: str = "alpaca"


class PriceFetcher:
    """Alpaca-first price fetcher with yfinance fallback."""

    def __init__(
        self,
        max_retries: int = 1,
        retry_backoff: float = 2.0,
        alpaca_client=None,
    ) -> None:
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._alpaca = alpaca_client
        self._alpaca_failed = False
        if not YF_AVAILABLE:
            log.info("yfinance not installed — PriceFetcher will rely on Alpaca only.")

    # ------------------------------------------------------------------

    @property
    def alpaca(self):
        """Lazily build an AlpacaClient. If creds are missing the fetcher
        silently degrades to yfinance-only (never crashes the tracker)."""
        if self._alpaca is None and not self._alpaca_failed:
            try:
                from ..execution.alpaca_client import AlpacaClient
                self._alpaca = AlpacaClient()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "PriceFetcher: Alpaca client unavailable (%s) — yfinance only", exc
                )
                self._alpaca_failed = True
        return self._alpaca

    # ------------------------------------------------------------------

    def fetch_latest(self, tickers: list[str]) -> dict[str, PriceSnapshot]:
        """Return ``{original_ticker: PriceSnapshot}`` for as many as possible.

        Alpaca last-trade first (per ticker), yfinance batch for the rest."""
        out: dict[str, PriceSnapshot] = {}
        if not tickers:
            return out
        now_iso = _utc_now_iso()

        # --- Pass 1: Alpaca last trade ---
        needs_yf: list[str] = []
        client = self.alpaca
        for t in tickers:
            asym = _alpaca_symbol(t) if client is not None else None
            price = None
            if asym is not None:
                try:
                    price = client.get_latest_trade(asym)
                except Exception as exc:  # noqa: BLE001
                    log.debug("alpaca latest %s failed: %s", asym, exc)
                    price = None
            if price and price > 0:
                out[t] = PriceSnapshot(t, float(price), now_iso, source="alpaca")
            else:
                needs_yf.append(t)

        # --- Pass 2: yfinance fallback for whatever Alpaca missed ---
        if needs_yf:
            yf_prices = self._yf_fetch_latest(needs_yf)
            for t in needs_yf:
                out[t] = yf_prices.get(t, PriceSnapshot(t, None, now_iso, source="none"))
        return out

    def fetch_closing_on_or_after(
        self,
        ticker: str,
        target_date: datetime,
    ) -> Optional[tuple[float, str]]:
        """Return (close, iso_ts) of the first trading day on/after ``target_date``.

        Used to look up 1d / 5d / 20d post-flag prices. Alpaca daily bars
        first; yfinance fallback if Alpaca can't price the symbol."""
        # --- Alpaca daily bars ---
        client = self.alpaca
        asym = _alpaca_symbol(ticker) if client is not None else None
        if asym is not None:
            try:
                start_dt = target_date.astimezone(timezone.utc)
                end_dt = start_dt + timedelta(days=10)  # weekends + holidays room
                bars = client.get_daily_bars(
                    asym,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    limit=10,
                )
                for bar in bars:
                    close = float(bar.get("c", 0) or 0)
                    if close > 0:
                        ts = str(bar.get("t") or "")
                        if not ts:
                            ts = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        return close, ts
            except Exception as exc:  # noqa: BLE001
                log.debug("alpaca daily bars %s failed: %s", asym, exc)

        # --- yfinance fallback ---
        return self._yf_closing_on_or_after(ticker, target_date)

    # ------------------------------------------------------------------
    # yfinance fallback paths (kept for non-US symbols Alpaca can't price)
    # ------------------------------------------------------------------

    def _yf_fetch_latest(self, tickers: list[str]) -> dict[str, PriceSnapshot]:
        out: dict[str, PriceSnapshot] = {}
        now_iso = _utc_now_iso()
        if not tickers or not YF_AVAILABLE:
            for t in tickers:
                out[t] = PriceSnapshot(t, None, now_iso, source="none")
            return out

        groups: dict[str, list[str]] = {}
        unmapped: list[str] = []
        for t in tickers:
            yf_sym = _normalize_ticker(t)
            if yf_sym is None:
                unmapped.append(t)
                continue
            groups.setdefault(yf_sym, []).append(t)

        for orig in unmapped:
            out[orig] = PriceSnapshot(orig, None, now_iso, source="none")

        if not groups:
            return out

        prices = self._batch_last_prices(list(groups.keys()))
        for yf_sym, price in prices.items():
            for orig in groups.get(yf_sym, []):
                out[orig] = PriceSnapshot(orig, price, now_iso, source="yfinance")
        for yf_sym, originals in groups.items():
            for orig in originals:
                if orig not in out:
                    out[orig] = PriceSnapshot(orig, None, now_iso, source="none")
        return out

    def _yf_closing_on_or_after(
        self, ticker: str, target_date: datetime,
    ) -> Optional[tuple[float, str]]:
        yf_sym = _normalize_ticker(ticker)
        if not yf_sym or not YF_AVAILABLE:
            return None
        try:
            start = target_date.astimezone(timezone.utc)
            end = start + timedelta(days=10)
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

    def _batch_last_prices(self, symbols: list[str]) -> dict[str, Optional[float]]:
        """Hit yfinance once for a batch of symbols. Returns last close price."""
        if not symbols:
            return {}

        out: dict[str, Optional[float]] = {s: None for s in symbols}
        attempt = 0
        while attempt <= self.max_retries:
            attempt += 1
            try:
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
