"""Volume + volatility + cross-asset features for the ML model.

For each (ticker, signal_date) we compute:
  - volume_ratio_5d_20d  : ratio of 5d avg volume to 20d avg volume
                          (>1 means volume has been picking up)
  - realized_vol_20d     : 20-day annualised realized volatility (%)
  - spy_5d_return        : SPY's 5-day return ending on signal_date
  - vix_level            : VIX closing level on signal_date
  - qqq_5d_return        : QQQ's 5-day return (tech proxy)

These run via a bulk yfinance fetcher that:
  1. Pulls daily Close+Volume for every unique ticker in one batch operation
  2. Pulls SPY, QQQ, ^VIX daily over the full window
  3. Caches in memory by ticker → DataFrame
  4. Looks up per-(ticker, date) features in O(1)

The fetcher is shared by ``ml/train.py`` (one big fetch before training)
and ``ml/predict.py`` (incremental fetches for new tickers as live signals
arrive).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)


# Indexes we always fetch — these become per-signal context features.
# Macro pack (Tier 2 #17): adds ^TNX (10Y), ^IRX (3M) for yield-curve
# slope, UUP (dollar index), USO (oil), GLD (gold) for cross-asset regime
# context.
CROSS_ASSETS = ["SPY", "QQQ", "^VIX", "^TNX", "^IRX", "UUP", "USO", "GLD"]


@dataclass
class MarketFeatures:
    volume_ratio_5d_20d: Optional[float] = None
    realized_vol_20d:    Optional[float] = None
    spy_5d_return:       Optional[float] = None
    qqq_5d_return:       Optional[float] = None
    vix_level:           Optional[float] = None
    # Macro pack
    yield_curve_slope:   Optional[float] = None  # 10Y minus 3M, in percentage points
    dxy_5d_return:       Optional[float] = None
    oil_5d_return:       Optional[float] = None
    gold_5d_return:      Optional[float] = None

    def to_list(self) -> list[float]:
        """In FEATURE_NAMES order. None → 0.0 (LightGBM/HGB tolerates this)."""
        return [
            self.volume_ratio_5d_20d if self.volume_ratio_5d_20d is not None else 1.0,
            self.realized_vol_20d    if self.realized_vol_20d is not None    else 25.0,
            self.spy_5d_return       if self.spy_5d_return is not None       else 0.0,
            self.qqq_5d_return       if self.qqq_5d_return is not None       else 0.0,
            self.vix_level           if self.vix_level is not None           else 18.0,
            self.yield_curve_slope   if self.yield_curve_slope is not None   else 0.0,
            self.dxy_5d_return       if self.dxy_5d_return is not None       else 0.0,
            self.oil_5d_return       if self.oil_5d_return is not None       else 0.0,
            self.gold_5d_return      if self.gold_5d_return is not None       else 0.0,
        ]


@dataclass
class MarketFeatureCache:
    """In-memory cache of daily history keyed by yfinance ticker."""
    by_ticker: dict[str, object] = field(default_factory=dict)
    # cross-asset DataFrames stored under their tickers
    batch_size: int = 50
    request_pause: float = 0.5

    def warm(self, tickers: list[str], *, start: date, end: date) -> None:
        """Fetch Close+Volume for all tickers + cross-assets over [start, end]."""
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            log.error("yfinance not installed — market features unavailable")
            return

        unique = sorted({t.strip().upper() for t in tickers if t})
        unique = [t for t in unique if t not in self.by_ticker]
        # Always fetch the cross-assets
        for ca in CROSS_ASSETS:
            if ca not in self.by_ticker:
                unique.append(ca)
        unique = sorted(set(unique))
        log.info("warming market-feature cache: %d new tickers, %s → %s",
                 len(unique), start, end)
        if not unique:
            return

        for i in range(0, len(unique), self.batch_size):
            batch = unique[i : i + self.batch_size]
            log.info("  batch %d/%d (%d tickers)",
                     i // self.batch_size + 1,
                     (len(unique) + self.batch_size - 1) // self.batch_size,
                     len(batch))
            self._fetch_batch(batch, start=start, end=end)
            time.sleep(self.request_pause)
        log.info("market-feature cache now holds %d tickers", len(self.by_ticker))

    def _fetch_batch(self, batch: list[str], *, start: date, end: date) -> None:
        try:
            import yfinance as yf  # type: ignore
            df = yf.download(
                tickers=" ".join(batch),
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("market-feature batch fetch failed: %s", exc)
            return

        if df is None or df.empty:
            return

        if len(batch) == 1:
            sym = batch[0]
            cols = [c for c in ("Close", "Volume") if c in df.columns]
            if cols:
                self.by_ticker[sym] = df[cols].dropna()
            return

        for sym in batch:
            try:
                sub = df[sym]
                cols = [c for c in ("Close", "Volume") if c in sub.columns]
                if cols:
                    self.by_ticker[sym] = sub[cols].dropna()
            except (KeyError, AttributeError):
                continue

    # ------------------------------------------------------------------

    def features_for(self, ticker: str, when: date) -> MarketFeatures:
        """Compute MarketFeatures for one (ticker, date) point."""
        result = MarketFeatures()
        # The user's ticker may be in T212 shape (AAPL_US_EQ); normalize.
        sym = ticker.strip().upper()
        if "_" in sym:
            parts = sym.split("_")
            if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
                sym = "_".join(parts[:-2]).replace("_", "-")

        df = self.by_ticker.get(sym)
        if df is not None and not df.empty:
            try:
                # Find first trading row on or after `when`
                mask = df.index.date <= when
                # We need an anchor AT OR BEFORE the signal date (last day with data)
                idx_positions = mask.nonzero()[0] if hasattr(mask, "nonzero") else [
                    i for i, b in enumerate(mask) if b
                ]
                if len(idx_positions) > 0:
                    anchor = int(idx_positions[-1])  # last day on/before signal
                    # 20-day window ending at anchor
                    window20 = df.iloc[max(0, anchor - 19): anchor + 1]
                    window5  = df.iloc[max(0, anchor - 4):  anchor + 1]

                    # Volume ratio: 5d avg / 20d avg (1.0 = normal, >1 = pickup)
                    if "Volume" in df.columns and len(window20) >= 10:
                        vol5 = float(window5["Volume"].mean())
                        vol20 = float(window20["Volume"].mean())
                        if vol20 > 0:
                            result.volume_ratio_5d_20d = vol5 / vol20

                    # Realized volatility: stdev of daily log returns × √252
                    if "Close" in df.columns and len(window20) >= 10:
                        import math
                        closes = window20["Close"].tolist()
                        returns = []
                        for j in range(1, len(closes)):
                            prev, curr = closes[j-1], closes[j]
                            if prev and prev > 0 and curr and curr > 0:
                                returns.append(math.log(curr / prev))
                        if len(returns) >= 5:
                            mean = sum(returns) / len(returns)
                            var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
                            std = math.sqrt(var)
                            result.realized_vol_20d = std * math.sqrt(252) * 100.0  # in %
            except Exception:
                pass

        # Cross-asset features
        result.spy_5d_return = self._n_day_return("SPY", when, 5)
        result.qqq_5d_return = self._n_day_return("QQQ", when, 5)
        result.vix_level     = self._level_at("^VIX", when)

        # Macro pack — yield curve slope (10Y - 3M) and cross-asset moves
        tnx = self._level_at("^TNX", when)
        irx = self._level_at("^IRX", when)
        if tnx is not None and irx is not None:
            # ^TNX / ^IRX are quoted as 10× the yield in percent; difference
            # is already in percentage points after divide.
            result.yield_curve_slope = (tnx - irx) / 10.0
        result.dxy_5d_return  = self._n_day_return("UUP", when, 5)
        result.oil_5d_return  = self._n_day_return("USO", when, 5)
        result.gold_5d_return = self._n_day_return("GLD", when, 5)
        return result

    def _level_at(self, asset: str, when: date) -> Optional[float]:
        df = self.by_ticker.get(asset)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        try:
            mask = df.index.date <= when
            positions = mask.nonzero()[0] if hasattr(mask, "nonzero") else [
                i for i, b in enumerate(mask) if b
            ]
            if len(positions) == 0:
                return None
            return float(df["Close"].iloc[int(positions[-1])])
        except Exception:
            return None

    def _n_day_return(self, asset: str, when: date, n_days: int) -> Optional[float]:
        df = self.by_ticker.get(asset)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        try:
            mask = df.index.date <= when
            positions = mask.nonzero()[0] if hasattr(mask, "nonzero") else [
                i for i, b in enumerate(mask) if b
            ]
            if len(positions) < n_days + 1:
                return None
            end_pos = int(positions[-1])
            start_pos = end_pos - n_days
            if start_pos < 0:
                return None
            end_close = float(df["Close"].iloc[end_pos])
            start_close = float(df["Close"].iloc[start_pos])
            if start_close <= 0:
                return None
            return (end_close - start_close) / start_close * 100.0
        except Exception:
            return None
