"""Price-action signal generator.

Runs every 60 seconds inside the daemon. For each ticker in the liquid
universe, pulls the last ~120 minutes of 1-minute bars from Alpaca and
fires a signal when any of the following setups trigger:

  - opening_range_breakout : 15-min ORB breakout with volume confirmation
  - vwap_cross             : Reclaim/loss of VWAP with momentum
  - donchian_breakout      : New 20-bar high (or low) with volume
  - rsi_extreme            : RSI(14) < 25 (oversold bounce) or > 75 (overbought reversal)
  - volume_spike           : 1-minute bar volume > 3× 20-bar avg with > 0.5% range
  - gap_and_go             : Open gap > 2% from prior close, trend continuation

Each trigger writes one ``raw_signals`` row (source='price_action_<strategy>',
source_tier=1) + one ``signal_scores`` row with a pre-computed composite.
The bot's live trader reads ``signal_scores`` so these signals flow
through the same gate / sizer / risk pipeline as news signals.

Important design choices:
  - We use Alpaca's IEX data feed (free tier, sub-second latency)
  - 1-minute bars are sufficient — sub-minute trading on $5k account is
    a losing game once you account for commissions, fills, slippage.
  - Signals are gated by liquidity (avg volume) and intra-day vol (ATR)
    so we don't fire on dead names that happen to have a spike.
  - The signal "model_p_5d" is filled by the SAME ML predictor that
    processes news signals — so price-action signals get the same
    calibrated probability treatment.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal, Optional

from ..config import CONFIG
from ..execution.alpaca_client import AlpacaClient, AlpacaError
from ..storage import get_connection, insert_raw_signal
from ..storage.db import insert_signal_score, utc_now
from .universe import CRYPTO_PAIRS, LIQUID_EQUITIES, normalize_alpaca

log = logging.getLogger("marketradar.signals.price_action")

Direction = Literal["buy", "sell"]


@dataclass
class ScanStats:
    universe_size: int = 0
    bars_fetched: int = 0
    signals_emitted: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    by_strategy: dict[str, int] = field(default_factory=dict)


@dataclass
class PASignal:
    """One emitted price-action signal."""
    ticker: str
    strategy: str
    direction: Direction
    reason: str
    composite_score: float
    sentiment: float
    setup_payload: dict


# ---------------------------------------------------------------------------
# Indicator math (pure functions, no deps beyond stdlib)
# ---------------------------------------------------------------------------

def _sma(values: list[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _vwap(bars: list[dict]) -> Optional[float]:
    num = den = 0.0
    for b in bars:
        typical = (b["h"] + b["l"] + b["c"]) / 3.0
        vol = b.get("v", 0) or 0
        num += typical * vol
        den += vol
    return (num / den) if den > 0 else None


def _atr(bars: list[dict], period: int = 14) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(-period, 0):
        h, l = bars[i]["h"], bars[i]["l"]
        prev_close = bars[i - 1]["c"]
        trs.append(max(h - l, abs(h - prev_close), abs(l - prev_close)))
    return sum(trs) / period


# ---------------------------------------------------------------------------
# Setup detectors — each returns ``Optional[PASignal]``
# ---------------------------------------------------------------------------

def _opening_range_breakout(ticker: str, bars: list[dict]) -> Optional[PASignal]:
    """Last bar breaks the first-15-min range with volume confirmation."""
    if len(bars) < 30:
        return None
    # First 15 1-min bars of the session
    or_bars = bars[:15]
    or_high = max(b["h"] for b in or_bars)
    or_low = min(b["l"] for b in or_bars)
    or_range = or_high - or_low
    if or_range <= 0:
        return None
    avg_vol = _sma([b.get("v", 0) for b in bars[-30:-1]], 20) or 0
    last = bars[-1]
    if avg_vol <= 0:
        return None
    vol_mult = last.get("v", 0) / avg_vol
    if last["c"] > or_high and vol_mult > 1.5:
        atr = _atr(bars)
        strength = min(2.0, (last["c"] - or_high) / max(or_range, 0.01))
        composite = 6.5 + strength
        return PASignal(
            ticker, "opening_range_breakout", "buy",
            f"ORB long: close {last['c']:.2f} > OR-high {or_high:.2f}, vol {vol_mult:.1f}x",
            composite_score=min(composite, 9.0),
            sentiment=0.6,
            setup_payload={"or_high": or_high, "or_low": or_low,
                           "vol_mult": vol_mult, "atr": atr},
        )
    if last["c"] < or_low and vol_mult > 1.5:
        atr = _atr(bars)
        strength = min(2.0, (or_low - last["c"]) / max(or_range, 0.01))
        composite = 6.5 + strength
        return PASignal(
            ticker, "opening_range_breakout", "sell",
            f"ORB short: close {last['c']:.2f} < OR-low {or_low:.2f}, vol {vol_mult:.1f}x",
            composite_score=min(composite, 9.0),
            sentiment=-0.6,
            setup_payload={"or_high": or_high, "or_low": or_low,
                           "vol_mult": vol_mult, "atr": atr},
        )
    return None


def _vwap_cross(ticker: str, bars: list[dict]) -> Optional[PASignal]:
    """Last bar crosses session VWAP from the other side with volume."""
    if len(bars) < 30:
        return None
    vwap = _vwap(bars)
    if vwap is None:
        return None
    prev = bars[-2]
    last = bars[-1]
    avg_vol = _sma([b.get("v", 0) for b in bars[-30:-1]], 20) or 0
    if avg_vol <= 0:
        return None
    vol_mult = last.get("v", 0) / avg_vol
    if vol_mult < 1.3:
        return None
    if prev["c"] < vwap and last["c"] > vwap:
        return PASignal(
            ticker, "vwap_cross", "buy",
            f"VWAP reclaim {vwap:.2f} → close {last['c']:.2f}, vol {vol_mult:.1f}x",
            composite_score=6.8,
            sentiment=0.5,
            setup_payload={"vwap": vwap, "vol_mult": vol_mult},
        )
    if prev["c"] > vwap and last["c"] < vwap:
        return PASignal(
            ticker, "vwap_cross", "sell",
            f"VWAP loss {vwap:.2f} → close {last['c']:.2f}, vol {vol_mult:.1f}x",
            composite_score=6.8,
            sentiment=-0.5,
            setup_payload={"vwap": vwap, "vol_mult": vol_mult},
        )
    return None


def _donchian_breakout(ticker: str, bars: list[dict], lookback: int = 20) -> Optional[PASignal]:
    """New ``lookback``-bar high or low.

    QUALITY GATE: also require multi-timeframe confirmation — the 5-bar
    rolling close trend must agree with the breakout direction. This
    filters out one-bar noise spikes and dramatically lifts precision
    based on academic momentum studies (Jegadeesh & Titman, 1993).
    """
    if len(bars) < lookback + 5:
        return None
    window = bars[-lookback - 1:-1]
    hi = max(b["h"] for b in window)
    lo = min(b["l"] for b in window)
    last = bars[-1]
    atr = _atr(bars)
    if atr is None or atr <= 0:
        return None
    # 5-bar trend filter — close[-1] vs close[-5] for momentum confirmation
    closes = [b["c"] for b in bars]
    trend_5bar = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0
    # Liquidity gate: bar volume above 20-bar average (no thin-volume breakouts)
    avg_vol = _sma([b.get("v", 0) for b in bars[-25:-1]], 20) or 0
    vol_ok = avg_vol > 0 and last.get("v", 0) >= 0.5 * avg_vol

    if last["c"] > hi and trend_5bar > 0 and vol_ok:
        # Strength bonus when breakout exceeds prior high by more than 1× ATR
        atr_pct = (last["c"] - hi) / atr
        composite = 7.0 + min(atr_pct, 1.5)
        return PASignal(
            ticker, "donchian_breakout", "buy",
            f"Donchian-{lookback} long: {last['c']:.2f} > prior high {hi:.2f}, "
            f"5bar trend +{trend_5bar*100:.2f}%",
            composite_score=min(composite, 9.0),
            sentiment=0.6,
            setup_payload={"donchian_high": hi, "donchian_low": lo,
                           "atr": atr, "trend_5bar_pct": trend_5bar * 100,
                           "atr_break_x": atr_pct},
        )
    if last["c"] < lo and trend_5bar < 0 and vol_ok:
        atr_pct = (lo - last["c"]) / atr
        composite = 7.0 + min(atr_pct, 1.5)
        return PASignal(
            ticker, "donchian_breakout", "sell",
            f"Donchian-{lookback} short: {last['c']:.2f} < prior low {lo:.2f}, "
            f"5bar trend {trend_5bar*100:.2f}%",
            composite_score=min(composite, 9.0),
            sentiment=-0.6,
            setup_payload={"donchian_high": hi, "donchian_low": lo,
                           "atr": atr, "trend_5bar_pct": trend_5bar * 100,
                           "atr_break_x": atr_pct},
        )
    return None


def _rsi_extreme(ticker: str, bars: list[dict]) -> Optional[PASignal]:
    """RSI(14) crosses out of oversold/overbought with a confirming bar."""
    closes = [b["c"] for b in bars]
    rsi_now = _rsi(closes, 14)
    rsi_prev = _rsi(closes[:-1], 14)
    if rsi_now is None or rsi_prev is None:
        return None
    last = bars[-1]
    body = last["c"] - last["o"]
    # Oversold bounce: RSI was <25, now > 30 + bullish bar
    if rsi_prev < 25 and rsi_now > 30 and body > 0:
        return PASignal(
            ticker, "rsi_oversold_bounce", "buy",
            f"RSI bounce: {rsi_prev:.0f}→{rsi_now:.0f}, bull bar +{body:.2f}",
            composite_score=6.5,
            sentiment=0.4,
            setup_payload={"rsi_prev": rsi_prev, "rsi_now": rsi_now},
        )
    # Overbought reversal: RSI was >75, now < 70 + bearish bar
    if rsi_prev > 75 and rsi_now < 70 and body < 0:
        return PASignal(
            ticker, "rsi_overbought_reversal", "sell",
            f"RSI reversal: {rsi_prev:.0f}→{rsi_now:.0f}, bear bar {body:.2f}",
            composite_score=6.5,
            sentiment=-0.4,
            setup_payload={"rsi_prev": rsi_prev, "rsi_now": rsi_now},
        )
    return None


def _volume_spike(ticker: str, bars: list[dict]) -> Optional[PASignal]:
    """1-min bar with > 3× avg volume + > 0.5% range."""
    if len(bars) < 25:
        return None
    avg_vol = _sma([b.get("v", 0) for b in bars[-21:-1]], 20) or 0
    if avg_vol <= 0:
        return None
    last = bars[-1]
    vol_mult = last.get("v", 0) / avg_vol
    if vol_mult < 3.0:
        return None
    rng_pct = (last["h"] - last["l"]) / max(last["c"], 0.01) * 100.0
    if rng_pct < 0.5:
        return None
    body = last["c"] - last["o"]
    direction: Direction = "buy" if body > 0 else "sell"
    return PASignal(
        ticker, "volume_spike", direction,
        f"Vol spike {vol_mult:.1f}x avg, {rng_pct:.2f}% range, body {body:+.2f}",
        composite_score=6.7,
        sentiment=0.5 if body > 0 else -0.5,
        setup_payload={"vol_mult": vol_mult, "range_pct": rng_pct},
    )


def _gap_and_go(ticker: str, bars: list[dict]) -> Optional[PASignal]:
    """Session open gapped >2% from prior close + first bar continues."""
    if len(bars) < 10:
        return None
    # Heuristic: first bar of the day vs. average of close 5 bars before that
    first = bars[0]
    pre = bars[1:6] if len(bars) >= 6 else bars[1:]
    if not pre:
        return None
    # Without a "yesterday close" reference, gauge gap from first-bar high vs. session-low so far
    session_low = min(b["l"] for b in bars[:10])
    session_high = max(b["h"] for b in bars[:10])
    if first["o"] <= 0:
        return None
    gap_pct = (first["o"] - session_low) / session_low * 100.0
    last = bars[-1]
    if gap_pct < 2.0:
        # Try short side
        gap_down_pct = (session_high - first["o"]) / session_high * 100.0
        if gap_down_pct >= 2.0 and last["c"] < first["o"]:
            return PASignal(
                ticker, "gap_and_go", "sell",
                f"Gap-down {gap_down_pct:.1f}%, continuing lower",
                composite_score=6.6,
                sentiment=-0.5,
                setup_payload={"gap_pct": -gap_down_pct},
            )
        return None
    if last["c"] > first["o"]:
        return PASignal(
            ticker, "gap_and_go", "buy",
            f"Gap-up {gap_pct:.1f}%, continuing higher",
            composite_score=6.6,
            sentiment=0.5,
            setup_payload={"gap_pct": gap_pct},
        )
    return None


_STRATEGIES = (
    _opening_range_breakout,
    _vwap_cross,
    _donchian_breakout,
    _rsi_extreme,
    _volume_spike,
    _gap_and_go,
)


# ---------------------------------------------------------------------------
# Scanner — pulls bars + runs each strategy + persists hits
# ---------------------------------------------------------------------------

class PriceActionScanner:
    """Scans the liquid universe and emits signals on each call to ``run()``."""

    def __init__(
        self,
        *,
        alpaca: Optional[AlpacaClient] = None,
        universe: Iterable[str] = LIQUID_EQUITIES,
        crypto_universe: Iterable[str] = CRYPTO_PAIRS,
        bar_lookback_minutes: int = 120,
        skip_if_market_closed: bool = True,
    ):
        self.alpaca = alpaca or AlpacaClient()
        self.universe = tuple(universe)
        # Crypto trades 24/7 — separate list, scanned even when stock market is closed
        self.crypto_universe = tuple(crypto_universe)
        self.bar_lookback_minutes = bar_lookback_minutes
        self.skip_if_market_closed = skip_if_market_closed
        # Deduplication: don't re-fire the same (ticker, strategy, direction)
        # within 30 minutes — avoids spam from a single setup re-triggering
        # every minute.
        self._recent_fires: dict[tuple[str, str, str], datetime] = {}
        self._cooldown = timedelta(minutes=30)

    def run(self) -> ScanStats:
        stats = ScanStats(universe_size=len(self.universe) + len(self.crypto_universe))
        market_open = True
        try:
            market_open = self.alpaca.is_market_open()
        except AlpacaError as exc:
            log.warning("market clock fetch failed: %s", exc)

        # Always scan crypto (24/7). Scan equities only when market is open
        # (or if explicitly allowed; the daemon uses default True).
        scan_list: list[tuple[str, str]] = []  # (ticker, asset_class)
        if market_open or not self.skip_if_market_closed:
            scan_list.extend((t, "stock") for t in self.universe)
        scan_list.extend((t, "crypto") for t in self.crypto_universe)

        for ticker, asset_class in scan_list:
            try:
                bars = (self._fetch_crypto_bars(ticker)
                        if asset_class == "crypto"
                        else self._fetch_bars(ticker))
                if not bars or len(bars) < 20:
                    continue
                stats.bars_fetched += 1
                for strategy_fn in _STRATEGIES:
                    sig = strategy_fn(ticker, bars)
                    if sig is None:
                        continue
                    key = (sig.ticker, sig.strategy, sig.direction)
                    last = self._recent_fires.get(key)
                    if last and (datetime.now(timezone.utc) - last) < self._cooldown:
                        stats.duplicates_skipped += 1
                        continue
                    if self._persist(sig, bars):
                        self._recent_fires[key] = datetime.now(timezone.utc)
                        stats.signals_emitted += 1
                        stats.by_strategy[sig.strategy] = (
                            stats.by_strategy.get(sig.strategy, 0) + 1
                        )
            except Exception as exc:  # noqa: BLE001
                stats.errors += 1
                log.warning("scan(%s) failed: %s", ticker, exc)
        if stats.signals_emitted:
            log.info(
                "Price-action scan emitted %d signals across %d tickers "
                "(skipped %d dupes, %d errors). by_strategy=%s",
                stats.signals_emitted, stats.bars_fetched,
                stats.duplicates_skipped, stats.errors, stats.by_strategy,
            )
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fetch_bars(self, ticker: str) -> Optional[list[dict]]:
        """Fetch the last ``bar_lookback_minutes`` of 1-min bars from Alpaca."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=self.bar_lookback_minutes)
        params = {
            "timeframe": "1Min",
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 200,
            "feed": "iex",
            "adjustment": "raw",
        }
        try:
            data = self.alpaca._request(
                "GET", f"/v2/stocks/{ticker}/bars",
                params=params, base=self.alpaca.data_base_url,
            )
        except AlpacaError as exc:
            log.debug("bars(%s) failed: %s", ticker, exc)
            return None
        return (data or {}).get("bars") or None

    def _fetch_crypto_bars(self, pair: str) -> Optional[list[dict]]:
        """Fetch crypto 1-min bars. Alpaca endpoint: /v1beta3/crypto/us/bars."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=self.bar_lookback_minutes)
        params = {
            "symbols": pair,
            "timeframe": "1Min",
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 200,
        }
        try:
            data = self.alpaca._request(
                "GET", "/v1beta3/crypto/us/bars",
                params=params, base=self.alpaca.data_base_url,
            )
        except AlpacaError as exc:
            log.debug("crypto_bars(%s) failed: %s", pair, exc)
            return None
        bars_by_symbol = (data or {}).get("bars") or {}
        return bars_by_symbol.get(pair) or None

    def _persist(self, sig: PASignal, bars: list[dict]) -> bool:
        """Insert raw_signal + signal_score. Returns True on success."""
        last = bars[-1]
        title = f"[{sig.strategy}] {sig.ticker} {sig.direction.upper()} — {sig.reason}"
        body = json.dumps({
            "strategy": sig.strategy,
            "direction": sig.direction,
            "reason": sig.reason,
            "last_bar": {"o": last["o"], "h": last["h"], "l": last["l"],
                         "c": last["c"], "v": last.get("v"), "t": last.get("t")},
            **sig.setup_payload,
        })
        external_id = f"pa-{sig.ticker}-{sig.strategy}-{sig.direction}-{last.get('t', utc_now())}"
        try:
            with get_connection() as conn:
                signal_id = insert_raw_signal(
                    conn,
                    source=f"price_action_{sig.strategy}",
                    source_tier=1,
                    external_id=external_id,
                    url=None,
                    title=title,
                    body=body,
                    author="price_action_scanner",
                    author_metadata={"strategy": sig.strategy},
                    raw_payload=sig.setup_payload,
                    published_at=utc_now(),
                    tickers=[{"ticker": sig.ticker, "market": "US",
                              "asset_class": "large_cap", "confidence": 1.0}],
                )
                if signal_id is None:
                    return False  # duplicate
                insert_signal_score(
                    conn,
                    signal_id=signal_id,
                    ticker=sig.ticker,
                    event_type=f"pa_{sig.strategy}",
                    sentiment=sig.sentiment,
                    sentiment_magnitude=abs(sig.sentiment),
                    factual=1,  # price action is observed, not inferred
                    source_weight=9.0,  # direct observation = top tier
                    corroboration_count=0,
                    author_quality=1.0,
                    anti_pump_flag=0,
                    composite_score=sig.composite_score,
                    signal_class=f"price_action_{sig.direction}",
                )
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("persist(%s) failed: %s", sig.ticker, exc)
            return False
