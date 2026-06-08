"""Market-wide movers scanner — the direct catalyst-"bang" detector.

The existing price-action scanner only watches ~85 liquid names, so it is blind
to the micro-cap rockets that ARE the edge (INHD +1897%, NPT +677%, MTEN, ...).
This ingestor pulls Alpaca's market-wide screener — top % gainers and most-active
by volume — across the ENTIRE US market, no watchlist. It catches the move itself
the moment a name is ripping, regardless of whether we also saw the news.

Verified 2026-06-08: the /screener/stocks/movers endpoint returned INHD at
+1897% as the #1 gainer — exactly the name we were trying to catch.

These signals are ticker-native (every row has an authoritative symbol). Under
the catalyst-only policy a pure momentum mover with no news does NOT auto-trade
(the news-catalyst gate filters it); its value is (a) surfacing every bang so a
paired news catalyst gets corroboration, and (b) completeness — we now SEE every
rocket in the market. Tier 1 (authoritative exchange screener data).

Docs: https://docs.alpaca.markets/reference/mostactives , /movers
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import requests

from ..config import CONFIG
from .base import Ingestor, ParsedSignal, TickerMention

log = logging.getLogger(__name__)


class MarketMoversIngestor(Ingestor):
    """Alpaca market-wide top-gainers + most-actives → ``raw_signals``. Tier 1."""

    source = "market_movers"
    source_tier = 1

    MOVERS_PATH = "/v1beta1/screener/stocks/movers"
    ACTIVES_PATH = "/v1beta1/screener/stocks/most-actives"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        data_base_url: str = "https://data.alpaca.markets",
        top: int = 50,
        min_gain_pct: float = 20.0,   # only record real moves, not +3% noise
        active_top: int = 20,
        timeout: float = 15.0,
    ) -> None:
        super().__init__()
        key = (api_key or CONFIG.alpaca_api_key or "").strip()
        secret = (api_secret or CONFIG.alpaca_api_secret or "").strip()
        if not key or not secret:
            raise RuntimeError("Alpaca credentials missing — cannot pull movers screener")
        self.data_base_url = data_base_url.rstrip("/")
        self.top = top
        self.min_gain_pct = min_gain_pct
        self.active_top = active_top
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
                "Accept": "application/json",
            }
        )

    def _get(self, path: str, params: dict[str, Any]) -> Optional[dict[str, Any]]:
        try:
            resp = self._session.get(
                f"{self.data_base_url}{path}", params=params, timeout=self.timeout
            )
        except requests.RequestException as exc:
            log.warning("[%s] fetch failed (%s): %s", self.source, path, exc)
            return None
        if resp.status_code in (401, 403):
            log.warning("[%s] auth rejected (%d)", self.source, resp.status_code)
            return None
        if resp.status_code == 429:
            log.warning("[%s] rate limited (429) — backing off", self.source)
            return None
        if resp.status_code >= 400:
            log.warning("[%s] HTTP %d on %s", self.source, resp.status_code, path)
            return None
        return resp.json()

    # ------------------------------------------------------------------

    def fetch(self) -> Iterable[dict[str, Any]]:
        n_gain = n_active = 0
        movers = self._get(self.MOVERS_PATH, {"top": self.top})
        if movers:
            for g in movers.get("gainers", []) or []:
                try:
                    if float(g.get("percent_change") or 0) >= self.min_gain_pct:
                        n_gain += 1
                        yield {"kind": "gainer", **g}
                except (TypeError, ValueError):
                    continue
        actives = self._get(self.ACTIVES_PATH, {"by": "volume", "top": self.active_top})
        if actives:
            for a in actives.get("most_actives", []) or []:
                n_active += 1
                yield {"kind": "active", **a}
        log.info("[%s] fetched %d gainers(>=%.0f%%) + %d most-active",
                 self.source, n_gain, self.min_gain_pct, n_active)

    # ------------------------------------------------------------------

    def parse(self, raw_entry: dict[str, Any]) -> Optional[ParsedSignal]:
        symbol = str(raw_entry.get("symbol") or "").strip().upper()
        if not symbol:
            return None
        kind = raw_entry.get("kind")
        # One signal per (symbol, kind, UTC day) — detect each mover once a day;
        # re-polls dedup. A genuinely new day re-surfaces a still-running name.
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if kind == "gainer":
            pct = float(raw_entry.get("percent_change") or 0)
            price = raw_entry.get("price")
            title = f"MOVER: {symbol} +{pct:.0f}% to ${price}"
            body = (
                f"{symbol} is a top US market gainer today: +{pct:.1f}% to ${price} "
                f"(change ${raw_entry.get('change')}). Market-wide momentum / volatility "
                f"signal — check for a paired catalyst."
            )
            external_id = f"{self.source}|gainer|{symbol}|{day}"
        else:  # active (most-active by volume)
            vol = raw_entry.get("volume")
            trades = raw_entry.get("trade_count")
            title = f"UNUSUAL VOLUME: {symbol}"
            body = (
                f"{symbol} is among the most-active US stocks by volume today: "
                f"{vol:,} shares across {trades:,} trades." if isinstance(vol, int)
                else f"{symbol} is among the most-active US stocks by volume today."
            )
            external_id = f"{self.source}|active|{symbol}|{day}"

        return ParsedSignal(
            external_id=external_id,
            title=title,
            body=body,
            url="https://data.alpaca.markets/v1beta1/screener/stocks/movers",
            author="alpaca_screener",
            author_metadata={"kind": kind},
            published_at=None,
            raw_payload=dict(raw_entry),
            tickers=[TickerMention(ticker=symbol, market="US", confidence=1.0)],
        )
