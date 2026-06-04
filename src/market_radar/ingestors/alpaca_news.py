"""Alpaca (Benzinga) news firehose — market-wide catalyst capture.

Unlike the RSS ingestor (curated feeds + Google-News topical queries) and the
price-action scanner (LIQUID_EQUITIES, ~85 names), this ingestor pulls Alpaca's
news API with NO symbol filter — every ticker, every headline, in near real
time. This is the feed that carries the press-release wires (PR Newswire,
Business Wire, GlobeNewswire, Benzinga editorial) which break micro-cap
catalysts — a partnership, contract win, FDA action, buyout — minutes after
they hit the tape.

Why this exists: on 2026-06-04 VERU (a $36M micro-cap) ran +159% on a public
08:34 ET press release ("clinical supply agreement with Novo Nordisk"). Alpaca
carried it in real time, but the bot was blind — it scanned only the curated
liquid universe and never ingested this feed. The catalyst sat on the wire for
~3.5 hours before the stock was halted. This ingestor closes that blind spot
so the scoring/notification pipeline sees the whole market, not a watchlist.

Alpaca tags each article with structured ``symbols`` (authoritative), so we
trust those directly rather than scraping tickers from the headline. Broad
"movers"/roundup articles (many symbols) are recorded but NOT ticker-tagged,
so they don't generate per-ticker scoring noise — the real catalyst PRs carry
1-3 symbols (VERU's Novo deal tagged ``[NVO, VERU]``).

NOTE (eyes-only): downstream, ``alpaca_news`` is in the live trader's default
``LIVE_BLOCKED_SOURCES`` — these signals are scored + alerted but NOT traded
until the edge is validated and a min-price/liquidity gate is added. See
``live_trader.LiveTraderConfig.blocked_sources``.

Docs: https://docs.alpaca.markets/reference/news-3
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import requests

from ..config import CONFIG
from .base import Ingestor, ParsedSignal, TickerMention

log = logging.getLogger(__name__)

# Articles tagged with more symbols than this are market roundups
# ("12 Health Care Stocks Moving Today"), not single-name catalysts. We still
# record them, but don't ticker-tag (→ not scored/traded per ticker). Real
# catalyst PRs carry few symbols.
MAX_SYMBOLS_FOR_TAGGING = 8

# How far back each poll looks. Generous overlap vs the ~60s cadence so a burst
# of >50 articles between polls is still caught; dedup on (source, external_id)
# absorbs the repeats.
LOOKBACK_MINUTES = 15

# Bound the work per poll during news bursts.
MAX_PAGES = 4
PER_PAGE = 50


class AlpacaNewsIngestor(Ingestor):
    """Market-wide Alpaca/Benzinga news → ``raw_signals``.

    Tier 2 (mainstream wire). ``source_weight`` is registered at 7.5 in
    ``scoring/source_weights.py``.
    """

    source = "alpaca_news"
    source_tier = 2

    NEWS_PATH = "/v1beta1/news"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        data_base_url: str = "https://data.alpaca.markets",
        lookback_minutes: int = LOOKBACK_MINUTES,
        timeout: float = 15.0,
    ) -> None:
        super().__init__()
        key = (api_key or CONFIG.alpaca_api_key or "").strip()
        secret = (api_secret or CONFIG.alpaca_api_secret or "").strip()
        if not key or not secret:
            raise RuntimeError("Alpaca credentials missing — cannot pull news feed")
        self.data_base_url = data_base_url.rstrip("/")
        self.lookback_minutes = lookback_minutes
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------

    def fetch(self) -> Iterable[dict[str, Any]]:
        start = (
            datetime.now(timezone.utc) - timedelta(minutes=self.lookback_minutes)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        page_token: Optional[str] = None
        seen = 0
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "start": start,
                "limit": PER_PAGE,
                "sort": "desc",
                "include_content": "false",
                "exclude_contentless": "false",
            }
            if page_token:
                params["page_token"] = page_token
            try:
                resp = self._session.get(
                    f"{self.data_base_url}{self.NEWS_PATH}",
                    params=params,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                log.warning("[%s] fetch failed: %s", self.source, exc)
                return
            if resp.status_code in (401, 403):
                log.warning("[%s] auth rejected (%d)", self.source, resp.status_code)
                return
            if resp.status_code == 429:
                log.warning("[%s] rate limited (429) — backing off this cycle", self.source)
                return
            if resp.status_code >= 400:
                log.warning("[%s] HTTP %d: %s", self.source, resp.status_code, resp.text[:200])
                return

            data = resp.json()
            items = data.get("news") or []
            for item in items:
                yield item
            seen += len(items)

            page_token = data.get("next_page_token")
            if not page_token or not items:
                break

        log.info("[%s] fetched %d articles (since %s)", self.source, seen, start)

    # ------------------------------------------------------------------

    def parse(self, raw_entry: dict[str, Any]) -> Optional[ParsedSignal]:
        news_id = raw_entry.get("id")
        if news_id is None:
            return None

        headline = (raw_entry.get("headline") or "").strip()
        summary = (raw_entry.get("summary") or "").strip()
        if not headline and not summary:
            return None

        symbols = [str(s).strip().upper() for s in (raw_entry.get("symbols") or []) if s]
        # Roundup/index articles tag many symbols — record but don't ticker-tag.
        tickers: list[TickerMention] = []
        if 0 < len(symbols) <= MAX_SYMBOLS_FOR_TAGGING:
            tickers = [
                TickerMention(ticker=s, market="US", asset_class=None, confidence=0.9)
                for s in symbols
            ]

        return ParsedSignal(
            external_id=f"{self.source}|{news_id}",
            title=headline or None,
            body=summary or None,
            url=raw_entry.get("url"),
            author=raw_entry.get("author") or raw_entry.get("source"),
            author_metadata={
                "wire": raw_entry.get("source"),
                "symbols_tagged": [t.ticker for t in tickers],
                "symbols_all": symbols,
            },
            published_at=raw_entry.get("created_at"),
            raw_payload={
                "id": news_id,
                "headline": headline,
                "summary": summary,
                "wire": raw_entry.get("source"),
                "symbols": symbols,
                "url": raw_entry.get("url"),
                "created_at": raw_entry.get("created_at"),
                "updated_at": raw_entry.get("updated_at"),
            },
            tickers=tickers,
        )
