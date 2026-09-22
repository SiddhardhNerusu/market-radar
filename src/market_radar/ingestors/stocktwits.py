"""StockTwits ingestor — public trending + per-symbol streams.

StockTwits' public API does not require auth for read-only access. The
trending endpoint returns recent posts across the most-discussed tickers,
which is ideal for daemon-style polling.

Messages come pre-tagged with the tickers they reference (via the
``symbols`` field), so we don't have to re-run our own ticker extractor.
Messages also include an optional ``entities.sentiment`` field
("Bullish" / "Bearish"), which we record as a per-signal hint for the
scoring layer (but don't trust as ground truth — StockTwits users
self-tag and the labels are often inaccurate).
"""
from __future__ import annotations

import os

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import requests

from .base import Ingestor, ParsedSignal, TickerMention

log = logging.getLogger(__name__)


TRENDING_URL = "https://api.stocktwits.com/api/2/streams/trending.json"
STREAM_TIMEOUT = 15.0


class StockTwitsTrendingIngestor(Ingestor):
    source = "stocktwits_trending"
    source_tier = 3

    USER_AGENT = os.getenv("RESEARCH_CONTACT_UA", "market-radar/0.1 (research; set RESEARCH_CONTACT_UA)")

    def __init__(self, *, timeout: float = STREAM_TIMEOUT) -> None:
        super().__init__()
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.USER_AGENT,
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------

    def fetch(self) -> Iterable[dict[str, Any]]:
        try:
            resp = self._session.get(TRENDING_URL, timeout=self.timeout)
            if resp.status_code == 429:
                log.warning("[%s] rate-limited (429)", self.source)
                return
            if resp.status_code >= 400:
                log.warning("[%s] HTTP %d body=%s", self.source, resp.status_code, resp.text[:200])
                return
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("[%s] fetch failed: %s", self.source, exc)
            return

        messages = (data or {}).get("messages") or []
        log.info("[%s] %d trending messages", self.source, len(messages))
        for msg in messages:
            if isinstance(msg, dict):
                yield msg

    # ------------------------------------------------------------------

    def parse(self, raw_entry: Any) -> Optional[ParsedSignal]:
        if not isinstance(raw_entry, dict):
            return None

        msg_id = raw_entry.get("id")
        if not msg_id:
            return None
        external_id = f"stocktwits_{msg_id}"

        body = raw_entry.get("body") or ""
        if not body:
            return None

        created_at = raw_entry.get("created_at")
        published = self._parse_iso(created_at)
        user = raw_entry.get("user") or {}
        author = user.get("username")

        # StockTwits-supplied symbols (more reliable than text extraction)
        symbols_raw = raw_entry.get("symbols") or []
        tickers: list[TickerMention] = []
        for s in symbols_raw:
            if not isinstance(s, dict):
                continue
            sym = s.get("symbol")
            if not sym or not isinstance(sym, str):
                continue
            tickers.append(
                TickerMention(
                    ticker=sym.upper().split(".")[0],   # strip exchange suffix
                    market=None,
                    asset_class=None,
                    confidence=0.85,  # high but slightly below cashtag because
                                      # StockTwits sometimes auto-tags loosely
                )
            )

        # Sentiment hint (if user labeled their post)
        entities = raw_entry.get("entities") or {}
        sentiment_obj = entities.get("sentiment") if isinstance(entities, dict) else None
        sentiment_label = (
            sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
        )

        author_metadata = {
            "username": author,
            "user_id": user.get("id"),
            "followers": user.get("followers"),
            "following": user.get("following"),
            "ideas": user.get("ideas"),
            "join_date": user.get("join_date"),
            "official": user.get("official"),
            "trade_status": user.get("trade_status"),
            "classification": user.get("classification"),
        }

        return ParsedSignal(
            external_id=external_id,
            title=None,
            body=body[:2000],
            url=f"https://stocktwits.com/{author}/message/{msg_id}" if author else None,
            author=author,
            author_metadata=author_metadata,
            published_at=published,
            raw_payload={
                "id": msg_id,
                "body": body,
                "stocktwits_sentiment": sentiment_label,
                "symbols": [s.get("symbol") for s in symbols_raw if isinstance(s, dict)],
                "likes_count": (raw_entry.get("likes") or {}).get("total")
                if isinstance(raw_entry.get("likes"), dict) else None,
                "reshares_count": (raw_entry.get("reshares") or {}).get("reshared_count")
                if isinstance(raw_entry.get("reshares"), dict) else None,
                "conversation_id": raw_entry.get("conversation_id"),
            },
            tickers=tickers,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_iso(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        # StockTwits uses RFC3339 like "2026-05-12T11:34:09Z"
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                dt = datetime.strptime(value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
        return value
