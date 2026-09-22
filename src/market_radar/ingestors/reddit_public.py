"""Reddit ingestor — public JSON endpoint (no auth required).

Uses the ``/r/{subreddit}/new.json`` endpoint, which is read-only and free.
Each post is a Tier 3 signal — we run the ticker extractor over the title
and selftext, then record the post with author metadata.

**Author quality is set to a default 0.5** for v1 because the public JSON
endpoint doesn't return author karma or account age. When the user adds
Reddit API credentials (``REDDIT_CLIENT_ID`` + ``REDDIT_CLIENT_SECRET``),
the PRAW-based path in ``reddit_authed.py`` (to be added) will overwrite
this with real karma/age.

Anti-pump heuristics applied at ingest time:
  - subreddit = wallstreetbets AND post is sub-1-hour old AND ticker is
    extracted as length 4–5 → mark anti_pump_flag in payload for the
    scoring layer (we don't drop the row, just tag it)
"""
from __future__ import annotations

import os

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import requests

from .base import Ingestor, ParsedSignal, TickerMention
from .ticker_extractor import TICKER_EXTRACTOR

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubredditSpec:
    name: str            # e.g., 'wallstreetbets'
    label: str
    source_weight: float  # 1-10
    treat_as_pump_prone: bool = False


SUBREDDITS: list[SubredditSpec] = [
    # Higher quality, more research-oriented
    SubredditSpec("SecurityAnalysis", "r/SecurityAnalysis", 5.0),
    SubredditSpec("investing", "r/investing", 4.0),
    SubredditSpec("stocks", "r/stocks", 4.0),
    SubredditSpec("ValueInvesting", "r/ValueInvesting", 4.5),
    SubredditSpec("dividends", "r/dividends", 4.0),
    SubredditSpec("economics", "r/economics", 4.0),
    SubredditSpec("Economics", "r/Economics", 4.0),
    SubredditSpec("finance", "r/finance", 4.0),
    SubredditSpec("StockMarket", "r/StockMarket", 4.0),
    # Sentiment-heavy / pump-prone
    SubredditSpec("wallstreetbets", "r/wallstreetbets", 3.0, treat_as_pump_prone=True),
    SubredditSpec("options", "r/options", 3.5),
    SubredditSpec("Daytrading", "r/Daytrading", 3.0, treat_as_pump_prone=True),
    SubredditSpec("swingtrading", "r/swingtrading", 3.5),
    SubredditSpec("thetagang", "r/thetagang", 4.0),
    SubredditSpec("pennystocks", "r/pennystocks", 2.0, treat_as_pump_prone=True),
    SubredditSpec("Shortsqueeze", "r/Shortsqueeze", 2.0, treat_as_pump_prone=True),
    # Sector-specific
    SubredditSpec("biotechplays", "r/biotechplays", 3.5, treat_as_pump_prone=True),
    SubredditSpec("Biotechnology", "r/Biotechnology", 4.0),
    SubredditSpec("SPACs", "r/SPACs", 3.0, treat_as_pump_prone=True),
    SubredditSpec("energystocks", "r/energystocks", 3.5),
    SubredditSpec("realestateinvesting", "r/realestateinvesting", 3.5),
]


REDDIT_JSON_URL = "https://www.reddit.com/r/{sub}/new.json?limit=40"
# Reddit asks for a descriptive User-Agent in the format <platform>:<id>:<ver> by <reddit-handle>
# Without this, we get 429/403 much faster.
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "macos:market-radar:0.1 (research; set REDDIT_USER_AGENT)")


class RedditPublicIngestor(Ingestor):
    """Public-JSON Reddit ingestor — no auth required."""

    source = "reddit_public_aggregate"
    source_tier = 3

    def __init__(
        self,
        subs: Optional[list[SubredditSpec]] = None,
        *,
        timeout: float = 15.0,
        per_sub_pause: float = 1.0,
    ) -> None:
        super().__init__()
        self.subs = subs or SUBREDDITS
        self.timeout = timeout
        self.per_sub_pause = per_sub_pause
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": REDDIT_USER_AGENT,
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------

    def fetch(self) -> Iterable[dict[str, Any]]:
        for spec in self.subs:
            url = REDDIT_JSON_URL.format(sub=spec.name)
            try:
                resp = self._session.get(url, timeout=self.timeout)
                if resp.status_code == 429:
                    log.warning("[%s] %s rate-limited (429), skipping", self.source, spec.name)
                    time.sleep(5)
                    continue
                if resp.status_code in (401, 403):
                    log.warning(
                        "[%s] %s blocked us (%d). Reddit may require auth for this sub.",
                        self.source, spec.name, resp.status_code,
                    )
                    continue
                if resp.status_code >= 400:
                    log.warning("[%s] %s HTTP %d", self.source, spec.name, resp.status_code)
                    continue
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                log.warning("[%s] %s fetch failed: %s", self.source, spec.name, exc)
                continue

            children = ((data or {}).get("data") or {}).get("children") or []
            log.info("[%s] r/%s → %d posts", self.source, spec.name, len(children))
            for child in children:
                post = (child or {}).get("data") or {}
                if not isinstance(post, dict):
                    continue
                yield {"spec": spec, "post": post}

            time.sleep(self.per_sub_pause)

    # ------------------------------------------------------------------

    def parse(self, raw_entry: dict[str, Any]) -> Optional[ParsedSignal]:
        spec: SubredditSpec = raw_entry["spec"]
        post: dict[str, Any] = raw_entry["post"]

        post_id = post.get("id")
        if not post_id:
            return None
        external_id = f"reddit_t3_{post_id}"

        title = post.get("title") or ""
        selftext = post.get("selftext") or ""
        permalink = post.get("permalink")
        url = (
            f"https://www.reddit.com{permalink}"
            if permalink and permalink.startswith("/")
            else permalink or post.get("url")
        )
        author = post.get("author") or None
        if author in ("[deleted]", "AutoModerator"):
            return None

        created_utc = post.get("created_utc")
        published = self._iso_from_epoch(created_utc)

        author_metadata = {
            # The public JSON has limited author data — these are coarse
            # heuristics, but we still record them for the dashboard.
            "subreddit": spec.name,
            "subreddit_subscribers": post.get("subreddit_subscribers"),
            "score": post.get("score"),
            "num_comments": post.get("num_comments"),
            "upvote_ratio": post.get("upvote_ratio"),
            "is_self": post.get("is_self"),
            "stickied": post.get("stickied"),
            "over_18": post.get("over_18"),
            "spoiler": post.get("spoiler"),
            "is_video": post.get("is_video"),
        }

        # Don't ingest stickied mod posts or daily/megathreads
        if post.get("stickied"):
            return None

        # Extract tickers
        extracted = TICKER_EXTRACTOR.extract(title, selftext)

        # Anti-pump heuristic: pump-prone sub + post < 1h old + only-extracted
        # tickers are 4-5 char (typical penny ticker shape)
        post_age_seconds = (
            (datetime.now(timezone.utc).timestamp() - created_utc)
            if isinstance(created_utc, (int, float))
            else None
        )
        looks_pumpy = bool(
            spec.treat_as_pump_prone
            and (post_age_seconds is not None and post_age_seconds < 3600)
            and extracted
            and all(len(t.ticker) >= 4 for t in extracted)
        )

        tickers = [
            TickerMention(
                ticker=t.ticker,
                market="US",
                asset_class=None,
                confidence=t.confidence * (0.7 if looks_pumpy else 1.0),
            )
            for t in extracted
        ]

        return ParsedSignal(
            external_id=external_id,
            title=title or None,
            body=(selftext or None) if selftext else None,
            url=url,
            author=author,
            author_metadata=author_metadata,
            published_at=published,
            raw_payload={
                "subreddit": spec.name,
                "subreddit_label": spec.label,
                "subreddit_weight": spec.source_weight,
                "treat_as_pump_prone": spec.treat_as_pump_prone,
                "anti_pump_flag": looks_pumpy,
                "title": title,
                "selftext": selftext[:2000] if selftext else None,  # cap stored body
                "url": url,
                "score": post.get("score"),
                "num_comments": post.get("num_comments"),
                "upvote_ratio": post.get("upvote_ratio"),
                "ticker_extractions": [
                    {
                        "ticker": t.ticker,
                        "confidence": t.confidence,
                        "source": t.source,
                    }
                    for t in extracted
                ],
            },
            tickers=tickers,
        )

    # ------------------------------------------------------------------

    # Like RssNewsIngestor: write `source` per-subreddit rather than the
    # aggregator label, so dashboard analytics work per sub.
    def poll(self):  # type: ignore[override]
        from .base import PollResult
        from ..storage import get_connection, insert_raw_signal
        from ..storage.db import record_daemon_health

        result = PollResult()
        try:
            raw_entries = list(self.fetch())
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] fetch failed", self.source)
            with get_connection() as conn:
                record_daemon_health(conn, source=self.source, success=False, error=str(exc))
            result.errors += 1
            return result

        result.fetched = len(raw_entries)
        if not raw_entries:
            with get_connection() as conn:
                record_daemon_health(conn, source=self.source, success=True)
            return result

        with get_connection() as conn:
            for raw in raw_entries:
                try:
                    signal = self.parse(raw)
                except Exception as exc:  # noqa: BLE001
                    log.warning("[%s] parse failed: %s", self.source, exc, exc_info=True)
                    result.errors += 1
                    continue
                if signal is None:
                    continue
                sub_source = f"reddit_{raw['spec'].name.lower()}"
                try:
                    inserted_id = insert_raw_signal(
                        conn,
                        source=sub_source,
                        source_tier=self.source_tier,
                        external_id=signal.external_id,
                        url=signal.url,
                        title=signal.title,
                        body=signal.body,
                        author=signal.author,
                        author_metadata=signal.author_metadata,
                        raw_payload=signal.raw_payload,
                        published_at=signal.published_at,
                        tickers=[t.to_dict() for t in signal.tickers],
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "[%s/%s] insert failed for %s: %s",
                        self.source, sub_source, signal.external_id, exc,
                    )
                    result.errors += 1
                    continue
                if inserted_id is None:
                    result.duplicates += 1
                else:
                    result.inserted += 1
            record_daemon_health(conn, source=self.source, success=result.errors == 0)

        log.info(
            "[%s] poll done: fetched=%d inserted=%d dup=%d err=%d",
            self.source, result.fetched, result.inserted,
            result.duplicates, result.errors,
        )
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _iso_from_epoch(value: Any) -> Optional[str]:
        if not isinstance(value, (int, float)):
            return None
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OverflowError, OSError, ValueError):
            return None
