"""RSS news aggregator — Tier 2 financial news sources.

Each entry in ``FEEDS`` is an independently polled feed. Sources are picked
to be reliable, redundant (multiple sources → easier corroboration scoring),
and free to access. We deliberately avoid Seeking Alpha because their feed
has gotten increasingly aggressive about blocking non-browser clients.

Each parsed article goes through ``ticker_extractor`` to identify the
companies it's about. Articles with no ticker match are still recorded
(with empty signal_tickers) so we can re-process later if our ticker
universe grows.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import feedparser
import requests

from .base import Ingestor, ParsedSignal, TickerMention
from .ticker_extractor import TICKER_EXTRACTOR

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedSpec:
    source_id: str          # canonical source name written into raw_signals.source
    url: str
    label: str              # human-readable label for logs
    source_weight: float    # 1-10 for the scorer (Tier 2 baseline is 7)
    ticker_hint: Optional[str] = None  # optional ticker if the feed is per-symbol


# Tier 2 feed registry — order doesn't matter, dedup is per (source, external_id).
FEEDS: list[FeedSpec] = [
    # MarketWatch
    FeedSpec(
        source_id="marketwatch_topstories",
        url="http://feeds.marketwatch.com/marketwatch/topstories/",
        label="MarketWatch — Top Stories",
        source_weight=7.5,
    ),
    FeedSpec(
        source_id="marketwatch_realtime",
        url="http://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
        label="MarketWatch — Real-Time Headlines",
        source_weight=7.5,
    ),
    FeedSpec(
        source_id="marketwatch_marketpulse",
        url="http://feeds.marketwatch.com/marketwatch/marketpulse/",
        label="MarketWatch — Market Pulse",
        source_weight=7.5,
    ),
    # CNBC
    FeedSpec(
        source_id="cnbc_topnews",
        url="https://www.cnbc.com/id/100003114/device/rss/rss.html",
        label="CNBC — Top News",
        source_weight=7.0,
    ),
    FeedSpec(
        source_id="cnbc_business",
        url="https://www.cnbc.com/id/10001147/device/rss/rss.html",
        label="CNBC — Business",
        source_weight=7.0,
    ),
    FeedSpec(
        source_id="cnbc_markets",
        url="https://www.cnbc.com/id/15839069/device/rss/rss.html",
        label="CNBC — Markets",
        source_weight=7.0,
    ),
    FeedSpec(
        source_id="cnbc_earnings",
        url="https://www.cnbc.com/id/15839135/device/rss/rss.html",
        label="CNBC — Earnings",
        source_weight=7.5,
    ),
    # Yahoo Finance top stories
    FeedSpec(
        source_id="yahoo_finance_news",
        url="https://finance.yahoo.com/news/rssindex",
        label="Yahoo Finance — News Index",
        source_weight=6.5,
    ),
    # Reuters via Google News (Reuters' own RSS was deprecated)
    FeedSpec(
        source_id="reuters_google",
        url="https://news.google.com/rss/search?q=site:reuters.com+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Reuters (via Google News, last 24h)",
        source_weight=8.0,
    ),
    # Bloomberg via Google News
    FeedSpec(
        source_id="bloomberg_google",
        url="https://news.google.com/rss/search?q=site:bloomberg.com+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Bloomberg (via Google News, last 24h)",
        source_weight=8.5,
    ),
    # The Wall Street Journal via Google News
    FeedSpec(
        source_id="wsj_google",
        url="https://news.google.com/rss/search?q=site:wsj.com+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="WSJ (via Google News, last 24h)",
        source_weight=8.0,
    ),
    # Benzinga (often picks up things wires don't)
    FeedSpec(
        source_id="benzinga_news",
        url="https://www.benzinga.com/feed",
        label="Benzinga — News",
        source_weight=6.0,
    ),
    # Investing.com
    FeedSpec(
        source_id="investing_general",
        url="https://www.investing.com/rss/news.rss",
        label="Investing.com — News",
        source_weight=6.0,
    ),
    FeedSpec(
        source_id="investing_stockmarket",
        url="https://www.investing.com/rss/news_25.rss",
        label="Investing.com — Stock Market",
        source_weight=6.0,
    ),
    # Reuters business via Google News (broader query)
    FeedSpec(
        source_id="reuters_business_google",
        url="https://news.google.com/rss/search?q=site:reuters.com+stocks+OR+earnings+OR+merger+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Reuters business (via Google News)",
        source_weight=8.0,
    ),
    # ── Catalyst-themed Google News queries — surface high-impact events
    # across all reputable sources at once.
    FeedSpec(
        source_id="gnews_ma_topic",
        url="https://news.google.com/rss/search?q=acquisition+OR+merger+OR+takeover+stock+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — M&A topic",
        source_weight=7.0,
    ),
    FeedSpec(
        source_id="gnews_earnings_topic",
        url="https://news.google.com/rss/search?q=%22beats+earnings%22+OR+%22misses+earnings%22+OR+%22Q1+earnings%22+OR+%22Q4+earnings%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Earnings topic",
        source_weight=7.0,
    ),
    FeedSpec(
        source_id="gnews_fda_topic",
        url="https://news.google.com/rss/search?q=%22FDA+approval%22+OR+%22FDA+rejection%22+OR+%22Complete+Response+Letter%22+OR+%22Phase+3%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — FDA / biotech topic",
        source_weight=7.5,
    ),
    FeedSpec(
        source_id="gnews_buyback_topic",
        url="https://news.google.com/rss/search?q=%22share+buyback%22+OR+%22stock+repurchase%22+OR+%22authorizes+buyback%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Buyback topic",
        source_weight=6.5,
    ),
    FeedSpec(
        source_id="gnews_fed_macro",
        url="https://news.google.com/rss/search?q=%22Federal+Reserve%22+OR+CPI+OR+%22jobs+report%22+OR+%22interest+rates%22+OR+FOMC+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Fed / macro",
        source_weight=8.0,
    ),
    FeedSpec(
        source_id="gnews_analyst_topic",
        url="https://news.google.com/rss/search?q=%22upgraded+to%22+OR+%22downgraded+to%22+OR+%22price+target%22+OR+%22analyst+rating%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Analyst actions",
        source_weight=6.5,
    ),
    FeedSpec(
        source_id="gnews_insider_topic",
        url="https://news.google.com/rss/search?q=%22insider+buying%22+OR+%22insider+selling%22+OR+%22Form+4%22+OR+%22CEO+buys%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Insider activity",
        source_weight=7.0,
    ),
    FeedSpec(
        source_id="gnews_ipo_topic",
        url="https://news.google.com/rss/search?q=%22IPO%22+OR+%22goes+public%22+OR+%22direct+listing%22+OR+%22prices+IPO%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — IPO topic",
        source_weight=6.5,
    ),
    FeedSpec(
        source_id="gnews_layoffs_restructure",
        url="https://news.google.com/rss/search?q=%22layoffs%22+OR+%22restructuring%22+OR+%22job+cuts%22+OR+%22workforce+reduction%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Layoffs / restructure",
        source_weight=6.5,
    ),
    FeedSpec(
        source_id="gnews_lawsuit_regulatory",
        url="https://news.google.com/rss/search?q=%22SEC+charges%22+OR+%22DOJ+probe%22+OR+%22class+action%22+OR+%22antitrust%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Legal / regulatory",
        source_weight=7.0,
    ),
    # Additional aggregators
    FeedSpec(
        source_id="seekingalpha_news",
        url="https://seekingalpha.com/feed.xml",
        label="Seeking Alpha — News",
        source_weight=6.5,
    ),
    FeedSpec(
        source_id="zacks_news",
        url="https://www.zacks.com/feed.php",
        label="Zacks — Analyst News",
        source_weight=6.5,
    ),
    FeedSpec(
        source_id="reuters_world_google",
        url="https://news.google.com/rss/search?q=site:reuters.com+when:6h&hl=en-US&gl=US&ceid=US:en",
        label="Reuters — last 6h (more freq)",
        source_weight=8.0,
    ),
    # ── More retail-oriented news sources ──
    FeedSpec(
        source_id="streetinsider_general",
        url="https://www.streetinsider.com/streetinsider_news.rss",
        label="StreetInsider — catalyst news",
        source_weight=7.0,
    ),
    FeedSpec(
        source_id="thestreet_news",
        url="https://www.thestreet.com/.rss/full/",
        label="TheStreet",
        source_weight=6.5,
    ),
    FeedSpec(
        source_id="motley_fool",
        url="https://www.fool.com/feeds/index.aspx",
        label="Motley Fool",
        source_weight=5.5,
    ),
    FeedSpec(
        source_id="investorplace",
        url="https://investorplace.com/feed/",
        label="InvestorPlace",
        source_weight=5.5,
    ),
    FeedSpec(
        source_id="marketbeat_ratings",
        url="https://www.marketbeat.com/news/rss.aspx",
        label="MarketBeat — Ratings",
        source_weight=6.5,
    ),
    FeedSpec(
        source_id="forbes_markets",
        url="https://www.forbes.com/markets/feed/",
        label="Forbes — Markets",
        source_weight=6.0,
    ),
    FeedSpec(
        source_id="bi_markets",
        url="https://www.businessinsider.com/sai.rss",
        label="Business Insider — Markets",
        source_weight=6.0,
    ),
    FeedSpec(
        source_id="briefing_in_play",
        url="https://www.briefing.com/InPlay/InPlay.xml",
        label="Briefing.com — In Play (market hours commentary)",
        source_weight=7.0,
    ),
    # ── Catalyst-specific topical queries we hadn't covered ──
    FeedSpec(
        source_id="gnews_short_squeeze",
        url="https://news.google.com/rss/search?q=%22short+squeeze%22+OR+%22short+interest%22+OR+%22naked+short%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Short squeeze",
        source_weight=6.0,
    ),
    FeedSpec(
        source_id="gnews_clinical_trial",
        url="https://news.google.com/rss/search?q=%22clinical+trial%22+OR+%22Phase+2%22+OR+%22Phase+3%22+stock+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Clinical trial",
        source_weight=7.0,
    ),
    FeedSpec(
        source_id="gnews_guidance",
        url="https://news.google.com/rss/search?q=%22raises+guidance%22+OR+%22lowers+guidance%22+OR+%22cuts+outlook%22+OR+%22raises+outlook%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Guidance change",
        source_weight=7.0,
    ),
    FeedSpec(
        source_id="gnews_ceo_change",
        url="https://news.google.com/rss/search?q=%22CEO+resigns%22+OR+%22appoints+CEO%22+OR+%22steps+down%22+OR+%22new+CEO%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — CEO change",
        source_weight=7.0,
    ),
    FeedSpec(
        source_id="gnews_china_tariffs",
        url="https://news.google.com/rss/search?q=%22China+tariffs%22+OR+%22trade+war%22+OR+%22China+stocks%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — China / tariffs",
        source_weight=7.5,
    ),
    FeedSpec(
        source_id="gnews_dividends",
        url="https://news.google.com/rss/search?q=%22special+dividend%22+OR+%22dividend+increase%22+OR+%22dividend+cut%22+OR+%22suspends+dividend%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Dividend events",
        source_weight=6.5,
    ),
    FeedSpec(
        source_id="gnews_split_spinoff",
        url="https://news.google.com/rss/search?q=%22stock+split%22+OR+%22spinoff%22+OR+%22spin-off%22+OR+%22reverse+split%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Splits / spinoffs",
        source_weight=6.5,
    ),
    FeedSpec(
        source_id="gnews_short_seller",
        url="https://news.google.com/rss/search?q=%22Hindenburg%22+OR+%22Muddy+Waters%22+OR+%22short+report%22+OR+%22Citron+Research%22+when:1d&hl=en-US&gl=US&ceid=US:en",
        label="Google News — Short seller reports",
        source_weight=7.5,
    ),
]


class RssNewsIngestor(Ingestor):
    """Single Ingestor that handles all RSS feeds in ``FEEDS``.

    We treat each feed as a sub-source by setting ``raw_signals.source`` to
    the feed's ``source_id``. The base class's `daemon_health` row tracks
    them collectively as ``rss_news_aggregate``. Per-feed health is logged
    but not (yet) tabulated.
    """

    source = "rss_news_aggregate"
    source_tier = 2

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.0 Safari/605.1.15 "
        "(MARKET RADAR research)"
    )

    def __init__(
        self,
        feeds: Optional[list[FeedSpec]] = None,
        *,
        timeout: float = 20.0,
        per_feed_pause: float = 0.5,
    ) -> None:
        super().__init__()
        self.feeds = feeds or FEEDS
        self.timeout = timeout
        self.per_feed_pause = per_feed_pause
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.USER_AGENT,
                "Accept": (
                    "application/rss+xml,application/atom+xml,application/xml;"
                    "q=0.9,*/*;q=0.8"
                ),
            }
        )

    # ------------------------------------------------------------------

    def fetch(self) -> Iterable[dict[str, Any]]:
        for spec in self.feeds:
            try:
                resp = self._session.get(spec.url, timeout=self.timeout)
                if resp.status_code in (403, 401):
                    log.warning("[%s] %s blocked us (%d)", self.source, spec.source_id, resp.status_code)
                    continue
                if resp.status_code >= 400:
                    log.warning(
                        "[%s] %s HTTP %d",
                        self.source, spec.source_id, resp.status_code,
                    )
                    continue
            except requests.RequestException as exc:
                log.warning("[%s] %s fetch failed: %s", self.source, spec.source_id, exc)
                continue

            parsed = feedparser.parse(resp.content)
            if parsed.bozo and not parsed.entries:
                log.warning(
                    "[%s] %s malformed feed: %s",
                    self.source, spec.source_id,
                    getattr(parsed, "bozo_exception", "unknown"),
                )
                continue

            log.info("[%s] %s → %d entries", self.source, spec.source_id, len(parsed.entries))
            for entry in parsed.entries:
                yield {"spec": spec, "entry": entry}
            time.sleep(self.per_feed_pause)

    # ------------------------------------------------------------------

    def parse(self, raw_entry: dict[str, Any]) -> Optional[ParsedSignal]:
        spec: FeedSpec = raw_entry["spec"]
        entry = raw_entry["entry"]

        title = getattr(entry, "title", None) or ""
        link = getattr(entry, "link", None)
        summary = getattr(entry, "summary", None) or ""
        author = getattr(entry, "author", None)
        published = self._published_iso(entry)

        external_id = self._stable_id(spec, entry)
        if not external_id:
            return None

        # Skip empties — some feeds dump empty placeholder entries
        if not title and not summary:
            return None

        # Extract tickers from title+body
        extracted = TICKER_EXTRACTOR.extract(title, summary)
        tickers = [
            TickerMention(
                ticker=t.ticker,
                market="US",
                asset_class=None,
                confidence=t.confidence,
            )
            for t in extracted
        ]

        return ParsedSignal(
            external_id=external_id,
            title=title or None,
            body=summary or None,
            url=link,
            author=author,
            author_metadata={"feed": spec.source_id, "label": spec.label},
            published_at=published,
            raw_payload={
                "feed": spec.source_id,
                "feed_label": spec.label,
                "feed_weight": spec.source_weight,
                "title": title,
                "summary": summary,
                "link": link,
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

    # The base class writes `source = self.source` (the aggregator label),
    # but we want each row to carry its specific feed id for dashboards and
    # per-feed analytics. Override poll to set source per entry.

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

                feed_source = raw["spec"].source_id
                try:
                    inserted_id = insert_raw_signal(
                        conn,
                        source=feed_source,
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
                        self.source, feed_source, signal.external_id, exc,
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
    def _stable_id(spec: FeedSpec, entry: Any) -> Optional[str]:
        eid = getattr(entry, "id", None)
        if eid:
            return f"{spec.source_id}|{eid}"
        link = getattr(entry, "link", None)
        if link:
            return f"{spec.source_id}|{link}"
        title = getattr(entry, "title", None)
        published = getattr(entry, "published", None)
        if title:
            h = hashlib.sha256(f"{title}|{published or ''}".encode("utf-8")).hexdigest()[:16]
            return f"{spec.source_id}|hash:{h}"
        return None

    @staticmethod
    def _published_iso(entry: Any) -> Optional[str]:
        for attr in ("published_parsed", "updated_parsed"):
            val = getattr(entry, attr, None)
            if val:
                try:
                    dt = datetime(*val[:6], tzinfo=timezone.utc)
                    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except (TypeError, ValueError):
                    continue
        for attr in ("published", "updated"):
            val = getattr(entry, attr, None)
            if isinstance(val, str):
                return val
        return None
