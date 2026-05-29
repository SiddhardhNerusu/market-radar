"""Tiered signal ingestors. Each module exports an Ingestor subclass."""
from .base import Ingestor, ParsedSignal, PollResult, TickerMention
from .reddit_public import RedditPublicIngestor
from .rss_news import RssNewsIngestor
from .sec_edgar import SecEdgarIngestor
from .stocktwits import StockTwitsTrendingIngestor
from .ticker_extractor import TICKER_EXTRACTOR, TickerExtractor

__all__ = [
    "Ingestor",
    "ParsedSignal",
    "PollResult",
    "TickerMention",
    "SecEdgarIngestor",
    "RssNewsIngestor",
    "RedditPublicIngestor",
    "StockTwitsTrendingIngestor",
    "TickerExtractor",
    "TICKER_EXTRACTOR",
]
