"""Tiered signal ingestors. Each module exports an Ingestor subclass."""
from .alpaca_news import AlpacaNewsIngestor
from .base import Ingestor, ParsedSignal, PollResult, TickerMention
from .halts import NasdaqHaltsIngestor
from .market_movers import MarketMoversIngestor
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
    "AlpacaNewsIngestor",
    "NasdaqHaltsIngestor",
    "MarketMoversIngestor",
    "TickerExtractor",
    "TICKER_EXTRACTOR",
]
