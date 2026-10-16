"""Tiered signal ingestors. Each module exports an Ingestor subclass."""
from .alpaca_news import AlpacaNewsIngestor
from .base import Ingestor, ParsedSignal, PollResult, TickerMention
from .halts import NasdaqHaltsIngestor
from .market_movers import MarketMoversIngestor
from .sec_edgar import SecEdgarIngestor
from .ticker_extractor import TICKER_EXTRACTOR, TickerExtractor

__all__ = [
    "Ingestor",
    "ParsedSignal",
    "PollResult",
    "TickerMention",
    "SecEdgarIngestor",
    "AlpacaNewsIngestor",
    "NasdaqHaltsIngestor",
    "MarketMoversIngestor",
    "TickerExtractor",
    "TICKER_EXTRACTOR",
]
