"""Price-action + macro signal generators.

These run alongside the news/social ingestors but generate signals
from observed price action on a curated liquid universe. They
materially outnumber news signals (~30-100/day vs ~3/day) and have
tighter intra-bar latency.

Modules
-------
- ``price_action`` : opening-range breakout, VWAP cross, Donchian
                     breakout, RSI extremes, volume spike, gap-and-go.
- ``macro_regime`` : VIX level, SPY trend, 10Y-2Y spread, sector RS.
                     Used as a global gate/multiplier by the live trader.
- ``universe``     : curated list of liquid US stocks + ETFs + crypto.
"""
from .price_action import PriceActionScanner, ScanStats
from .universe import LIQUID_UNIVERSE

__all__ = ["PriceActionScanner", "ScanStats", "LIQUID_UNIVERSE"]
