"""Macro signal generators.

Modules
-------
- ``macro_regime`` : VIX level, SPY trend, 10Y-2Y spread, sector RS.
                     Used as a global gate/multiplier by the trader.
"""
from .macro_regime import Regime, get_regime

__all__ = ["Regime", "get_regime"]
