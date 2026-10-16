"""Trade execution layer.

Bridges the signal/ML/risk pipeline to a real broker (Alpaca).

Modules
-------
- ``alpaca_client``  : raw HTTP client for Alpaca's REST trading API.
- ``sizer``          : Kelly-fractional position sizing + ATR-based stop / TP.
- ``order_gateway``  : idempotent order submission + bookkeeping.
- ``pnl_reconcile``  : reconciles local trade records against broker fills.
"""
from .alpaca_client import AlpacaClient, AlpacaError, BracketOrder, Order, Position
from .sizer import SizingResult, size_trade

__all__ = [
    "AlpacaClient",
    "AlpacaError",
    "BracketOrder",
    "Order",
    "Position",
    "SizingResult",
    "size_trade",
]
