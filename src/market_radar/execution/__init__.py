"""Live trade execution layer.

Bridges the existing signal/ML/risk pipeline to a real broker (Alpaca).

Modules
-------
- ``alpaca_client``  : raw HTTP client for Alpaca's REST trading API.
- ``sizer``          : Kelly-fractional position sizing + ATR-based stop / TP.
- ``live_risk``      : ``LiveRiskManager`` subclass that feeds Alpaca account
                       state (equity, gross exposure, daily P&L, daily trade
                       count) into the existing 7-rule ``RiskManager``.
- ``live_trader``    : main loop that polls fresh ``signal_scores`` rows,
                       runs the selective gate + risk gate, sizes the trade,
                       and submits a bracket order to Alpaca.

Run with::

    python scripts/run_live_trader.py --paper

NEVER set ``--live`` until the system has run paper trading for at least
two weeks and shown a positive Sharpe with realistic slippage / commissions.
"""
from .alpaca_client import AlpacaClient, AlpacaError, BracketOrder, Order, Position
from .live_risk import LiveRiskManager
from .live_trader import LiveTrader, TraderConfig
from .sizer import SizingResult, size_trade

__all__ = [
    "AlpacaClient",
    "AlpacaError",
    "BracketOrder",
    "LiveRiskManager",
    "LiveTrader",
    "Order",
    "Position",
    "SizingResult",
    "TraderConfig",
    "size_trade",
]
