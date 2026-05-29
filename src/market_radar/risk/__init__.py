"""Risk infrastructure for MARKET RADAR.

Every real-money trade MUST pass through RiskManager.evaluate() before
hitting an execution adapter.  See manager.py for full contract.
"""
from .manager import RiskManager, TradeProposal, RiskDecision, SECTOR_MAP

__all__ = ["RiskManager", "TradeProposal", "RiskDecision", "SECTOR_MAP"]
