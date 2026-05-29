"""Alpaca options trading layer — defined-risk vertical debit spreads.

Why this exists
---------------
Stock-only trading on a $6,300 account is hard-capped at ~£20-£60/day average.
Options spreads multiply capital efficiency ~3-5×: a $50 debit spread risks
$50 to capture potentially $150 of profit on the right move. With the bot's
measured 75% hit rate at ``model_p>=0.65``, vertical debit spreads have a
positive expected value of ~$70/trade — the realistic path toward £150/day.

Risk shape
----------
Only ``debit spreads`` are supported in v1 (long ATM + short OTM, same expiry).
Max loss = debit paid (defined). Max gain = (width − debit). No naked legs,
no margin/assignment risk on the short leg as long as we close before expiry.

Modules
-------
- ``alpaca_options`` : extends AlpacaClient with chain fetch, quote, multi-leg submit
- ``spreads``        : strategy picker (bull-call / bear-put) + strike selection
                       + asymmetric Kelly sizing
- ``spread_trader``  : the loop integration — read signal → build spread →
                       size → risk-gate → submit → reconcile.

Defaults sized for a $6,300 account with 5% per-position cap = $315 max loss.
Whitelist of underlyings deliberately tiny (10 names) — only the very most
liquid option chains. Anything else falls through to the stock path.
"""
from .alpaca_options import (
    AlpacaOptionsClient,
    OptionContract,
    OptionQuote,
    OptionLeg,
    MultiLegOrder,
)
from .spreads import (
    SpreadSpec,
    SpreadSizing,
    build_vertical_debit_spread,
    OPTIONS_UNDERLYINGS,
)

__all__ = [
    "AlpacaOptionsClient",
    "OptionContract",
    "OptionQuote",
    "OptionLeg",
    "MultiLegOrder",
    "SpreadSpec",
    "SpreadSizing",
    "build_vertical_debit_spread",
    "OPTIONS_UNDERLYINGS",
]
