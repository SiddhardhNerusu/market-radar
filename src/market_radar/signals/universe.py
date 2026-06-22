"""Curated trading universe — names with deep liquidity, tight spreads,
and active options chains. Engineered for a small-account algo bot
where slippage is a real cost.

Selection criteria:
  - Average daily $-volume > $500M (we never move the market)
  - Bid-ask spread typically < 5 bps
  - Listed options with tight bid-ask
  - Not on overnight halt list / not in heavy reverse-split risk
"""
from __future__ import annotations

# ----------------------------------------------------------------------
# US equities — top liquidity. Refresh quarterly.
# ----------------------------------------------------------------------
US_LARGE_CAPS: tuple[str, ...] = (
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "AVGO",
    "ORCL", "ADBE", "AMD", "CRM", "QCOM", "INTC", "CSCO", "NFLX",
    # Banks / financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "V", "MA", "AXP",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT",
    # Consumer
    "WMT", "COST", "HD", "LOW", "PG", "KO", "PEP", "NKE", "MCD", "SBUX",
    "DIS", "TGT",
    # Energy
    "XOM", "CVX", "COP",
    # Industrials / cyclicals
    "BA", "CAT", "GE", "F", "GM",
    # Momentum favourites
    "TSLA", "PLTR", "COIN", "RIVN", "MARA", "RIOT", "SOFI",
)

# Liquid ETFs — index plays + sector rotation
US_ETFS: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA", "VTI",   # broad
    "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE",  # sectors
    "TLT", "HYG", "GLD", "SLV", "UNG", "USO",  # macro
    "ARKK",  # high-beta growth
    "SOXX", "SMH",  # semis
)

# Crypto pairs — ALL Alpaca-supported tradeable coins.
# Includes high-vol memecoins (DOGE, SHIB), DeFi (AAVE, UNI, MKR), and majors.
# Stablecoins (USDC, USDT) excluded because they don't move.
# Alpaca crypto trading: 24/7, fractional sizes, NO leverage, NO PDT.
CRYPTO_PAIRS: tuple[str, ...] = (
    # Majors (deepest liquidity)
    "BTC/USD", "ETH/USD",
    # Layer 1 alts (medium-high vol)
    "LTC/USD", "BCH/USD", "DOT/USD", "AVAX/USD", "XTZ/USD",
    # Smart-contract / oracles
    "LINK/USD", "GRT/USD",
    # DeFi (often pump on news) — MKR removed (not tradeable on Alpaca paper)
    "AAVE/USD", "UNI/USD", "CRV/USD", "SUSHI/USD", "YFI/USD",
    # Memecoins — highest "explode" potential, also highest dump risk
    "DOGE/USD", "SHIB/USD",
    # Utility
    "BAT/USD",
)

# EQUITY-ONLY since the full-audit rebuild: the crypto lane is decommissioned
# (Alpaca spot can't be shorted; the lane was unbookable + half-untradeable), so the
# scanner no longer scans crypto at source — this stops ~500 crypto names/day being
# scored only to be blocked at the execution gate. CRYPTO_PAIRS is kept for reference
# but is no longer part of the scanned universe.
LIQUID_EQUITIES: tuple[str, ...] = US_LARGE_CAPS + US_ETFS
LIQUID_UNIVERSE: tuple[str, ...] = LIQUID_EQUITIES

# Sector mapping for the macro regime / risk manager.
SECTOR_OF: dict[str, str] = {
    **{t: "tech" for t in (
        "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AMD", "AVGO",
        "ORCL", "CRM", "ADBE", "QCOM", "INTC", "CSCO", "PLTR", "NFLX",
    )},
    **{t: "banks" for t in ("JPM", "BAC", "WFC", "GS", "MS", "C", "BLK",
                            "V", "MA", "AXP")},
    **{t: "pharma" for t in ("UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
                             "TMO", "ABT")},
    **{t: "retail" for t in ("WMT", "COST", "HD", "LOW", "TGT", "NKE",
                             "MCD", "SBUX", "DIS")},
    **{t: "energy" for t in ("XOM", "CVX", "COP", "USO", "UNG", "XLE")},
    **{t: "auto" for t in ("TSLA", "F", "GM", "RIVN")},
    **{t: "crypto" for t in ("COIN", "MARA", "RIOT")},
}


def normalize_alpaca(ticker: str) -> str:
    """Map our internal ticker forms to Alpaca's expected form.

    Alpaca uses ``BRK.B`` not ``BRK-B``. Crypto uses ``BTC/USD`` not ``BTCUSD``.
    """
    t = ticker.strip().upper()
    if t in {"BRK-B", "BRK.B"}:
        return "BRK/B"  # Alpaca's odd format
    return t
