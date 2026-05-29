"""US mega-cap ticker list — S&P 100 + selected high-coverage tickers.

Signals on these tickers get a small composite boost because:
  1. They have deep liquidity (small edge → tradable)
  2. They get heavy news coverage (high corroboration)
  3. Most retail dashboards focus on these

Refreshed manually; rebuild from a free source quarterly.
"""

MEGACAPS: frozenset[str] = frozenset({
    # FAANG / mega tech
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "ORCL", "ADBE", "CRM", "AMD", "QCOM", "INTC", "TXN", "CSCO",
    "INTU", "IBM", "NOW", "AMAT", "MU", "LRCX", "KLAC", "PANW", "SNPS",
    "CDNS", "ANET", "FTNT", "WDAY", "ADSK",

    # Mega financials
    "BRK.B", "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "C",
    "AXP", "BLK", "SCHW", "USB", "PNC", "TFC", "SPGI", "MCO", "CB", "PGR",
    "TRV", "AIG", "MET", "PRU",

    # Healthcare giants
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "ELV", "CVS", "CI", "GILD", "MDT", "ISRG", "REGN", "VRTX", "SYK",
    "ZTS", "BSX",

    # Consumer / retail / industrial
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "HD", "LOW", "TGT", "NKE",
    "SBUX", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "BKNG", "CHTR",

    # Energy / commodities
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "PXD",
    "WMB", "KMI",

    # Industrials
    "BA", "CAT", "DE", "GE", "HON", "LMT", "RTX", "UPS", "FDX", "MMM",
    "ETN", "EMR", "ITW", "GD", "NOC", "PH", "ROP", "CMI", "PCAR",

    # Other large caps
    "BRK", "F", "GM", "TSM", "ASML", "TSL",
})
