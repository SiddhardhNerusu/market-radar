"""Graph-derived features — sector peer momentum, co-mention peers,
crypto-equity sympathetic moves.

All computed from data we already have:
  - Co-mention graph: built from ``raw_signals`` × ``signal_tickers``
    by counting how often each ticker pair appears in titles within
    the same 24h window. Top-5 most-co-mentioned peers per ticker over
    the trailing 90 days are this signal's "co-mention peers."
  - Sector peers: hardcoded GICS-industry rollup for the top liquid
    tickers. Median 5-day return of same-industry peers (excluding
    the signal's ticker itself) on signal date.
  - Crypto-equity: for the 25-ticker crypto-sensitive set (RIOT, MARA,
    COIN, MSTR, HUT, …), BTC's 5-day return is used as a co-move feature.

These are the "graph" inputs both to (a) the main HGB model and (b) the
separate graph-only model that predicts purely from graph features.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)


# Hardcoded crypto-correlated tickers. BTC moves these in the same direction
# more often than not; keeping the list small and curated.
CRYPTO_CORRELATED: set[str] = {
    "RIOT", "MARA", "CLSK", "HUT", "BITF", "BTBT", "CIFR", "WULF",
    "COIN", "MSTR", "GLXY", "HOOD",
    "GBTC", "IBIT", "FBTC", "ARKB", "BITO",
    "MOGO", "EXOD", "SQ", "PYPL",
}

# Compact GICS-industry-ish rollup for ~150 large liquid US names. Keeps
# the sector-peer feature meaningful for the most-traded tickers.
# Each value is a list of peer tickers in the same industry bucket.
SECTOR_PEERS: dict[str, list[str]] = {
    # Mega-cap tech
    "AAPL": ["MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA"],
    "MSFT": ["AAPL", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "ORCL"],
    "GOOGL": ["AAPL", "MSFT", "META", "AMZN", "NVDA", "GOOG"],
    "GOOG": ["AAPL", "MSFT", "META", "AMZN", "NVDA", "GOOGL"],
    "META": ["GOOGL", "GOOG", "AMZN", "AAPL", "MSFT", "SNAP", "PINS"],
    "AMZN": ["MSFT", "GOOGL", "META", "AAPL", "NVDA", "WMT"],
    "NVDA": ["AMD", "AVGO", "INTC", "QCOM", "MU", "MSFT", "TSM"],
    "AMD":  ["NVDA", "INTC", "AVGO", "QCOM", "MU", "MRVL"],
    "INTC": ["NVDA", "AMD", "AVGO", "QCOM", "MU", "TSM"],
    "TSM":  ["NVDA", "INTC", "AMD", "AVGO", "QCOM", "ASML"],
    "AVGO": ["NVDA", "AMD", "QCOM", "MRVL", "ADI", "TXN"],
    "QCOM": ["AVGO", "NVDA", "MU", "ADI", "TXN", "MRVL"],
    "MU":   ["NVDA", "INTC", "QCOM", "AVGO", "WDC", "STX"],
    "ASML": ["NVDA", "TSM", "AMAT", "LRCX", "KLAC"],
    "AMAT": ["LRCX", "KLAC", "ASML", "NVDA"],
    "LRCX": ["AMAT", "KLAC", "ASML"],
    # Mega-cap auto / EV
    "TSLA": ["RIVN", "LCID", "F", "GM", "NIO", "XPEV", "LI"],
    "F":    ["GM", "TSLA", "RIVN", "STLA", "TM"],
    "GM":   ["F",  "TSLA", "RIVN", "STLA"],
    "RIVN": ["TSLA", "LCID", "F", "GM"],
    "LCID": ["TSLA", "RIVN", "F", "GM"],
    "NIO":  ["XPEV", "LI", "TSLA", "RIVN", "LCID"],
    # Banks
    "JPM": ["BAC", "C", "WFC", "GS", "MS"],
    "BAC": ["JPM", "C", "WFC", "GS", "MS"],
    "WFC": ["JPM", "BAC", "C", "USB", "PNC"],
    "C":   ["JPM", "BAC", "WFC", "GS", "MS"],
    "GS":  ["MS", "JPM", "BAC", "C"],
    "MS":  ["GS", "JPM", "BAC", "C"],
    # Retail / consumer
    "WMT": ["TGT", "COST", "AMZN", "HD", "LOW"],
    "TGT": ["WMT", "COST", "AMZN"],
    "COST": ["WMT", "TGT", "AMZN"],
    "HD":  ["LOW", "WMT", "TGT"],
    "LOW": ["HD", "WMT", "TGT"],
    "NKE": ["LULU", "UAA", "DKS"],
    "LULU": ["NKE", "UAA"],
    # Healthcare / biotech mega-cap
    "JNJ": ["PFE", "MRK", "LLY", "ABBV", "BMY"],
    "PFE": ["JNJ", "MRK", "LLY", "ABBV", "BMY"],
    "MRK": ["PFE", "JNJ", "LLY", "ABBV", "BMY"],
    "LLY": ["MRK", "JNJ", "PFE", "ABBV", "NVO"],
    "ABBV": ["JNJ", "PFE", "MRK", "BMY"],
    "BMY":  ["PFE", "MRK", "ABBV", "JNJ"],
    "NVO":  ["LLY"],
    # Travel / leisure
    "AAL": ["UAL", "DAL", "LUV"],
    "UAL": ["AAL", "DAL", "LUV"],
    "DAL": ["AAL", "UAL", "LUV"],
    "LUV": ["DAL", "UAL", "AAL"],
    "BKNG": ["EXPE", "ABNB"],
    "ABNB": ["BKNG", "EXPE"],
    # Crypto-correlated (treat as own sector for sympathy)
    "RIOT": ["MARA", "CLSK", "HUT", "BITF", "WULF", "BTBT", "CIFR"],
    "MARA": ["RIOT", "CLSK", "HUT", "BITF", "WULF", "BTBT", "CIFR"],
    "CLSK": ["RIOT", "MARA", "HUT", "BITF", "WULF"],
    "COIN": ["MSTR", "HOOD", "RIOT", "MARA"],
    "MSTR": ["COIN", "RIOT", "MARA"],
    # Defense
    "LMT": ["RTX", "NOC", "GD", "BA"],
    "RTX": ["LMT", "NOC", "GD", "BA"],
    "NOC": ["LMT", "RTX", "GD", "BA"],
    "GD":  ["LMT", "RTX", "NOC", "BA"],
    "BA":  ["LMT", "RTX", "NOC", "GD"],
    # Energy
    "XOM": ["CVX", "COP", "PXD", "BP", "SHEL"],
    "CVX": ["XOM", "COP", "PXD", "BP", "SHEL"],
    "COP": ["XOM", "CVX", "PXD", "BP"],
    # ETFs (mostly used as benchmarks; peers = sector ETF set)
    "SPY": ["IVV", "VOO", "QQQ", "DIA", "IWM"],
    "QQQ": ["SPY", "VOO", "DIA", "IWM"],
    "IWM": ["SPY", "QQQ", "VOO", "DIA"],
    "DIA": ["SPY", "QQQ", "IWM"],
}


def _normalize_ticker(t: str) -> str:
    t = (t or "").upper()
    if "_" in t:
        parts = t.split("_")
        if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
            t = "_".join(parts[:-2]).replace("_", "-")
    return t


# ---- Co-mention graph -----------------------------------------------------

def build_co_mention_graph(*, days: int = 90, top_k: int = 5) -> dict[str, list[str]]:
    """Build a peer-mapping of {ticker: [top_k most-co-mentioned peers]}.

    Two tickers "co-mention" when they appear in distinct raw_signals
    with the same content_hash within a 24h window. We aggregate over
    the trailing ``days`` days.
    """
    from ..storage import get_connection

    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT rs.id AS sid, rs.content_hash, st.ticker, rs.published_at
                FROM raw_signals rs
                JOIN signal_tickers st ON st.signal_id = rs.id
                WHERE rs.published_at >= ?
                """,
                (cutoff,),
            ).fetchall()
    except Exception:  # noqa: BLE001
        return {}

    # Group by content_hash → set of tickers (and skip same-signal pairs)
    by_hash: dict[str, set[str]] = defaultdict(set)
    by_signal: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        ticker = _normalize_ticker(r["ticker"] or "")
        if not ticker:
            continue
        if r["content_hash"]:
            by_hash[r["content_hash"]].add(ticker)
        by_signal[r["sid"]].add(ticker)

    co_counts: dict[str, Counter] = defaultdict(Counter)
    for tickers in list(by_hash.values()) + list(by_signal.values()):
        tickers = list(tickers)
        for i, a in enumerate(tickers):
            for b in tickers[i + 1:]:
                co_counts[a][b] += 1
                co_counts[b][a] += 1

    peers: dict[str, list[str]] = {}
    for t, counter in co_counts.items():
        peers[t] = [pair for pair, _ in counter.most_common(top_k)]
    return peers


# ---- Compute graph features per row --------------------------------------

def _median(xs: list[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs = sorted(xs)
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2


def attach_graph_features(rows: list[dict], market_cache,
                           *, co_mention_peers: Optional[dict] = None) -> None:
    """In-place attach sector_peer_5d_return, co_mention_peer_5d_return,
    crypto_btc_5d_return, is_crypto_correlated.

    ``market_cache`` is a ``MarketFeatureCache`` already warmed for the
    tickers in question. ``co_mention_peers`` defaults to a freshly-built
    graph from the last 90 days.
    """
    if co_mention_peers is None:
        co_mention_peers = build_co_mention_graph()

    # BTC reference for crypto-correlation feature. We piggyback on the
    # market cache; if the user warms it for BTC-USD it'll be there.
    btc_cache_key = "BTC-USD"

    for r in rows:
        ticker = _normalize_ticker(r.get("ticker") or "")
        ts_str = r.get("price_at_flag_ts") or r.get("published_at") or r.get("scored_at")
        if not ticker or not isinstance(ts_str, str):
            r.setdefault("sector_peer_5d_return", 0.0)
            r.setdefault("co_mention_peer_5d_return", 0.0)
            r.setdefault("crypto_btc_5d_return", 0.0)
            r.setdefault("is_crypto_correlated", 0.0)
            continue
        try:
            d = datetime.strptime(ts_str[:10], "%Y-%m-%d").date()
        except ValueError:
            continue

        # Sector peers (excluding self)
        sector_peer_returns: list[float] = []
        for peer in SECTOR_PEERS.get(ticker, [])[:8]:
            if peer == ticker:
                continue
            try:
                ret = market_cache._n_day_return(peer, d, 5)
            except AttributeError:
                ret = None
            if ret is not None:
                sector_peer_returns.append(ret)

        # Co-mention peers (graph built above)
        com_peer_returns: list[float] = []
        for peer in co_mention_peers.get(ticker, [])[:5]:
            try:
                ret = market_cache._n_day_return(peer, d, 5)
            except AttributeError:
                ret = None
            if ret is not None:
                com_peer_returns.append(ret)

        # Crypto-equity feature
        is_crypto = 1.0 if ticker in CRYPTO_CORRELATED else 0.0
        btc_ret = None
        if is_crypto:
            try:
                btc_ret = market_cache._n_day_return(btc_cache_key, d, 5)
            except AttributeError:
                btc_ret = None

        r["sector_peer_5d_return"] = _median(sector_peer_returns) or 0.0
        r["co_mention_peer_5d_return"] = _median(com_peer_returns) or 0.0
        r["crypto_btc_5d_return"] = btc_ret or 0.0
        r["is_crypto_correlated"] = is_crypto


# ---- Feature subset for the graph-only model -----------------------------

GRAPH_ONLY_FEATURE_NAMES: list[str] = [
    "sector_peer_5d_return",
    "co_mention_peer_5d_return",
    "crypto_btc_5d_return",
    "is_crypto_correlated",
    "spy_5d_return",
    "qqq_5d_return",
    "yield_curve_slope",
    "dxy_5d_return",
    "oil_5d_return",
    "gold_5d_return",
    "vix_level",
    "volume_ratio_5d_20d",
    "realized_vol_20d",
    "unique_sources_24h",
    "institutional_buyers_minus_sellers_qoq",
]
