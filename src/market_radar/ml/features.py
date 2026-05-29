"""Feature extraction for the ML model.

Each row in ``signal_scores`` becomes one training/inference example. We
extract a fixed-width feature vector based on the signal's metadata + the
parent ``raw_signals`` row. The feature schema is locked here so the
training and prediction paths produce identical input shapes.

Categorical features are integer-encoded with a small known vocabulary;
unknown categories map to 0 (the "other" slot). This makes the same model
work for fresh signals even if we add new event types later.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

# Locked vocabulary for categorical features. Order matters — never reorder
# or you break the model. Append-only.
EVENT_TYPES = [
    "_unknown_",
    "m_a_announcement", "m_a_rumor", "fda_approval", "fda_rejection",
    "earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut",
    "analyst_upgrade", "analyst_downgrade", "insider_buy", "insider_sell",
    "insider_transaction", "activist_position", "passive_5pct_stake",
    "ipo_registration", "ipo_registration_amend", "macro", "lawsuit",
    "leadership_change", "buyback", "dividend", "material_event",
    "material_event_amend", "proxy_statement", "routine_prospectus",
    "routine_proxy", "speculation", "other",
]
EVENT_TYPE_TO_IDX = {et: i for i, et in enumerate(EVENT_TYPES)}


FEATURE_NAMES: list[str] = [
    # ── Signal-intrinsic features (computed during scoring) ──
    "event_type_idx",
    "source_tier",
    "source_weight",
    "ticker_confidence",
    "sentiment",
    "sentiment_magnitude",
    "factual",
    "corroboration_count",
    "author_quality",
    "anti_pump_flag",
    "composite_score",
    "megacap",
    # ── Time-of-event features ──
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "weekly_offset",     # hours since Monday midnight UTC (0..168)
    # ── Market-context features (per ticker, fetched from yfinance) ──
    "volume_ratio_5d_20d",   # 1.0 = normal, >1 = unusual pickup
    "realized_vol_20d",      # 20d annualised vol in % (typical 15..40)
    # ── Cross-asset macro context ──
    "spy_5d_return",         # SPY's 5-day return preceding signal
    "qqq_5d_return",         # QQQ's 5-day return
    "vix_level",             # VIX value on signal date
    # ── LLM label-quality features ──
    "has_llm_classification",   # 1 if LLM classified this row, else 0
    "llm_confidence",           # LLM's self-reported confidence (0..1)
    # ── Insider-cluster context (Tier 1 upgrade — 2026-05-13) ──
    # Number of distinct Form-4 insiders that filed on the same ticker in
    # the 30 days ending at this signal's published_at. Computed by
    # ``compute_insider_clusters`` and attached to the row dict before
    # extract_features() is called. 0 if no insider context available.
    # Capped at 20.
    "insider_cluster_size_30d",
    # ── Multi-source corroboration window (Tier 2 #19 — 2026-05-13) ──
    # Number of *distinct sources* (not just signal rows) that flagged
    # the same ticker in the trailing 24h window. Strengthens the
    # existing ``corroboration_count`` which only counts signals.
    "unique_sources_24h",
    # ── Macro feature pack (Tier 2 #17 — 2026-05-13) ──
    "yield_curve_slope",       # ^TNX (10Y) – ^IRX (3M) at signal date (pp)
    "dxy_5d_return",           # UUP 5-day return % (dollar proxy)
    "oil_5d_return",           # USO 5-day return %
    "gold_5d_return",          # GLD 5-day return %
    # ── External catalyst (Tier 1 #7 — 2026-05-13) ──
    # Days until next known catalyst (PDUFA / earnings / etc.) for this
    # ticker. Clipped to [-30, 60]; 0 if no upcoming catalyst within 60d.
    "days_until_catalyst",
    # ── Short-interest context (Tier 1 #10 — 2026-05-13) ──
    "short_interest_pct_float",    # % of public float reported short
    "days_to_cover",               # short interest / avg daily volume
    # ── Attention proxies (Tier 2 #14/#15 — 2026-05-13) ──
    "gtrends_zscore",            # Google Trends weekly z-score (free pytrends)
    "wikipedia_pageviews_zscore",  # Wikipedia 30d pageview z-score
    # ── PEAD (Tier 2 #11 — 2026-05-13) ──
    "eps_surprise_pct",          # most recent EPS surprise (Finnhub)
    "days_since_earnings",       # days since the last earnings report (clipped to 0..90)
    # ── 13F institutional crowding (Tier 2 #13 — 2026-05-13) ──
    "institutional_buyers_minus_sellers_qoq",  # net new institutional buyers vs prior Q
    # ── TA-Lib indicators (round-2 free batch — 2026-05-13) ──
    "rsi_14",                  # 0-100, mean-reversion proxy
    "macd_histogram",          # signed acceleration
    "macd_above_signal",       # 1 if MACD > signal line, else 0
    "bb_position_20d",         # 0-1 position in 20d Bollinger band
    "atr_pct",                 # ATR(14) / close, vol-adjustment
    "adx_14",                  # 0-100 trend strength
    "vwap_distance_pct",       # (close - vwap) / vwap × 100
    "breakout_20d",            # 1 if close > max(prior 20d close), -1 if <, 0 if neither
    # ── Calendar features ──
    "days_until_fomc",         # -30..+30; 99 if no upcoming meeting known
    "days_until_cpi",          # -15..+15; 99 if unknown
    "days_until_nfp",          # -15..+15; 99 if unknown
    "is_turn_of_month",        # 1 if within ±3 days of month boundary
    # ── Graph features ──
    "sector_peer_5d_return",       # median 5d return of GICS-industry peers
    "co_mention_peer_5d_return",   # median 5d return of top-5 co-mentioned peers
    "crypto_btc_5d_return",        # BTC 5d return (0 if ticker isn't crypto-correlated)
    "is_crypto_correlated",        # 1 if ticker in crypto-correlated set
    # ── Earnings Whispers alignment ──
    "whisper_minus_consensus_pct",  # +ve if whisper is more bullish than consensus
    # ── Fails-to-deliver ──
    "fails_to_deliver_pct_float",   # latest FTD as % of float, 0 if unknown
    # ── Insider transaction codes ──
    "insider_recent_buys_30d",      # count of P-coded transactions, 30d window
    "insider_recent_sells_30d",     # count of S-coded transactions, 30d window
    "insider_role_score",           # 3=CEO/CFO, 2=Director, 1=other, 0=unknown
    # ── News quality ──
    "news_novelty_score",           # 1 - max cosine sim to prior 24h, 0..1
    "finbert_sentiment_24h",        # mean per-message FinBERT score in 24h, -1..1
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if f == f else default  # NaN guard
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hour_of_day(iso_ts: Optional[str]) -> int:
    if not iso_ts:
        return 0
    try:
        dt = datetime.strptime(iso_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt.hour
    except (TypeError, ValueError):
        return 0


def _day_of_week(iso_ts: Optional[str]) -> int:
    if not iso_ts:
        return 0
    try:
        dt = datetime.strptime(iso_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt.weekday()
    except (TypeError, ValueError):
        return 0


def _weekly_offset(iso_ts: Optional[str]) -> float:
    if not iso_ts:
        return 0.0
    try:
        dt = datetime.strptime(iso_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return float(dt.weekday() * 24 + dt.hour)
    except (TypeError, ValueError):
        return 0.0


def compute_insider_clusters(rows: list[dict[str, Any]]) -> None:
    """In-place: attach ``insider_cluster_size_30d`` to each row.

    For each input row, counts distinct Form-4 insider filings on the
    same ticker in the 30 days ending at the row's published_at /
    scored_at. Distinct insiders are identified by the Form-4 title
    (which contains the reporting person's name).

    Loads the full Form-4 history once from ``raw_signals``, then matches
    against the per-row ticker + timestamp. O(rows × form4s_for_ticker).

    Safe to call in both training and prediction paths. If the SQL fails
    (e.g. table missing in a fresh DB) every row gets a default of 0.
    """
    from collections import defaultdict
    from datetime import datetime, timedelta

    from ..storage import get_connection

    # Default everyone to 0 first so callers don't need a presence check.
    for r in rows:
        r.setdefault("insider_cluster_size_30d", 0)

    if not rows:
        return

    try:
        with get_connection() as conn:
            form4 = conn.execute("""
                SELECT st.ticker, rs.title, rs.published_at
                FROM raw_signals rs
                JOIN signal_tickers st ON st.signal_id = rs.id
                WHERE rs.source LIKE 'sec_edgar%'
                  AND rs.title LIKE '4 - %'
                  AND rs.published_at IS NOT NULL
            """).fetchall()
    except Exception:  # noqa: BLE001 — fall back to all-zeros on schema error
        return

    by_ticker: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for f in form4:
        try:
            ts = datetime.strptime(f["published_at"], "%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError):
            continue
        ticker = (f["ticker"] or "").upper()
        if not ticker:
            continue
        by_ticker[ticker].append((ts, f["title"] or ""))
    for ticker in by_ticker:
        by_ticker[ticker].sort()  # sort by ts so we can short-circuit

    for r in rows:
        ticker = (r.get("ticker") or "").upper()
        if not ticker or ticker not in by_ticker:
            continue
        ts_str = r.get("published_at") or r.get("scored_at")
        if not ts_str:
            continue
        try:
            t1 = datetime.strptime(ts_str[:19] + "Z", "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        t0 = t1 - timedelta(days=30)
        seen: set[str] = set()
        for ts, title in by_ticker[ticker]:
            if ts > t1:
                break
            if ts >= t0 and title:
                seen.add(title)
        # Cap to keep the feature compact for tree-based models
        r["insider_cluster_size_30d"] = min(len(seen), 20)


def compute_corroboration_windows(rows: list[dict[str, Any]],
                                   *, window_hours: int = 24) -> None:
    """In-place: attach ``unique_sources_24h`` to each row.

    For each input row, counts the number of *distinct sources* in
    ``raw_signals`` that flagged the same ticker in the trailing
    ``window_hours`` window ending at the row's ``published_at`` /
    ``scored_at``.

    Different from ``corroboration_count`` (which counts signals): this
    counts *unique sources*. A ticker that hits 8 different RSS feeds is
    a stronger signal than the same ticker hitting one source 8 times.
    """
    from collections import defaultdict
    from datetime import datetime, timedelta

    from ..storage import get_connection

    for r in rows:
        r.setdefault("unique_sources_24h", 0)
    if not rows:
        return

    try:
        with get_connection() as conn:
            sigs = conn.execute("""
                SELECT st.ticker, rs.source, rs.published_at
                FROM raw_signals rs
                JOIN signal_tickers st ON st.signal_id = rs.id
                WHERE rs.published_at IS NOT NULL
            """).fetchall()
    except Exception:  # noqa: BLE001
        return

    by_ticker: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for s in sigs:
        try:
            ts = datetime.strptime(s["published_at"], "%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError):
            continue
        ticker = (s["ticker"] or "").upper()
        if not ticker:
            continue
        by_ticker[ticker].append((ts, s["source"] or ""))
    for ticker in by_ticker:
        by_ticker[ticker].sort()

    window = timedelta(hours=window_hours)
    for r in rows:
        ticker = (r.get("ticker") or "").upper()
        if not ticker or ticker not in by_ticker:
            continue
        ts_str = r.get("published_at") or r.get("scored_at")
        if not ts_str:
            continue
        try:
            t1 = datetime.strptime(ts_str[:19] + "Z", "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        t0 = t1 - window
        sources: set[str] = set()
        for ts, source in by_ticker[ticker]:
            if ts > t1:
                break
            if ts >= t0 and source:
                sources.add(source)
        r["unique_sources_24h"] = min(len(sources), 30)


def extract_features(row: dict[str, Any]) -> list[float]:
    """Return one feature vector for a (signal_score, raw_signal) joined row.

    Reads market-context features (volume_ratio, realized_vol, spy_5d_return,
    qqq_5d_return, vix_level) from row if present — otherwise uses safe
    defaults (volume_ratio=1.0, realized_vol=25.0, market returns=0.0,
    vix=18.0). This makes the same extractor work pre- and post-market-feature
    backfill without changing the model schema.

    ``row`` must include keys: event_type, source_tier, source_weight,
    ticker_confidence, sentiment, sentiment_magnitude, factual,
    corroboration_count, author_quality, anti_pump_flag, composite_score,
    signal_class, scored_at (or published_at). Optional market-context keys:
    volume_ratio_5d_20d, realized_vol_20d, spy_5d_return, qqq_5d_return,
    vix_level.
    """
    event_idx = EVENT_TYPE_TO_IDX.get(row.get("event_type") or "", 0)
    megacap = 1.0 if (row.get("signal_class") or "").endswith("megacap") else 0.0
    ts = row.get("scored_at") or row.get("published_at")
    dow = _day_of_week(ts)
    return [
        # signal-intrinsic
        float(event_idx),
        float(_safe_int(row.get("source_tier"))),
        _safe_float(row.get("source_weight")),
        _safe_float(row.get("ticker_confidence"), 1.0),
        _safe_float(row.get("sentiment")),
        _safe_float(row.get("sentiment_magnitude")),
        float(_safe_int(row.get("factual"))),
        float(_safe_int(row.get("corroboration_count"))),
        _safe_float(row.get("author_quality"), 0.7),
        float(_safe_int(row.get("anti_pump_flag"))),
        _safe_float(row.get("composite_score")),
        megacap,
        # time-of-event
        float(_hour_of_day(ts)),
        float(dow),
        1.0 if dow >= 5 else 0.0,
        _weekly_offset(ts),
        # market-context (safe defaults if missing)
        _safe_float(row.get("volume_ratio_5d_20d"), 1.0),
        _safe_float(row.get("realized_vol_20d"),    25.0),
        _safe_float(row.get("spy_5d_return"),        0.0),
        _safe_float(row.get("qqq_5d_return"),        0.0),
        _safe_float(row.get("vix_level"),           18.0),
        # LLM label-quality
        float(_safe_int(row.get("has_llm_classification"))),
        _safe_float(row.get("llm_confidence"), 0.0),
        # Insider cluster context
        float(_safe_int(row.get("insider_cluster_size_30d"))),
        # Multi-source corroboration window
        float(_safe_int(row.get("unique_sources_24h"))),
        # Macro pack (defaults imply "no macro stress / no macro signal")
        _safe_float(row.get("yield_curve_slope"), 0.0),
        _safe_float(row.get("dxy_5d_return"),    0.0),
        _safe_float(row.get("oil_5d_return"),    0.0),
        _safe_float(row.get("gold_5d_return"),   0.0),
        # External catalyst
        _safe_float(row.get("days_until_catalyst"), 0.0),
        # Short-interest context
        _safe_float(row.get("short_interest_pct_float"), 0.0),
        _safe_float(row.get("days_to_cover"),            0.0),
        # Attention proxies
        _safe_float(row.get("gtrends_zscore"),            0.0),
        _safe_float(row.get("wikipedia_pageviews_zscore"), 0.0),
        # PEAD context
        _safe_float(row.get("eps_surprise_pct"),  0.0),
        _safe_float(row.get("days_since_earnings"), 90.0),  # 90d default = "no recent earnings"
        # 13F crowding
        _safe_float(row.get("institutional_buyers_minus_sellers_qoq"), 0.0),
        # TA-Lib indicators
        _safe_float(row.get("rsi_14"),             50.0),
        _safe_float(row.get("macd_histogram"),     0.0),
        _safe_float(row.get("macd_above_signal"),  0.0),
        _safe_float(row.get("bb_position_20d"),    0.5),
        _safe_float(row.get("atr_pct"),            2.0),
        _safe_float(row.get("adx_14"),             20.0),
        _safe_float(row.get("vwap_distance_pct"),  0.0),
        _safe_float(row.get("breakout_20d"),       0.0),
        # Calendar
        _safe_float(row.get("days_until_fomc"),    99.0),
        _safe_float(row.get("days_until_cpi"),     99.0),
        _safe_float(row.get("days_until_nfp"),     99.0),
        _safe_float(row.get("is_turn_of_month"),   0.0),
        # Graph
        _safe_float(row.get("sector_peer_5d_return"),     0.0),
        _safe_float(row.get("co_mention_peer_5d_return"), 0.0),
        _safe_float(row.get("crypto_btc_5d_return"),      0.0),
        _safe_float(row.get("is_crypto_correlated"),      0.0),
        # Whisper
        _safe_float(row.get("whisper_minus_consensus_pct"), 0.0),
        # Fails-to-deliver
        _safe_float(row.get("fails_to_deliver_pct_float"),  0.0),
        # Insider codes
        _safe_float(row.get("insider_recent_buys_30d"),   0.0),
        _safe_float(row.get("insider_recent_sells_30d"),  0.0),
        _safe_float(row.get("insider_role_score"),        0.0),
        # News quality
        _safe_float(row.get("news_novelty_score"),        0.5),
        _safe_float(row.get("finbert_sentiment_24h"),     0.0),
    ]


def extract_features_df(rows: Iterable[dict[str, Any]]):
    """Build a pandas DataFrame of features + labels for training."""
    import pandas as pd  # type: ignore
    X = [extract_features(r) for r in rows]
    return pd.DataFrame(X, columns=FEATURE_NAMES)
