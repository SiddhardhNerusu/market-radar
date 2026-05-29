"""External-data feature attach helpers.

Each function reads from one staging table populated by an ingestor or
refresh script and attaches the resulting feature(s) to the row dicts
passed in. Safe to call against a fresh DB where the staging table is
empty or doesn't exist yet — the function is a no-op and rows keep
their default feature values.

Wired into ``ml/train.py`` (via ``_enrich_with_external_features``) and
``ml/predict.py`` (via the same path in ``predict_pending``). Each is
designed to fail closed (default values, no exceptions propagated).

Staging tables expected:
  - ``short_interest``           (Tier 1 #10)
  - ``catalysts``                (Tier 1 #7)
  - ``institutional_holdings``   (Tier 2 #13)
  - ``attention_data``           (Tier 2 #14 + #15)
  - ``earnings_data``            (Tier 2 #11)
"""
from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable, Optional

import sys  # noqa: F401 — kept for backward compatibility

from ..storage import get_connection

log = logging.getLogger(__name__)


# ---- helpers --------------------------------------------------------------

def _normalize_ticker(ticker: str) -> str:
    """Strip T212-style suffix (e.g. AAPL_US_EQ → AAPL)."""
    t = (ticker or "").upper()
    if "_" in t:
        parts = t.split("_")
        if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
            t = "_".join(parts[:-2]).replace("_", "-")
    return t


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s[:19] + "Z", "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


# ---- short interest -------------------------------------------------------

def attach_short_interest(rows: list[dict]) -> None:
    """Attach ``short_interest_pct_float`` + ``days_to_cover``."""
    for r in rows:
        r.setdefault("short_interest_pct_float", 0.0)
        r.setdefault("days_to_cover", 0.0)
    if not rows:
        return
    try:
        with get_connection() as conn:
            if not _table_exists(conn, "short_interest"):
                return
            si = conn.execute(
                "SELECT ticker, report_date, short_pct_float, days_to_cover "
                "FROM short_interest"
            ).fetchall()
    except sqlite3.OperationalError:
        return

    # Latest snapshot per ticker. Multiple report dates exist; we take the
    # most recent on or before the row's published_at.
    by_ticker: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for r in si:
        by_ticker[(r["ticker"] or "").upper()].append((
            r["report_date"], r["short_pct_float"] or 0.0, r["days_to_cover"] or 0.0,
        ))
    for t in by_ticker:
        by_ticker[t].sort()  # ascending by report_date

    for row in rows:
        ticker = _normalize_ticker(row.get("ticker") or "")
        if not ticker or ticker not in by_ticker:
            continue
        ts_str = row.get("published_at") or row.get("scored_at") or ""
        ts_date = ts_str[:10]
        latest = None
        for rd, sp, dtc in by_ticker[ticker]:
            if rd <= ts_date:
                latest = (sp, dtc)
            else:
                break
        if latest:
            row["short_interest_pct_float"] = float(latest[0])
            row["days_to_cover"] = float(latest[1])


# ---- catalysts ------------------------------------------------------------

def attach_catalyst_features(rows: list[dict]) -> None:
    """Attach ``days_until_catalyst`` (clipped to [-30, 60])."""
    for r in rows:
        r.setdefault("days_until_catalyst", 60.0)
    if not rows:
        return
    try:
        with get_connection() as conn:
            if not _table_exists(conn, "catalysts"):
                return
            cats = conn.execute(
                "SELECT ticker, decision_date FROM catalysts"
            ).fetchall()
    except sqlite3.OperationalError:
        return

    by_ticker: dict[str, list[datetime]] = defaultdict(list)
    for c in cats:
        d = _parse_ts(c["decision_date"])
        if d is None:
            continue
        by_ticker[(c["ticker"] or "").upper()].append(d)
    for t in by_ticker:
        by_ticker[t].sort()

    for row in rows:
        ticker = _normalize_ticker(row.get("ticker") or "")
        if not ticker or ticker not in by_ticker:
            continue
        ts = _parse_ts(row.get("published_at") or row.get("scored_at"))
        if ts is None:
            continue
        # Find the next future catalyst within 60 days
        next_d: Optional[float] = None
        for d in by_ticker[ticker]:
            delta = (d - ts).days
            if delta < -30:
                continue
            if delta > 60:
                break
            next_d = float(delta)
            break
        if next_d is not None:
            row["days_until_catalyst"] = next_d


# ---- attention proxies (Trends + Wikipedia) -------------------------------

def attach_attention_features(rows: list[dict]) -> None:
    """Attach ``gtrends_zscore`` + ``wikipedia_pageviews_zscore``."""
    for r in rows:
        r.setdefault("gtrends_zscore", 0.0)
        r.setdefault("wikipedia_pageviews_zscore", 0.0)
    if not rows:
        return
    try:
        with get_connection() as conn:
            if not _table_exists(conn, "attention_data"):
                return
            rs = conn.execute(
                "SELECT ticker, observed_at, gtrends_zscore, wiki_zscore "
                "FROM attention_data"
            ).fetchall()
    except sqlite3.OperationalError:
        return

    by_ticker: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for r in rs:
        by_ticker[(r["ticker"] or "").upper()].append((
            r["observed_at"], r["gtrends_zscore"] or 0.0, r["wiki_zscore"] or 0.0,
        ))
    for t in by_ticker:
        by_ticker[t].sort()

    for row in rows:
        ticker = _normalize_ticker(row.get("ticker") or "")
        if not ticker or ticker not in by_ticker:
            continue
        ts_date = (row.get("published_at") or row.get("scored_at") or "")[:10]
        latest = None
        for ad, gz, wz in by_ticker[ticker]:
            if ad <= ts_date:
                latest = (gz, wz)
            else:
                break
        if latest:
            row["gtrends_zscore"] = float(latest[0])
            row["wikipedia_pageviews_zscore"] = float(latest[1])


# ---- PEAD / earnings surprise ---------------------------------------------

def attach_pead_features(rows: list[dict]) -> None:
    """Attach ``eps_surprise_pct`` + ``days_since_earnings``."""
    for r in rows:
        r.setdefault("eps_surprise_pct", 0.0)
        r.setdefault("days_since_earnings", 90.0)
    if not rows:
        return
    try:
        with get_connection() as conn:
            if not _table_exists(conn, "earnings_data"):
                return
            er = conn.execute(
                "SELECT ticker, report_date, eps_surprise_pct FROM earnings_data"
            ).fetchall()
    except sqlite3.OperationalError:
        return

    by_ticker: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for r in er:
        d = _parse_ts(r["report_date"])
        if d is None:
            continue
        by_ticker[(r["ticker"] or "").upper()].append((d, r["eps_surprise_pct"] or 0.0))
    for t in by_ticker:
        by_ticker[t].sort()

    for row in rows:
        ticker = _normalize_ticker(row.get("ticker") or "")
        if not ticker or ticker not in by_ticker:
            continue
        ts = _parse_ts(row.get("published_at") or row.get("scored_at"))
        if ts is None:
            continue
        latest = None
        for rd, surprise in by_ticker[ticker]:
            if rd <= ts:
                latest = (rd, surprise)
            else:
                break
        if latest:
            row["eps_surprise_pct"] = float(latest[1])
            row["days_since_earnings"] = min(float((ts - latest[0]).days), 90.0)


# ---- 13F institutional holdings -------------------------------------------

def attach_institutional_features(rows: list[dict]) -> None:
    """Attach ``institutional_buyers_minus_sellers_qoq``."""
    for r in rows:
        r.setdefault("institutional_buyers_minus_sellers_qoq", 0.0)
    if not rows:
        return
    try:
        with get_connection() as conn:
            if not _table_exists(conn, "institutional_holdings"):
                return
            ih = conn.execute(
                "SELECT ticker, quarter_end, new_buyers, new_sellers, net_position "
                "FROM institutional_holdings"
            ).fetchall()
    except sqlite3.OperationalError:
        return

    by_ticker: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in ih:
        net = float(r["new_buyers"] or 0) - float(r["new_sellers"] or 0)
        by_ticker[(r["ticker"] or "").upper()].append((r["quarter_end"], net))
    for t in by_ticker:
        by_ticker[t].sort()

    for row in rows:
        ticker = _normalize_ticker(row.get("ticker") or "")
        if not ticker or ticker not in by_ticker:
            continue
        ts_date = (row.get("published_at") or row.get("scored_at") or "")[:10]
        latest = None
        for q, net in by_ticker[ticker]:
            if q <= ts_date:
                latest = net
            else:
                break
        if latest is not None:
            row["institutional_buyers_minus_sellers_qoq"] = float(latest)


# ---- entry point ----------------------------------------------------------

def attach_fails_to_deliver(rows: list[dict]) -> None:
    """Attach ``fails_to_deliver_pct_float`` (defaults 0)."""
    for r in rows:
        r.setdefault("fails_to_deliver_pct_float", 0.0)
    if not rows:
        return
    try:
        with get_connection() as conn:
            if not _table_exists(conn, "fails_to_deliver"):
                return
            ftd = conn.execute(
                "SELECT ticker, settlement_date, quantity FROM fails_to_deliver"
            ).fetchall()
    except sqlite3.OperationalError:
        return
    by_ticker: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for f in ftd:
        by_ticker[(f["ticker"] or "").upper()].append(
            (f["settlement_date"], float(f["quantity"] or 0)))
    for t in by_ticker:
        by_ticker[t].sort()
    for row in rows:
        ticker = _normalize_ticker(row.get("ticker") or "")
        if not ticker or ticker not in by_ticker:
            continue
        ts_date = (row.get("published_at") or row.get("scored_at") or "")[:10]
        latest = None
        for sd, q in by_ticker[ticker]:
            if sd <= ts_date:
                latest = q
            else:
                break
        # We don't have float to divide by — emit raw FTD share count
        # scaled by 1e-6 so it's roughly % of mid-cap float.
        if latest is not None:
            row["fails_to_deliver_pct_float"] = float(latest) * 1e-6


def attach_earnings_whispers(rows: list[dict]) -> None:
    for r in rows:
        r.setdefault("whisper_minus_consensus_pct", 0.0)
    if not rows:
        return
    try:
        with get_connection() as conn:
            if not _table_exists(conn, "earnings_whispers"):
                return
            ew = conn.execute(
                "SELECT ticker, report_date, whisper_minus_consensus FROM earnings_whispers"
            ).fetchall()
    except sqlite3.OperationalError:
        return
    by_ticker: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in ew:
        if r["whisper_minus_consensus"] is None:
            continue
        by_ticker[(r["ticker"] or "").upper()].append(
            (r["report_date"] or "", float(r["whisper_minus_consensus"])))
    for t in by_ticker:
        by_ticker[t].sort()
    for row in rows:
        ticker = _normalize_ticker(row.get("ticker") or "")
        if not ticker or ticker not in by_ticker:
            continue
        ts_date = (row.get("published_at") or row.get("scored_at") or "")[:10]
        latest = None
        for d, w in by_ticker[ticker]:
            if d <= ts_date:
                latest = w
            else:
                break
        if latest is not None:
            row["whisper_minus_consensus_pct"] = float(latest)


def attach_insider_codes(rows: list[dict]) -> None:
    """Attach insider_recent_buys_30d / insider_recent_sells_30d /
    insider_role_score from the insider_transactions table.
    """
    for r in rows:
        r.setdefault("insider_recent_buys_30d", 0.0)
        r.setdefault("insider_recent_sells_30d", 0.0)
        r.setdefault("insider_role_score", 0.0)
    if not rows:
        return
    try:
        with get_connection() as conn:
            if not _table_exists(conn, "insider_transactions"):
                return
            ix = conn.execute(
                "SELECT ticker, transaction_code, shares, role_score, report_date "
                "FROM insider_transactions"
            ).fetchall()
    except sqlite3.OperationalError:
        return

    by_ticker: dict[str, list[tuple[datetime, str, float, int]]] = defaultdict(list)
    for r in ix:
        d = _parse_ts(r["report_date"])
        if d is None:
            continue
        by_ticker[(r["ticker"] or "").upper()].append(
            (d, r["transaction_code"] or "", float(r["shares"] or 0),
             int(r["role_score"] or 0)))
    for t in by_ticker:
        by_ticker[t].sort()

    for row in rows:
        ticker = _normalize_ticker(row.get("ticker") or "")
        if not ticker or ticker not in by_ticker:
            continue
        ts = _parse_ts(row.get("published_at") or row.get("scored_at"))
        if ts is None:
            continue
        t0 = ts - timedelta(days=30)
        buys = sells = max_role = 0
        for d, code, shares, role in by_ticker[ticker]:
            if d > ts:
                break
            if d >= t0:
                if code == "P":
                    buys += 1
                elif code == "S":
                    sells += 1
                max_role = max(max_role, role)
        row["insider_recent_buys_30d"]  = float(buys)
        row["insider_recent_sells_30d"] = float(sells)
        row["insider_role_score"]       = float(max_role)


def attach_news_features(rows: list[dict]) -> None:
    """Attach news_novelty_score + finbert_sentiment_24h from news_features."""
    for r in rows:
        r.setdefault("news_novelty_score", 0.5)
        r.setdefault("finbert_sentiment_24h", 0.0)
    if not rows:
        return
    try:
        with get_connection() as conn:
            if not _table_exists(conn, "news_features"):
                return
            # Per-signal-id direct lookup is cheapest
            sids = [r.get("signal_id") for r in rows if r.get("signal_id")]
            if sids:
                placeholders = ",".join("?" * len(sids))
                lookup = {nf["signal_id"]: nf for nf in conn.execute(
                    f"SELECT signal_id, novelty_score, finbert_sentiment "
                    f"FROM news_features WHERE signal_id IN ({placeholders})",
                    sids,
                ).fetchall()}
                for r in rows:
                    nf = lookup.get(r.get("signal_id"))
                    if nf:
                        r["news_novelty_score"] = float(nf["novelty_score"] or 0.5)
                        r["finbert_sentiment_24h"] = float(nf["finbert_sentiment"] or 0.0)
    except sqlite3.OperationalError:
        return


def attach_all_external_features(rows: list[dict]) -> None:
    """Convenience wrapper that calls every attach_* helper in turn."""
    attach_short_interest(rows)
    attach_catalyst_features(rows)
    attach_attention_features(rows)
    attach_pead_features(rows)
    attach_institutional_features(rows)
    attach_fails_to_deliver(rows)
    attach_earnings_whispers(rows)
    attach_insider_codes(rows)
    attach_news_features(rows)
