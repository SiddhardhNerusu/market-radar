"""Confluence-based sizing multipliers.

Pull from the alt-data tables (short interest, insider transactions, catalysts,
attention, FTDs) and return a multiplier in [0.3, 1.5] that the trader applies
to position size. Each individual signal moves the multiplier by 10-20%.

Empty tables are no-ops — this helper degrades gracefully when the refresh
scripts haven't run yet. As you populate more alt-data, the bot
*automatically* starts using it for sizing.

Caching: 5-min in-process cache per (ticker, direction) avoids hammering SQLite
when the same ticker appears across multiple loop iterations.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Literal, Optional

from ..storage import get_connection

log = logging.getLogger("marketradar.signals.confluence")

Direction = Literal["buy", "sell"]

_CACHE: dict[tuple[str, str], tuple[datetime, float, list[str]]] = {}
_TTL = timedelta(minutes=5)
_MIN_MULT = 0.3
_MAX_MULT = 1.5


def get_confluence_multiplier(
    symbol: str, direction: Direction, *,
    return_reasons: bool = False,
):
    """Return a sizing multiplier (and optional human-readable reasons).

    Multiplier is the product of independent boosts/cuts:
      - short_interest > 15% + bullish = squeeze potential (×1.20)
      - short_interest > 30% + bearish = squeeze risk      (×0.70)
      - insider cluster (3+ buys/30d) + bullish            (×1.15)
      - catalyst < 3 days away (earnings/FDA/PDUFA)         (×0.50)
      - catalyst < 7 days away                              (×0.80)
      - attention z-score > 2σ + bullish                    (×1.10)
      - high FTD % + bullish = squeeze potential            (×1.10)

    Clamped to [0.3, 1.5].
    """
    symbol = symbol.upper().strip()
    # Crypto pairs (BTC/USD, ETH/USD, etc.) have no alt-data — skip queries.
    if "/" in symbol:
        return (1.0, []) if return_reasons else 1.0
    key = (symbol, direction)
    now = datetime.utcnow()
    cached = _CACHE.get(key)
    if cached and (now - cached[0]) < _TTL:
        return (cached[1], cached[2]) if return_reasons else cached[1]

    mult = 1.0
    reasons: list[str] = []
    try:
        with get_connection() as conn:
            mult, reasons = _compute(conn, symbol, direction)
    except sqlite3.OperationalError as exc:
        # Missing tables (very old DB) — degrade silently
        log.debug("confluence skipped (missing tables): %s", exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("confluence error for %s: %s", symbol, exc)

    mult = max(_MIN_MULT, min(_MAX_MULT, mult))
    _CACHE[key] = (now, mult, reasons)
    return (mult, reasons) if return_reasons else mult


def _compute(conn, symbol: str, direction: Direction) -> tuple[float, list[str]]:
    mult = 1.0
    reasons: list[str] = []

    # ---- Short interest (FINRA bi-monthly) ----
    si_row = conn.execute(
        "SELECT short_pct_float FROM short_interest WHERE ticker=? "
        "ORDER BY report_date DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    if si_row and si_row[0] is not None:
        si_pct = float(si_row[0])
        if direction == "buy" and si_pct >= 15:
            mult *= 1.20
            reasons.append(f"short_squeeze_potential(SI={si_pct:.0f}%)")
        elif direction == "sell" and si_pct >= 30:
            mult *= 0.70
            reasons.append(f"squeeze_risk_on_short(SI={si_pct:.0f}%)")

    # ---- Insider cluster (Form 4 buys via SEC EDGAR) ----
    # Schema: insider_transactions(ticker, insider_name, transaction_code,
    #         shares, price, is_acquired, ..., report_date)
    # 'P' = Open-market purchase (the bullish kind we care about)
    try:
        insider_row = conn.execute(
            "SELECT COUNT(DISTINCT insider_name), SUM(shares*price) "
            "FROM insider_transactions "
            "WHERE ticker=? AND transaction_code='P' "
            "AND is_acquired=1 "
            "AND report_date >= date('now','-30 days')",
            (symbol,),
        ).fetchone()
        if insider_row and insider_row[0]:
            n_buyers = int(insider_row[0])
            total_usd = float(insider_row[1] or 0)
            if n_buyers >= 3 and direction == "buy":
                mult *= 1.15
                reasons.append(f"insider_cluster({n_buyers} buyers/30d, ${total_usd/1000:.0f}k)")
            elif total_usd >= 1_000_000 and direction == "buy":
                # Single large insider buy ($1M+) is also signal
                mult *= 1.10
                reasons.append(f"large_insider_buy(${total_usd/1000000:.1f}M)")
    except sqlite3.OperationalError:
        pass

    # ---- Catalysts (earnings, FDA, PDUFA) ----
    # Schema: catalysts(ticker, decision_date, catalyst_type, description, source)
    try:
        cat_row = conn.execute(
            "SELECT julianday(decision_date) - julianday('now') AS days, catalyst_type "
            "FROM catalysts WHERE ticker=? AND decision_date >= date('now') "
            "ORDER BY decision_date ASC LIMIT 1",
            (symbol,),
        ).fetchone()
        if cat_row and cat_row[0] is not None:
            days_to = float(cat_row[0])
            cat_type = cat_row[1] or "catalyst"
            if days_to < 3:
                mult *= 0.50
                reasons.append(f"imminent_{cat_type}({days_to:.0f}d)")
            elif days_to < 7:
                mult *= 0.80
                reasons.append(f"near_{cat_type}({days_to:.0f}d)")
    except sqlite3.OperationalError:
        pass

    # ---- Attention spike (Google/Wiki z-score) ----
    # Schema: attention_data(ticker, observed_at, gtrends_value, gtrends_zscore,
    #         wiki_pageviews, wiki_zscore, ingested_at)
    try:
        att_row = conn.execute(
            "SELECT MAX(gtrends_zscore), MAX(wiki_zscore) FROM attention_data "
            "WHERE ticker=? AND observed_at >= datetime('now','-3 days')",
            (symbol,),
        ).fetchone()
        if att_row:
            gt_z = float(att_row[0] or 0)
            wk_z = float(att_row[1] or 0)
            best_z = max(gt_z, wk_z)
            if best_z >= 2.0 and direction == "buy":
                mult *= 1.10
                reasons.append(f"attention_spike(z={best_z:.1f})")
    except sqlite3.OperationalError:
        pass

    # ---- Fails-to-deliver (squeeze proxy) ----
    # Schema: fails_to_deliver(ticker, settlement_date, quantity, price)
    # We compute notional FTDs and look for a spike vs. recent average.
    try:
        ftd_rows = conn.execute(
            "SELECT quantity FROM fails_to_deliver WHERE ticker=? "
            "ORDER BY settlement_date DESC LIMIT 30",
            (symbol,),
        ).fetchall()
        if ftd_rows and len(ftd_rows) >= 5:
            qtys = [float(r[0] or 0) for r in ftd_rows]
            latest = qtys[0]
            avg_prior = sum(qtys[1:]) / len(qtys[1:])
            if avg_prior > 0 and latest > avg_prior * 3 and direction == "buy":
                mult *= 1.10
                reasons.append(f"ftd_spike({latest/avg_prior:.1f}x avg)")
    except sqlite3.OperationalError:
        pass

    return mult, reasons
