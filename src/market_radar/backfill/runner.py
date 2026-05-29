"""Orchestrate the historical backfill.

End-to-end pipeline:

  1. Walk the past N quarters of SEC EDGAR form.idx files
  2. Filter to forms we care about + map CIK → ticker via cached map
  3. Bulk-fetch yfinance daily history for every unique ticker
  4. For each filing: snapshot anchor price + 1d/5d/20d closes/returns
  5. Insert into raw_signals + signal_scores + signal_outcomes

Idempotent — re-running skips filings already present (UNIQUE on
``(source, external_id)``).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from ..ingestors.cik_lookup import CIK_LOOKUP
from ..scoring.composite import _utc_now_iso
from ..storage import get_connection, init_db, insert_raw_signal
from .prices import HistoricalPriceCache
from .sec_index import BACKFILL_FORMS, FilingRecord, walk_quarters

log = logging.getLogger(__name__)


@dataclass
class BackfillStats:
    quarters: int = 0
    raw_filings: int = 0
    ticker_matched: int = 0
    inserted_signals: int = 0
    dup_signals: int = 0
    scored: int = 0
    outcomes_priced: int = 0
    outcomes_unpriced: int = 0
    errors: int = 0


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _current_quarter(d: date) -> tuple[int, int]:
    return d.year, (d.month - 1) // 3 + 1


def _filing_external_id(record: FilingRecord) -> str:
    """Stable id used for dedup. Filename is unique per filing on EDGAR."""
    return f"backfill|{record.form}|{record.filename}"


def _backfill_source(record: FilingRecord) -> str:
    """Distinguishes backfilled rows from live rows."""
    return f"sec_edgar_backfill_{record.form.lower().replace(' ', '_').replace('/', '_')}"


def run_backfill(
    *,
    quarters: int = 8,
    cap_filings: Optional[int] = None,
    skip_existing: bool = True,
) -> BackfillStats:
    """Run the full backfill pipeline.

    Args:
        quarters: how many trailing quarters to ingest.
        cap_filings: hard cap on number of filings processed (for testing).
        skip_existing: if True, skip filings whose external_id is already
            in raw_signals.
    """
    stats = BackfillStats()
    init_db()

    end_year, end_qtr = _current_quarter(_today())

    # -------- 1. Walk form.idx files --------
    records: list[FilingRecord] = []
    for record in walk_quarters(end_year=end_year, end_qtr=end_qtr, quarters=quarters):
        records.append(record)
        if cap_filings and len(records) >= cap_filings:
            break
    stats.raw_filings = len(records)
    stats.quarters = quarters

    if not records:
        log.warning("Backfill found no records — aborting")
        return stats

    # -------- 2. Map CIK → ticker; drop unmappable --------
    keep: list[tuple[FilingRecord, str]] = []
    for r in records:
        ticker = CIK_LOOKUP.get_ticker(r.cik)
        if ticker:
            keep.append((r, ticker.upper()))
    stats.ticker_matched = len(keep)
    log.info("Mapped %d / %d filings to a ticker", len(keep), len(records))

    if not keep:
        return stats

    # -------- 3. Dedup against existing DB rows --------
    if skip_existing:
        with get_connection() as conn:
            existing = set(
                row["external_id"]
                for row in conn.execute(
                    "SELECT external_id FROM raw_signals WHERE source LIKE 'sec_edgar_backfill_%'"
                ).fetchall()
                if row["external_id"]
            )
        before = len(keep)
        keep = [(r, t) for r, t in keep if _filing_external_id(r) not in existing]
        log.info("Skipping %d already-backfilled filings", before - len(keep))

    if not keep:
        log.info("Nothing new to backfill")
        return stats

    # -------- 4. Warm price cache --------
    unique_tickers = sorted({t for _, t in keep})
    earliest = min(r.filed for r, _ in keep)
    latest = max(r.filed for r, _ in keep)
    # Pad latest by 35 days so 20-trading-day lookups still resolve
    fetch_until = min(latest + timedelta(days=35), _today())
    log.info(
        "Backfilling %d filings on %d unique tickers from %s to %s",
        len(keep), len(unique_tickers), earliest, latest,
    )
    pc = HistoricalPriceCache()
    pc.warm(unique_tickers, start=earliest, end=fetch_until)

    # -------- 5. Insert one filing at a time --------
    with get_connection() as conn:
        for record, ticker in keep:
            try:
                _insert_one(conn, record, ticker, pc, stats)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "backfill insert failed cik=%s form=%s: %s",
                    record.cik, record.form, exc,
                )
                stats.errors += 1

    log.info(
        "Backfill done: %d filings → %d signals inserted (%d dups), "
        "%d scored, %d priced / %d unpriced, %d errors",
        stats.raw_filings, stats.inserted_signals, stats.dup_signals,
        stats.scored, stats.outcomes_priced, stats.outcomes_unpriced, stats.errors,
    )
    return stats


def _insert_one(
    conn,
    record: FilingRecord,
    ticker: str,
    pc: HistoricalPriceCache,
    stats: BackfillStats,
) -> None:
    """Insert raw_signal + signal_score + signal_outcome for one filing."""
    from ..scoring.composite import _score_one  # noqa: PLC0415 — late import to avoid circular

    source = _backfill_source(record)
    external_id = _filing_external_id(record)
    title = f"{record.form} - {record.company} ({record.cik:010d})"
    url = record.filing_url()
    published = datetime.combine(record.filed, datetime.min.time(), tzinfo=timezone.utc)
    published_iso = published.strftime("%Y-%m-%dT%H:%M:%SZ")

    raw_payload = {
        "form": record.form,
        "form_event": record.form_event,
        "cik": record.cik,
        "company_name": record.company,
        "filed_date": record.filed.isoformat(),
        "filing_url": url,
        "is_backfill": True,
    }

    signal_id = insert_raw_signal(
        conn,
        source=source,
        source_tier=1,
        external_id=external_id,
        url=url,
        title=title,
        body=None,
        author=record.company,
        author_metadata={"cik": record.cik, "form": record.form},
        raw_payload=raw_payload,
        published_at=published_iso,
        tickers=[{
            "ticker": ticker,
            "market": "US",
            "asset_class": None,
            "confidence": 1.0,
        }],
    )

    if signal_id is None:
        stats.dup_signals += 1
        return
    stats.inserted_signals += 1

    # Score using the same scorer the live pipeline uses
    row = {
        "signal_id": signal_id,
        "source": source,
        "source_tier": 1,
        "title": title,
        "body": None,
        "author": record.company,
        "author_metadata_json": None,
        "raw_payload_json": json.dumps(raw_payload),
        "ingested_at": published_iso,
        "published_at": published_iso,
        "ticker": ticker,
        "ticker_confidence": 1.0,
        "asset_class": None,
    }
    # Use the live scorer but pretend corroboration is 0 (we don't backfill
    # cross-source corroboration; that's a live-only signal)
    composite, fields = _score_one(conn, _RowProxy(row), corroboration_window_hours=4)
    conn.execute(
        """
        INSERT INTO signal_scores (
            signal_id, ticker, event_type, sentiment, sentiment_magnitude,
            factual, source_weight, corroboration_count, author_quality,
            anti_pump_flag, composite_score, signal_class, scored_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal_id,
            ticker,
            fields["event_type"],
            fields["sentiment"],
            fields["sentiment_magnitude"],
            fields["factual"],
            fields["source_weight"],
            fields["corroboration_count"],
            fields["author_quality"],
            fields["anti_pump_flag"],
            composite,
            fields["signal_class"],
            _utc_now_iso(),
        ),
    )
    score_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    stats.scored += 1

    # Outcome lookup against the in-memory price cache
    lookup = pc.lookup(ticker, record.filed)
    fully_resolved = 1 if lookup.return_20d_pct is not None else 0
    conn.execute(
        """
        INSERT INTO signal_outcomes (
            score_id, ticker, price_at_flag, price_at_flag_ts,
            price_1d, return_1d_pct,
            price_5d, return_5d_pct,
            price_20d, return_20d_pct,
            fully_resolved
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            score_id, ticker,
            lookup.price_at_flag, lookup.price_at_flag_ts,
            lookup.price_1d, lookup.return_1d_pct,
            lookup.price_5d, lookup.return_5d_pct,
            lookup.price_20d, lookup.return_20d_pct,
            fully_resolved,
        ),
    )

    if lookup.price_at_flag is not None:
        stats.outcomes_priced += 1
    else:
        stats.outcomes_unpriced += 1


class _RowProxy:
    """Lightweight sqlite3.Row-style accessor over a plain dict."""

    def __init__(self, d: dict):
        self._d = d

    def __getitem__(self, key):
        return self._d[key]
