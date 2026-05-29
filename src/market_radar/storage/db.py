"""SQLite access layer.

All timestamps stored as ISO 8601 UTC strings. Use utc_now() for consistency.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from ..config import CONFIG, PROJECT_ROOT


SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
LLM_SCHEMA_PATH = PROJECT_ROOT / "sql" / "llm_schema.sql"
EXECUTION_SCHEMA_PATH = PROJECT_ROOT / "sql" / "execution_schema.sql"
OPTIONS_SCHEMA_PATH = PROJECT_ROOT / "sql" / "options_schema.sql"


def utc_now() -> str:
    """Return current UTC time as ISO 8601 string with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)  # autocommit
    conn.row_factory = sqlite3.Row
    # Four pragmas pinned on every connection — concurrency hardening
    # after the 2026-05-14 corruption incident where a daemon writer +
    # an LLM-backfill writer raced on the same WAL.
    #
    #   journal_mode = WAL      — preferred on local APFS/ext4.
    #   synchronous  = NORMAL   — fsync on checkpoint only, not every commit;
    #                             safe for WAL, much faster under contention.
    #   busy_timeout = 10000    — 10 s of automatic retry on SQLITE_BUSY.
    #   foreign_keys = ON       — enforce FK constraints app-wide.
    #
    # WAL falls back to TRUNCATE on mounts that don't support file
    # locking (Cowork sandbox, certain network shares).
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        try:
            conn.execute("PRAGMA journal_mode = TRUNCATE;")
        except sqlite3.OperationalError:
            pass
    try:
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.execute("PRAGMA foreign_keys = ON;")
    except sqlite3.OperationalError:
        pass
    return conn


@contextmanager
def get_connection(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Context-managed SQLite connection. Use this for all DB access."""
    db_path = path or CONFIG.db_path
    conn = _connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(path: Optional[Path] = None) -> None:
    """Create all tables from schema.sql. Idempotent.

    Also applies in-place migrations for new columns added since the
    DB was originally created (since SQLite ``CREATE TABLE IF NOT EXISTS``
    won't add new columns to existing tables).
    """
    db_path = path or CONFIG.db_path
    schema_sql = SCHEMA_PATH.read_text()
    with get_connection(db_path) as conn:
        conn.executescript(schema_sql)
        # LLM enrichment tables (separate file for cleaner organization)
        if LLM_SCHEMA_PATH.exists():
            conn.executescript(LLM_SCHEMA_PATH.read_text())
        # Execution layer tables (bot orders + decisions + daily P&L)
        if EXECUTION_SCHEMA_PATH.exists():
            conn.executescript(EXECUTION_SCHEMA_PATH.read_text())
        # Options layer tables (bot_option_decisions / _spreads / _legs)
        if OPTIONS_SCHEMA_PATH.exists():
            conn.executescript(OPTIONS_SCHEMA_PATH.read_text())
        _migrate_columns(conn)


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """Add columns/tables/indices that may be missing on older DBs.

    All operations are idempotent — safe to call on every init_db().
    """
    needed: list[tuple[str, str, str]] = [
        ("signal_scores", "model_p_5d", "REAL"),
        ("signal_scores", "model_version", "TEXT"),
        ("raw_signals",   "content_hash", "TEXT"),
        # Multi-horizon predictions (Tier 1 #6 — 2026-05-13)
        ("signal_scores", "model_p_1d",  "REAL"),
        ("signal_scores", "model_p_20d", "REAL"),
        # Per-event-type routing (Tier 1 #5 — 2026-05-13)
        ("signal_scores", "model_bucket", "TEXT"),
        # T212 currency normalization (2026-05-14)
        ("t212_positions", "quote_currency", "TEXT"),
        ("t212_positions", "native_market_value", "REAL"),
        ("t212_positions", "market_value_usd", "REAL"),
        # Real-time notifications (2026-05-14)
        ("notifications_sent", "signal_id", "INTEGER"),
        ("notifications_sent", "ticker", "TEXT"),
        ("notifications_sent", "direction", "TEXT"),
        ("notifications_sent", "action", "TEXT"),
        ("notifications_sent", "p", "REAL"),
        ("notifications_sent", "channels_sent", "TEXT"),
        ("notifications_sent", "ingested_at", "TEXT"),
        ("notifications_sent", "latency_seconds", "REAL"),
    ]
    for table, column, coltype in needed:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            except sqlite3.OperationalError:
                pass

    # Index on content_hash for fast cross-source dedup lookups
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_signals_content_hash "
            "ON raw_signals(content_hash, ingested_at)"
        )
    except sqlite3.OperationalError:
        pass

    # --- New staging tables for upgrade items (2026-05-13) -----------------
    new_tables_sql = [
        # FINRA short interest (Tier 1 #10)
        """
        CREATE TABLE IF NOT EXISTS short_interest (
            ticker             TEXT NOT NULL,
            report_date        TEXT NOT NULL,
            short_interest     REAL,
            avg_daily_volume   REAL,
            days_to_cover      REAL,
            float_shares       REAL,
            short_pct_float    REAL,
            ingested_at        TEXT NOT NULL,
            PRIMARY KEY (ticker, report_date)
        )
        """,
        # FDA / earnings catalysts (Tier 1 #7)
        """
        CREATE TABLE IF NOT EXISTS catalysts (
            ticker          TEXT NOT NULL,
            decision_date   TEXT NOT NULL,
            catalyst_type   TEXT NOT NULL,         -- 'pdufa', 'earnings', 'fomc', etc.
            description     TEXT,
            source          TEXT,
            ingested_at     TEXT NOT NULL,
            PRIMARY KEY (ticker, decision_date, catalyst_type)
        )
        """,
        # 13F institutional holdings flow (Tier 2 #13)
        """
        CREATE TABLE IF NOT EXISTS institutional_holdings (
            ticker          TEXT NOT NULL,
            quarter_end     TEXT NOT NULL,
            new_buyers      INTEGER DEFAULT 0,
            new_sellers     INTEGER DEFAULT 0,
            net_position    INTEGER DEFAULT 0,
            total_holders   INTEGER DEFAULT 0,
            ingested_at     TEXT NOT NULL,
            PRIMARY KEY (ticker, quarter_end)
        )
        """,
        # Attention proxies — Google Trends + Wikipedia pageviews (Tier 2 #14/#15)
        """
        CREATE TABLE IF NOT EXISTS attention_data (
            ticker          TEXT NOT NULL,
            observed_at     TEXT NOT NULL,        -- ISO date of the observation
            gtrends_value   REAL,                 -- weekly interest (Google Trends)
            gtrends_zscore  REAL,                 -- z-score vs trailing 13w mean
            wiki_pageviews  INTEGER,              -- last-week pageviews
            wiki_zscore     REAL,                 -- z-score vs trailing 30d mean
            ingested_at     TEXT NOT NULL,
            PRIMARY KEY (ticker, observed_at)
        )
        """,
        # Earnings calendar & surprises (Tier 2 #11 PEAD)
        """
        CREATE TABLE IF NOT EXISTS earnings_data (
            ticker            TEXT NOT NULL,
            report_date       TEXT NOT NULL,
            eps_actual        REAL,
            eps_estimate      REAL,
            eps_surprise_pct  REAL,
            revenue_actual    REAL,
            revenue_estimate  REAL,
            revenue_surprise_pct REAL,
            ingested_at       TEXT NOT NULL,
            PRIMARY KEY (ticker, report_date)
        )
        """,
        # Drift detection — rolling AUC observations
        """
        CREATE TABLE IF NOT EXISTS model_drift_observations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_at     TEXT NOT NULL,
            model_version   TEXT,
            window_n        INTEGER,
            rolling_auc     REAL,
            ph_stat         REAL,           -- Page-Hinkley running stat
            alerted         INTEGER DEFAULT 0
        )
        """,
    ]
    for stmt in new_tables_sql:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            # Non-fatal; logged at low level so noisy migrations don't spam
            pass

    new_indices = [
        "CREATE INDEX IF NOT EXISTS idx_catalysts_decision_date ON catalysts(decision_date, ticker)",
        "CREATE INDEX IF NOT EXISTS idx_short_interest_ticker ON short_interest(ticker, report_date DESC)",
        "CREATE INDEX IF NOT EXISTS idx_earnings_data_ticker ON earnings_data(ticker, report_date DESC)",
        "CREATE INDEX IF NOT EXISTS idx_attention_data_ticker ON attention_data(ticker, observed_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_inst_holdings_ticker ON institutional_holdings(ticker, quarter_end DESC)",
    ]
    for idx in new_indices:
        try:
            conn.execute(idx)
        except sqlite3.OperationalError:
            pass


# ---------------------------------------------------------------------------
# Raw signal insertion (Tier 1/2/3 ingestors all funnel through here)
# ---------------------------------------------------------------------------


_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def content_hash_for(title: Optional[str], body: Optional[str]) -> Optional[str]:
    """Compute a stable hash of normalised title+body text.

    Used to detect when the *same story* appears across multiple sources
    (e.g. 8 outlets all repeating one Bloomberg headline). Same hash =
    same story, even when URLs and external_ids differ.

    Normalization: lowercase, strip punctuation, collapse whitespace, take
    first 200 chars after normalisation. Truncation reduces noise from
    body length differences. Returns None for empty content.
    """
    text = " ".join(filter(None, [title or "", body or ""]))
    if not text:
        return None
    text = text.lower()
    text = _NORMALIZE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return None
    text = text[:200]  # truncate to first 200 normalised chars
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def insert_raw_signal(
    conn: sqlite3.Connection,
    *,
    source: str,
    source_tier: int,
    external_id: Optional[str],
    url: Optional[str],
    title: Optional[str],
    body: Optional[str],
    author: Optional[str],
    author_metadata: Optional[dict[str, Any]],
    raw_payload: Optional[dict[str, Any]],
    published_at: Optional[str],
    tickers: list[dict[str, Any]],
) -> Optional[int]:
    """Insert a raw signal + its ticker mentions.

    Returns the inserted row id, or None if the signal was a duplicate
    (UNIQUE constraint on source+external_id).
    """
    ingested_at = utc_now()
    chash = content_hash_for(title, body)
    try:
        cursor = conn.execute(
            """
            INSERT INTO raw_signals (
                source, source_tier, external_id, url, title, body, author,
                author_metadata, raw_payload, ingested_at, published_at,
                content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                source_tier,
                external_id,
                url,
                title,
                body,
                author,
                json.dumps(author_metadata) if author_metadata else None,
                json.dumps(raw_payload) if raw_payload else None,
                ingested_at,
                published_at,
                chash,
            ),
        )
    except sqlite3.IntegrityError:
        return None  # duplicate

    signal_id = cursor.lastrowid
    for t in tickers:
        try:
            conn.execute(
                """
                INSERT INTO signal_tickers (
                    signal_id, ticker, market, asset_class, confidence
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    t["ticker"],
                    t.get("market"),
                    t.get("asset_class"),
                    t.get("confidence", 1.0),
                ),
            )
        except sqlite3.IntegrityError:
            pass  # duplicate ticker on same signal
    return signal_id


def insert_signal_score(
    conn: sqlite3.Connection,
    *,
    signal_id: int,
    ticker: str,
    event_type: Optional[str],
    sentiment: Optional[float],
    sentiment_magnitude: Optional[float],
    factual: Optional[int],
    source_weight: float,
    corroboration_count: int,
    author_quality: float,
    anti_pump_flag: int,
    composite_score: float,
    signal_class: Optional[str],
) -> int:
    cursor = conn.execute(
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
            event_type,
            sentiment,
            sentiment_magnitude,
            factual,
            source_weight,
            corroboration_count,
            author_quality,
            anti_pump_flag,
            composite_score,
            signal_class,
            utc_now(),
        ),
    )
    return cursor.lastrowid


def record_daemon_health(
    conn: sqlite3.Connection,
    *,
    source: str,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Track per-source poll health for the dashboard's daemon status panel."""
    now = utc_now()
    if success:
        conn.execute(
            """
            INSERT INTO daemon_health (source, last_poll_at, last_success_at, last_error, consecutive_errors)
            VALUES (?, ?, ?, NULL, 0)
            ON CONFLICT(source) DO UPDATE SET
                last_poll_at = excluded.last_poll_at,
                last_success_at = excluded.last_success_at,
                last_error = NULL,
                consecutive_errors = 0
            """,
            (source, now, now),
        )
    else:
        conn.execute(
            """
            INSERT INTO daemon_health (source, last_poll_at, last_error, consecutive_errors)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(source) DO UPDATE SET
                last_poll_at = excluded.last_poll_at,
                last_error = excluded.last_error,
                consecutive_errors = consecutive_errors + 1
            """,
            (source, now, error),
        )
