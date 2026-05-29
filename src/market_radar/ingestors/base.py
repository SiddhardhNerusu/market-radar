"""Base class shared by every ingestor.

An ingestor is responsible for:
  1. Polling its source for new content (RSS, API, etc.)
  2. Parsing each entry into a structured ``ParsedSignal``
  3. Extracting one or more tickers (with confidence)
  4. Writing the result to the ``raw_signals`` and ``signal_tickers`` tables
     via ``storage.insert_raw_signal`` (which handles dedup)

The base class handles error reporting + health tracking. Concrete
ingestors override ``fetch()`` and ``parse()``.
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..storage import get_connection, insert_raw_signal
from ..storage.db import record_daemon_health

log = logging.getLogger(__name__)


@dataclass
class TickerMention:
    ticker: str
    market: Optional[str] = None
    asset_class: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "market": self.market,
            "asset_class": self.asset_class,
            "confidence": self.confidence,
        }


@dataclass
class ParsedSignal:
    """One unit of content the ingestor wants to record."""
    external_id: str               # source-specific id used for dedup
    title: Optional[str] = None
    body: Optional[str] = None
    url: Optional[str] = None
    author: Optional[str] = None
    author_metadata: Optional[dict[str, Any]] = None
    published_at: Optional[str] = None  # ISO 8601 if known
    raw_payload: Optional[dict[str, Any]] = None
    tickers: list[TickerMention] = field(default_factory=list)


@dataclass
class PollResult:
    fetched: int = 0
    inserted: int = 0
    duplicates: int = 0
    errors: int = 0


class Ingestor(abc.ABC):
    """Base class. Subclasses must set ``source`` and ``source_tier``
    and implement ``fetch`` + ``parse``.
    """

    source: str = ""
    source_tier: int = 0

    # If the source supports query parameters or per-request shape control
    # this is where subclasses can read it from kwargs at construction.

    def __init__(self) -> None:
        if not self.source:
            raise ValueError(f"{type(self).__name__} must set class attr 'source'")
        if self.source_tier not in (1, 2, 3, 4):
            raise ValueError(
                f"{type(self).__name__}.source_tier must be 1/2/3/4 (got {self.source_tier!r})"
            )

    # ------------------------------------------------------------------
    # Public API used by the daemon
    # ------------------------------------------------------------------

    def poll(self) -> PollResult:
        """Run one polling cycle: fetch → parse → persist. Never raises;
        errors are logged + recorded in daemon_health, and the result is
        returned with ``errors > 0``."""
        result = PollResult()
        try:
            raw_entries = list(self.fetch())
        except Exception as exc:  # noqa: BLE001 — top-level catch by design
            log.exception("[%s] fetch failed", self.source)
            with get_connection() as conn:
                record_daemon_health(conn, source=self.source, success=False, error=str(exc))
            result.errors += 1
            return result

        result.fetched = len(raw_entries)
        if not raw_entries:
            with get_connection() as conn:
                record_daemon_health(conn, source=self.source, success=True)
            return result

        with get_connection() as conn:
            for entry in raw_entries:
                try:
                    signal = self.parse(entry)
                except Exception as exc:  # noqa: BLE001
                    log.warning("[%s] parse failed: %s", self.source, exc, exc_info=True)
                    result.errors += 1
                    continue

                if signal is None:
                    continue

                try:
                    inserted_id = insert_raw_signal(
                        conn,
                        source=self.source,
                        source_tier=self.source_tier,
                        external_id=signal.external_id,
                        url=signal.url,
                        title=signal.title,
                        body=signal.body,
                        author=signal.author,
                        author_metadata=signal.author_metadata,
                        raw_payload=signal.raw_payload,
                        published_at=signal.published_at,
                        tickers=[t.to_dict() for t in signal.tickers],
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "[%s] insert failed for %s: %s",
                        self.source, signal.external_id, exc,
                    )
                    result.errors += 1
                    continue

                if inserted_id is None:
                    result.duplicates += 1
                else:
                    result.inserted += 1

            record_daemon_health(conn, source=self.source, success=result.errors == 0)

        log.info(
            "[%s] poll done: fetched=%d inserted=%d dup=%d err=%d",
            self.source, result.fetched, result.inserted,
            result.duplicates, result.errors,
        )
        return result

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fetch(self) -> Iterable[Any]:
        """Return an iterable of raw source-specific entries (dicts, feedparser
        entries, JSON items, etc.)."""

    @abc.abstractmethod
    def parse(self, raw_entry: Any) -> Optional[ParsedSignal]:
        """Convert a raw entry to a ParsedSignal. Return None to skip."""
