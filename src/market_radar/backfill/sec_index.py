"""Fetch + parse SEC EDGAR full-index ``form.idx`` files.

Each quarter's file is at:
    https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{n}/form.idx

The file lists every EDGAR filing in that quarter with form type, company,
CIK, filed date, and the path to the filing on EDGAR. Fixed-width-ish format.

We cache downloads on disk so re-runs are cheap.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import requests

from ..config import PROJECT_ROOT
from ..ingestors.cik_lookup import DEFAULT_UA

log = logging.getLogger(__name__)


CACHE_DIR = PROJECT_ROOT / "data" / "sec_backfill_cache"

INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/form.idx"

# Form types worth backfilling. Form 4 deliberately excluded because the
# CIK in form.idx is the *filer's* (an insider person), not the issuer
# company — getting the issuer requires fetching each filing's body.
BACKFILL_FORMS = {
    "8-K": "material_event",
    "8-K/A": "material_event_amend",
    "SC 13D": "activist_position",
    "SC 13G": "passive_5pct_stake",
    "SC 13D/A": "activist_position_amend",
    "SC 13G/A": "passive_5pct_stake_amend",
    "S-1": "ipo_registration",
    "S-1/A": "ipo_registration_amend",
    "DEF 14A": "proxy_statement",
    "425": "m_a_communication",
}


@dataclass
class FilingRecord:
    form: str               # e.g. "8-K"
    form_event: str         # taxonomy bucket, e.g. "material_event"
    company: str
    cik: int
    filed: date
    filename: str

    def filing_url(self) -> str:
        # form.idx gives us "edgar/data/{cik}/...-index.htm" — prepend Archives base
        return f"https://www.sec.gov/Archives/{self.filename}"


def cache_path(year: int, qtr: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"form_{year}_q{qtr}.idx"


def fetch_quarter_index(year: int, qtr: int, *, user_agent: str = DEFAULT_UA) -> Path:
    """Download (with caching) a quarter's form.idx and return the local path."""
    path = cache_path(year, qtr)
    if path.exists() and path.stat().st_size > 0:
        log.debug("Using cached index for %s Q%s (%d bytes)", year, qtr, path.stat().st_size)
        return path

    url = INDEX_URL.format(year=year, qtr=qtr)
    log.info("Fetching SEC full-index %s Q%s …", year, qtr)
    resp = requests.get(
        url,
        headers={"User-Agent": user_agent, "Accept": "text/plain"},
        timeout=60,
    )
    if resp.status_code == 404:
        # Future quarter — not yet published
        log.info("  no index yet for %s Q%s (404)", year, qtr)
        path.write_text("")
        return path
    resp.raise_for_status()
    path.write_text(resp.text)
    log.info("  cached %d bytes to %s", len(resp.text), path)
    time.sleep(0.2)  # pacing
    return path


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


_SPLIT_RE = re.compile(r"\s{2,}")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_idx_file(path: Path) -> Iterable[FilingRecord]:
    """Yield ``FilingRecord`` for every backfill-relevant form in the file.

    form.idx is column-aligned with variable spacing. Robust parse: split each
    data line on runs of 2+ whitespace into 5 fields:
        [Form Type, Company Name, CIK, Date Filed, File Name]

    Form types like ``SC 13D``, ``DEF 14A``, ``1-A POS`` survive this split
    because the internal spaces between their words are single-space, not 2+.
    """
    if not path.exists():
        return
    text = path.read_text(errors="replace")
    in_data = False
    for line in text.splitlines():
        if not in_data:
            if line.startswith("---"):
                in_data = True
            continue
        if not line.strip():
            continue

        fields = _SPLIT_RE.split(line.strip())
        if len(fields) < 5:
            continue
        form_type, company, cik_str, date_str, filename = fields[:5]

        if form_type not in BACKFILL_FORMS:
            continue
        if not _DATE_RE.match(date_str):
            continue
        try:
            cik = int(cik_str)
        except ValueError:
            continue
        try:
            filed = date.fromisoformat(date_str)
        except ValueError:
            continue

        yield FilingRecord(
            form=form_type,
            form_event=BACKFILL_FORMS[form_type],
            company=company,
            cik=cik,
            filed=filed,
            filename=filename,
        )


# ---------------------------------------------------------------------------
# High-level walker
# ---------------------------------------------------------------------------


def walk_quarters(
    *,
    end_year: int,
    end_qtr: int,
    quarters: int = 8,
    user_agent: str = DEFAULT_UA,
) -> Iterable[FilingRecord]:
    """Yield FilingRecord across the last ``quarters`` quarters ending at
    (end_year, end_qtr) inclusive."""
    y, q = end_year, end_qtr
    targets: list[tuple[int, int]] = []
    for _ in range(quarters):
        targets.append((y, q))
        q -= 1
        if q == 0:
            q = 4
            y -= 1
    targets.reverse()

    for year, qtr in targets:
        try:
            path = fetch_quarter_index(year, qtr, user_agent=user_agent)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to fetch %s Q%s: %s", year, qtr, exc)
            continue
        count = 0
        for record in parse_idx_file(path):
            count += 1
            yield record
        log.info("  → %s Q%s yielded %d backfill-relevant records", year, qtr, count)
