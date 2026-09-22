"""Refresh the ``catalysts`` table with upcoming FDA PDUFA dates and
similar pre-known biotech catalysts.

Scrapes a single free calendar (MarketBeat's public PDUFA page) — no
API key required, ~1 request/refresh. The same flow can be pointed at
BiopharmaWatch or CatalystAlert by swapping the URL + parser.

Usage::

    python scripts/refresh_fda_catalysts.py
    python scripts/refresh_fda_catalysts.py --dry-run

Feature impact:
  ``ml/external_features.attach_catalyst_features`` reads from this
  table and emits ``days_until_catalyst``.

Best-effort by design — if the upstream HTML schema changes, we log and
return zero new catalysts rather than fail loudly.
"""
from __future__ import annotations

import os

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

from market_radar.storage import get_connection, init_db  # noqa: E402


log = logging.getLogger("refresh_fda_catalysts")


MARKETBEAT_URL = "https://www.marketbeat.com/fda-calendar/upcoming/"
USER_AGENT = os.getenv("RESEARCH_CONTACT_UA", "market-radar research (set RESEARCH_CONTACT_UA)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise_date(s: str) -> Optional[str]:
    """Parse a variety of date strings into ISO 8601 date."""
    s = s.strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


_DISEASE_ABBR = {
    "ADHD", "ALS", "BCA", "CKD", "CML", "COPD", "COVID", "DMD",
    "FDA", "GVHD", "HCC", "HIV", "HOCM", "IBS", "JIA", "LDL",
    "MDD", "MDS", "MS", "NAFLD", "NASH", "NDA", "PD", "PDUFA",
    "PSC", "RA", "RSV", "RTOR", "SCD", "TB", "T1D", "T2D",
    "UC", "VTE", "WAC",
}


def _ticker_from_row(text: str) -> Optional[str]:
    """Pull a ticker from a MarketBeat row cell.

    MarketBeat rows look like ``"CING Cingulate | $4.83 ..."`` — the
    ticker is the first all-caps token at the start. Fall back to
    cashtag/parens patterns. Reject obvious disease abbreviations.
    """
    text = text.strip()
    # Pattern 1: ticker at the start, immediately followed by space + Mixed-case word
    m = re.match(r"\s*([A-Z]{1,5})\s+[A-Z][a-z]", text)
    if m and m.group(1) not in _DISEASE_ABBR:
        return m.group(1)
    # Pattern 2: cashtag
    m = re.search(r"\$([A-Z]{1,5})\b", text)
    if m and m.group(1) not in _DISEASE_ABBR:
        return m.group(1)
    # Pattern 3: TICKER followed by another all-caps token (e.g. "CING NDA")
    m = re.match(r"\s*([A-Z]{2,5})\s+([A-Z][A-Za-z0-9.&,\-/']+)", text)
    if m and m.group(1) not in _DISEASE_ABBR:
        return m.group(1)
    # Pattern 4: parens-wrapped, but only if not a disease abbreviation
    for cand in re.findall(r"\(([A-Z]{1,5})\)", text):
        if cand not in _DISEASE_ABBR:
            return cand
    return None


def fetch_marketbeat_pdufa() -> list[dict]:
    """Pull upcoming PDUFA dates from MarketBeat's public table.

    Returns a list of dicts: {ticker, decision_date, catalyst_type,
    description}.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
    try:
        r = requests.get(MARKETBEAT_URL, headers=headers, timeout=20)
        r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("MarketBeat fetch failed: %s", exc)
        return []

    try:
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:  # noqa: BLE001
        soup = BeautifulSoup(r.text, "html.parser")

    rows: list[dict] = []
    # MarketBeat renders catalysts in <table> rows. Be defensive — try
    # multiple table classes / shapes.
    tables = soup.find_all("table")
    for tbl in tables:
        for tr in tbl.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td"])]
            if not cells:
                continue
            # Look for a ticker-shaped token in the first 2 cells and a
            # date-shaped token in any cell.
            ticker = None
            decision_date = None
            description_bits = []
            for c in cells:
                if ticker is None:
                    t = _ticker_from_row(c)
                    if t:
                        ticker = t
                if decision_date is None:
                    d = _normalise_date(c)
                    if d:
                        decision_date = d
                description_bits.append(c)
            if ticker and decision_date:
                rows.append({
                    "ticker": ticker,
                    "decision_date": decision_date,
                    "catalyst_type": "pdufa",
                    "description": " | ".join(description_bits)[:300],
                })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    init_db()

    rows = fetch_marketbeat_pdufa()
    log.info("Parsed %d PDUFA rows from MarketBeat", len(rows))
    if not rows:
        log.warning("No catalysts found — upstream HTML may have changed. "
                    "Inspect %s manually.", MARKETBEAT_URL)
        return 0

    if args.dry_run:
        log.info("Dry-run; first 5 rows:")
        for r in rows[:5]:
            log.info("  %s", r)
        return 0

    with get_connection() as conn:
        n = 0
        for r in rows:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO catalysts
                    (ticker, decision_date, catalyst_type, description, source, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (r["ticker"], r["decision_date"], r["catalyst_type"],
                     r["description"], "marketbeat", _utc_now()),
                )
                n += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("insert failed: %s", exc)
    log.info("Inserted/replaced %d catalysts.", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
