"""Rebuild Form 4 history from the SEC's bulk *Insider Transactions Data Sets*.

https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
publishes one zip per quarter (``{yyyy}q{n}_form345.zip``) holding every Form
3/4/5 as TSV tables: SUBMISSION (accession, filing date, issuer ticker, the
10b5-1 checkbox), NONDERIV_TRANS (code, shares, price, A/D), REPORTINGOWNER
(relationship, title). That is exactly ``insider_transactions`` without fetching
~250k individual filings, so the published_at event study can run on years of
history in minutes.

Caveats, stated so the study is read correctly:
- FILING_DATE has no clock time. Rows are stored with a date-only
  ``published_at``; the event study and the lane both treat a date-only stamp as
  "after the close" and roll the bet to the next trading day (conservative).
  ``--accept-times`` enriches qualifying rows with the exact EDGAR acceptance
  timestamp (one small request per filing, rate-limited).
- The data set lags: the latest quarter appears ~6 weeks after quarter end. The
  live ``sec_edgar`` ingestor covers the gap going forward.
- ``AFF10B5ONE`` is only populated for filings after the 2023 rule change; older
  rows are treated as not-a-plan (unknown), which slightly over-includes.

Pure parsing lives in ``parse_quarter``; ``write_rows`` is the only DB writer.
"""
from __future__ import annotations

import csv
import io
import logging
import os
import re
import sqlite3
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import requests

from ..config import CONFIG
from ..ingestors.insider_transactions import ensure_table

log = logging.getLogger("marketradar.backfill.form4ds")

DATASET_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{yyyy}q{q}_form345.zip"
SOURCE = "sec_edgar_form4_ds"
DEFAULT_UA = os.getenv("SEC_USER_AGENT", "").strip() or os.getenv("RESEARCH_CONTACT_UA", "").strip() \
    or "market-radar research (contact@example.com)"
CACHE_DIR = CONFIG.project_root / "data" / "sec_form345"

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def parse_sec_date(s: str) -> Optional[str]:
    """'30-SEP-2025' -> '2025-09-30'. Also accepts ISO. None if unparseable."""
    s = (s or "").strip()
    if not s:
        return None
    m = re.match(r"^(\d{1,2})-([A-Z]{3})-(\d{4})$", s.upper())
    if m:
        d, mon, y = int(m.group(1)), _MONTHS.get(m.group(2)), int(m.group(3))
        if mon:
            try:
                return date(y, mon, d).isoformat()
            except ValueError:
                return None
    try:
        return date.fromisoformat(s[:10]).isoformat()
    except ValueError:
        return None


def quarters_between(start: date, end: date) -> list[tuple[int, int]]:
    out = []
    y, q = start.year, (start.month - 1) // 3 + 1
    ey, eq = end.year, (end.month - 1) // 3 + 1
    while (y, q) <= (ey, eq):
        out.append((y, q))
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


@dataclass
class Form4Row:
    accession: str
    filing_date: str            # ISO date
    issuer_name: str
    ticker: str
    is_10b5_1: int
    insider_name: str
    is_officer: int
    is_director: int
    is_10pct: int
    officer_title: str
    role_score: int
    transaction_code: str
    shares: float
    price: float
    is_acquired: int
    trans_date: Optional[str]
    issuer_cik: str = ""


def _role_score(title: str, is_director: bool, is_officer: bool, is_10pct: bool) -> int:
    t = (title or "").upper()
    if any(k in t for k in ("CEO", "CFO", "CHIEF")):
        return 3
    if is_director:
        return 2
    if is_officer or is_10pct:
        return 1
    return 0


def _f(v) -> float:
    try:
        return float(str(v).replace(",", "")) if v not in (None, "") else 0.0
    except ValueError:
        return 0.0


def _read_tsv(z: zipfile.ZipFile, name: str) -> Iterator[dict]:
    with z.open(name) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        yield from csv.DictReader(text, delimiter="\t")


def parse_quarter(zip_path: Path, *, codes: Iterable[str] = ("P",)) -> list[Form4Row]:
    """Parse one quarterly zip into flat rows (one per non-derivative transaction
    whose code is in ``codes``; default code P only). Pure except for reading the zip."""
    codes = {c.upper() for c in codes}
    with zipfile.ZipFile(zip_path) as z:
        subs: dict[str, dict] = {}
        for r in _read_tsv(z, "SUBMISSION.tsv"):
            if (r.get("DOCUMENT_TYPE") or "").strip() not in ("4", "4/A"):
                continue
            subs[r["ACCESSION_NUMBER"]] = r
        owners: dict[str, dict] = {}
        for r in _read_tsv(z, "REPORTINGOWNER.tsv"):
            acc = r["ACCESSION_NUMBER"]
            if acc not in subs:
                continue
            rel = (r.get("RPTOWNER_RELATIONSHIP") or "").lower()
            o = owners.setdefault(acc, {"name": r.get("RPTOWNERNAME") or "", "officer": False,
                                        "director": False, "tenpct": False, "title": ""})
            o["officer"] |= "officer" in rel
            o["director"] |= "director" in rel
            o["tenpct"] |= "10%" in rel or "tenpercent" in rel or "ten percent" in rel
            if not o["title"]:
                o["title"] = r.get("RPTOWNER_TITLE") or ""
        out: list[Form4Row] = []
        for r in _read_tsv(z, "NONDERIV_TRANS.tsv"):
            code = (r.get("TRANS_CODE") or "").strip().upper()
            if code not in codes:
                continue
            acc = r["ACCESSION_NUMBER"]
            s = subs.get(acc)
            if not s:
                continue
            ticker = (s.get("ISSUERTRADINGSYMBOL") or "").strip().upper()
            if not ticker or ticker in ("NONE", "N/A") or len(ticker) > 6:
                continue
            fdate = parse_sec_date(s.get("FILING_DATE") or "")
            if not fdate:
                continue
            o = owners.get(acc, {"name": "", "officer": False, "director": False, "tenpct": False, "title": ""})
            aff = (s.get("AFF10B5ONE") or "").strip().lower()
            out.append(Form4Row(
                accession=acc, filing_date=fdate, issuer_name=(s.get("ISSUERNAME") or "").strip(),
                ticker=ticker, is_10b5_1=1 if aff in ("1", "true") else 0,
                insider_name=o["name"], is_officer=int(o["officer"]), is_director=int(o["director"]),
                is_10pct=int(o["tenpct"]), officer_title=o["title"],
                role_score=_role_score(o["title"], o["director"], o["officer"], o["tenpct"]),
                transaction_code=code, shares=_f(r.get("TRANS_SHARES")), price=_f(r.get("TRANS_PRICEPERSHARE")),
                is_acquired=1 if (r.get("TRANS_ACQUIRED_DISP_CD") or "").strip().upper() == "A" else 0,
                trans_date=parse_sec_date(r.get("TRANS_DATE") or ""),
                issuer_cik=(s.get("ISSUERCIK") or "").strip(),
            ))
    return out


def download_quarter(year: int, q: int, *, user_agent: str = DEFAULT_UA,
                     cache_dir: Path = CACHE_DIR, session: Optional[requests.Session] = None) -> Optional[Path]:
    """Download (cached) one quarter's zip. Returns None on 404 (not yet published)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{year}q{q}_form345.zip"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = DATASET_URL.format(yyyy=year, q=q)
    sess = session or requests.Session()
    r = sess.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=120)
    if r.status_code == 404:
        log.info("%dq%d not published yet (404)", year, q)
        return None
    r.raise_for_status()
    dest.write_bytes(r.content)
    log.info("downloaded %s (%.1f MB)", dest.name, len(r.content) / 1e6)
    return dest


def write_rows(conn: sqlite3.Connection, rows: Iterable[Form4Row], *, now: Optional[str] = None) -> dict:
    """Upsert rows into raw_signals (one stub per accession) + insider_transactions.
    Idempotent: re-running a quarter changes nothing."""
    ensure_table(conn)
    stamp = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_sig = n_tx = 0
    sig_ids: dict[str, int] = {}
    for r in rows:
        sid = sig_ids.get(r.accession)
        if sid is None:
            cur = conn.execute(
                """INSERT OR IGNORE INTO raw_signals
                     (source, source_tier, external_id, url, title, body, ingested_at, published_at)
                   VALUES (?, 1, ?, ?, ?, NULL, ?, ?)""",
                (SOURCE, r.accession,
                 f"https://www.sec.gov/Archives/edgar/data/{r.issuer_cik.lstrip('0') or '0'}/"
                 f"{r.accession.replace('-', '')}/{r.accession}-index.htm",
                 f"4 - {r.issuer_name} ({r.ticker})", stamp, r.filing_date),
            )
            if cur.rowcount:
                n_sig += 1
            sid = conn.execute("SELECT id FROM raw_signals WHERE source=? AND external_id=?",
                               (SOURCE, r.accession)).fetchone()[0]
            sig_ids[r.accession] = sid
        cur = conn.execute(
            """INSERT OR IGNORE INTO insider_transactions
                 (signal_id, ticker, insider_name, transaction_code, shares, price, is_acquired,
                  officer_title, is_officer, is_director, is_10pct, role_score, report_date,
                  ingested_at, is_10b5_1)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, r.ticker, r.insider_name, r.transaction_code, r.shares, r.price, r.is_acquired,
             r.officer_title, r.is_officer, r.is_director, r.is_10pct, r.role_score,
             r.trans_date or r.filing_date, stamp, r.is_10b5_1),
        )
        n_tx += cur.rowcount
    return {"signals_inserted": n_sig, "transactions_inserted": n_tx}


_ACCEPT_RE = re.compile(r"Accepted</div>\s*<div class=\"info\">\s*(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})", re.I)


def fetch_accept_time(issuer_cik: str, accession: str, *, user_agent: str = DEFAULT_UA,
                      session: Optional[requests.Session] = None) -> Optional[str]:
    """EDGAR acceptance timestamp (ET) for one filing, as an ISO-8601 UTC string.
    This is the real 'published_at'. One ~10 KB request; callers rate-limit."""
    url = (f"https://www.sec.gov/Archives/edgar/data/{issuer_cik.lstrip('0') or '0'}/"
           f"{accession.replace('-', '')}/{accession}-index.htm")
    sess = session or requests.Session()
    try:
        r = sess.get(url, headers={"User-Agent": user_agent}, timeout=30)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    m = _ACCEPT_RE.search(r.text)
    if not m:
        return None
    from zoneinfo import ZoneInfo
    et = datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}").replace(tzinfo=ZoneInfo("America/New_York"))
    return et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def enrich_accept_times(conn: sqlite3.Connection, *, min_notional: float, min_price: float,
                        max_requests: int = 2000, rps: float = 6.0, user_agent: str = DEFAULT_UA) -> int:
    """Replace date-only published_at with the EDGAR acceptance time for data-set
    rows that would QUALIFY for the lane (so we never spend requests on noise)."""
    rows = conn.execute(
        """SELECT rs.id, rs.external_id, rs.url
           FROM raw_signals rs
           WHERE rs.source = ? AND length(rs.published_at) = 10
             AND EXISTS (SELECT 1 FROM insider_transactions ix
                         WHERE ix.signal_id = rs.id AND ix.transaction_code='P' AND ix.is_acquired=1
                           AND ix.price >= ? AND ix.shares * ix.price >= ?)
           ORDER BY rs.published_at DESC LIMIT ?""",
        (SOURCE, min_price, min_notional, max_requests)).fetchall()
    sess = requests.Session()
    n = 0
    for r in rows:
        m = re.search(r"/edgar/data/(\d+)/", r["url"] or "")
        cik = m.group(1) if m else "0"
        ts = fetch_accept_time(cik, r["external_id"], user_agent=user_agent, session=sess)
        time.sleep(1.0 / rps)
        if ts:
            conn.execute("UPDATE raw_signals SET published_at=? WHERE id=?", (ts, r["id"]))
            n += 1
    return n
