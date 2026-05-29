"""SEC filing body fetcher — pulls actual filing content given an EDGAR URL.

The live RSS ingestor (``sec_edgar.py``) and the historical backfill both
record only filing metadata (title + accession + URL); neither downloads
the actual document. This module fills that gap: given a SEC filing URL,
it resolves the filing's primary document, downloads it, strips
HTML/SGML markup, and returns cleaned text capped at ``MAX_BODY_CHARS``.

Why this exists:
  Without bodies, the LLM classifier sees only the title (e.g.
  ``"8-K - Apple Inc."``) and returns ``event_type="other"`` ~96% of
  the time. With bodies, it can read ``"Item 2.02 Results of Operations
  and Financial Condition"`` + the press-release content and correctly
  classify the event.

Used by:
  - ``scripts/fetch_sec_bodies.py`` — bulk backfill of body=NULL rows
  - ``sec_edgar.py`` live ingestor — post-parse hook for new filings

EDGAR endpoints used:
  1. ``/Archives/edgar/data/<CIK>/<ACC_NO_DASHES>/index.json``
     JSON manifest of all documents in a filing.
  2. ``/Archives/edgar/data/<CIK>/<ACC_NO_DASHES>/<DOC>``
     Direct URL for any document referenced in the manifest.
  3. ``/Archives/edgar/data/<CIK>/<DASHED_ACC>.txt`` (and the new-style
     variant inside the accession dir) — concatenated SGML full
     submission, used as a fallback when (1) fails.

SEC rate limit: 10 req/sec hard cap. We pace at ~6 rps (≈170ms between
HTTP calls by default) to leave headroom.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import warnings
from bs4 import BeautifulSoup
try:
    from bs4 import XMLParsedAsHTMLWarning  # available in bs4 >= 4.11
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:  # older bs4
    pass

from ..config import PROJECT_ROOT
from .cik_lookup import DEFAULT_UA

log = logging.getLogger(__name__)


# ---- tuning knobs ---------------------------------------------------------

MAX_BODY_CHARS = 8000           # cap fed to LLM (bounds prompt token cost)
MIN_USEFUL_CHARS = 200          # below this we treat the fetch as failed
INDEX_TIMEOUT_S = 15.0
DOC_TIMEOUT_S = 30.0
PRIMARY_DOC_CAP_BYTES = 2_000_000   # skip docs bigger than 2 MB
FULL_SUBMISSION_CAP_BYTES = 1_500_000
DEFAULT_INTER_REQUEST_S = 0.17  # ~6 rps; SEC's hard cap is 10 rps
MAX_RETRIES = 2                 # for transient 429/5xx
RETRY_BACKOFF_S = 1.5

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "sec_body_cache"


# ---- URL parsing ----------------------------------------------------------

# Accession-number variants observed in raw_signals.url:
#   .../data/<CIK>/<NODASH18>/<DASHED>-index.htm        (live RSS)
#   .../data/<CIK>/<DASHED>.txt                          (backfill 1)
#   .../data/<CIK>/<NODASH18>/<DASHED>.txt               (backfill 2)
_RE_CIK = re.compile(r"/data/(\d+)/")
_RE_ACC_NODASH = re.compile(r"/(\d{18})(?:/|$|\.)")
_RE_ACC_DASHED = re.compile(r"(\d{10}-\d{2}-\d{6})")


@dataclass(frozen=True)
class FilingRef:
    cik: int
    accession_no_dashes: str       # 18-digit form
    accession_dashed: str          # 10-2-6 form


def parse_url(url: Optional[str]) -> Optional[FilingRef]:
    """Pull CIK + accession out of an EDGAR URL.

    Returns ``None`` if either component is missing.
    """
    if not url:
        return None
    cik_m = _RE_CIK.search(url)
    if not cik_m:
        return None
    try:
        cik = int(cik_m.group(1))
    except ValueError:
        return None

    no_dash_m = _RE_ACC_NODASH.search(url)
    if no_dash_m:
        no_dash = no_dash_m.group(1)
        dashed = f"{no_dash[:10]}-{no_dash[10:12]}-{no_dash[12:]}"
        return FilingRef(cik=cik, accession_no_dashes=no_dash,
                         accession_dashed=dashed)

    dashed_m = _RE_ACC_DASHED.search(url)
    if dashed_m:
        dashed = dashed_m.group(1)
        no_dash = dashed.replace("-", "")
        if len(no_dash) == 18:
            return FilingRef(cik=cik, accession_no_dashes=no_dash,
                             accession_dashed=dashed)

    return None


# ---- text cleaners --------------------------------------------------------

# DOCUMENT block matcher for SGML full-submission files
_RE_DOC = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
_RE_DOC_TYPE = re.compile(r"<TYPE>\s*([^<\r\n]+)", re.IGNORECASE)
_RE_DOC_TEXT = re.compile(r"<TEXT>(.*?)</TEXT>", re.IGNORECASE | re.DOTALL)
_RE_WS = re.compile(r"[ \t ]+")
_RE_BLANK_LINES = re.compile(r"\n\s*\n+")


def _strip_html(html_text: str) -> str:
    """BeautifulSoup-based HTML/SGML to text. Robust against malformed input."""
    if not html_text:
        return ""
    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:  # noqa: BLE001 — lxml can choke on weird SGML
        soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "head", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = _RE_WS.sub(" ", text)
    text = _RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _strip_xml_keep_structure(xml_text: str) -> str:
    """Lightweight XML stripper that keeps tag NAMES as readable tokens.

    Used for Form 4 / Form 3 / Form 5 which file structured XML where the
    important info IS the (tag, value) pairs: ``<transactionCode>P</...>``
    means a Purchase. A pure ``get_text`` would lose the tag names.
    """
    if not xml_text:
        return ""
    # Replace opening tags with their tag name surrounded by spaces; drop
    # closing tags + self-closing markers + XML decl.
    text = re.sub(r"<\?xml[^>]*\?>", "", xml_text)
    text = re.sub(r"</[A-Za-z_:][^>]*>", "\n", text)
    text = re.sub(
        r"<([A-Za-z_:][A-Za-z0-9_:\-\.]*)[^>]*>",
        lambda m: f" {m.group(1)} ",
        text,
    )
    text = _RE_WS.sub(" ", text)
    text = _RE_BLANK_LINES.sub("\n", text)
    return text.strip()


# ---- primary-document selection ------------------------------------------

def _score_doc_name(name: str, form_type: Optional[str]) -> int:
    """Higher score => more likely to be the filing's primary document.

    Index pages, images, stylesheets are scored < 0 so they're discarded.
    """
    if not name:
        return -1
    n = name.lower()
    # Drop EDGAR index/header pages (don't contain real filing content)
    if n.endswith(("-index.htm", "-index.html", "-index-headers.html",
                   "/index.json", "index.json", ".hdr.sgml")):
        return -1
    if n.endswith((".jpg", ".jpeg", ".png", ".gif", ".css", ".js",
                   ".pdf", ".zip")):
        return -1

    # Form 3/4/5 ownership filings use a structured XML primary doc.
    is_ownership_xml = (
        n.endswith(".xml") and (
            n == "ownership.xml"
            or "form3" in n or "form4" in n or "form5" in n
            or n.startswith("wf-form") or n.startswith("primary_doc")
            or "f345" in n
        )
    )
    if is_ownership_xml:
        # Strongest possible boost — it's effectively the canonical primary
        return 12

    score = 0
    if n.endswith((".htm", ".html")):
        score += 5
    elif n.endswith(".xml"):
        score += 3
    elif n.endswith(".txt"):
        score += 2
    else:
        score += 1

    if form_type:
        ft = re.sub(r"[^a-z0-9]", "", form_type.lower())
        if ft and ft in re.sub(r"[^a-z0-9]", "", n):
            score += 4

    if "exhibit" in n or n.startswith("ex-") or re.match(r"^ex\d", n):
        score -= 2

    return score


def _pick_primary(items: list[dict], form_type: Optional[str]) -> Optional[dict]:
    if not items:
        return None
    scored = []
    for it in items:
        name = it.get("name", "") or ""
        s = _score_doc_name(name, form_type)
        if s < 0:
            continue
        try:
            size = int(it.get("size", 0) or 0)
        except (TypeError, ValueError):
            size = 0
        scored.append((s, size, it))
    if not scored:
        return None
    # Higher score wins; tie -> larger size wins
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


# ---- politeness gate ------------------------------------------------------

_RATE_LOCK = threading.Lock()
_LAST_REQUEST_AT = [0.0]


def _polite_sleep(min_interval_s: float) -> None:
    with _RATE_LOCK:
        gap = time.monotonic() - _LAST_REQUEST_AT[0]
        if gap < min_interval_s:
            time.sleep(min_interval_s - gap)
        _LAST_REQUEST_AT[0] = time.monotonic()


# ---- fetcher --------------------------------------------------------------

@dataclass
class FetchResult:
    body: Optional[str]
    source_strategy: str           # "cache" / "index_json" / "full_txt" / "miss"
    primary_doc: Optional[str]     # filename of doc the body came from


class SecBodyFetcher:
    """Resolve a SEC filing URL to clean body text, with on-disk cache.

    Cache layout: one ``.txt`` file per accession (no-dashes form) under
    ``data/sec_body_cache/``. Empty file = "previously tried and failed,
    skip for now."
    """

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_UA,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        min_interval_s: float = DEFAULT_INTER_REQUEST_S,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self._session = session or requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })

    # ----- public API -----

    def fetch(
        self,
        url: Optional[str],
        *,
        form_type: Optional[str] = None,
        use_cache: bool = True,
    ) -> FetchResult:
        """Fetch & clean the primary document for a filing URL.

        Returns a FetchResult. ``body`` is None when the document could
        not be retrieved or was too small to be useful.
        """
        ref = parse_url(url)
        if ref is None:
            return FetchResult(body=None, source_strategy="miss",
                               primary_doc=None)

        if use_cache:
            cached = self._read_cache(ref)
            if cached is not None:
                return FetchResult(
                    body=cached or None,
                    source_strategy="cache",
                    primary_doc=None,
                )

        # Strategy A — manifest-based
        result = self._fetch_via_index_json(ref, form_type=form_type)
        if result.body and len(result.body) >= MIN_USEFUL_CHARS:
            self._write_cache(ref, result.body)
            return result

        # Strategy B — full submission .txt
        result = self._fetch_via_full_submission(ref)
        if result.body and len(result.body) >= MIN_USEFUL_CHARS:
            self._write_cache(ref, result.body)
            return result

        # Cache the empty result so we don't keep retrying
        self._write_cache(ref, "")
        return FetchResult(body=None, source_strategy="miss",
                           primary_doc=None)

    def fetch_body(self, url: Optional[str], *,
                   form_type: Optional[str] = None,
                   use_cache: bool = True) -> Optional[str]:
        """Convenience wrapper returning only the body string (or None)."""
        return self.fetch(url, form_type=form_type, use_cache=use_cache).body

    # ----- internals -----

    def _cache_path(self, ref: FilingRef) -> Path:
        return self.cache_dir / f"{ref.accession_no_dashes}.txt"

    def _read_cache(self, ref: FilingRef) -> Optional[str]:
        path = self._cache_path(ref)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _write_cache(self, ref: FilingRef, body: str) -> None:
        try:
            self._cache_path(ref).write_text(body, encoding="utf-8")
        except OSError as exc:
            log.debug("cache write failed for %s: %s",
                      ref.accession_no_dashes, exc)

    def _get_with_retry(self, url: str, *, timeout: float) -> Optional[requests.Response]:
        """GET with polite spacing + small retry on 429 / 5xx / transient
        connection errors. Returns None on hard 404 or after retries
        exhausted.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES + 1):
            _polite_sleep(self.min_interval_s)
            try:
                r = self._session.get(url, timeout=timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_S * (attempt + 1))
                    continue
                log.debug("GET %s failed after retries: %s", url, exc)
                return None

            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt < MAX_RETRIES:
                    sleep_for = RETRY_BACKOFF_S * (attempt + 1)
                    if r.status_code == 429:
                        sleep_for = max(sleep_for, 5.0)
                    log.debug("GET %s got %d, retrying in %.1fs",
                              url, r.status_code, sleep_for)
                    time.sleep(sleep_for)
                    continue
                log.debug("GET %s gave up at status %d", url, r.status_code)
                return None
            if not r.ok:
                log.debug("GET %s status=%d", url, r.status_code)
                return None
            return r
        if last_exc:
            log.debug("GET %s exhausted retries: %s", url, last_exc)
        return None

    def _fetch_via_index_json(self, ref: FilingRef,
                              *, form_type: Optional[str]) -> FetchResult:
        manifest_url = (
            f"https://www.sec.gov/Archives/edgar/data/{ref.cik}/"
            f"{ref.accession_no_dashes}/index.json"
        )
        r = self._get_with_retry(manifest_url, timeout=INDEX_TIMEOUT_S)
        if r is None:
            return FetchResult(None, "miss", None)
        try:
            manifest = r.json()
        except ValueError as exc:
            log.debug("index.json not JSON for %s: %s", manifest_url, exc)
            return FetchResult(None, "miss", None)

        items = manifest.get("directory", {}).get("item", []) or []
        primary = _pick_primary(items, form_type)
        if not primary:
            return FetchResult(None, "miss", None)

        name = primary.get("name", "")
        try:
            size = int(primary.get("size", 0) or 0)
        except (TypeError, ValueError):
            size = 0
        if size and size > PRIMARY_DOC_CAP_BYTES:
            log.debug("primary doc too large (%d bytes) %s/%s, skipping",
                      size, ref.accession_no_dashes, name)
            return FetchResult(None, "miss", name)

        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/{ref.cik}/"
            f"{ref.accession_no_dashes}/{name}"
        )
        r = self._get_with_retry(doc_url, timeout=DOC_TIMEOUT_S)
        if r is None:
            return FetchResult(None, "miss", name)

        raw = r.text or ""
        if name.lower().endswith(".xml"):
            text = _strip_xml_keep_structure(raw)
        else:
            text = _strip_html(raw)
        body = text[:MAX_BODY_CHARS] if text else None
        return FetchResult(body=body, source_strategy="index_json",
                           primary_doc=name)

    def _fetch_via_full_submission(self, ref: FilingRef) -> FetchResult:
        # Try new-style location first, then legacy.
        candidates = [
            (f"https://www.sec.gov/Archives/edgar/data/{ref.cik}/"
             f"{ref.accession_no_dashes}/{ref.accession_dashed}.txt"),
            f"https://www.sec.gov/Archives/edgar/data/{ref.cik}/{ref.accession_dashed}.txt",
        ]
        for url in candidates:
            r = self._get_with_retry(url, timeout=DOC_TIMEOUT_S)
            if r is None:
                continue
            raw = r.text or ""
            if not raw:
                continue
            if len(raw) > FULL_SUBMISSION_CAP_BYTES:
                raw = raw[:FULL_SUBMISSION_CAP_BYTES]
            extracted = _extract_first_document_text(raw)
            if extracted is None:
                # Last-ditch: strip the whole envelope
                stripped = _strip_html(raw)
            else:
                # If the extracted block is an XML form, preserve structure
                if extracted.lstrip().lower().startswith("<?xml") \
                        or "<ownershipDocument" in extracted:
                    stripped = _strip_xml_keep_structure(extracted)
                else:
                    stripped = _strip_html(extracted)
            if stripped and len(stripped) >= MIN_USEFUL_CHARS:
                body = stripped[:MAX_BODY_CHARS]
                return FetchResult(body=body, source_strategy="full_txt",
                                   primary_doc=ref.accession_dashed + ".txt")
        return FetchResult(None, "miss", None)


def _extract_first_document_text(submission_text: str) -> Optional[str]:
    """Return the TEXT block of the first useful DOCUMENT in an SGML
    full-submission file. Prefers non-graphic, non-zip documents. Returns
    None if no DOCUMENT blocks parse out cleanly.
    """
    if not submission_text:
        return None
    docs = _RE_DOC.findall(submission_text)
    if not docs:
        return None
    skip_types = {"GRAPHIC", "EXCEL", "ZIP", "JSON", "XBRL"}
    for doc in docs[:5]:
        type_m = _RE_DOC_TYPE.search(doc)
        dtype = (type_m.group(1).strip().upper() if type_m else "")
        if dtype in skip_types:
            continue
        text_m = _RE_DOC_TEXT.search(doc)
        if not text_m:
            continue
        block = text_m.group(1).strip()
        if block and len(block) >= MIN_USEFUL_CHARS:
            return block
    return None


# ---- module-level convenience --------------------------------------------

_DEFAULT_FETCHER: Optional[SecBodyFetcher] = None


def get_default_fetcher() -> SecBodyFetcher:
    global _DEFAULT_FETCHER
    if _DEFAULT_FETCHER is None:
        _DEFAULT_FETCHER = SecBodyFetcher()
    return _DEFAULT_FETCHER


def fetch_body_for_url(url: str, *, form_type: Optional[str] = None,
                       use_cache: bool = True) -> Optional[str]:
    """Top-level helper. Uses a module-singleton fetcher."""
    return get_default_fetcher().fetch_body(url, form_type=form_type,
                                            use_cache=use_cache)
