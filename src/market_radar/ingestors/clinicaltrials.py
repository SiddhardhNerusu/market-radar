"""ClinicalTrials.gov v2 ingestor — biotech catalyst coverage (blueprint #5).

A Phase 2/3 readout is the single biggest single-day move-maker in biotech: a
positive readout can double a micro-cap overnight, a failure can erase 80%. The
rest of the pipeline (RSS / Alpaca news / SEC 8-Ks) only sees these AFTER a PR
hits the wire — by which point the move has often already happened. The
ClinicalTrials.gov registry posts the *structured* trial record (phase, status,
completion date, and — critically — whether **results have been posted**) and is
the authoritative primary source. Watching it lets us seed a catalyst row the
moment a Phase 2/3 study flips to "results posted", independent of any PR.

This ingestor writes into the ``catalysts`` table (NOT raw_signals) — same table
the FDA/PDUFA + earnings catalysts land in, which the confluence sizer and the ML
``days_until_catalyst`` feature already read. ``catalyst_type='clinical_trial'``,
``source='clinicaltrials_v2'``.

API (no key, public):
    GET https://clinicaltrials.gov/api/v2/studies
Params we use:
    query.term = AREA[Phase](PHASE2 OR PHASE3)
                 AND AREA[LastUpdatePostDate]RANGE[<since>,MAX]
                 -- phase + recency filter (there is NO ``filter.phase`` param;
                    phase MUST go through the AREA[] search syntax)
    aggFilters = results:with     -- only studies that have POSTED results
    fields     = <trimmed protocolSection paths> + hasResults
    sort       = LastUpdatePostDate:desc
    pageSize   = up to 1000 (token-paginated via nextPageToken)
    countTotal = true
    format     = json

Sponsor → ticker is best-effort: a small contains/substring lookup over the
public sponsors we care about. Unmapped sponsors (academic centres, NIH/NCI,
foreign pharma without a US listing) get ticker=None — we still record the
catalyst so a later PR/news event can corroborate against it. Mapping is data,
not behaviour, so it lives in a dict here and degrades gracefully.

Never raises on fetch failure — returns 0 (mirrors the daemon-friendly contract
of the other ingestors).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import requests

from ..storage import get_connection

log = logging.getLogger(__name__)

API_URL = "https://clinicaltrials.gov/api/v2/studies"
SOURCE = "clinicaltrials_v2"
CATALYST_TYPE = "clinical_trial"

# The catalysts table's ``ticker`` is part of the PRIMARY KEY and therefore
# NOT NULL — a literal None can't be stored. So an unmapped sponsor is recorded
# under a non-ticker SENTINEL keyed on the NCT id (e.g. "CT:NCT02466971"). This
# keeps the catalyst row for later corroboration/coverage without ever masquerading
# as a real symbol: a "CT:" prefix can't collide with any real US ticker, so the
# confluence sizer / ML ``days_until_catalyst`` lookups (which query by real
# ticker) simply never match it. parse() still reports ticker=None as the honest
# mapping result; poll() substitutes the sentinel only at write time.
UNMAPPED_TICKER_PREFIX = "CT:"

# Trimmed field set keeps the payload small + the parser contract explicit.
_FIELDS = ",".join(
    [
        "protocolSection.identificationModule.nctId",
        "protocolSection.identificationModule.briefTitle",
        "protocolSection.designModule.phases",
        "protocolSection.statusModule.overallStatus",
        "protocolSection.statusModule.lastUpdatePostDateStruct",
        "protocolSection.statusModule.primaryCompletionDateStruct",
        "protocolSection.sponsorCollaboratorsModule.leadSponsor",
        "hasResults",
    ]
)

# Best-effort sponsor-name → US ticker map. Keys are lowercase substrings; a
# sponsor org matches if it CONTAINS the key (so "Regeneron Pharmaceuticals,
# Inc." matches "regeneron"). Intentionally small + biased to the large/mid-cap
# biotech & pharma names whose readouts actually move a tradeable US listing —
# academic / NIH / unlisted-foreign sponsors are deliberately left unmapped
# (ticker=None) rather than guessed. Extend as coverage gaps surface; this is a
# lookup table (data), never a control-flow branch.
_SPONSOR_TICKERS: dict[str, str] = {
    "regeneron": "REGN",
    "vertex pharmaceutical": "VRTX",
    "moderna": "MRNA",
    "biontech": "BNTX",
    "gilead": "GILD",
    "amgen": "AMGN",
    "biogen": "BIIB",
    "incyte": "INCY",
    "alnylam": "ALNY",
    "sarepta": "SRPT",
    "exelixis": "EXEL",
    "halozyme": "HALO",
    "neurocrine": "NBIX",
    "united therapeutics": "UTHR",
    "jazz pharmaceutical": "JAZZ",
    "ionis": "IONS",
    "bridgebio": "BBIO",
    "intra-cellular": "ITCI",
    "ascendis": "ASND",
    "insmed": "INSM",
    "krystal biotech": "KRYS",
    "madrigal": "MDGL",
    "arrowhead": "ARWR",
    "blueprint medicines": "BPMC",
    "novavax": "NVAX",
    "pfizer": "PFE",
    "merck sharp": "MRK",          # MSD/Merck & Co. lists trials as "Merck Sharp & Dohme"
    "eli lilly": "LLY",
    "bristol-myers squibb": "BMY",
    "bristol myers squibb": "BMY",
    "abbvie": "ABBV",
    "johnson & johnson": "JNJ",
    "janssen": "JNJ",              # J&J's pharma arm sponsors under "Janssen"
    "astrazeneca": "AZN",
    "novartis": "NVS",
    "glaxosmithkline": "GSK",
    "sanofi": "SNY",
    "bayer": "BAYRY",
    "takeda": "TAK",
    "roche": "RHHBY",
    "genentech": "RHHBY",          # Roche subsidiary
    "novo nordisk": "NVO",
    "boehringer": "BINGY",
}


def map_sponsor_to_ticker(sponsor: Optional[str]) -> Optional[str]:
    """Best-effort sponsor-org → US ticker. Returns None when unmapped.

    Case-insensitive substring match against ``_SPONSOR_TICKERS``. Prefers the
    longest matching key so a more specific sponsor name wins over a short
    generic one. Never raises.
    """
    if not sponsor:
        return None
    name = sponsor.strip().lower()
    if not name:
        return None
    best_key: Optional[str] = None
    for key in _SPONSOR_TICKERS:
        if key in name and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return _SPONSOR_TICKERS[best_key] if best_key else None


class ClinicalTrialsIngestor:
    """ClinicalTrials.gov v2 → ``catalysts`` table. Biotech Phase 2/3 readouts.

    Not an ``Ingestor`` subclass: it writes structured catalyst rows (ticker may
    be None) rather than ``raw_signals``/``signal_tickers``, so the base
    fetch→parse→insert_raw_signal contract doesn't apply. It exposes ``poll()``
    so the daemon can drive it the same way.
    """

    source = SOURCE

    def __init__(
        self,
        *,
        lookback_days: int = 7,
        page_size: int = 200,
        max_pages: int = 5,
        timeout: float = 25.0,
    ) -> None:
        self.lookback_days = max(1, int(lookback_days))
        self.page_size = max(1, min(int(page_size), 1000))
        self.max_pages = max(1, int(max_pages))
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "market-radar/1.0 (biotech catalyst monitor)",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------

    def _since_date(self) -> str:
        since = datetime.now(timezone.utc).date() - timedelta(days=self.lookback_days)
        return since.isoformat()

    def fetch(self) -> Iterable[dict[str, Any]]:
        """Yield raw study dicts for recently-updated Phase 2/3 studies WITH
        posted results. Token-paginated. Never raises — logs + stops on error."""
        term = (
            f"AREA[Phase](PHASE2 OR PHASE3) "
            f"AND AREA[LastUpdatePostDate]RANGE[{self._since_date()},MAX]"
        )
        page_token: Optional[str] = None
        pages = 0
        total_yielded = 0
        while pages < self.max_pages:
            params: dict[str, Any] = {
                "query.term": term,
                "aggFilters": "results:with",
                "fields": _FIELDS,
                "sort": "LastUpdatePostDate:desc",
                "pageSize": self.page_size,
                "countTotal": "true",
                "format": "json",
            }
            if page_token:
                params["pageToken"] = page_token
            try:
                resp = self._session.get(API_URL, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("[%s] fetch failed: %s", self.source, exc)
                return
            if resp.status_code >= 400:
                log.warning("[%s] HTTP %d: %s", self.source, resp.status_code,
                            resp.text[:200])
                return
            try:
                payload = resp.json()
            except ValueError as exc:
                log.warning("[%s] JSON decode failed: %s", self.source, exc)
                return

            studies = payload.get("studies") or []
            for study in studies:
                if isinstance(study, dict):
                    total_yielded += 1
                    yield study

            page_token = payload.get("nextPageToken")
            pages += 1
            if not page_token or not studies:
                break

        log.info("[%s] fetched %d studies across %d page(s)",
                 self.source, total_yielded, pages)

    # ------------------------------------------------------------------

    @staticmethod
    def parse(study: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Map one raw study dict to a catalyst row (or None to skip).

        Returns a dict with keys: ticker, decision_date, description, sponsor,
        nct_id. ``ticker`` may be None when the sponsor isn't mapped. Decision
        date is the primary-completion date when present, else the last-update
        date — both are the closest structured proxy for "when the readout
        landed". A row with neither date is skipped (no key for the table).
        """
        proto = study.get("protocolSection") or {}
        ident = proto.get("identificationModule") or {}
        status = proto.get("statusModule") or {}
        design = proto.get("designModule") or {}
        spons = proto.get("sponsorCollaboratorsModule") or {}

        nct_id = (ident.get("nctId") or "").strip()
        if not nct_id:
            return None

        primary = (status.get("primaryCompletionDateStruct") or {}).get("date")
        last_update = (status.get("lastUpdatePostDateStruct") or {}).get("date")
        decision_date = (primary or last_update or "").strip()
        if not decision_date:
            return None

        sponsor = ((spons.get("leadSponsor") or {}).get("name") or "").strip() or None
        ticker = map_sponsor_to_ticker(sponsor)

        phases = design.get("phases") or []
        phase_str = "/".join(p.replace("PHASE", "Phase ") for p in phases) or "Phase 2/3"
        overall = (status.get("overallStatus") or "").strip()
        brief = (ident.get("briefTitle") or "").strip()

        lead = f"{phase_str} clinical trial {nct_id}"
        if sponsor:
            lead += f" (sponsor {sponsor})"
        tail_bits = []
        if overall:
            tail_bits.append(overall.replace("_", " ").lower())
        tail_bits.append("results posted")
        description = f"{lead} — {', '.join(tail_bits)}"
        if brief:
            description = f"{description}: {brief}"

        return {
            "ticker": ticker,
            "decision_date": decision_date,
            "description": description,
            "sponsor": sponsor,
            "nct_id": nct_id,
        }

    # ------------------------------------------------------------------

    def poll(self) -> int:
        """Run one fetch → parse → upsert cycle. Returns rows written.

        Never raises. Studies whose sponsor maps to no ticker are still inserted
        (ticker=None) so they exist for later corroboration. Parse runs BEFORE
        the writer is opened (matches the base ingestor's "no network/CPU under
        the SQLite writer" rule)."""
        try:
            raw = list(self.fetch())
        except Exception as exc:  # noqa: BLE001 — daemon-safe top-level catch
            log.warning("[%s] poll fetch error: %s", self.source, exc)
            return 0

        rows: list[dict[str, Any]] = []
        for study in raw:
            try:
                parsed = self.parse(study)
            except Exception as exc:  # noqa: BLE001
                log.debug("[%s] parse error: %s", self.source, exc)
                continue
            if parsed is not None:
                rows.append(parsed)

        if not rows:
            return 0

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        wrote = 0
        try:
            with get_connection() as conn:
                for row in rows:
                    try:
                        # ticker is a NOT-NULL PK column, so unmapped studies are
                        # stored under a "CT:<nct>" sentinel (never a real symbol)
                        # — keyed on the NCT id so each unmapped study is distinct
                        # and re-polls upsert rather than duplicate.
                        ticker = row["ticker"] or (
                            UNMAPPED_TICKER_PREFIX + row["nct_id"]
                        )
                        conn.execute(
                            """
                            INSERT INTO catalysts
                              (ticker, decision_date, catalyst_type, description,
                               source, ingested_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(ticker, decision_date, catalyst_type)
                            DO UPDATE SET
                              description = excluded.description,
                              source      = excluded.source,
                              ingested_at = excluded.ingested_at
                            """,
                            (
                                ticker,
                                row["decision_date"],
                                CATALYST_TYPE,
                                row["description"],
                                SOURCE,
                                now_iso,
                            ),
                        )
                        wrote += 1
                    except Exception as exc:  # noqa: BLE001
                        log.debug("[%s] insert %s failed: %s",
                                  self.source, row.get("nct_id"), exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] DB write error: %s", self.source, exc)
            return wrote

        log.info("[%s] poll done: parsed=%d wrote=%d", self.source, len(rows), wrote)
        return wrote
