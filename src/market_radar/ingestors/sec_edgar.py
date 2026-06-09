"""SEC EDGAR ingestor — pulls recent filings of high-signal types.

We poll EDGAR's per-form-type Atom feeds and treat each filing as a Tier 1
factual signal. Coverage:

  - 8-K     Material events (earnings, M&A, leadership change, etc.)
  - 4       Insider transactions (buys/sells by officers/directors/10% holders)
  - SC 13D  Beneficial ownership > 5% (activist filings)
  - SC 13G  Passive 5%+ stakes
  - S-1     IPO registration
  - DEF 14A Proxy statements

SEC requires a descriptive User-Agent header and rate-limits to ~10 req/sec.

References:
  - https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent
  - https://www.sec.gov/os/accessing-edgar-data
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import feedparser
import requests

from .base import Ingestor, ParsedSignal, TickerMention
from .cik_lookup import CIK_LOOKUP, DEFAULT_UA
from .sec_body_fetcher import SecBodyFetcher

log = logging.getLogger(__name__)


# Feed URL template — Atom output, current filings of one type
FEED_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type={form}&output=atom&count=40"
)


# Form types worth ingesting. The label drives a coarse event_type bucket
# used downstream by the scorer.
FORM_TYPES = {
    "8-K": "material_event",
    "6-K": "material_event",   # foreign issuers' material-news filing (their 8-K equivalent).
                                # Many micro-cap runners are foreign (Chinese/Israeli/etc.) and
                                # file catalysts as 6-K — without this we're blind to all of them
                                # (e.g. RGNT's European-launch 6-K that ran it +141% on 2026-06-09).
    "4": "insider_transaction",
    "SC 13D": "activist_position",
    "SC 13G": "passive_5pct_stake",
    "S-1": "ipo_registration",
    "DEF 14A": "proxy_statement",
}


# Matches "Company Name (0000123456) (Filer)" or "8-K - Company Name (0000123456)"
TITLE_CIK_RE = re.compile(r"\((\d{6,10})\)")


class SecEdgarIngestor(Ingestor):
    source = "sec_edgar"
    source_tier = 1

    def __init__(
        self,
        *,
        forms: Optional[list[str]] = None,
        user_agent: str = DEFAULT_UA,
        timeout: float = 20.0,
        fetch_bodies: bool = True,
        body_fetcher: Optional[SecBodyFetcher] = None,
    ) -> None:
        super().__init__()
        self.forms = forms or list(FORM_TYPES.keys())
        self.user_agent = user_agent
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/atom+xml,application/xml;q=0.9,*/*;q=0.8",
                "Host": "www.sec.gov",
            }
        )
        # Body fetching is best-effort: failures don't block ingestion.
        # Disable via fetch_bodies=False in tests or constrained environments.
        self.fetch_bodies = fetch_bodies
        self._body_fetcher = body_fetcher if body_fetcher is not None else (
            SecBodyFetcher(user_agent=user_agent) if fetch_bodies else None
        )

    # ------------------------------------------------------------------

    def fetch(self) -> Iterable[dict[str, Any]]:
        """Yield raw entries from each form-type feed.

        We add a tiny gap between feeds to stay polite under SEC's 10 req/sec
        guidance.
        """
        for form in self.forms:
            url = FEED_URL.format(form=form.replace(" ", "+"))
            try:
                resp = self._session.get(url, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as exc:
                log.warning("[%s] feed %s failed: %s", self.source, form, exc)
                continue

            parsed = feedparser.parse(resp.content)
            if parsed.bozo and not parsed.entries:
                log.warning(
                    "[%s] feed %s malformed: %s",
                    self.source, form, getattr(parsed, "bozo_exception", "unknown"),
                )
                continue

            for entry in parsed.entries:
                yield {"form": form, "entry": entry}

            time.sleep(0.2)  # gentle pacing — well under SEC's 10 rps

    # ------------------------------------------------------------------

    def parse(self, raw_entry: dict[str, Any]) -> Optional[ParsedSignal]:
        form = raw_entry.get("form", "")
        entry = raw_entry.get("entry") or {}

        external_id = (
            getattr(entry, "id", None)
            or getattr(entry, "link", None)
            or self._fallback_id(entry, form)
        )
        if not external_id:
            return None

        title = getattr(entry, "title", None)
        link = getattr(entry, "link", None)
        summary = getattr(entry, "summary", None)
        updated = self._iso_or_none(getattr(entry, "updated", None))

        # Resolve the actual filing body. Live ingestion only sees the
        # RSS summary (HTML metadata stub), which is useless to the LLM
        # classifier. The body fetcher resolves the primary document via
        # EDGAR's index.json or full-submission .txt and returns clean
        # text. Best-effort: a failed fetch leaves body=summary so the
        # row still records.
        body: Optional[str] = summary
        if self.fetch_bodies and self._body_fetcher is not None and link:
            try:
                fetched = self._body_fetcher.fetch_body(link, form_type=form)
            except Exception as exc:  # noqa: BLE001 — defensive: never block ingest
                log.debug("[%s] body fetch raised for %s: %s",
                          self.source, link, exc)
                fetched = None
            if fetched:
                body = fetched

        cik = self._extract_cik(title or "")
        ticker = CIK_LOOKUP.get_ticker(cik) if cik else None
        company_name = CIK_LOOKUP.get_name(cik) if cik else None

        # If we can't match a ticker, we still record the filing (with no
        # ticker mention) — it can be re-processed once the CIK lookup
        # is refreshed.
        tickers: list[TickerMention] = []
        if ticker:
            tickers.append(
                TickerMention(
                    ticker=ticker,
                    market="US",
                    asset_class=None,  # filled in by scorer/classifier later
                    confidence=1.0,
                )
            )

        return ParsedSignal(
            external_id=external_id,
            title=title,
            body=body,
            url=link,
            author=company_name,
            author_metadata={"cik": cik, "form": form} if cik else {"form": form},
            published_at=updated,
            raw_payload={
                "form": form,
                "form_event": FORM_TYPES.get(form, "other"),
                "cik": cik,
                "company_name": company_name,
                "title": title,
                "summary": summary,
                "link": link,
            },
            tickers=tickers,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cik(title: str) -> Optional[int]:
        match = TITLE_CIK_RE.search(title)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _iso_or_none(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        # feedparser already gives RFC3339-ish strings; normalize to a Z suffix
        # if possible.
        try:
            # try several common formats
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%SZ",
                "%a, %d %b %Y %H:%M:%S %z",
            ):
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    continue
        except Exception:  # noqa: BLE001
            pass
        return value

    @staticmethod
    def _fallback_id(entry: Any, form: str) -> Optional[str]:
        link = getattr(entry, "link", None)
        updated = getattr(entry, "updated", None)
        if link and updated:
            return f"{form}|{link}|{updated}"
        return None
