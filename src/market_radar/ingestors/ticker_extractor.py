"""Extract US-equity tickers from free-text news content.

False positives matter here — a noisy ticker mention creates a noisy signal
that contaminates the whole edge-measurement pipeline. We prefer high-
confidence patterns over broad matching:

  1. **Cashtag** ``$AAPL`` (highest confidence, 0.95)
  2. **Exchange prefix** ``(NASDAQ: AAPL)`` / ``NYSE:AAPL`` (0.9)
  3. **Known-ticker token** — an all-uppercase word 1–5 chars that exists in
     the SEC ticker universe AND is not in our common-words blocklist (0.7
     in title, 0.55 in body).

Company-name → ticker matching is intentionally NOT implemented in v1
because the false-positive risk ("Apple Records" → AAPL) is too high
without a curated name map.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .cik_lookup import CIK_LOOKUP

log = logging.getLogger(__name__)


# Words and acronyms that happen to look like tickers but almost never refer
# to a company in financial news. We never extract these as tickers. This
# list grows as we observe false positives.
COMMON_WORDS_BLOCKLIST = {
    # Pronouns / prepositions / conjunctions / articles
    "A", "I", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS",
    "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    # Common short words
    "AM", "PM", "FOR", "THE", "AND", "BUT", "NOT", "ALL", "ANY", "ARE", "BIG",
    "BUY", "CAN", "CEO", "CFO", "COO", "CTO", "DAY", "DUE", "END", "FAR", "FED",
    "FEW", "FIX", "FOR", "GET", "GOT", "HAD", "HAS", "HER", "HIM", "HIS", "HOT",
    "HOW", "ITS", "LET", "LOW", "MAN", "MAY", "NEW", "NOW", "OFF", "OLD", "ONE",
    "OUT", "OWN", "PER", "PUT", "RUN", "SAW", "SAY", "SEE", "SET", "SHE", "TAX",
    "TEN", "TOP", "TWO", "USE", "WAS", "WAY", "WHO", "WHY", "YOU",
    # Finance jargon often misread
    "ESG", "ETF", "IPO", "ATH", "ATL", "EOD", "EOY", "EPS", "FOMO",
    "FTSE", "GDP", "HFT", "MEV", "QE", "ROI", "RSI",
    "SEC", "SPAC", "SPX", "TLT", "USD", "VIX", "WSB", "YOY", "QOQ", "MOM",
    "CPI", "PPI", "PMI", "GDP", "ISM", "NFP", "FOMC", "ECB", "BOJ", "BOE",
    "PCE", "PIK", "PIPE", "RIA", "REIT", "ARR", "MRR", "DCF", "TAM", "SAM",
    "SOM", "IRR", "CAGR", "EBITDA", "EBIT", "ESOP", "RSU", "PSU", "AUM",
    # Tech/media acronyms often in financial news
    "API", "SDK", "OEM", "ODM", "SaaS", "PaaS", "IaaS", "AWS", "GCP", "ML",
    "GenAI", "LLM", "GPU", "TPU", "FAQ", "URL", "VPN", "DNS", "CDN",
    # Generic acronyms that overlap with tickers
    "GLP", "TV", "CD",  # GLP-1 drugs, television, compact disc — usually not GLP/TV/CD the tickers
    # Country/region/regulator acronyms
    "EEU", "EMEA", "APAC", "LATAM", "OECD", "WTO", "IMF", "WHO", "NATO",
    # Months / days
    "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    # Common confusable acronyms
    "AI", "API", "ATM", "CDC", "CIA", "DOJ", "EU", "FBI", "FDA", "GOP", "IPO",
    "IRS", "LLC", "LLP", "LP", "NASA", "NYC", "NYT", "OPEC", "PR", "Q1", "Q2",
    "Q3", "Q4", "TSA", "UK", "UN", "US", "USA", "WHO", "WSJ",
}


CASHTAG_RE = re.compile(r"(?<![\w/])\$([A-Z]{1,5})(?:\.[A-Z]{1,2})?(?![\w])")
EXCHANGE_PREFIX_RE = re.compile(
    r"\b(?:NASDAQ|NYSE(?:MKT)?|AMEX|OTCBB|OTCMKTS|ARCA|BATS|TSX|LSE|ETR|FRA)"
    r"\s*[:\-]?\s*([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b",
    re.IGNORECASE,
)

# Known-caps matcher used on free text. The negative lookbehind/lookahead
# requirements are strict to avoid false positives:
#   - must be a true word boundary (not part of CamelCase or all-caps run)
#   - must NOT be followed by a "." (kills U.S., E.U., U.N., A.I., etc.)
#   - must NOT be preceded by "." (kills .NET, .com style fragments)
# We require 3+ characters at this path; 1- and 2-char tickers are only
# matched via cashtags and exchange prefixes where they're unambiguous.
ALL_CAPS_TOKEN_RE = re.compile(
    r"(?<![A-Z\.])\b([A-Z]{3,5})\b(?!\.)"
)


@dataclass
class ExtractedTicker:
    ticker: str
    confidence: float
    source: str  # 'cashtag' / 'exchange_prefix' / 'known_caps_title' / 'known_caps_body'


class TickerExtractor:
    """Stateful ticker extractor.

    Built once and reused across the daemon so we don't repeatedly load
    the SEC ticker universe.
    """

    def __init__(self) -> None:
        # Lazy-loaded: avoid downloading the SEC file just to construct the
        # object.
        self._known: Optional[set[str]] = None

    @property
    def known_tickers(self) -> set[str]:
        if self._known is None:
            # Trigger CIK_LOOKUP load
            CIK_LOOKUP._load()  # noqa: SLF001 — internal, but stable
            self._known = {t.upper() for t in CIK_LOOKUP._map.values()}
            log.info("TickerExtractor loaded %d known tickers", len(self._known))
        return self._known

    # ------------------------------------------------------------------

    def extract(
        self,
        title: Optional[str],
        body: Optional[str],
    ) -> list[ExtractedTicker]:
        """Return a deduped list of tickers found in title+body, highest-
        confidence-first."""
        title_text = title or ""
        body_text = body or ""
        combined = f"{title_text}\n{body_text}"

        best: dict[str, ExtractedTicker] = {}

        def _add(t: str, conf: float, source: str) -> None:
            t = t.upper()
            if t in COMMON_WORDS_BLOCKLIST:
                return
            if t not in self.known_tickers:
                # Not a real ticker, ignore (this kills FOMO/QQQ/etc. matches
                # that aren't actual companies)
                return
            existing = best.get(t)
            if existing is None or existing.confidence < conf:
                best[t] = ExtractedTicker(ticker=t, confidence=conf, source=source)

        # 1. Cashtags — highest confidence
        for m in CASHTAG_RE.finditer(combined):
            _add(m.group(1), 0.95, "cashtag")

        # 2. Exchange-prefixed mentions — also very high confidence
        for m in EXCHANGE_PREFIX_RE.finditer(combined):
            _add(m.group(1).upper(), 0.9, "exchange_prefix")

        # 3. All-caps tokens that match a known ticker
        #    (lower confidence; we require it to also pass the blocklist
        #    and known-ticker check, both done inside _add)
        for m in ALL_CAPS_TOKEN_RE.finditer(title_text):
            _add(m.group(1), 0.7, "known_caps_title")
        for m in ALL_CAPS_TOKEN_RE.finditer(body_text):
            _add(m.group(1), 0.55, "known_caps_body")

        return sorted(best.values(), key=lambda x: x.confidence, reverse=True)


# Module singleton
TICKER_EXTRACTOR = TickerExtractor()
