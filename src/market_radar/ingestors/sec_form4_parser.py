"""Form 4 ownership.xml parser → structured transaction codes.

Form 4 XML structure (relevant fields):
  - reportingOwner.reportingOwnerRelationship.{isDirector,isOfficer,isTenPercentOwner}
  - reportingOwner.reportingOwnerRelationship.officerTitle  (CEO/CFO/etc.)
  - issuer.issuerTradingSymbol
  - nonDerivativeTable.nonDerivativeTransaction[*].transactionCoding.transactionCode
      (P=Purchase, S=Sale, A=Award, M=Exercise, F=Tax, G=Gift, ...)
  - nonDerivativeTransaction[*].transactionAmounts.transactionShares.value
  - nonDerivativeTransaction[*].transactionAmounts.transactionPricePerShare.value

We extract these per transaction and feed them to the
``insider_transactions`` table for the
``attach_insider_codes`` enrichment in ``ml/external_features.py``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Form4Transaction:
    transaction_code: str        # P/S/A/M/F/G/D/...
    shares: float = 0.0
    price_per_share: float = 0.0
    is_acquired: bool = False    # True for P/A; False for S/F/D


@dataclass
class Form4Parsed:
    issuer_ticker: Optional[str] = None
    issuer_cik: Optional[str] = None
    insider_name: Optional[str] = None
    is_officer: bool = False
    is_director: bool = False
    is_10pct: bool = False
    is_10b5_1: bool = False       # trade under a pre-planned Rule 10b5-1 plan (routine, uninformative)
    officer_title: Optional[str] = None
    period_of_report: Optional[str] = None
    transactions: list[Form4Transaction] = field(default_factory=list)

    @property
    def insider_role_score(self) -> int:
        title = (self.officer_title or "").upper()
        if any(t in title for t in ("CEO", "CFO", "CHIEF")):
            return 3
        if self.is_director:
            return 2
        if self.is_officer or self.is_10pct:
            return 1
        return 0


def _extract(xml_text: str, tag: str) -> Optional[str]:
    """Extract a value for a given tag.

    Handles two shapes:
      A. Raw XML — ``<tag>val</tag>`` or ``<tag><value>val</value></tag>``
      B. Stripped text from sec_body_fetcher's _strip_xml_keep_structure,
         which converts tags to whitespace-delimited tokens:
         ``tagName value tagName value``. We extract by finding the
         tag name token and returning the next non-tag token(s).
    """
    # Shape A — raw XML
    m = re.search(fr"<{tag}>\s*(?:<value>\s*)?([^<]+?)\s*(?:</value>)?\s*</{tag}>",
                  xml_text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Shape B — stripped tokens. Find "tag\s+<value-tokens-until-next-tag>"
    # where the value is the run of non-whitespace separated by whitespace,
    # capped at ~80 chars (Form 4 fields are short).
    tok = re.escape(tag)
    m = re.search(fr"\b{tok}\b\s+([^\n<>]{{1,80}}?)(?=\s+[A-Za-z][A-Za-z0-9_:\-\.]*\s|$)",
                  xml_text)
    if m:
        cand = m.group(1).strip()
        # Filter out obvious tag-name continuations
        if cand and not cand.startswith("<"):
            return cand
    return None


def _extract_all(xml_text: str, tag: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(
        fr"<{tag}>\s*(?:<value>\s*)?([^<]+?)\s*(?:</value>)?\s*</{tag}>",
        xml_text, re.IGNORECASE | re.DOTALL)]


def parse_form4_xml(xml_text: str) -> Optional[Form4Parsed]:
    """Parse a Form 4 body (raw XML OR stripped-tag-token text) into
    structured fields. Returns None on completely unparseable input.
    """
    if not xml_text or "ownershipdocument" not in xml_text.lower():
        return None
    out = Form4Parsed()
    out.issuer_ticker = _extract(xml_text, "issuerTradingSymbol")
    out.issuer_cik = _extract(xml_text, "issuerCik")
    out.insider_name = _extract(xml_text, "rptOwnerName")
    out.officer_title = _extract(xml_text, "officerTitle")
    out.period_of_report = _extract(xml_text, "periodOfReport")
    out.is_officer = (_extract(xml_text, "isOfficer") or "0").strip() in {"1", "true", "True"}
    out.is_director = (_extract(xml_text, "isDirector") or "0").strip() in {"1", "true", "True"}
    out.is_10pct = (_extract(xml_text, "isTenPercentOwner") or "0").strip() in {"1", "true", "True"}
    # 10b5-1 plan: a pre-arranged trading plan makes the trade ROUTINE (scheduled,
    # uninformative) — the opposite of an opportunistic insider buy (the durable
    # edge). The literal "10b5-1" appears in the structured flag / footnotes of
    # plan trades; "rule10b5One false" (no dash-1) is a non-plan trade.
    out.is_10b5_1 = bool(re.search(r"10b5[\s\-–_]*1\b", xml_text, re.IGNORECASE))

    # Walk transactions. The body fetcher's stripped text doesn't preserve
    # <nonDerivativeTransaction> blocks — fields appear in document order
    # with the tag names as tokens. We extract by finding each
    # transactionCode occurrence and reading the shares/price values that
    # appear within the next ~200 chars.
    block_re = re.compile(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>",
                           re.IGNORECASE | re.DOTALL)
    blocks_found = list(block_re.finditer(xml_text))

    if blocks_found:
        # Raw-XML shape
        for m in blocks_found:
            body = m.group(1)
            code = _extract(body, "transactionCode") or ""
            try:
                shares = float(_extract(body, "transactionShares") or 0)
            except ValueError:
                shares = 0.0
            try:
                price = float(_extract(body, "transactionPricePerShare") or 0)
            except ValueError:
                price = 0.0
            acquired_disposed = _extract(body, "transactionAcquiredDisposedCode") or ""
            out.transactions.append(Form4Transaction(
                transaction_code=code.upper(),
                shares=shares,
                price_per_share=price,
                is_acquired=acquired_disposed.upper() == "A",
            ))
    else:
        # Stripped-token shape.  The XML stripper preserves document order
        # but drops <nonDerivativeTransaction> wrappers; the body becomes
        # a stream of "tagName value..." tokens.  Two real-data quirks:
        #   1. transactionShares (and the other numeric/letter fields)
        #      are wrapped in a <value> element in the XML, which the
        #      stripper renders as a literal ``value`` token between the
        #      field name and the number.  Example:
        #          transactionShares \n  value 899.0000
        #   2. transactionCode is NOT wrapped in <value>; its single-
        #      letter code appears directly after the tag name.
        #
        # Correct approach: extract every occurrence of each field
        # separately, then zip them in document order.  The N-th
        # transactionCode corresponds to the N-th transactionShares,
        # N-th transactionPricePerShare, and N-th
        # transactionAcquiredDisposedCode.  ``transactionShares`` etc.
        # appear BEFORE their ``transactionCode`` in the actual XML, so
        # a window-after-code search misses them 100% of the time.
        codes = [m.group(1) for m in re.finditer(
            r"\btransactionCode\b\s+([A-Z])\b", xml_text,
        )]
        # Match both "tag value 123" and "tag 123" — the stripper may
        # or may not emit the literal ``value`` separator depending on
        # the source XML's <value> wrapping.
        shares_tokens = [m.group(1) for m in re.finditer(
            r"\btransactionShares\b\s+(?:value\s+)?([0-9.,]+)", xml_text,
        )]
        prices_tokens = [m.group(1) for m in re.finditer(
            r"\btransactionPricePerShare\b\s+(?:value\s+)?([0-9.,]+)", xml_text,
        )]
        ad_tokens = [m.group(1) for m in re.finditer(
            r"\btransactionAcquiredDisposedCode\b\s+(?:value\s+)?([AD])\b",
            xml_text,
        )]

        n = len(codes)
        for i in range(n):
            try:
                shares = (float(shares_tokens[i].replace(",", ""))
                          if i < len(shares_tokens) else 0.0)
            except ValueError:
                shares = 0.0
            try:
                price = (float(prices_tokens[i].replace(",", ""))
                         if i < len(prices_tokens) else 0.0)
            except ValueError:
                price = 0.0
            is_acquired = (ad_tokens[i] == "A") if i < len(ad_tokens) else False
            out.transactions.append(Form4Transaction(
                transaction_code=codes[i].upper(),
                shares=shares,
                price_per_share=price,
                is_acquired=is_acquired,
            ))

    if not out.transactions and not out.issuer_ticker:
        return None
    return out
