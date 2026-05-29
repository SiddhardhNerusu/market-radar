"""Detect routine / administrative SEC filings that shouldn't dominate the feed.

Examples of patterns we downgrade:
  - 424B-series prospectus supplements from commodity / index trust filers
    (Teucrium, SPDR, iShares, Vanguard, ProShares, Invesco PowerShares)
  - DEF 14A from fund-of-fund structures (John Hancock Funds, etc.)
  - S-1/A from already-known issuers (just an amendment)

The runtime classification is by company-name substring match against a
curated blocklist. Cheap, deterministic, and easy to expand as we find
more patterns in the live feed.
"""
from __future__ import annotations

import re
from typing import Optional


# Tokens that, when present in the *issuer* name, indicate the filer is a
# trust or fund vehicle whose filings are usually administrative.
ROUTINE_ISSUER_TOKENS = {
    # Commodity ETF trusts
    "teucrium",
    "commodity trust",
    "commodities trust",
    # Index ETF families
    "spdr",
    "ishares",
    "vanguard",
    "proshares",
    "powershares",
    "invesco",
    "schwab strategic",
    "wisdomtree",
    "first trust",
    "global x",
    # Mutual fund / closed-end fund umbrellas
    "john hancock funds",
    "fidelity advisor",
    "fidelity rutland",
    "putnam funds",
    "blackrock funds",
    "morgan stanley funds",
    "nuveen",
    "vaneck vectors",
    "calamos",
    "lazard funds",
    "tortoise",
    "delaware funds",
    "voya partners",
    "voya investors",
    "advisors series trust",
    "trust for advisor",
    # SPAC shell companies
    "acquisition corp",
    "acquisition corporation",
    "spac",
    "holdco",
}


# Form types that, when issued by a routine-filer issuer, should be
# auto-downgraded. (8-Ks from ETF trusts still matter — only the
# prospectus-style forms are downgraded.)
ROUTINE_FORM_TYPES = {
    "424B",   # any 424B-series
    "DEF 14A",
    "DEFA14A",
    "DEFR14A",
    "S-1/A",
    "S-3/A",
}


# Big-bank / broker-dealer issuers who file 424B-series prospectus
# supplements routinely as part of their MTN (medium-term note) shelf
# programs. These aren't news — they're regular debt issuance paperwork.
ROUTINE_BANK_ISSUERS = {
    "citigroup", "citi inc", "citi finance",
    "jpmorgan", "jp morgan",
    "goldman sachs",
    "morgan stanley",
    "bank of america",
    "wells fargo",
    "us bancorp", "u.s. bancorp", "u s bancorp",
    "pnc financial",
    "truist financial",
    "regions financial",
    "fifth third",
    "huntington bancshares",
    "keycorp",
    "m&t bank",
    "northern trust",
    "state street",
    "bny mellon", "bank of new york",
    "barclays",
    "deutsche bank",
    "ubs ag", "ubs group",
    "credit suisse",
    "hsbc",
    "rbc", "royal bank of canada",
    "td bank", "toronto-dominion",
    "bmo financial",
    "scotiabank",
}


def is_routine_filing(
    *,
    form: Optional[str],
    issuer_name: Optional[str],
) -> bool:
    """Return True if this looks like a routine administrative filing.

    Three patterns trigger routine status:
      1. 424B-series or DEF 14A from a known trust/fund vehicle
         (Teucrium, SPDR, iShares, fund families, etc.)
      2. 424B-series from a known big-bank / broker-dealer who files
         these continuously for their MTN / structured-products programs
      3. 424B-series with no recognizable issuer match — these are *less*
         routine but still administrative. Mild flag.
    """
    if not form or not issuer_name:
        return False
    issuer_lower = issuer_name.lower()
    form_norm = form.upper().strip()

    # All 424B variants are prospectus supplements — fundamentally admin.
    if form_norm.startswith("424B"):
        if any(tok in issuer_lower for tok in ROUTINE_ISSUER_TOKENS):
            return True
        if any(tok in issuer_lower for tok in ROUTINE_BANK_ISSUERS):
            return True
        # 424B from a non-bank operating company COULD be a real capital
        # raise — those occasionally matter. Don't auto-downgrade those.
        return False

    if form_norm in ROUTINE_FORM_TYPES:
        if any(tok in issuer_lower for tok in ROUTINE_ISSUER_TOKENS):
            return True
        return False

    # DEF 14A from any fund-vehicle issuer is routine
    if "14A" in form_norm or form_norm == "DEF 14A":
        if any(tok in issuer_lower for tok in ROUTINE_ISSUER_TOKENS):
            return True
    return False


def routine_event_type(form: Optional[str]) -> str:
    """Map a routine filing to the right event_type taxonomy bucket."""
    if not form:
        return "routine_prospectus"
    form_norm = form.upper().strip()
    if form_norm.startswith("424B"):
        return "routine_prospectus"
    if "14A" in form_norm:
        return "routine_proxy"
    return "routine_prospectus"
