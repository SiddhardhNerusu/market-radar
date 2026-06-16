"""8-K Item-code extraction (ingestion deep-dive opportunity #3).

An 8-K's Item codes ARE its event taxonomy — free, structured, low-false-positive
— yet the pipeline left ``event_subtype`` 100% NULL and lumped every 8-K into one
mushy ``other``/``material_event`` bucket. This extracts the codes from the body
the fetcher already retrieved so each event class (earnings 2.02, officer change
5.02, restatement 4.02, material agreement 1.01, delisting 3.01, …) can be scored
and measured on its own.

Pure + dependency-free so it is unit-testable and usable both at ingest time and
in a one-off backfill over existing bodies.
"""
from __future__ import annotations

import re
from typing import Optional

# Current SEC 8-K item taxonomy → human label.
ITEM_LABELS: dict[str, str] = {
    "1.01": "material_definitive_agreement",
    "1.02": "termination_of_material_agreement",
    "1.03": "bankruptcy_or_receivership",
    "1.04": "mine_safety",
    "2.01": "completion_of_acquisition_or_disposition",
    "2.02": "results_of_operations",          # earnings release
    "2.03": "creation_of_direct_financial_obligation",
    "2.04": "triggering_event_debt_acceleration",
    "2.05": "costs_associated_with_exit",
    "2.06": "material_impairment",
    "3.01": "notice_of_delisting",
    "3.02": "unregistered_equity_sale",        # dilution
    "3.03": "modification_of_securityholder_rights",
    "4.01": "change_in_accountant",
    "4.02": "non_reliance_restatement",        # restatement — strongly negative
    "5.01": "change_in_control",
    "5.02": "officer_or_director_change",
    "5.03": "amendment_to_bylaws",
    "5.07": "shareholder_vote_results",
    "7.01": "regulation_fd_disclosure",
    "8.01": "other_events",
    "9.01": "financial_statements_and_exhibits",
}

# Priority for picking ONE dominant subtype when an 8-K carries several items —
# most event-bearing / tradeable first, boilerplate (9.01/8.01/7.01) last.
_PRIORITY: tuple[str, ...] = (
    "1.03", "4.02", "2.06", "3.01", "2.04", "5.01", "2.01", "1.01", "2.02",
    "5.02", "1.02", "3.02", "4.01", "2.05", "2.03", "3.03", "5.03", "5.07",
    "1.04", "7.01", "8.01", "9.01",
)

_ITEM_RE = re.compile(r"\bitem\s+(\d\.\d{2})\b", re.IGNORECASE)


def extract_item_codes(text: Optional[str]) -> list[str]:
    """Return sorted unique 8-K Item codes (e.g. ['2.02', '9.01']) found in text.
    Only codes in the known taxonomy are returned (filters OCR/format noise)."""
    if not text:
        return []
    found = {m for m in _ITEM_RE.findall(text) if m in ITEM_LABELS}
    return sorted(found)


def dominant_subtype(codes: list[str]) -> Optional[str]:
    """Pick the single most event-bearing item as the event_subtype label."""
    if not codes:
        return None
    for code in _PRIORITY:
        if code in codes:
            return ITEM_LABELS[code]
    return ITEM_LABELS.get(sorted(codes)[0])


def subtype_for_text(text: Optional[str]) -> Optional[str]:
    """Convenience: body text → dominant event_subtype label (or None)."""
    return dominant_subtype(extract_item_codes(text))


# 8-K item subtype -> trading event_type (blueprint #3: deterministic routing).
# Maps the dominant Item code to a real event class so 8-Ks stop dumping into
# the 'other' bucket. Boilerplate items (8.01 other, 9.01 exhibits, 7.01 Reg FD)
# have NO entry -> they correctly remain 'other'.
SUBTYPE_TO_EVENT: dict[str, str] = {
    "results_of_operations":                    "earnings_announcement",
    "officer_or_director_change":               "leadership_change",
    "completion_of_acquisition_or_disposition": "m_a_announcement",
    "change_in_control":                        "m_a_announcement",
    "material_definitive_agreement":            "material_agreement",
    "termination_of_material_agreement":        "material_agreement",
    "notice_of_delisting":                      "delisting",
    "non_reliance_restatement":                 "restatement",
    "bankruptcy_or_receivership":               "bankruptcy",
    "unregistered_equity_sale":                 "dilution",
    "material_impairment":                      "impairment",
    "triggering_event_debt_acceleration":       "debt_distress",
    "change_in_accountant":                     "auditor_change",
    "shareholder_vote_results":                 "shareholder_vote",
}

# New 8-K-derived event types that are reliably BEARISH (avoid-long overlay,
# consistent with the deep-dive's finding that delisting/dilution/restatement
# net strongly negative). Exposed so the trade gate can block them.
BEARISH_8K_EVENTS: tuple[str, ...] = (
    "delisting", "restatement", "bankruptcy", "dilution", "impairment",
    "debt_distress", "auditor_change",
)


def event_type_for_8k(body: Optional[str]) -> Optional[str]:
    """Deterministic event_type for an 8-K body via its Item codes. Returns the
    mapped event_type, or None when the 8-K carries only boilerplate items
    (8.01/9.01/7.01) — in which case the caller leaves it 'other'."""
    return SUBTYPE_TO_EVENT.get(dominant_subtype(extract_item_codes(body)) or "")


# Non-8-K SEC form -> event_type (coarse, deterministic). 424B* are routine
# prospectuses (the form-tag bug's victims), not insider buys.
_FORM_EVENT: dict[str, str] = {
    "SC 13D": "activist_position", "SC 13G": "passive_5pct_stake",
    "S-1": "ipo_registration", "DEF 14A": "proxy_statement",
    "425": "m_a_announcement",
}


def sec_event_type(form: Optional[str], body: Optional[str]) -> Optional[str]:
    """Deterministic event_type for a SEC filing from its (corrected) form +
    body. None means 'let the LLM/heuristics decide' (genuinely ambiguous)."""
    f = (form or "").strip()
    if f in ("8-K", "8-K/A", "6-K", "6-K/A"):
        return event_type_for_8k(body)
    if f.startswith("424B"):
        return "routine_prospectus"
    return _FORM_EVENT.get(f)

