"""Lock the SEC form-tag fix (ingestion blueprint #1): EDGAR's &type= filter
prefix-matches, so the real form must come from the entry, not the query."""
from market_radar.ingestors.sec_edgar import SecEdgarIngestor


class _Tag:
    def __init__(self, term):
        self.term = term


class _Entry:
    def __init__(self, title="", tags=None):
        self.title = title
        if tags is not None:
            self.tags = tags


def test_real_form_from_title_prefix_beats_query_form():
    # The bug: type=4 feed also returns 424B2 prospectuses; query form is '4'.
    e = _Entry(title="424B2 - BofA Finance LLC (0001682472) (Filer)")
    assert SecEdgarIngestor._real_form(e, "4") == "424B2"


def test_real_form_genuine_form4():
    e = _Entry(title="4 - SOME CORP (0000123456) (Reporting)")
    assert SecEdgarIngestor._real_form(e, "4") == "4"


def test_real_form_prefers_category_term():
    e = _Entry(title="ignored", tags=[_Tag("8-K")])
    assert SecEdgarIngestor._real_form(e, "SC 13D") == "8-K"


def test_real_form_falls_back_to_query_when_no_signal():
    e = _Entry(title="no dash here")
    assert SecEdgarIngestor._real_form(e, "DEF 14A") == "DEF 14A"
