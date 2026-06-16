"""Lock the 8-K Item-code extractor (ingestion deep-dive #3)."""
from market_radar.ingestors.sec_item_codes import (
    dominant_subtype,
    extract_item_codes,
    subtype_for_text,
)


def test_extracts_multiple_codes_dedup_sorted():
    body = ("UNITED STATES ... Item 2.02 Results of Operations ... "
            "Item 9.01 Financial Statements and Exhibits ... Item 2.02 again")
    assert extract_item_codes(body) == ["2.02", "9.01"]


def test_case_insensitive_and_trailing_punctuation():
    assert extract_item_codes("ITEM 5.02. Departure of officers") == ["5.02"]
    assert extract_item_codes("item 1.01, entry into agreement") == ["1.01"]


def test_ignores_non_taxonomy_and_plain_item():
    # 'Item 1' (no decimal) and an out-of-taxonomy 9.99 must not match.
    assert extract_item_codes("Item 1 and Item 9.99 and see item 7 below") == []


def test_dominant_prefers_event_bearing_over_boilerplate():
    # 9.01 (exhibits) is boilerplate; 2.02 (earnings) wins.
    assert dominant_subtype(["2.02", "9.01"]) == "results_of_operations"
    # restatement (4.02) outranks earnings.
    assert dominant_subtype(["2.02", "4.02", "9.01"]) == "non_reliance_restatement"


def test_empty_and_none():
    assert extract_item_codes(None) == []
    assert extract_item_codes("") == []
    assert dominant_subtype([]) is None
    assert subtype_for_text("no items here") is None


def test_subtype_for_text_end_to_end():
    assert subtype_for_text("blah Item 5.02 Departure ... Item 9.01") == "officer_or_director_change"
