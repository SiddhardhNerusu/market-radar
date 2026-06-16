"""Lock the deterministic SEC classification (ingestion blueprint #3): 8-K Item
codes + SEC forms route to real event_types with no LLM, so the SEC bulk stops
falling into 'other'."""
from market_radar.ingestors.sec_item_codes import (
    BEARISH_8K_EVENTS,
    event_type_for_8k,
    sec_event_type,
)


def test_8k_earnings_item_maps_to_earnings():
    assert event_type_for_8k("... Item 2.02 Results of Operations ... Item 9.01") \
        == "earnings_announcement"


def test_8k_officer_change_maps_to_leadership_change():
    assert event_type_for_8k("Item 5.02 Departure of Directors or Officers") \
        == "leadership_change"


def test_8k_dilution_and_delisting_are_bearish():
    dil = event_type_for_8k("Item 3.02 Unregistered Sales of Equity Securities")
    assert dil == "dilution" and dil in BEARISH_8K_EVENTS
    delist = event_type_for_8k("Item 3.01 Notice of Delisting")
    assert delist == "delisting" and delist in BEARISH_8K_EVENTS


def test_8k_boilerplate_only_stays_other():
    # 8.01 + 9.01 carry no event meaning -> None -> caller keeps 'other'.
    assert event_type_for_8k("Item 8.01 Other Events ... Item 9.01 Exhibits") is None


def test_sec_event_type_by_form():
    assert sec_event_type("SC 13D", "") == "activist_position"
    assert sec_event_type("SC 13G", "") == "passive_5pct_stake"
    assert sec_event_type("424B2", "") == "routine_prospectus"
    assert sec_event_type("8-K", "Item 2.02 Results") == "earnings_announcement"
    # Form-4 is handled by its own deterministic XML path, not here -> None.
    assert sec_event_type("4", "anything") is None
