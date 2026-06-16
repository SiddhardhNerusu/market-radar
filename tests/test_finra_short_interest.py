"""FINRA consolidated short-interest ingestor — fixture-driven (no live network).

Locks the coverage blueprint #5 ingest: a representative FINRA API response parses
into >=1 ``short_interest`` row and UPSERTs idempotently into a temp DB. Also pins
the two field-quirks we handle: the 999.99 days-to-cover sentinel is nulled, and an
UPSERT does NOT clobber a float/short-%-of-float a different ingestor wrote.
"""
from contextlib import contextmanager

import pytest

from market_radar.ingestors import finra_short_interest as fsi
from market_radar.ingestors.finra_short_interest import FinraShortInterestIngestor
from market_radar.storage.db import get_connection as real_gc
from market_radar.storage.db import init_db


# A representative slice of the live consolidated dataset (real field names /
# shapes captured from api.finra.org). Mixes a clean NYSE row, a Nasdaq(NNM) row,
# a thin name with the 999.99 days-to-cover sentinel, an OTC row that uses the
# alternate ticker field, and two junk rows that must be skipped.
FIXTURE_ROWS = [
    {
        "symbolCode": "AA",
        "issueName": "Alcoa Corporation",
        "marketClassCode": "NYSE",
        "settlementDate": "2026-05-29",
        "currentShortPositionQuantity": 10624431,
        "previousShortPositionQuantity": 9476203,
        "averageDailyVolumeQuantity": 9564290,
        "daysToCoverQuantity": 1.11,
        "changePercent": 12.12,
    },
    {
        "symbolCode": "GME",
        "issueName": "GameStop Corp",
        "marketClassCode": "NNM",
        "settlementDate": "2026-05-29",
        "currentShortPositionQuantity": 55000000,
        "averageDailyVolumeQuantity": 5000000,
        "daysToCoverQuantity": 11.0,
        "changePercent": 3.5,
    },
    {
        # Thin name: FINRA reports the 999.99 "effectively infinite" sentinel.
        "symbolCode": "AACAF",
        "issueName": "AAC Technologies Holdings Inc",
        "marketClassCode": "OTC",
        "settlementDate": "2026-05-29",
        "currentShortPositionQuantity": 4612254,
        "averageDailyVolumeQuantity": 33,
        "daysToCoverQuantity": 999.99,
        "changePercent": 5.5,
    },
    {
        # OTC-only dataset uses the long SIP field name instead of symbolCode.
        "securitiesInformationProcessorSymbolIdentifier": "AABB",
        "issueName": "Asia Broadband Inc",
        "marketClassCode": "OTC",
        "settlementDate": "2026-05-29",
        "currentShortPositionQuantity": 58815,
        "averageDailyVolumeQuantity": 23138923,
        "daysToCoverQuantity": 1,
    },
    {"issueName": "No ticker — must skip", "settlementDate": "2026-05-29"},
    {"symbolCode": "NODATE", "currentShortPositionQuantity": 100},  # no date — skip
]


class _FakeResp:
    def __init__(self, rows, status=200, total=None):
        self._rows = rows
        self.status_code = status
        self.headers = {} if total is None else {"record-total": str(total)}

    def json(self):
        return self._rows


class _FakeSession:
    """Stands in for requests.Session.

    - A ``limit==1`` probe for the newest candidate date returns 1 row (so the
      ingestor latches that date); older candidate dates return empty.
    - A full page request for that date returns the whole fixture.
    """

    def __init__(self, rows, latest_date="2026-05-29"):
        self._rows = rows
        self._latest = latest_date
        self.headers = {}
        self.calls = []

    def post(self, url, json=None, timeout=None):  # noqa: A002 — mirror requests API
        self.calls.append(json)
        cf = (json or {}).get("compareFilters") or [{}]
        date_val = cf[0].get("fieldValue")
        limit = (json or {}).get("limit")
        if date_val != self._latest:
            return _FakeResp([], total=0)            # not the published date
        if limit == 1:
            return _FakeResp(self._rows[:1], total=len(self._rows))  # probe hit
        return _FakeResp(self._rows, total=len(self._rows))          # full page


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    init_db(db)

    @contextmanager
    def _gc(path=None):
        with real_gc(db) as c:
            yield c

    monkeypatch.setattr(
        "market_radar.ingestors.finra_short_interest.get_connection", _gc
    )
    return db


def _build_ingestor(rows):
    ing = FinraShortInterestIngestor()
    ing._session = _FakeSession(rows)  # inject fake — no live network
    return ing


def test_parse_row_maps_fields_and_nulls_sentinel():
    p = fsi.parse_row(FIXTURE_ROWS[0])
    assert p["ticker"] == "AA"
    assert p["report_date"] == "2026-05-29"
    assert p["short_interest"] == 10624431.0
    assert p["avg_daily_volume"] == 9564290.0
    assert p["days_to_cover"] == 1.11
    assert p["float_shares"] is None and p["short_pct_float"] is None

    # 999.99 sentinel -> None
    thin = fsi.parse_row(FIXTURE_ROWS[2])
    assert thin["ticker"] == "AACAF" and thin["days_to_cover"] is None

    # alternate ticker field name is accepted
    otc = fsi.parse_row(FIXTURE_ROWS[3])
    assert otc["ticker"] == "AABB"

    # junk rows skipped
    assert fsi.parse_row(FIXTURE_ROWS[4]) is None
    assert fsi.parse_row(FIXTURE_ROWS[5]) is None


def test_poll_parses_fixture_and_upserts(temp_db):
    ing = _build_ingestor(FIXTURE_ROWS)
    wrote = ing.poll()

    # 4 valid rows (AA, GME, AACAF, AABB); 2 junk skipped.
    assert wrote == 4

    with real_gc(temp_db) as c:
        rows = c.execute(
            "SELECT ticker, report_date, short_interest, avg_daily_volume, "
            "days_to_cover, float_shares, short_pct_float "
            "FROM short_interest ORDER BY ticker"
        ).fetchall()

    assert len(rows) >= 1
    by_ticker = {r["ticker"]: r for r in rows}
    assert set(by_ticker) == {"AA", "GME", "AACAF", "AABB"}

    aa = by_ticker["AA"]
    assert aa["report_date"] == "2026-05-29"
    assert aa["short_interest"] == 10624431.0
    assert aa["days_to_cover"] == 1.11

    # sentinel nulled in the persisted row too
    assert by_ticker["AACAF"]["days_to_cover"] is None


def test_poll_is_idempotent_and_upsert_preserves_float(temp_db):
    # Seed a float/short-%-of-float that a *different* ingestor would own.
    with real_gc(temp_db) as c:
        c.execute(
            "INSERT INTO short_interest (ticker, report_date, float_shares, "
            "short_pct_float, ingested_at) VALUES (?, ?, ?, ?, ?)",
            ("AA", "2026-05-29", 1.8e8, 5.9, "2026-05-30T00:00:00Z"),
        )

    ing = _build_ingestor(FIXTURE_ROWS)
    first = ing.poll()
    second = ing.poll()  # re-poll: same PK -> UPDATE, not a second insert
    assert first == 4 and second == 4

    with real_gc(temp_db) as c:
        n = c.execute("SELECT COUNT(*) FROM short_interest").fetchone()[0]
        aa = c.execute(
            "SELECT short_interest, float_shares, short_pct_float "
            "FROM short_interest WHERE ticker='AA'"
        ).fetchone()

    assert n == 4, "idempotent: re-poll must UPSERT, not duplicate"
    # FINRA fields written, but the externally-owned float survived the UPSERT.
    assert aa["short_interest"] == 10624431.0
    assert aa["float_shares"] == 1.8e8
    assert aa["short_pct_float"] == 5.9


def test_poll_returns_zero_on_empty_source(temp_db):
    # No date returns rows -> graceful 0, no raise, no rows written.
    ing = FinraShortInterestIngestor()
    ing._session = _FakeSession([], latest_date="1900-01-01")
    assert ing.poll() == 0
    with real_gc(temp_db) as c:
        assert c.execute("SELECT COUNT(*) FROM short_interest").fetchone()[0] == 0


def test_candidate_dates_are_recent_and_descending():
    import datetime as _dt
    cands = fsi._candidate_settlement_dates(_dt.date(2026, 6, 15))
    assert cands == sorted(cands, reverse=True)  # newest-first
    assert all(c <= "2026-06-15" for c in cands)  # never future
    # mid-month (15th) and an end-of-month date for June should be present
    assert "2026-06-15" in cands
    assert any(c.startswith("2026-05-2") or c.startswith("2026-05-3") for c in cands)
