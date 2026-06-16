"""Lock the ClinicalTrials.gov v2 biotech-catalyst ingestor (blueprint #5).

No live network: a FIXTURE v2 payload (the real response shape, trimmed) is fed
through fetch() so the test asserts the parse + the upsert into the ``catalysts``
table. Acceptance: fixture parses into >=1 catalysts row. Also locks the
best-effort sponsor->ticker mapping (mapped sponsor gets a ticker; an unmapped
academic sponsor lands with ticker=None but is still recorded).
"""
from contextlib import contextmanager

import pytest

from market_radar.ingestors.clinicaltrials import (
    ClinicalTrialsIngestor,
    map_sponsor_to_ticker,
)
from market_radar.storage.db import get_connection as real_gc
from market_radar.storage.db import init_db


# A trimmed-but-realistic /api/v2/studies?aggFilters=results:with response.
# Study 1: mapped sponsor (Regeneron -> REGN), has primaryCompletionDate.
# Study 2: unmapped academic sponsor -> ticker stays None, dated by lastUpdate.
# Study 3: no nctId -> must be skipped.
FIXTURE = {
    "totalCount": 2,
    "studies": [
        {
            "hasResults": True,
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT04695977",
                    "briefTitle": "A Study of REGN-COV2 in Adults",
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "lastUpdatePostDateStruct": {"date": "2026-06-14", "type": "ACTUAL"},
                    "primaryCompletionDateStruct": {"date": "2026-05-30", "type": "ACTUAL"},
                },
                "designModule": {"phases": ["PHASE2", "PHASE3"]},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Regeneron Pharmaceuticals", "class": "INDUSTRY"}
                },
            },
        },
        {
            "hasResults": True,
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT02466971",
                    "briefTitle": "Phase 3 Oncology Trial",
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "lastUpdatePostDateStruct": {"date": "2026-06-10", "type": "ACTUAL"},
                    # no primaryCompletionDate -> falls back to lastUpdate
                },
                "designModule": {"phases": ["PHASE3"]},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "National Cancer Institute (NCI)", "class": "NIH"}
                },
            },
        },
        {
            # Malformed: no nctId -> parse() returns None, must not insert.
            "hasResults": True,
            "protocolSection": {
                "identificationModule": {"briefTitle": "Missing id"},
                "statusModule": {
                    "lastUpdatePostDateStruct": {"date": "2026-06-09"}
                },
                "designModule": {"phases": ["PHASE2"]},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Whatever"}},
            },
        },
    ],
    "nextPageToken": None,
}


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    init_db(db)

    @contextmanager
    def _gc(path=None):
        with real_gc(db) as c:
            yield c

    monkeypatch.setattr(
        "market_radar.ingestors.clinicaltrials.get_connection", _gc
    )
    return db


def _ingestor_with_fixture(monkeypatch):
    ing = ClinicalTrialsIngestor()
    # Stub fetch() so NO live network is touched in the test.
    monkeypatch.setattr(ing, "fetch", lambda: list(FIXTURE["studies"]))
    return ing


def test_sponsor_mapping():
    assert map_sponsor_to_ticker("Regeneron Pharmaceuticals") == "REGN"
    assert map_sponsor_to_ticker("MERCK SHARP & DOHME LLC") == "MRK"
    assert map_sponsor_to_ticker("National Cancer Institute (NCI)") is None
    assert map_sponsor_to_ticker(None) is None
    assert map_sponsor_to_ticker("") is None


def test_parse_skips_missing_nct():
    bad = FIXTURE["studies"][2]
    assert ClinicalTrialsIngestor.parse(bad) is None


def test_parse_prefers_primary_completion_date():
    parsed = ClinicalTrialsIngestor.parse(FIXTURE["studies"][0])
    assert parsed is not None
    assert parsed["ticker"] == "REGN"
    assert parsed["decision_date"] == "2026-05-30"  # primaryCompletion, not lastUpdate
    assert parsed["nct_id"] == "NCT04695977"
    assert "Phase 2/Phase 3" in parsed["description"]


def test_poll_inserts_catalysts(temp_db, monkeypatch):
    ing = _ingestor_with_fixture(monkeypatch)
    wrote = ing.poll()

    # Acceptance: fixture parses into >=1 catalysts row. Two valid studies, one
    # skipped (no nctId).
    assert wrote == 2

    with real_gc(temp_db) as c:
        rows = c.execute(
            "SELECT ticker, decision_date, catalyst_type, description, source "
            "FROM catalysts WHERE source=? ORDER BY decision_date",
            ("clinicaltrials_v2",),
        ).fetchall()

    assert len(rows) == 2

    # Mapped industry sponsor -> real ticker present.
    regn = next(r for r in rows if r["ticker"] == "REGN")
    assert regn["catalyst_type"] == "clinical_trial"
    assert regn["source"] == "clinicaltrials_v2"
    assert regn["decision_date"] == "2026-05-30"

    # Unmapped academic sponsor -> stored under the "CT:<nct>" sentinel (the
    # catalysts.ticker column is NOT NULL), but still recorded for coverage.
    nci = next(r for r in rows if str(r["ticker"]).startswith("CT:"))
    assert nci["ticker"] == "CT:NCT02466971"
    assert nci["decision_date"] == "2026-06-10"  # fell back to lastUpdate
    assert "NCT02466971" in nci["description"]
    # parse() still reports the honest mapping result (None) before write-time.
    assert ClinicalTrialsIngestor.parse(FIXTURE["studies"][1])["ticker"] is None


def test_poll_upsert_is_idempotent(temp_db, monkeypatch):
    ing = _ingestor_with_fixture(monkeypatch)
    ing.poll()
    ing.poll()  # second run must upsert, not duplicate
    with real_gc(temp_db) as c:
        n = c.execute(
            "SELECT COUNT(*) FROM catalysts WHERE source='clinicaltrials_v2'"
        ).fetchone()[0]
    assert n == 2


def test_poll_never_raises_on_fetch_failure(temp_db, monkeypatch):
    ing = ClinicalTrialsIngestor()

    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(ing, "fetch", _boom)
    assert ing.poll() == 0  # swallowed, returns 0
