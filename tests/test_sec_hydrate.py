"""Lock the async SEC body-hydration latency fix (ingestion blueprint #2):
stub-first ingest does NO HTTP in the poll loop, and the out-of-band hydrate job
fills real bodies without ever holding the writer across the network.
"""
from contextlib import contextmanager

import pytest

from market_radar.ingestors import sec_hydrate
from market_radar.ingestors.sec_edgar import SecEdgarIngestor
from market_radar.storage.db import get_connection as real_gc
from market_radar.storage.db import init_db, insert_raw_signal


def test_stub_mode_constructs_no_body_fetcher():
    # Acceptance #1: with fetch_bodies=False there is NO body fetcher, so parse()
    # cannot reach any requests.* call inside the poll loop.
    ing = SecEdgarIngestor(fetch_bodies=False)
    assert ing._body_fetcher is None


class _FakeFetcher:
    def __init__(self, body="REAL 8-K FILING TEXT — Item 5.02 ..."):
        self.body = body
        self.calls = 0

    def fetch_body(self, url, *, form_type=None, use_cache=True):
        self.calls += 1
        return self.body


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    init_db(db)

    @contextmanager
    def _gc(path=None):
        with real_gc(db) as c:
            yield c
    monkeypatch.setattr("market_radar.ingestors.sec_hydrate.get_connection", _gc)
    return db


def _insert_stub(db, *, hydrated, link):
    with real_gc(db) as c:
        return insert_raw_signal(
            c, source="sec_edgar", source_tier=1, external_id=link, url=link,
            title="8-K - SomeCo (0001234567)", body="RSS metadata stub",
            author=None, author_metadata=None,
            raw_payload={"link": link, "form": "8-K", "body_hydrated": hydrated},
            published_at=None, tickers=[])


def test_hydrate_fills_unhydrated_stub(temp_db):
    rid = _insert_stub(temp_db, hydrated=0, link="https://sec.gov/a")
    fake = _FakeFetcher()
    n = sec_hydrate.hydrate_sec_bodies(fetcher=fake)
    assert n == 1 and fake.calls == 1
    import json
    with real_gc(temp_db) as c:
        row = c.execute("SELECT body, raw_payload FROM raw_signals WHERE id=?",
                        (rid,)).fetchone()
    assert "REAL 8-K FILING TEXT" in row["body"]
    assert json.loads(row["raw_payload"])["body_hydrated"] is True


def test_hydrate_skips_already_hydrated(temp_db):
    _insert_stub(temp_db, hydrated=1, link="https://sec.gov/b")
    fake = _FakeFetcher()
    n = sec_hydrate.hydrate_sec_bodies(fetcher=fake)
    assert n == 0 and fake.calls == 0, "must not refetch an already-hydrated body"
