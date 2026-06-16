"""Lock daemon health history + stalled-source detection (ingestion blueprint #8):
a source that fetches OK but has silently STOPPED producing is flagged; sources
that simply never produce (or lack enough history) are not."""
import pytest

from market_radar.storage.db import get_connection as real_gc
from market_radar.storage.db import (
    init_db,
    record_daemon_health_history,
    source_uptime_7d,
    stalled_sources,
)


@pytest.fixture
def temp_db(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    return db


def _hist(conn, src, success, inserted, n=1):
    for _ in range(n):
        record_daemon_health_history(conn, source=src, success=success, inserted=inserted)


def test_stalled_known_producer_is_flagged(temp_db):
    with real_gc(temp_db) as c:
        _hist(c, "stalled_src", True, 3, n=5)    # produced earlier...
        _hist(c, "stalled_src", True, 0, n=12)   # ...then went silent
        _hist(c, "quiet_src",   True, 0, n=12)   # never produced anything
        _hist(c, "healthy_src", True, 4, n=12)   # still producing
        stalled = stalled_sources(c, min_polls=10)
    assert "stalled_src" in stalled
    assert "quiet_src" not in stalled, "a source that never produces is not 'broken'"
    assert "healthy_src" not in stalled


def test_too_few_polls_not_flagged(temp_db):
    with real_gc(temp_db) as c:
        _hist(c, "new_src", True, 2, n=3)
        _hist(c, "new_src", True, 0, n=4)
        assert stalled_sources(c, min_polls=10) == []


def test_uptime_query_runs(temp_db):
    with real_gc(temp_db) as c:
        _hist(c, "s", True, 1, n=8)
        _hist(c, "s", False, 0, n=2)
        rows = source_uptime_7d(c)
    assert rows and rows[0]["source"] == "s" and rows[0]["uptime_pct"] == 80.0
