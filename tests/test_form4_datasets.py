"""SEC Insider Transactions Data Set backfill: parsing, role mapping, idempotent writes,
and that the lane's candidate selection reads the result."""
import io
import zipfile
from datetime import date

from market_radar.backfill.form4_datasets import (
    Form4Row, parse_quarter, parse_sec_date, quarters_between, write_rows)
from market_radar.lane import candidates as cands
from market_radar.storage.db import get_connection, init_db

SUB = ("ACCESSION_NUMBER\tFILING_DATE\tPERIOD_OF_REPORT\tDOCUMENT_TYPE\tISSUERCIK\tISSUERNAME\tISSUERTRADINGSYMBOL\tAFF10B5ONE\n"
       "0001-25-000001\t30-SEP-2025\t26-SEP-2025\t4\t0000001959\tWK Kellogg Co\tKLG\tfalse\n"
       "0001-25-000002\t30-SEP-2025\t26-SEP-2025\t4\t0000002222\tPlan Co\tPLN\t1\n"
       "0001-25-000003\t30-SEP-2025\t26-SEP-2025\t3\t0000003333\tForm3 Co\tFTH\t0\n")
OWN = ("ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
       "0001-25-000001\t1\tBrown Bruce\tOfficer\tChief Customer Officer\n"
       "0001-25-000002\t2\tDoe Jane\tDirector\t\n")
TRX = ("ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
       "0001-25-000001\t26-SEP-2025\tP\t2000\t21.36\tA\n"
       "0001-25-000001\t26-SEP-2025\tS\t500\t22.00\tD\n"
       "0001-25-000002\t26-SEP-2025\tP\t10000\t15.00\tA\n"
       "0001-25-000003\t26-SEP-2025\tP\t100\t5.00\tA\n")


def _zip(tmp_path):
    zp = tmp_path / "2025q3_form345.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("SUBMISSION.tsv", SUB)
        z.writestr("REPORTINGOWNER.tsv", OWN)
        z.writestr("NONDERIV_TRANS.tsv", TRX)
    return zp


def test_parse_sec_date_and_quarters():
    assert parse_sec_date("30-SEP-2025") == "2025-09-30"
    assert parse_sec_date("2025-09-30") == "2025-09-30"
    assert parse_sec_date("") is None
    assert quarters_between(date(2025, 7, 1), date(2026, 1, 15)) == [(2025, 3), (2025, 4), (2026, 1)]


def test_parse_quarter_keeps_form4_code_p_with_roles(tmp_path):
    rows = parse_quarter(_zip(tmp_path))
    assert [r.ticker for r in rows] == ["KLG", "PLN"]          # Form 3 excluded; S excluded by default
    klg, pln = rows
    assert klg.filing_date == "2025-09-30" and klg.is_acquired == 1 and klg.is_10b5_1 == 0
    assert klg.role_score == 3 and klg.is_officer == 1 and klg.insider_name == "Brown Bruce"
    assert pln.is_10b5_1 == 1 and pln.role_score == 2 and pln.is_director == 1
    assert len(parse_quarter(_zip(tmp_path), codes=("P", "S"))) == 3


def test_write_rows_idempotent_and_feeds_lane_candidates(tmp_path):
    path = tmp_path / "x.db"
    init_db(path)
    rows = parse_quarter(_zip(tmp_path))
    with get_connection(path) as c:
        r1 = write_rows(c, rows, now="2026-01-01T00:00:00Z")
        r2 = write_rows(c, rows, now="2026-01-01T00:00:00Z")
        assert r1 == {"signals_inserted": 2, "transactions_inserted": 2}
        assert r2 == {"signals_inserted": 0, "transactions_inserted": 0}
        sig = tuple(c.execute("SELECT source, external_id, published_at FROM raw_signals ORDER BY id").fetchone())
        assert sig == ("sec_edgar_form4_ds", "0001-25-000001", "2025-09-30")
        from datetime import datetime, timezone
        found = cands.fetch_recent_insider_rows(c, since_utc=datetime(2025, 9, 1, tzinfo=timezone.utc))
        out = cands.aggregate_purchases(found, min_notional_usd=25_000, min_price=10.0)
        # KLG: $42.7k, no plan -> candidate, rolled to next trading day (date-only)
        # PLN: 10b5-1 plan -> excluded
        assert [(o.ticker, o.event_day, round(o.notional_usd)) for o in out] == [("KLG", "2025-10-01", 42720)]
