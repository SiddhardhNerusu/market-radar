"""The daemon job that feeds the insider lane: hydrated Form 4 bodies ->
insider_transactions rows, idempotently, with the 10b5-1 flag."""
from market_radar.ingestors.insider_transactions import refresh_insider_transactions
from market_radar.storage.db import get_connection, init_db

FORM4 = """<?xml version="1.0"?><ownershipDocument>
<issuer><issuerCik>0000001</issuerCik><issuerTradingSymbol>ACME</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId><rptOwnerName>Jane Doe</rptOwnerName></reportingOwnerId>
<reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>1</isOfficer><officerTitle>CEO</officerTitle></reportingOwnerRelationship></reportingOwner>
<periodOfReport>2026-06-09</periodOfReport>
<nonDerivativeTable><nonDerivativeTransaction>
<transactionCoding><transactionCode>P</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>2000</value></transactionShares>
<transactionPricePerShare><value>50.25</value></transactionPricePerShare>
<transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
</nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>""" + " " * 200


def test_refresh_parses_once_and_is_idempotent(tmp_path):
    path = tmp_path / "x.db"
    init_db(path)
    with get_connection(path) as c:
        c.execute("INSERT INTO raw_signals (id, source, source_tier, title, body, published_at, ingested_at) VALUES (?,?,?,?,?,?,?)",
                  (7, "sec_edgar", 1, "4 - ACME", FORM4, "2026-06-10T14:00:00Z", "2026-06-10T14:00:30Z"))
        out = refresh_insider_transactions(conn=c)
        assert out == {"bodies": 1, "inserted": 1, "skipped": 0}
        row = tuple(c.execute("SELECT ticker, transaction_code, shares, price, is_acquired, role_score, is_10b5_1, report_date FROM insider_transactions").fetchone())
        assert row == ("ACME", "P", 2000.0, 50.25, 1, 3, 0, "2026-06-09")
        again = refresh_insider_transactions(conn=c)
        assert again["bodies"] == 0
        assert c.execute("SELECT COUNT(*) FROM insider_transactions").fetchone()[0] == 1


def test_refresh_flags_10b5_1_plan_trades(tmp_path):
    path = tmp_path / "y.db"
    init_db(path)
    body = FORM4.replace("<periodOfReport>", "<footnote>Transaction under a Rule 10b5-1 plan</footnote><periodOfReport>")
    with get_connection(path) as c:
        c.execute("INSERT INTO raw_signals (id, source, source_tier, title, body, published_at, ingested_at) VALUES (?,?,?,?,?,?,?)",
                  (8, "sec_edgar", 1, "4 - ACME", body, "2026-06-10T14:00:00Z", "2026-06-10T14:00:30Z"))
        refresh_insider_transactions(conn=c)
        assert c.execute("SELECT is_10b5_1 FROM insider_transactions").fetchone()[0] == 1
