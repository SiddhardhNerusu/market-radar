"""Lock Form-4 10b5-1 detection (ingestion blueprint #5): a pre-planned Rule
10b5-1 trade is ROUTINE/uninformative; an opportunistic buy is not. Separating
them is the durable insider edge (Cohen-Malloy-Pomorski)."""
from market_radar.ingestors.sec_form4_parser import parse_form4_xml


def test_10b5_1_detected_in_footnote_sale():
    xml = (
        "<ownershipDocument><issuerTradingSymbol>ACME</issuerTradingSymbol>"
        "<rptOwnerName>Jane Doe</rptOwnerName>"
        "<officerTitle>CFO</officerTitle><isOfficer>1</isOfficer>"
        "Footnote: This sale was effected pursuant to a Rule 10b5-1 trading plan."
        "<nonDerivativeTransaction>"
        "<transactionCode>S</transactionCode>"
        "<transactionShares>5000</transactionShares>"
        "<transactionPricePerShare>20</transactionPricePerShare>"
        "<transactionAcquiredDisposedCode>D</transactionAcquiredDisposedCode>"
        "</nonDerivativeTransaction>"
    )
    p = parse_form4_xml(xml)
    assert p is not None and p.issuer_ticker == "ACME"
    assert p.is_10b5_1 is True


def test_opportunistic_buy_is_not_10b5_1():
    xml = (
        "<ownershipDocument><issuerTradingSymbol>BETA</issuerTradingSymbol>"
        "<rptOwnerName>John Roe</rptOwnerName><isDirector>1</isDirector>"
        " rule10b5One false "
        "<nonDerivativeTransaction>"
        "<transactionCode>P</transactionCode>"
        "<transactionShares>1000</transactionShares>"
        "<transactionPricePerShare>5</transactionPricePerShare>"
        "<transactionAcquiredDisposedCode>A</transactionAcquiredDisposedCode>"
        "</nonDerivativeTransaction>"
    )
    p = parse_form4_xml(xml)
    assert p is not None
    assert p.is_10b5_1 is False, "'10b5One false' (no dash-1) is NOT a plan trade"
    assert p.transactions and p.transactions[0].transaction_code == "P"
    assert p.transactions[0].is_acquired is True
