"""Lock the outcome label-hygiene rules (P1 rebuild). The corrupted labels
(avg return 211%, max 3,002,400%) came from dividing by sub-penny anchors and
split-unadjusted prices; classify_return is the source-of-truth guard."""
from market_radar.outcomes.tracker import classify_return


def test_subpenny_anchor_is_corrupt():
    # $0.0001 anchor -> the +3,002,400% bug. Must be flagged, no return computed.
    assert classify_return(0.0001, 3.00) == (None, 1)


def test_anchor_below_one_dollar_is_corrupt():
    assert classify_return(0.99, 2.00)[1] == 1
    assert classify_return(1.00, 1.50)[1] == 0  # exactly $1 is allowed


def test_normal_return_is_clean():
    ret, corrupt = classify_return(10.0, 11.0)
    assert corrupt == 0 and abs(ret - 10.0) < 1e-9


def test_implausible_return_is_corrupt():
    # +9900% on a $1 name over a few days -> split/ticker-reuse artifact.
    assert classify_return(1.0, 100.0) == (None, 1)


def test_unresolved_close_is_not_corrupt():
    # Close not yet available (0/None) is "not resolved", not "corrupt".
    assert classify_return(10.0, 0) == (None, 0)
    assert classify_return(10.0, None) == (None, 0)


def test_garbage_anchor_is_corrupt():
    assert classify_return(None, 5.0) == (None, 1)
    assert classify_return("nan", 5.0) == (None, 1)


def test_legit_catalyst_winner_survives():
    # The strategy's TARGET trade: a real +75% catalyst pop must NOT be flagged.
    ret, corrupt = classify_return(4.0, 7.0)
    assert corrupt == 0 and abs(ret - 75.0) < 1e-9
