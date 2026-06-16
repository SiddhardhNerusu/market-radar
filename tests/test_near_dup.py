"""Lock the SimHash near-duplicate clustering contract (blueprint #6).

The whole point of this module is to collapse what exact ``content_hash``
cannot: stories that are *almost* identical (one word/number changed). These
tests pin the three behaviours that matter:
  1. synthetic near-dups (a real headline reprinted with small edits) cluster.
  2. clearly-distinct stories do NOT bleed into each other.
  3. byte-identical texts always collapse to one cluster.
"""
from market_radar.dedup import cluster, hamming, simhash


def test_identical_text_has_identical_fingerprint():
    a = "Acme Corp reports record quarterly earnings, beats estimates"
    assert simhash(a) == simhash(a)
    assert hamming(simhash(a), simhash(a)) == 0


# A realistic structured-note body: the shared boilerplate dwarfs the few
# tokens (CUSIP / coupon / maturity) that actually differ between filings. This
# is where SimHash is meant to operate — NOT 10-token headlines, where a single
# word change flips a large fraction of the fingerprint by design.
_424B2_BOILER = (
    "424B2 pricing supplement filed pursuant to rule 424 b 2 registration "
    "statement number 333 the information in this preliminary pricing supplement "
    "is not complete and may be changed contingent income auto callable "
    "securities linked to the least performing of the common stock subject to "
    "the credit risk of the issuer and the guarantee of the parent company "
    "payment at maturity additional terms of the notes interest payment dates "
    "valuation dates final valuation date stated principal amount per security "
)


def test_near_dup_fingerprints_are_close_distinct_are_far():
    base = _424B2_BOILER + "JPMorgan Chase Financial Co LLC CUSIP 48133A2K9 maturity June 2030 coupon 9.15"
    near = _424B2_BOILER + "JPMorgan Chase Financial Co LLC CUSIP 48133A3L1 maturity July 2031 coupon 8.40"
    far = ("Tesla recalls 1.2 million vehicles over autopilot software defect "
           "NHTSA investigation widens federal safety probe into driver assist")
    assert hamming(simhash(base), simhash(near)) <= 3
    assert hamming(simhash(base), simhash(far)) > 3


def test_synthetic_near_dups_share_a_cluster():
    # Several near-identical filings that differ only in CUSIP/coupon/date — the
    # exact case content_hash misses. They must land in ONE cluster.
    items = [
        ("a", _424B2_BOILER + "JPMorgan Chase Financial Co LLC CUSIP 48133A2K9 maturity June 2030 coupon 9.15"),
        ("b", _424B2_BOILER + "JPMorgan Chase Financial Co LLC CUSIP 48133A3L1 maturity July 2031 coupon 8.40"),
        ("c", _424B2_BOILER + "JPMorgan Chase Financial Co LLC CUSIP 48133A4M2 maturity Aug 2030 coupon 10.05"),
        ("d", _424B2_BOILER + "JPMorgan Chase Financial Co LLC CUSIP 48133A5N3 maturity Sep 2032 coupon 7.95"),
    ]
    out = cluster(items)
    roots = {out["a"], out["b"], out["c"], out["d"]}
    assert len(roots) == 1, f"near-dups failed to merge: {out}"


def test_distinct_texts_do_not_cluster():
    items = [
        ("x", "Federal Reserve holds interest rates steady at March FOMC meeting"),
        ("y", "Apple unveils Vision Pro 2 headset with lighter design and lower price"),
        ("z", "Boeing 737 MAX deliveries halted after fresh fuselage quality finding"),
    ]
    out = cluster(items)
    assert len({out["x"], out["y"], out["z"]}) == 3, f"distinct texts merged: {out}"


def test_identical_texts_collapse_to_one_cluster():
    text = "424B2 - Morgan Stanley Finance LLC (0001666268) (Filer)"
    items = [("f1", text), ("f2", text), ("f3", text), ("f4", text)]
    out = cluster(items)
    assert len(set(out.values())) == 1


def test_many_repeated_filings_collapse_but_a_different_one_splits():
    # Many copies of one filing + one genuinely different filing -> 2 clusters,
    # not N. This is the 424B2-firehose acceptance shape in miniature.
    items = [(f"jpm{i}",
              _424B2_BOILER + f"JPMorgan Chase Financial Co LLC CUSIP 48133A{i}K maturity 2030 coupon 9")
             for i in range(20)]
    items.append(("recall",
                  "Ford recalls 870000 trucks over brake fluid leak fire risk "
                  "NHTSA opens investigation into faulty hydraulic line supplier"))
    out = cluster(items)
    jpm_roots = {out[f"jpm{i}"] for i in range(20)}
    assert len(jpm_roots) == 1
    assert out["recall"] not in jpm_roots


def test_min_tokens_keeps_short_texts_as_singletons():
    # Short headlines are SimHash-noisy; with min_tokens set they must NOT merge
    # into one false cluster even when superficially similar.
    items = [
        ("s1", "AAPL up 3%"),
        ("s2", "TSLA up 3%"),
        ("s3", "NVDA up 3%"),
    ]
    out = cluster(items, min_tokens=20)
    assert len(set(out.values())) == 3, f"short texts false-merged: {out}"
    # Long near-dups still cluster under the same min_tokens guard.
    long_items = [
        ("a", _424B2_BOILER + "JPMorgan Chase Financial Co LLC CUSIP 48133A2K9 maturity June 2030 coupon 9.15"),
        ("b", _424B2_BOILER + "JPMorgan Chase Financial Co LLC CUSIP 48133A3L1 maturity July 2031 coupon 8.40"),
    ]
    out2 = cluster(long_items, min_tokens=20)
    assert out2["a"] == out2["b"]


def test_bands_must_exceed_max_hamming():
    import pytest
    with pytest.raises(ValueError):
        cluster([("a", "hello world")], bands=3, max_hamming=3)


def test_empty_text_fingerprint_is_zero():
    assert simhash("") == 0
    assert simhash(None) == 0  # type: ignore[arg-type]
