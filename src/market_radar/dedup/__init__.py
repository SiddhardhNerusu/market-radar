"""Near-duplicate clustering (blueprint #6).

Exact ``content_hash`` (see :func:`market_radar.storage.db.content_hash_for`)
only collapses byte-identical normalised text. It misses the *bulk* of the
real duplication, which is **near**-duplicate: e.g. hundreds of structured-note
424B2 pricing supplements that share 95% of their boilerplate and differ only
in a CUSIP / dollar amount / date, or one wire story re-printed by a dozen RSS
feeds with slightly different framing. SimHash + LSH banding gives us a cheap,
dependency-free way to cluster those together.

Public API::

    from market_radar.dedup import simhash, hamming, cluster
"""
from .near_dup import cluster, hamming, simhash

__all__ = ["simhash", "hamming", "cluster"]
