"""SimHash near-duplicate fingerprinting + LSH-banded union-find clustering.

Why SimHash (and not the exact ``content_hash``):
    ``content_hash_for`` in ``storage.db`` sha256s the first 200 normalised
    chars. Two stories that say the same thing with one word changed get two
    completely different hashes, so exact-hash dedup leaves ~93% of trailing
    raw_signals "distinct". SimHash is a *locality-sensitive* fingerprint: the
    Hamming distance between two SimHashes is proportional to how different the
    underlying token bags are, so near-dups land within a few bits of each
    other and can be clustered.

Algorithm (all stdlib, no new deps):
    simhash(text): tokenise -> weight each distinct token by its count ->
        blake2b each token to a `bits`-wide hash -> for each bit position add
        +weight if the bit is 1 else -weight -> sign-collapse the accumulator
        back to a `bits`-wide integer fingerprint.
    cluster(items): LSH banding. Split each fingerprint into `bands` equal
        slices; two items that share *any* identical band slice are candidate
        near-dups (this is the classic LSH trick that avoids the O(n^2) all-
        pairs Hamming comparison). Confirm a candidate pair with an exact
        Hamming check (<= max_hamming) and union them in a union-find. Cluster
        id of each item is the union-find root's stable string id.

Designed to run over the trailing-14d ``raw_signals`` (~200k rows): banding
keeps it near-linear; only same-band collisions pay the Hamming cost.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Hashable, Iterable

__all__ = ["simhash", "hamming", "cluster", "tokenize"]


# Mirror the storage-layer normalisation philosophy (lowercase, alnum only),
# but keep tokens rather than collapsing to one string so the fingerprint sees
# the full token bag, not just the truncated prefix exact-hashing uses.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + split into alphanumeric tokens. Empty/None -> []."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _token_hash(token: str, bits: int) -> int:
    """Stable `bits`-wide hash of a token via blake2b.

    blake2b lets us request an exact digest width; we ask for enough bytes to
    cover `bits` and mask down. Deterministic across processes (unlike the
    salted builtin ``hash``), which matters for a persisted cluster id.
    """
    nbytes = (bits + 7) // 8
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=nbytes).digest()
    return int.from_bytes(digest, "big") & ((1 << bits) - 1)


def simhash(text: str, bits: int = 64) -> int:
    """Token-weighted SimHash fingerprint of `text` as a `bits`-wide int.

    Token frequency is the weight, so a repeated boilerplate phrase pulls the
    fingerprint more strongly than a one-off token — exactly what we want for
    structured-note filings whose differences are a few rare tokens against a
    huge shared boilerplate body.

    Empty text returns 0 (every empty doc collapses together, which is fine —
    they carry no signal).
    """
    tokens = tokenize(text)
    if not tokens:
        return 0

    # Weight by count so we blake2b each distinct token once, not per-occurrence.
    counts: dict[str, int] = defaultdict(int)
    for tok in tokens:
        counts[tok] += 1

    accum = [0] * bits
    for tok, weight in counts.items():
        h = _token_hash(tok, bits)
        for i in range(bits):
            if (h >> i) & 1:
                accum[i] += weight
            else:
                accum[i] -= weight

    fingerprint = 0
    for i in range(bits):
        if accum[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming(a: int, b: int) -> int:
    """Hamming distance between two fingerprints (count of differing bits)."""
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------
# Union-find (disjoint set) with path compression + union by rank.
# ---------------------------------------------------------------------------


class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}
        self.rank: dict[Hashable, int] = {}

    def add(self, x: Hashable) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: Hashable) -> Hashable:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _band_keys(fp: int, bits: int, bands: int) -> list[int]:
    """Split a fingerprint into `bands` slices; return each slice's value.

    Two items sharing any identical slice are LSH candidate near-dups. With
    `bands` bands and a `max_hamming` threshold, any pair within `max_hamming`
    bits is guaranteed to share at least one identical band as long as
    bands > max_hamming (pigeonhole: <= max_hamming differing bits cannot
    touch all bands).
    """
    band_width = bits // bands
    mask = (1 << band_width) - 1
    keys = []
    for b in range(bands):
        keys.append((fp >> (b * band_width)) & mask)
    # Account for any leftover high bits when bits isn't divisible by bands —
    # fold them into the final band so no signal is dropped.
    leftover = bits - band_width * bands
    if leftover:
        keys[-1] |= ((fp >> (band_width * bands)) & ((1 << leftover) - 1)) << band_width
    return keys


# Above this bucket size we anchor-chain instead of doing all-pairs. The huge
# buckets are the boilerplate-firehose clusters (hundreds of near-identical
# 424B2s) where every member is within threshold of every other, so chaining to
# a single anchor is both correct and the only way to stay near-linear. Small
# buckets get the exact O(k^2) treatment so transitive near-dups never leak.
_ALLPAIRS_BUCKET_CAP = 64


def cluster(
    items: Iterable[tuple[Hashable, str]],
    *,
    bits: int = 64,
    bands: int = 8,
    max_hamming: int = 3,
    min_tokens: int = 0,
) -> dict[Hashable, str]:
    """Cluster ``(id, text)`` items into near-duplicate groups.

    Returns ``{id: cluster_id}`` where ``cluster_id`` is a stable string
    (``"c<root_id>"``) shared by every member of a near-dup cluster. A unique
    document gets its own singleton cluster keyed on itself.

    Parameters mirror standard SimHash-LSH practice:
      bits         fingerprint width (64).
      bands        LSH bands. Must exceed ``max_hamming`` for the pigeonhole
                   guarantee that all within-threshold pairs are candidates.
      max_hamming  max bit distance for two docs to be called near-dups (3).
      min_tokens   docs with fewer than this many tokens are NOT SimHash-
                   clustered — they each become their own singleton. SimHash is
                   unreliable on very short texts (a single word change flips a
                   large fraction of the fingerprint), so on a heterogeneous
                   corpus short headlines/tickers would otherwise smear into one
                   false mega-cluster. 0 disables the guard (default).
    """
    if bands <= max_hamming:
        # Pigeonhole guarantee would be lost — recall would silently drop.
        raise ValueError(
            f"bands ({bands}) must be > max_hamming ({max_hamming}) so every "
            "near-dup pair shares at least one identical band"
        )

    materialized = list(items)
    uf = _UnionFind()
    fps: dict[Hashable, int] = {}
    eligible: list[Hashable] = []
    for ident, text in materialized:
        uf.add(ident)
        fps[ident] = simhash(text, bits=bits)
        # Short texts stay singletons (never bucketed) so they can't false-merge.
        if min_tokens <= 0 or len(tokenize(text)) >= min_tokens:
            eligible.append(ident)

    # Bucket ids by each band slice; only ids colliding in a band are compared.
    buckets: dict[tuple[int, int], list[Hashable]] = defaultdict(list)
    for ident in eligible:
        for band_idx, key in enumerate(_band_keys(fps[ident], bits, bands)):
            buckets[(band_idx, key)].append(ident)

    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) <= _ALLPAIRS_BUCKET_CAP:
            # Small bucket: exact all-pairs so transitive near-dups (i~j, j~k
            # but i far from k) all union correctly via union-find.
            for i in range(len(members)):
                fi = fps[members[i]]
                for j in range(i + 1, len(members)):
                    mj = members[j]
                    if uf.find(members[i]) == uf.find(mj):
                        continue
                    if hamming(fi, fps[mj]) <= max_hamming:
                        uf.union(members[i], mj)
        else:
            # Huge boilerplate bucket: everything is mutually near-identical, so
            # anchor-chaining is correct and keeps this near-linear.
            anchor = members[0]
            for other in members[1:]:
                if uf.find(anchor) == uf.find(other):
                    continue
                if hamming(fps[anchor], fps[other]) <= max_hamming:
                    uf.union(anchor, other)

    return {ident: f"c{uf.find(ident)}" for ident in fps}
