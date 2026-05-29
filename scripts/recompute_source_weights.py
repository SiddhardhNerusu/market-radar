"""Recompute SOURCE_WEIGHTS from measured per-source 5-day hit rate.

Static credibility weights are a reasonable starting point but stale: a
source that looked good a year ago may have drifted, and we have 135k+
resolved outcomes that *measure* which sources actually predict moves.

This script:
  1. Loads (source, return_5d_pct) pairs from the last ``--lookback-days``
     days (default 180).
  2. For each source with at least ``--min-n`` resolved outcomes, computes
     the empirical hit rate (% positive) and the mean absolute return
     magnitude.
  3. Maps the hit rate to a learned weight via a sigmoid centred on the
     base rate, clipped to ``[--floor, --ceiling]`` of the source's
     static weight. So a source whose hit rate matches the base rate
     gets ~1.0x its static weight; consistently-predictive sources get
     up to ``ceiling``; consistently-anti-predictive ones get
     ``floor``.
  4. Writes the result to ``data/source_weights_learned.json``. The
     composite scorer can opt in by loading this file when present.

Default behaviour is a **dry run**: the script reports the proposed
shifts but doesn't write anything until you re-run with ``--apply``.

Usage::

    python scripts/recompute_source_weights.py
    python scripts/recompute_source_weights.py --apply
    python scripts/recompute_source_weights.py --lookback-days 90 --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.config import PROJECT_ROOT  # noqa: E402
from market_radar.scoring.source_weights import (  # noqa: E402
    SOURCE_WEIGHTS,
    UNKNOWN_SOURCE_WEIGHT,
    weight_for,
)
from market_radar.storage import get_connection  # noqa: E402


log = logging.getLogger("recompute_source_weights")


OUTPUT_PATH = PROJECT_ROOT / "data" / "source_weights_learned.json"


def _sigmoid(x: float, *, k: float = 12.0) -> float:
    """Tight sigmoid: most movement happens within ±10pp of the centre."""
    return 1.0 / (1.0 + math.exp(-k * x))


def _normalize_source(source: str) -> str:
    """Collapse all sec_edgar_backfill_* variants into 'sec_edgar' so
    historical EDGAR rows share one learned weight with live."""
    if source and source.startswith("sec_edgar"):
        return "sec_edgar"
    return source


def _learned_weight(
    *, source: str, hit_rate: float, base_rate: float,
    floor_mult: float, ceiling_mult: float,
) -> float:
    """Map empirical hit rate to a weight, multiplicative against the
    static credibility.

    multiplier = sigmoid(hit_rate - base_rate) ∈ (0, 1) →
                 rescale to (floor_mult, ceiling_mult)
    """
    s = _sigmoid(hit_rate - base_rate)
    mult = floor_mult + (ceiling_mult - floor_mult) * s
    static = SOURCE_WEIGHTS.get(source) or UNKNOWN_SOURCE_WEIGHT
    return float(static * mult)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lookback-days", type=int, default=180,
                   help="How many days of resolved outcomes to consider (default: 180)")
    p.add_argument("--min-n", type=int, default=30,
                   help="Minimum resolved outcomes per source to be eligible (default: 30)")
    p.add_argument("--floor", dest="floor_mult", type=float, default=0.5,
                   help="Worst-case multiplier vs. static weight (default: 0.5x)")
    p.add_argument("--ceiling", dest="ceiling_mult", type=float, default=1.6,
                   help="Best-case multiplier vs. static weight (default: 1.6x)")
    p.add_argument("--apply", action="store_true",
                   help="Actually write the learned weights file (default: dry run)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT rs.source AS source,
                   so.return_5d_pct AS r
            FROM signal_scores ss
            JOIN raw_signals    rs ON rs.id = ss.signal_id
            JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE so.return_5d_pct IS NOT NULL
              AND ss.scored_at >= datetime('now', ?)
            """,
            (f"-{int(args.lookback_days)} days",),
        ).fetchall()

    log.info("Loaded %d resolved (source, return_5d) pairs over last %d days",
             len(rows), args.lookback_days)

    by_source: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        src = _normalize_source(r["source"] or "")
        if not src:
            continue
        by_source[src].append(float(r["r"]))

    # Global base rate — anchors the sigmoid centre
    all_returns = [v for vs in by_source.values() for v in vs]
    if not all_returns:
        log.error("No resolved returns found — nothing to learn from.")
        return 1
    base_rate = sum(1 for v in all_returns if v > 0) / len(all_returns)
    log.info("Global base rate (P[return_5d > 0]): %.4f over %d obs",
             base_rate, len(all_returns))

    learned: dict[str, dict] = {}
    eligible = 0
    for src, returns in sorted(by_source.items(), key=lambda x: -len(x[1])):
        n = len(returns)
        if n < args.min_n:
            continue
        hits = sum(1 for v in returns if v > 0)
        hit_rate = hits / n
        mean_abs = sum(abs(v) for v in returns) / n
        weight = _learned_weight(
            source=src, hit_rate=hit_rate, base_rate=base_rate,
            floor_mult=args.floor_mult, ceiling_mult=args.ceiling_mult,
        )
        static = SOURCE_WEIGHTS.get(src) or UNKNOWN_SOURCE_WEIGHT
        learned[src] = {
            "n": n,
            "hit_rate": round(hit_rate, 4),
            "mean_abs_return_pct": round(mean_abs, 4),
            "static_weight": static,
            "learned_weight": round(weight, 3),
            "multiplier": round(weight / static if static else 0.0, 3),
        }
        eligible += 1

    log.info("Eligible sources (n >= %d): %d / %d",
             args.min_n, eligible, len(by_source))

    # Report the biggest shifts so the human can sanity-check
    shifts = sorted(
        learned.items(),
        key=lambda kv: abs(kv[1]["multiplier"] - 1.0),
        reverse=True,
    )
    log.info("Top 15 shifts vs static (multiplier=1.0 means unchanged):")
    log.info("  %-30s  %5s  %6s  %5s  %5s  %5s",
             "source", "n", "hitrt", "stat", "learn", "x")
    for src, d in shifts[:15]:
        log.info(
            "  %-30s  %5d  %.4f  %5.2f  %5.2f  %5.2fx",
            src, d["n"], d["hit_rate"], d["static_weight"],
            d["learned_weight"], d["multiplier"],
        )

    if not args.apply:
        log.info("Dry run — not writing %s. Re-run with --apply to save.",
                 OUTPUT_PATH)
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lookback_days": args.lookback_days,
        "min_n": args.min_n,
        "base_rate": base_rate,
        "floor_mult": args.floor_mult,
        "ceiling_mult": args.ceiling_mult,
        "weights": learned,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s (%d sources)", OUTPUT_PATH, len(learned))
    log.info("The composite scorer can opt into these by calling "
             "scoring.source_weights.weight_for() once it's been wired to "
             "load this file. (Not yet auto-applied — review the diff first.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
