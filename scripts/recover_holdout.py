"""Recover unresolved outcomes (the ~135k-row June holdout + backlog) so we get
the first genuine OUT-OF-SAMPLE test (ingestion deep-dive opportunity #1).

Entry prices are already captured and the 5-day windows have closed; this just
runs the existing resolver in big batches until the resolvable backlog stops
shrinking. Idempotent + safe — identical to the daemon's hourly resolver, faster.
Resolved outcomes are split-adjusted (alpaca adjustment='all') and corrupt-flagged
by the P1 hygiene fixes, so the recovered sample is clean.

Run in background:  PYTHONPATH=src .venv/bin/python scripts/recover_holdout.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.outcomes import update_due_outcomes
from market_radar.storage import get_connection

BATCH = 1500
MAX_ITERS = 300


def _remaining() -> int:
    with get_connection() as c:
        return c.execute(
            "SELECT COUNT(*) FROM signal_outcomes "
            "WHERE return_5d_pct IS NULL AND price_at_flag IS NOT NULL"
        ).fetchone()[0]


def main() -> int:
    start = _remaining()
    print(f"[recover] start: {start} unresolved-with-price", flush=True)
    prev = start
    stall = 0
    for i in range(MAX_ITERS):
        try:
            update_due_outcomes(batch_size=BATCH)
        except Exception as exc:  # noqa: BLE001
            print(f"[recover] iter {i} resolver error (continuing): {exc}", flush=True)
        rem = _remaining()
        print(f"[recover] iter {i}: remaining={rem} (resolved {prev - rem} this batch)", flush=True)
        if rem >= prev:
            stall += 1
            if stall >= 3:  # 3 batches with no progress => only poison pills left
                print("[recover] no further progress (poison pills) — stopping", flush=True)
                break
        else:
            stall = 0
        prev = rem
        if rem == 0:
            break
    end = _remaining()
    print(f"[recover] DONE: {start} -> {end} ({start - end} resolved)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
