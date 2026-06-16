"""Periodic DB maintenance (P3 hardening 2026-06-15): reclaim space + prune telemetry.

The ~1.3 GB SQLite file is hit every ~30s by ~15 jobs with no retention policy.
This:
  * Prunes high-volume TELEMETRY tables older than --retain-days. It NEVER
    touches training / edge / accounting data (raw_signals, signal_scores,
    signal_outcomes, bot_orders, bot_option_*, bot_account_snapshots,
    bot_daily_pnl) — only operational noise.
  * Runs ANALYZE + VACUUM to reclaim freelist pages and refresh planner stats.

WARNING: VACUUM takes an EXCLUSIVE lock and rewrites the whole file. Run it when
the daemon/trader are idle or stopped (e.g. a weekend launchd window), NOT in the
middle of an active trading session.

Usage:  PYTHONPATH=src .venv/bin/python scripts/db_maintenance.py [--retain-days N] [--no-vacuum]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.storage import get_connection

# (table, timestamp_column) — telemetry only; safe to age out. A wrong/missing
# column just skips that table (caught below) — no data-loss risk.
_TELEMETRY = [
    ("daemon_health_alerts", "created_at"),
    ("notifications_sent", "ingested_at"),
    ("llm_spend_daily", "spend_date"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retain-days", type=int, default=90)
    ap.add_argument("--no-vacuum", action="store_true",
                    help="prune only; skip the exclusive-lock VACUUM")
    args = ap.parse_args()
    cutoff = f"datetime('now', '-{int(args.retain_days)} days')"

    with get_connection() as conn:
        for tbl, ts_col in _TELEMETRY:
            try:
                before = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                conn.execute(f"DELETE FROM {tbl} WHERE {ts_col} < {cutoff}")
                after = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                print(f"  {tbl}: {before} -> {after} ({before - after} pruned > {args.retain_days}d)")
            except Exception as exc:  # noqa: BLE001 — missing table/col is non-fatal
                print(f"  {tbl}: skipped ({exc})")
        if not args.no_vacuum:
            print("  ANALYZE + VACUUM (exclusive lock; reclaiming space)…")
            conn.execute("ANALYZE")
            conn.execute("VACUUM")
            print("  VACUUM done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
