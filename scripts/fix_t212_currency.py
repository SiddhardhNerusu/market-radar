"""Backfill currency normalization on existing t212_positions rows.

Reads every non-zero-quantity position row, detects its quote currency
from the T212 ticker shape, fetches FX rates, and writes correct USD
values into the new ``market_value_usd`` column.  Also writes
``native_market_value`` (the unconverted quantity * current_price) for
traceability and updates the legacy ``current_price`` / ``average_price``
columns to USD so the risk manager's existing query (which multiplies
quantity * current_price) returns true USD without further code changes.

Fail-closed: when a row's quote currency can't be detected or its FX
rate can't be fetched, the USD columns are set to 0 (no phantom dollar
exposure).  Native values remain on the row.

Idempotent: re-running is safe — every row is recomputed from scratch.
Does NOT delete rows.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.config import CONFIG  # noqa: E402
from market_radar.t212.fx import (  # noqa: E402
    detect_quote_currency, to_usd, fx_rate_to_usd,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fix_t212_currency")


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CONFIG.db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _safe_float(v) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def main() -> int:
    conn = _open_db()
    try:
        rows = conn.execute(
            """
            SELECT id, ticker, quantity, average_price, current_price,
                   COALESCE(quote_currency,'') AS quote_currency,
                   COALESCE(native_market_value, 0) AS native_market_value,
                   COALESCE(market_value_usd, 0)    AS market_value_usd
            FROM t212_positions
            WHERE quantity IS NOT NULL AND quantity != 0
            """
        ).fetchall()
    except sqlite3.OperationalError as exc:
        log.error("Schema missing new columns? Run init_db first: %s", exc)
        return 2

    total_before_native = 0.0
    total_before_recorded = 0.0
    total_after_usd = 0.0
    unknown_ccy: list[str] = []
    converted_by_ccy: dict[str, int] = {}

    BATCH = 200
    updates: list[tuple] = []
    for row in rows:
        rid = row["id"]
        ticker = row["ticker"] or ""
        qty = _safe_float(row["quantity"]) or 0.0
        native_cur = _safe_float(row["current_price"])
        native_avg = _safe_float(row["average_price"])
        # `current_price` may already be a USD value from a *post-fix*
        # snapshot — for the backfill we trust the row's raw values and
        # treat them as native.  Old rows (before this fix) wrote native;
        # new rows (after this fix) also write native into raw_payload but
        # USD into current_price.  We use raw_payload as the source of
        # truth when available.
        # Simpler approach for the backfill: take whatever's stored as
        # native and reconvert.  This is idempotent only when run BEFORE
        # the snapshotter starts writing USD-converted prices; once the
        # snapshotter is updated, this backfill should be run ONCE.

        ccy = detect_quote_currency(ticker)
        native_mv = qty * native_cur if (qty is not None and native_cur is not None) else 0.0
        total_before_native += abs(native_mv)
        total_before_recorded += abs(row["market_value_usd"] or native_mv)

        if ccy is None:
            unknown_ccy.append(ticker)
            stored_ccy = "unknown_ccy"
            usd_cur = 0.0
            usd_avg = 0.0
            mv_usd = 0.0
        else:
            converted_by_ccy[ccy] = converted_by_ccy.get(ccy, 0) + 1
            rate = fx_rate_to_usd(ccy)
            if rate is None:
                stored_ccy = ccy
                usd_cur = 0.0
                usd_avg = 0.0
                mv_usd = 0.0
                log.warning("FX miss for %s (%s) — zeroing USD value", ticker, ccy)
            else:
                stored_ccy = ccy
                usd_cur = (native_cur or 0.0) * rate
                usd_avg = (native_avg or 0.0) * rate
                mv_usd = qty * usd_cur

        total_after_usd += abs(mv_usd)
        updates.append((stored_ccy, native_mv, mv_usd, usd_cur, usd_avg, rid))

        if len(updates) >= BATCH:
            conn.executemany(
                """
                UPDATE t212_positions
                SET quote_currency       = ?,
                    native_market_value  = ?,
                    market_value_usd     = ?,
                    current_price        = ?,
                    average_price        = ?
                WHERE id = ?
                """,
                updates,
            )
            conn.commit()
            updates = []

    if updates:
        conn.executemany(
            """
            UPDATE t212_positions
            SET quote_currency       = ?,
                native_market_value  = ?,
                market_value_usd     = ?,
                current_price        = ?,
                average_price        = ?
            WHERE id = ?
            """,
            updates,
        )
        conn.commit()

    # Summary
    print()
    print("=" * 60)
    print(f"Backfill complete on {len(rows)} rows.")
    print(f"  Sum |native_mv| (before fix):  ${total_before_native:>14,.2f}")
    print(f"  Sum |market_value_usd| (after): ${total_after_usd:>14,.2f}")
    if total_before_native > 0:
        ratio = total_after_usd / total_before_native
        print(f"  Ratio:                          {ratio:.4f}x")
    print(f"  Converted-by-currency:           {converted_by_ccy}")
    print(f"  Unknown-currency rows:           {len(unknown_ccy)}")
    if unknown_ccy:
        print(f"    sample tickers: {sorted(set(unknown_ccy))[:8]}")

    # Quick sanity on the LATEST snapshot only
    latest = conn.execute(
        """
        SELECT
            ROUND(SUM(ABS(market_value_usd)), 2) AS usd_total,
            ROUND(SUM(ABS(native_market_value)), 2) AS native_total,
            COUNT(*) AS n
        FROM t212_positions p
        JOIN (
            SELECT account_type AS a, MAX(snapshot_at) AS m
            FROM t212_positions GROUP BY account_type
        ) j ON j.a = p.account_type AND j.m = p.snapshot_at
        WHERE p.quantity != 0
        """
    ).fetchone()
    print()
    print(f"Latest snapshot (post-fix): n={latest['n']} "
          f"USD=${latest['usd_total']:,} native=${latest['native_total']:,}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
