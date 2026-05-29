"""Refresh the ``institutional_holdings`` table with 13F-derived
quarter-over-quarter institutional flow per ticker.

Two data paths:
  1. **Finnhub free tier** (preferred when ``FINNHUB_API_KEY`` is set):
     ``/stock/institutional-portfolio`` returns the most recent 13F
     positions of an institution by ticker. For our purposes the
     ``/stock/insider-transactions`` endpoint is more useful but it's
     paid-tier; this path tracks only the institutional flow.
  2. **SEC EDGAR fallback** (free, always works but slow): walk recent
     13F filings, parse their information tables, aggregate.

For now we use path (1) with a polite rate and fall back to no-op when
the key isn't present. Full path (2) is left as a TODO comment because
parsing 13F-HR XML correctly is a multi-hour job and the LLM redo + the
other Tier 1 items have higher leverage.

Usage::

    python scripts/refresh_13f_flow.py --top-tickers 100
    python scripts/refresh_13f_flow.py --tickers AAPL,TSLA --dry-run

Feature impact:
  ``ml/external_features.attach_institutional_features`` reads this
  table and emits ``institutional_buyers_minus_sellers_qoq``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from market_radar.config import CONFIG  # noqa: E402
from market_radar.storage import get_connection, init_db  # noqa: E402


log = logging.getLogger("refresh_13f_flow")
FINNHUB_BASE = "https://finnhub.io/api/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fnh_get(path: str, **params) -> Optional[dict]:
    if not CONFIG.finnhub_api_key:
        return None
    params["token"] = CONFIG.finnhub_api_key
    try:
        r = requests.get(f"{FINNHUB_BASE}{path}", params=params, timeout=25,
                         headers={"User-Agent": "MARKET RADAR research"})
        if r.status_code in (403, 429):
            log.debug("Finnhub %s -> %d", path, r.status_code)
            return None
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as exc:
        log.debug("Finnhub %s failed: %s", path, exc)
        return None


def fetch_13f_summary(ticker: str) -> Optional[dict]:
    """Returns the Finnhub institutional ownership rollup for one ticker."""
    return _fnh_get("/stock/institutional-ownership", symbol=ticker, limit=1)


def pick_tickers(top_n: int) -> list[str]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT ticker, COUNT(*) AS n FROM signal_scores
            WHERE scored_at >= datetime('now', '-90 days')
            GROUP BY ticker
            ORDER BY n DESC
            LIMIT ?
            """,
            (top_n,),
        ).fetchall()
    return [r["ticker"] for r in rows]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--top-tickers", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not CONFIG.finnhub_api_key:
        log.warning(
            "FINNHUB_API_KEY not set in .env — skipping 13F refresh. "
            "Add a free key from https://finnhub.io to populate the "
            "institutional_holdings table. (SEC-EDGAR direct-parse path "
            "is a separate TODO.)"
        )
        return 0
    init_db()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = pick_tickers(args.top_tickers)
    log.info("Refreshing 13F flow for %d tickers", len(tickers))

    inserted = 0
    with get_connection() as conn:
        for ticker in tickers:
            payload = fetch_13f_summary(ticker)
            time.sleep(0.4)
            if not payload:
                continue
            entries = payload.get("ownership") or payload.get("data") or []
            if not entries:
                continue
            # The Finnhub free endpoint returns a *list* of recent
            # quarter-end snapshots per institution. To get
            # buyers/sellers/net we aggregate across institutions.
            quarter_end = entries[0].get("reportDate")
            if not quarter_end:
                continue
            new_buyers = sum(1 for e in entries if (e.get("change") or 0) > 0)
            new_sellers = sum(1 for e in entries if (e.get("change") or 0) < 0)
            net_position = sum(int(e.get("change") or 0) for e in entries)
            total_holders = len(entries)

            if args.dry_run:
                log.info("  %s quarter=%s buyers=%d sellers=%d net=%d total=%d",
                         ticker, quarter_end, new_buyers, new_sellers,
                         net_position, total_holders)
                continue

            conn.execute(
                """
                INSERT OR REPLACE INTO institutional_holdings
                (ticker, quarter_end, new_buyers, new_sellers, net_position,
                 total_holders, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, quarter_end, new_buyers, new_sellers, net_position,
                 total_holders, _utc_now()),
            )
            inserted += 1
    log.info("Done. inserted=%d", inserted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
