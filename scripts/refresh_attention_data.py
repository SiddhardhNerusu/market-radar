"""Refresh the ``attention_data`` table with Google Trends + Wikipedia
pageview proxies for retail attention.

Both data sources are free. Both are rate-limited:
  - Google Trends (``pytrends``): ~5 requests/min before throttling
  - Wikipedia pageviews (MediaWiki REST API): ~100 req/sec is fine for
    a polite User-Agent

For each ticker we compute:
  - ``gtrends_value``  : current-week search interest (0..100 scale)
  - ``gtrends_zscore`` : z-score of current-week vs trailing 13 weeks
  - ``wiki_pageviews`` : last-week pageviews on the company's page
  - ``wiki_zscore``    : z-score of last week vs trailing 30 days

Usage::

    python scripts/refresh_attention_data.py --tickers AAPL,TSLA,NVDA
    python scripts/refresh_attention_data.py --top-tickers 50        # top by recent signal volume
    python scripts/refresh_attention_data.py --dry-run

The script picks tickers to refresh either from --tickers or from the
most-active tickers in the last week of signals. Wikipedia page names
default to the ticker symbol; pass --wiki-mapping if you want a curated
TICKER->Wikipedia-page map.

Feature impact:
  ``ml/external_features.attach_attention_features`` reads from this
  table and emits ``gtrends_zscore`` + ``wikipedia_pageviews_zscore``.
"""
from __future__ import annotations

import os

import argparse
import logging
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from market_radar.storage import get_connection, init_db  # noqa: E402


log = logging.getLogger("refresh_attention_data")


WIKI_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/{title}/daily/{start}/{end}"
USER_AGENT = os.getenv("RESEARCH_CONTACT_UA", "market-radar research (set RESEARCH_CONTACT_UA)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _zscore(values: list[float], current: float) -> Optional[float]:
    if len(values) < 4:
        return None
    mu = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else 0.0
    if sd == 0:
        return 0.0
    return (current - mu) / sd


def pick_tickers(*, explicit: Optional[list[str]], top_n: Optional[int]) -> list[str]:
    if explicit:
        return sorted({t.strip().upper() for t in explicit if t})
    if top_n:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT ticker, COUNT(*) AS n FROM signal_scores
                WHERE scored_at >= datetime('now', '-7 days')
                GROUP BY ticker
                ORDER BY n DESC
                LIMIT ?
                """,
                (top_n,),
            ).fetchall()
        return [r["ticker"] for r in rows]
    return []


# ---- Wikipedia ----

def fetch_wiki_pageviews(ticker: str, *, days: int = 30,
                         page_title: Optional[str] = None) -> Optional[list[int]]:
    """Return a list of daily pageview counts (most recent N days) for the
    ticker's likely Wikipedia article. Returns None if the article isn't
    found.
    """
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    title = page_title or ticker  # crude default — users can curate
    url = WIKI_API.format(
        title=title,
        start=start.strftime("%Y%m%d"),
        end=end.strftime("%Y%m%d"),
    )
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as exc:
        log.debug("Wiki fetch failed for %s: %s", ticker, exc)
        return None
    items = data.get("items") or []
    return [int(it.get("views", 0)) for it in items]


# ---- Google Trends ----

def fetch_gtrends(ticker: str) -> Optional[tuple[float, list[float]]]:
    """Return (current_week_value, trailing_13_week_values) for the
    ticker's Google Trends search interest. Uses ``pytrends``; the call
    pattern below is conservative on rate limit.
    """
    try:
        from pytrends.request import TrendReq  # type: ignore
    except ImportError:
        log.warning("pytrends not installed — skip Google Trends")
        return None
    pytrends = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=1.0)
    try:
        pytrends.build_payload([ticker], cat=0, timeframe="today 3-m", geo="US")
        df = pytrends.interest_over_time()
    except Exception as exc:  # noqa: BLE001
        log.debug("pytrends failed for %s: %s", ticker, exc)
        return None
    if df is None or df.empty or ticker not in df.columns:
        return None
    series = df[ticker].tolist()
    if not series:
        return None
    current = float(series[-1])
    history = [float(x) for x in series[-14:-1]]  # trailing 13 weeks
    return current, history


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--top-tickers", type=int, default=None)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--gtrends", action="store_true", default=False,
                   help="Enable Google Trends fetches (slow; rate-limited)")
    p.add_argument("--wiki-pause", type=float, default=0.5)
    p.add_argument("--gtrends-pause", type=float, default=15.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    init_db()

    tickers_arg = args.tickers.split(",") if args.tickers else None
    tickers = pick_tickers(explicit=tickers_arg, top_n=args.top_tickers)
    if not tickers:
        log.error("No tickers specified. Pass --tickers or --top-tickers.")
        return 1
    log.info("Refreshing attention data for %d tickers", len(tickers))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n_wiki = n_trends = 0

    with get_connection() as conn:
        for ticker in tickers:
            wiki_pv: Optional[list[int]] = None
            wiki_z: Optional[float] = None
            try:
                wiki_pv = fetch_wiki_pageviews(ticker, days=args.days)
            except Exception as exc:  # noqa: BLE001
                log.debug("wiki fetch error %s: %s", ticker, exc)
            if wiki_pv and len(wiki_pv) >= 8:
                last_week = sum(wiki_pv[-7:])
                prior = [sum(wiki_pv[i:i+7])
                         for i in range(0, len(wiki_pv) - 13, 7)
                         if i + 7 <= len(wiki_pv) - 7]
                wiki_z = _zscore(prior, float(last_week)) if prior else None
                n_wiki += 1
            time.sleep(args.wiki_pause)

            gt_current = gt_z = None
            if args.gtrends:
                try:
                    res = fetch_gtrends(ticker)
                except Exception as exc:  # noqa: BLE001
                    log.debug("gtrends fetch error %s: %s", ticker, exc)
                    res = None
                if res is not None:
                    current, history = res
                    gt_current = current
                    gt_z = _zscore(history, current) if history else None
                    n_trends += 1
                time.sleep(args.gtrends_pause)

            if args.dry_run:
                log.info("  %s wiki_last=%s wiki_z=%s gt=%s gt_z=%s",
                         ticker,
                         sum(wiki_pv[-7:]) if wiki_pv else None,
                         wiki_z, gt_current, gt_z)
                continue

            conn.execute(
                """
                INSERT OR REPLACE INTO attention_data
                (ticker, observed_at, gtrends_value, gtrends_zscore,
                 wiki_pageviews, wiki_zscore, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, today,
                    gt_current, gt_z,
                    sum(wiki_pv[-7:]) if wiki_pv else None,
                    wiki_z,
                    _utc_now(),
                ),
            )

    log.info("Done. wiki=%d gtrends=%d (of %d tickers)", n_wiki, n_trends, len(tickers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
