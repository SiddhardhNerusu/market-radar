"""Fetch daily bars for the trend-following ETF basket into data/research_bars.db.

Research-only data pull (does NOT touch data/market_radar.db).
- 11 liquid ETFs, 2016-01-01 -> present, timeframe=1Day, adjustment=all
- Tries feed=sip first; falls back to feed=iex per-symbol if SIP is forbidden
- Idempotent upsert into etf_bars(symbol,date,open,high,low,close,volume)
- Validates: row counts, first/last date, gaps > 7 calendar days, closes > 0
"""

import json
import os
import sqlite3
import sys
import time
from datetime import date, datetime

import requests
from dotenv import load_dotenv

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT, "data", "research_bars.db")
BASE_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"

SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "HYG", "GLD", "DBC", "VNQ"]
START = "2016-01-01"

# Known long US market closures within window (> 7 calendar-day gaps are
# flagged regardless; this list is only used to annotate, not suppress).


def get_headers() -> dict:
    load_dotenv(os.path.join(PROJECT, ".env"))
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        print("FATAL: missing Alpaca credentials in .env", file=sys.stderr)
        sys.exit(1)
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS etf_bars (
            symbol TEXT,
            date   TEXT,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.commit()
    return conn


def fetch_symbol(session: requests.Session, headers: dict, symbol: str, feed: str):
    """Fetch all pages of daily bars for one symbol. Returns (bars, feed_used).

    Raises requests.HTTPError on non-retryable failure other than SIP-forbidden,
    which triggers an automatic per-symbol fallback to iex.
    """
    bars = []
    page_token = None
    while True:
        params = {
            "timeframe": "1Day",
            "adjustment": "all",
            "feed": feed,
            "start": START,
            "limit": 10000,
        }
        if page_token:
            params["page_token"] = page_token
        for attempt in range(5):
            resp = session.get(
                BASE_URL.format(symbol=symbol), headers=headers, params=params, timeout=30
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            break
        if resp.status_code == 403 and feed == "sip":
            # SIP not permitted for this key -> caller retries with iex
            return None, None
        resp.raise_for_status()
        payload = resp.json()
        chunk = payload.get("bars") or []
        bars.extend(chunk)
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return bars, feed


def upsert(conn: sqlite3.Connection, symbol: str, bars: list) -> int:
    rows = []
    for b in bars:
        # 't' like '2016-01-04T05:00:00Z' -> date part
        d = b["t"][:10]
        rows.append((symbol, d, b["o"], b["h"], b["l"], b["c"], float(b["v"])))
    conn.executemany(
        """
        INSERT INTO etf_bars (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def validate(conn: sqlite3.Connection) -> dict:
    out = {"per_symbol": {}, "issues": []}
    for sym in SYMBOLS:
        cur = conn.execute(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM etf_bars WHERE symbol=?", (sym,)
        )
        n, dmin, dmax = cur.fetchone()
        bad_close = conn.execute(
            "SELECT COUNT(*) FROM etf_bars WHERE symbol=? AND (close IS NULL OR close <= 0)",
            (sym,),
        ).fetchone()[0]
        # gap check: consecutive trading dates > 7 calendar days apart
        dates = [
            r[0]
            for r in conn.execute(
                "SELECT date FROM etf_bars WHERE symbol=? ORDER BY date", (sym,)
            )
        ]
        gaps = []
        for a, b in zip(dates, dates[1:]):
            da = date.fromisoformat(a)
            db = date.fromisoformat(b)
            if (db - da).days > 7:
                gaps.append((a, b, (db - da).days))
        out["per_symbol"][sym] = {
            "rows": n,
            "first": dmin,
            "last": dmax,
            "bad_close": bad_close,
            "gaps_gt_7d": gaps,
        }
        if n == 0:
            out["issues"].append(f"{sym}: NO ROWS")
        if bad_close:
            out["issues"].append(f"{sym}: {bad_close} rows with close<=0 or NULL")
        for a, b, days in gaps:
            out["issues"].append(f"{sym}: {days}-day calendar gap {a} -> {b}")
    return out


def main():
    headers = get_headers()
    conn = init_db()
    session = requests.Session()

    feeds_used = {}
    for sym in SYMBOLS:
        bars, used = fetch_symbol(session, headers, sym, "sip")
        if bars is None:
            print(f"{sym}: SIP forbidden (403) -> falling back to feed=iex")
            bars, used = fetch_symbol(session, headers, sym, "iex")
        n = upsert(conn, sym, bars)
        feeds_used[sym] = used
        print(f"{sym}: fetched {len(bars)} bars via {used}, upserted {n}")

    report = validate(conn)
    report["feeds_used"] = feeds_used
    total = conn.execute("SELECT COUNT(*) FROM etf_bars").fetchone()[0]
    report["total_rows"] = total
    print("\n===VALIDATION===")
    print(json.dumps(report, indent=2))
    conn.close()


if __name__ == "__main__":
    main()
