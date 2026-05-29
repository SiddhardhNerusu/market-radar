"""Refresh ``fails_to_deliver`` table from SEC's free FTD list.

SEC publishes a CSV of fails-to-deliver every two weeks at
https://www.sec.gov/data/foiadocsfailsdatahtm. Materialized into a
table that feeds the ``fails_to_deliver_pct_float`` feature.

Run weekly via cron.
"""
from __future__ import annotations

import argparse, csv, io, logging, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402
from market_radar.storage import get_connection, init_db  # noqa: E402

log = logging.getLogger("refresh_fails_to_deliver")


# SEC has restructured the FTD data location several times. We try a
# couple of canonical patterns; whichever one works is fine.
SEC_FTD_LIST_CANDIDATES = [
    "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data",
    "https://www.sec.gov/foia/docs/failsdata.htm",
    "https://www.sec.gov/data/foiadocsfailsdatahtm",
]

UA = "MARKET RADAR research (redacted@example.com)"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _try_direct_zip_urls() -> list[str]:
    """SEC publishes monthly FTD zips with predictable filenames:
        https://www.sec.gov/files/data/fails-deliver-data/cnsfails<YYYY><MM><a|b>.zip
    'a' = first half of the month, 'b' = second half.
    We try the last 3 months × both halves and return any that 200-OK.
    """
    import requests as _r
    found: list[str] = []
    today = datetime.utcnow().date()
    for delta in range(0, 4):
        y = today.year
        m = today.month - delta
        while m < 1:
            m += 12
            y -= 1
        for half in ("b", "a"):
            for prefix in (
                f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{y:04d}{m:02d}{half}.zip",
                f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{y:04d}{m:02d}{half}.txt",
                f"https://www.sec.gov/foia/files/cnsfails{y:04d}{m:02d}{half}.zip",
            ):
                try:
                    h = _r.head(prefix, headers={"User-Agent": UA}, timeout=10,
                                allow_redirects=True)
                    if h.status_code == 200:
                        found.append(prefix)
                        return found  # take the most recent half-month we find
                except _r.RequestException:
                    continue
    return found


def fetch_latest_ftd_url() -> str | None:
    """Resolve a working URL for the latest FTD file."""
    import re
    # 1. Try scraping the documented index pages
    for url in SEC_FTD_LIST_CANDIDATES:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": UA})
            if r.status_code != 200:
                continue
        except requests.RequestException:
            continue
        # Match either old or new path style
        matches = re.findall(
            r'href="([^"]*?cnsfails\d{6}[ab]?\.(?:zip|txt))"', r.text
        )
        if matches:
            # Take the lexicographically-largest filename → most recent
            best = sorted(matches)[-1]
            if best.startswith("/"):
                return "https://www.sec.gov" + best
            return best

    # 2. Fall back to probing well-known direct URLs
    direct = _try_direct_zip_urls()
    if direct:
        return direct[0]
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    init_db()

    # Ensure the FTD table exists (idempotent)
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fails_to_deliver (
                ticker          TEXT NOT NULL,
                settlement_date TEXT NOT NULL,
                quantity        REAL,
                price           REAL,
                ingested_at     TEXT NOT NULL,
                PRIMARY KEY (ticker, settlement_date)
            )
        """)

    url = fetch_latest_ftd_url()
    if not url:
        log.warning("Couldn't find a recent FTD URL. SEC may have moved the data; "
                    "check https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data "
                    "manually.")
        return 1
    log.info("Fetching: %s", url)
    try:
        r = requests.get(url, timeout=120,
                         headers={"User-Agent": "MARKET RADAR research (redacted@example.com)"})
        r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("FTD download failed: %s", exc)
        return 1

    if url.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = [n for n in zf.namelist() if not n.startswith("__")]
            if not names:
                log.warning("Empty FTD zip")
                return 1
            with zf.open(names[0]) as fh:
                txt = fh.read().decode("latin-1", errors="replace")
    else:
        txt = r.content.decode("latin-1", errors="replace")

    # Files are pipe-delimited: SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY|DESCRIPTION|PRICE
    reader = csv.reader(io.StringIO(txt), delimiter="|")
    header = next(reader, [])
    inserted = 0
    with get_connection() as conn:
        for row in reader:
            if len(row) < 6:
                continue
            sd, cusip, sym, qty, desc, price = row[:6]
            sym = (sym or "").upper().strip()
            if not sym or not sd:
                continue
            try:
                q = float(qty)
            except (TypeError, ValueError):
                q = None
            try:
                pr = float(price)
            except (TypeError, ValueError):
                pr = None
            # Normalize settlement date YYYYMMDD → YYYY-MM-DD
            if len(sd) == 8 and sd.isdigit():
                sd = f"{sd[:4]}-{sd[4:6]}-{sd[6:8]}"
            if args.dry_run:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO fails_to_deliver "
                "(ticker, settlement_date, quantity, price, ingested_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (sym, sd, q, pr, _utc_now()),
            )
            inserted += 1
    log.info("Done. inserted=%d (dry_run=%s)", inserted, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
