#!/usr/bin/env python3
"""
Enrich live-timed events with EX-ANTE microstructure features:
  - shares outstanding (SEC companyfacts, as-of <= event_day)  -> float buckets
  - prior-day RVOL  = vol(day-1) / avg vol(day-21..day-2)       [fully ex-ante]
  - opening gap     = open(event_day)/close(day-1) - 1          [knowable at the open]
  - dollar volume day-1, price
  - EVENT-DAY RVOL  = vol(event_day)/trailing avg               [END-OF-DAY knowledge — labeled, kept separate]

Reads:  data/market_radar.db (READ-ONLY)
Writes: data/research_bars.db  table event_features (replaced each run)
Caches: SEC/Alpaca JSON in the scratchpad dir (SCRATCH env) so re-runs are cheap.

Universe / labels / sampling are pre-registered (see study brief):
  live-timed events, dedup one per (ticker, event_day);
  BANG r5>=50, CRASH r5<=-20, MEH otherwise;
  ALL bangs + ALL crashes + deterministic stride sample of ~245 MEH controls.
"""

import json
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from datetime import date, datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DB = os.path.join(PROJ, "data", "market_radar.db")
RESEARCH_DB = os.path.join(PROJ, "data", "research_bars.db")
SCRATCH = os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-502/-Users-sidnerusu-Desktop-Goyo-Server/348d0e40-56b9-407c-84ad-b2422002ba5c/scratchpad",
)
SEC_CACHE = os.path.join(SCRATCH, "sec_facts")
BARS_CACHE = os.path.join(SCRATCH, "alpaca_bars")
os.makedirs(SEC_CACHE, exist_ok=True)
os.makedirs(BARS_CACHE, exist_ok=True)

SEC_UA = "market-radar-research redacted@example.com"
SEC_MIN_INTERVAL = 0.13  # <=8 req/s
MEH_STRIDE = 40          # deterministic control sample: ~9794/40 ~= 245 controls

BARS_START = "2026-03-20"
BARS_END = "2026-06-14"


def load_env():
    envp = os.path.join(PROJ, ".env")
    if os.path.exists(envp):
        for line in open(envp):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def http_get(url, headers, retries=3, timeout=60):
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 403, 404, 422):
                return e.code, e.read()
            last = e
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  GET failed after {retries} tries: {url[:100]} ({last})", flush=True)
    return None, None


# ---------------------------------------------------------------- sample
def build_sample():
    con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        WITH ev AS (
          SELECT ss.ticker,
                 substr(COALESCE(rs.published_at, ss.scored_at),1,10) AS event_day,
                 so.return_5d_pct AS r5,
                 so.price_at_flag AS px,
                 ROW_NUMBER() OVER (
                   PARTITION BY ss.ticker, substr(COALESCE(rs.published_at, ss.scored_at),1,10)
                   ORDER BY ss.id) AS rn
          FROM signal_scores ss
          JOIN raw_signals rs   ON rs.id = ss.signal_id
          JOIN signal_outcomes so ON so.score_id = ss.id
          WHERE so.return_5d_pct IS NOT NULL
            AND COALESCE(so.data_corrupt,0) = 0
            AND so.price_at_flag BETWEEN 1 AND 50
            AND ABS(so.return_5d_pct) <= 400
            AND ABS(julianday(ss.scored_at) - julianday(rs.published_at)) <= 2
            AND rs.source NOT LIKE 'sec_edgar_backfill%'
        ),
        d AS (SELECT ticker, event_day, r5, px,
                     CASE WHEN r5 >= 50 THEN 'bang'
                          WHEN r5 <= -20 THEN 'crash'
                          ELSE 'meh' END AS label
              FROM ev WHERE rn = 1),
        meh AS (SELECT *, ROW_NUMBER() OVER (ORDER BY ticker, event_day) AS rid
                FROM d WHERE label = 'meh')
        SELECT ticker, event_day, r5, px, label FROM d WHERE label != 'meh'
        UNION ALL
        SELECT ticker, event_day, r5, px, label FROM meh WHERE rid % :stride = 1
        ORDER BY ticker, event_day
        """,
        {"stride": MEH_STRIDE},
    ).fetchall()
    con.close()
    events = [dict(r) for r in rows]
    n = {"bang": 0, "crash": 0, "meh": 0}
    for e in events:
        n[e["label"]] += 1
    print(f"sample: {len(events)} events  bangs={n['bang']} crashes={n['crash']} meh_controls={n['meh']}", flush=True)
    return events


# ---------------------------------------------------------------- SEC
def load_cik_map():
    path = os.path.join(SCRATCH, "company_tickers.json")
    data = None
    if os.path.exists(path):
        data = json.load(open(path))
    else:
        status, body = http_get("https://www.sec.gov/files/company_tickers.json",
                                {"User-Agent": SEC_UA})
        if status == 200:
            data = json.loads(body)
            json.dump(data, open(path, "w"))
        else:
            local = os.path.join(PROJ, "data", "sec_company_tickers.json")
            if os.path.exists(local):
                print("SEC ticker map fetch failed; using local cached copy", flush=True)
                data = json.load(open(local))
    if data is None:
        raise RuntimeError("no ticker->CIK map available")
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


_last_sec = [0.0]


def fetch_companyfacts(cik):
    cache = os.path.join(SEC_CACHE, f"CIK{cik:010d}.json")
    if os.path.exists(cache):
        try:
            return json.load(open(cache))
        except Exception:  # noqa: BLE001
            pass
    wait = SEC_MIN_INTERVAL - (time.time() - _last_sec[0])
    if wait > 0:
        time.sleep(wait)
    _last_sec[0] = time.time()
    status, body = http_get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        {"User-Agent": SEC_UA, "Accept-Encoding": "identity"})
    if status != 200 or not body:
        json.dump({"_miss": status}, open(cache, "w"))
        return {"_miss": status}
    try:
        data = json.loads(body)
    except Exception:  # noqa: BLE001
        json.dump({"_miss": "parse"}, open(cache, "w"))
        return {"_miss": "parse"}
    # slim before caching: keep only the two concepts we need
    slim = {"facts": {}}
    facts = data.get("facts", {})
    for taxo, concept in (("dei", "EntityCommonStockSharesOutstanding"),
                          ("us-gaap", "CommonStockSharesOutstanding")):
        c = facts.get(taxo, {}).get(concept)
        if c:
            slim["facts"].setdefault(taxo, {})[concept] = c
    json.dump(slim, open(cache, "w"))
    return slim


def so_entries(factdoc):
    """Return sorted list of (end_date_str, val, filed) share-count facts."""
    out = []
    facts = factdoc.get("facts", {})
    for taxo, concept in (("dei", "EntityCommonStockSharesOutstanding"),
                          ("us-gaap", "CommonStockSharesOutstanding")):
        c = facts.get(taxo, {}).get(concept)
        if not c:
            continue
        for unit, entries in c.get("units", {}).items():
            if unit != "shares":
                continue
            for e in entries:
                if e.get("end") and e.get("val") is not None:
                    out.append((e["end"], float(e["val"]), e.get("filed") or ""))
        if out:
            break  # prefer dei; fall back to us-gaap only if dei absent
    out.sort()
    return out


def so_asof(entries, event_day):
    """Latest fact end <= event_day. Multi-class approximation: among entries at
    the chosen end, keep the most recently filed, sum distinct values."""
    eligible = [e for e in entries if e[0] <= event_day]
    if not eligible:
        return None, None
    best_end = eligible[-1][0]
    at_end = [e for e in eligible if e[0] == best_end]
    max_filed = max(e[2] for e in at_end)
    vals = sorted({e[1] for e in at_end if e[2] == max_filed})
    return sum(vals), best_end


# ---------------------------------------------------------------- Alpaca
_feed_used = ["sip"]


def _fetch_bars_batch(batch, headers):
    """Fetch one symbol batch, paginated. On 400/422 (bad symbol poisons the
    request) split the batch and retry halves; drop single bad symbols."""
    got = {}
    feed = _feed_used[0]
    token = None
    from urllib.parse import quote
    while True:
        url = ("https://data.alpaca.markets/v2/stocks/bars?symbols="
               + ",".join(quote(s, safe="") for s in batch)
               + f"&timeframe=1Day&adjustment=all&feed={feed}"
               + f"&start={BARS_START}&end={BARS_END}&limit=10000&sort=asc")
        if token:
            url += f"&page_token={token}"
        status, body = http_get(url, headers)
        if status == 403 and feed == "sip":
            feed = _feed_used[0] = "iex"
            print("  SIP denied -> falling back to IEX feed", flush=True)
            continue
        if status in (400, 422):
            if len(batch) == 1:
                print(f"  bars: symbol rejected ({batch[0]})", flush=True)
                return got
            mid = len(batch) // 2
            got.update(_fetch_bars_batch(batch[:mid], headers))
            got.update(_fetch_bars_batch(batch[mid:], headers))
            return got
        if status != 200 or not body:
            print(f"  bars batch failed status={status} ({batch[0]}..)", flush=True)
            return got
        data = json.loads(body)
        for sym, blist in (data.get("bars") or {}).items():
            got.setdefault(sym, []).extend(
                [b["t"][:10], b["o"], b["c"], b["v"]] for b in blist)
        token = data.get("next_page_token")
        if not token:
            return got
        time.sleep(0.35)


def fetch_bars(tickers):
    """Return {sym: [(date, open, close, volume), ...]} sorted asc, plus feed used."""
    headers = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
               "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET"]}
    bars = {}
    syms = sorted(set(tickers))
    for i in range(0, len(syms), 50):
        batch = syms[i:i + 50]
        cache = os.path.join(BARS_CACHE, f"batch_{i:04d}_{len(syms)}.json")
        got = None
        if os.path.exists(cache):
            got = json.load(open(cache))
            if not got:          # earlier failed run cached an empty batch
                got = None
        if got is None:
            got = _fetch_bars_batch(batch, headers)
            json.dump(got, open(cache, "w"))
            time.sleep(0.35)  # stay well under Alpaca rate limits
        for sym, blist in got.items():
            bars[sym] = sorted(blist)
        done = min(i + 50, len(syms))
        print(f"  bars: {done}/{len(syms)} symbols", flush=True)
    return bars, _feed_used[0]


# ---------------------------------------------------------------- features
def compute_features(ev, sym_bars):
    """All bar-derived features for one event. Returns dict of columns."""
    out = {"prior_rvol": None, "gap_pct": None, "eod_rvol": None,
           "dollar_vol_prev": None, "n_trailing_bars": 0,
           "event_day_is_trading": None, "effective_day": None}
    if not sym_bars:
        return out
    dates = [b[0] for b in sym_bars]
    # effective event trading day = first bar date >= event_day (within 5 cal days)
    idx = None
    for j, d in enumerate(dates):
        if d >= ev["event_day"]:
            idx = j
            break
    if idx is None:
        return out
    d_eff = date.fromisoformat(dates[idx])
    d_evt = date.fromisoformat(ev["event_day"])
    if (d_eff - d_evt).days > 5:
        return out
    out["effective_day"] = dates[idx]
    out["event_day_is_trading"] = 1 if dates[idx] == ev["event_day"] else 0
    if idx < 1:
        return out
    prev = sym_bars[idx - 1]           # day-1
    evt = sym_bars[idx]                # event day
    window = sym_bars[max(0, idx - 21): idx - 1]   # day-21 .. day-2
    out["n_trailing_bars"] = len(window)
    out["dollar_vol_prev"] = prev[2] * prev[3]
    if prev[2] and prev[2] > 0:
        out["gap_pct"] = (evt[1] / prev[2] - 1.0) * 100.0
    if len(window) >= 10:
        avg_vol = sum(b[3] for b in window) / len(window)
        if avg_vol > 0:
            out["prior_rvol"] = prev[3] / avg_vol
            out["eod_rvol"] = evt[3] / avg_vol   # END-OF-DAY knowledge only
    return out


def main():
    load_env()
    t0 = time.time()
    events = build_sample()
    tickers = sorted({e["ticker"] for e in events})
    print(f"unique tickers: {len(tickers)}", flush=True)

    # --- SEC shares outstanding
    cik_map = load_cik_map()
    no_cik, no_facts = [], []
    ticker_so = {}   # ticker -> (entries, cik)
    for k, t in enumerate(tickers):
        cik = cik_map.get(t.upper())
        if cik is None:
            no_cik.append(t)
            continue
        doc = fetch_companyfacts(cik)
        entries = so_entries(doc)
        if not entries:
            no_facts.append(t)
        ticker_so[t] = (entries, cik)
        if (k + 1) % 50 == 0:
            print(f"  SEC facts: {k + 1}/{len(tickers)}", flush=True)
    print(f"SEC: no_cik={len(no_cik)} no_facts={len(no_facts)}", flush=True)

    # --- Alpaca bars (single fixed window covers every event)
    bars, feed_used = fetch_bars(tickers)
    print(f"bars fetched for {len(bars)}/{len(tickers)} symbols (feed={feed_used})", flush=True)

    # --- assemble rows
    rows = []
    for ev in events:
        f = compute_features(ev, bars.get(ev["ticker"]))
        so_val = so_date = staleness = stale_flag = None
        cik = None
        if ev["ticker"] in ticker_so:
            entries, cik = ticker_so[ev["ticker"]]
            so_val, so_date = so_asof(entries, ev["event_day"])
            if so_val is not None:
                staleness = (date.fromisoformat(ev["event_day"])
                             - date.fromisoformat(so_date)).days
                stale_flag = 1 if staleness > 120 else 0
        bucket = None
        if so_val is not None:
            bucket = "<20M" if so_val < 20e6 else ("20-100M" if so_val <= 100e6 else ">100M")
        rows.append((
            ev["ticker"], ev["event_day"], ev["label"], ev["r5"], ev["px"],
            str(cik) if cik else None, so_val, so_date, staleness, stale_flag, bucket,
            f["prior_rvol"], f["gap_pct"], f["eod_rvol"], f["dollar_vol_prev"],
            f["n_trailing_bars"], f["event_day_is_trading"], f["effective_day"],
            feed_used,
        ))

    con = sqlite3.connect(RESEARCH_DB)
    con.execute("DROP TABLE IF EXISTS event_features")
    con.execute("""
        CREATE TABLE event_features (
            ticker TEXT NOT NULL,
            event_day TEXT NOT NULL,          -- (ticker,event_day) = one deduped live-timed event
            label TEXT NOT NULL,              -- bang r5>=50 / crash r5<=-20 / meh (stride-sampled control)
            r5 REAL,                          -- 5d return pct from signal_outcomes
            px REAL,                          -- price_at_flag
            cik TEXT,
            so REAL,                          -- shares outstanding as-of latest SEC fact end <= event_day
            so_asof TEXT,                     -- fact end date used
            so_staleness_days INTEGER,
            so_stale_flag INTEGER,            -- 1 = nearest fact >120d old
            float_bucket TEXT,                -- <20M / 20-100M / >100M (shares outstanding proxy for float)
            prior_rvol REAL,                  -- EX-ANTE: vol(day-1)/avg vol(day-21..day-2)
            gap_pct REAL,                     -- EX-ANTE at the open: open(event_day)/close(day-1)-1, pct
            eod_rvol REAL,                    -- END-OF-DAY KNOWLEDGE ONLY: vol(event_day)/trailing avg
            dollar_vol_prev REAL,             -- close(day-1)*vol(day-1)
            n_trailing_bars INTEGER,          -- bars in day-21..day-2 window (rvol requires >=10)
            event_day_is_trading INTEGER,     -- 0 = event on weekend/holiday, gap uses next trading day open
            effective_day TEXT,               -- trading day used as event day for bar features
            bars_feed TEXT,                   -- sip or iex
            PRIMARY KEY (ticker, event_day)
        )""")
    con.executemany(
        "INSERT INTO event_features VALUES (" + ",".join("?" * 19) + ")", rows)
    con.commit()

    # --- coverage report
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    n = len(rows)
    print("\n=== COVERAGE ===", flush=True)
    print(f"rows written: {n}")
    for col in ("so", "prior_rvol", "gap_pct", "eod_rvol"):
        c = q(f"SELECT COUNT(*) FROM event_features WHERE {col} IS NOT NULL")
        print(f"  {col:12s}: {c}/{n} ({100.0 * c / n:.1f}%)")
    print(f"  so_stale>120d: {q('SELECT COUNT(*) FROM event_features WHERE so_stale_flag=1')}")
    print(f"  non-trading event days: {q('SELECT COUNT(*) FROM event_features WHERE event_day_is_trading=0')}")
    print(f"  distinct event days: {q('SELECT COUNT(DISTINCT event_day) FROM event_features')}")
    for lbl in ("bang", "crash", "meh"):
        c = q(f"SELECT COUNT(*) FROM event_features WHERE label='{lbl}'")
        cb = q(f"SELECT COUNT(*) FROM event_features WHERE label='{lbl}' AND prior_rvol IS NOT NULL AND so IS NOT NULL")
        print(f"  {lbl:5s}: n={c}  fully-featured(so+prior_rvol)={cb}")
    con.close()
    print(f"\nno_cik tickers ({len(no_cik)}): {','.join(no_cik[:40])}", flush=True)
    print(f"no_facts tickers ({len(no_facts)}): {','.join(no_facts[:40])}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
