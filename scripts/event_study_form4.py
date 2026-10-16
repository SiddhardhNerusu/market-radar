"""Published_at-anchored event study for large insider open-market purchases.

Replaces the scored_at-anchored, universe-demeaned, sticky-window analysis in
docs/EDGE_HUNT_2026-06-16.md with one that cannot flatter the signal:

  * events anchored on raw_signals.published_at (when the Form 4 went public)
  * one event per (ticker, event_day); same-day purchases aggregated
  * entry at the first regular-hours minute close >= published_at + latency
    (Alpaca IEX minute bars) or, without Alpaca keys / --no-minute, the
    event-day close (OPTIMISTIC for same-day filings, printed as such)
  * returns to the close of the Nth trading day after entry, minus SPY over
    the identical window (market-adjusted, beta=1), minus the round-trip
    cost tier from market_radar.backtest.replay
  * robust summary (median, trim10, drop-top-k with k = 10% of n),
    early/late split on distinct days, ticker-matched placebo permutation,
    day-cluster bootstrap CI, and the promotion gauntlet with Bonferroni
    applied in code (--families N)

Usage:
  PYTHONPATH=src .venv/bin/python scripts/event_study_form4.py \
      --since 2025-01-01 --horizons 5,20 --families 3 --out results/event_study_form4.json

Without ALPACA_API_KEY/ALPACA_API_SECRET the script falls back to yfinance
daily bars: minute-level entry is then unavailable and every event enters at
the event-day close. Bars are cached under data/event_study_cache/ (gitignored).
Exits with a clear message (no traceback) when the DB or the
insider_transactions table is missing.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.config import CONFIG  # noqa: E402
from market_radar.research import event_study as es  # noqa: E402
from market_radar.storage import get_connection  # noqa: E402

log = logging.getLogger("event_study_form4")
CACHE_DIR = ROOT / "data" / "event_study_cache"
MIN_DAYS = 15


# ---------------------------------------------------------------------------
# Bar source with on-disk cache
# ---------------------------------------------------------------------------
class BarSource:
    """Daily + minute bars from Alpaca (IEX, split/div adjusted) with a
    yfinance daily fallback when credentials are absent. Every fetch is
    cached as JSON under data/event_study_cache/ so re-runs are free."""

    def __init__(self, *, use_minute: bool):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.source = "yfinance"
        try:
            # Lazy + guarded: market_radar.execution.__init__ may import
            # modules that are being removed, and keys may be absent.
            from market_radar.execution.alpaca_client import AlpacaClient
            self.client = AlpacaClient()
            self.source = "alpaca"
        except Exception as exc:  # noqa: BLE001 - any failure -> fallback
            log.warning("Alpaca unavailable (%s: %s). Falling back to yfinance "
                        "daily bars; minute-level entry is NOT available, every "
                        "event will enter at the event-day close.",
                        type(exc).__name__, exc)
        self.use_minute = use_minute and self.client is not None
        self.minute_failures = 0

    # -- daily ---------------------------------------------------------------
    def daily(self, symbol: str, start: date, end: date) -> list[dict]:
        cache = CACHE_DIR / f"{symbol}_1Day_{start}_{end}.json"
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except json.JSONDecodeError:
                cache.unlink()
        if self.client is not None:
            bars = self.client.get_daily_bars(symbol, start.isoformat(), end.isoformat(), limit=10000)
            bars = [{"t": b["t"], "o": b.get("o"), "h": b.get("h"), "l": b.get("l"),
                     "c": b["c"], "v": b.get("v")} for b in bars]
        else:
            bars = self._yf_daily(symbol, start, end)
        if bars:
            cache.write_text(json.dumps(bars))
        return bars

    @staticmethod
    def _yf_daily(symbol: str, start: date, end: date) -> list[dict]:
        try:
            import yfinance as yf
        except ImportError:
            log.error("yfinance not installed and no Alpaca keys: no price source")
            return []
        try:
            df = yf.download(symbol, start=start.isoformat(),
                             end=(end + timedelta(days=1)).isoformat(),
                             auto_adjust=True, progress=False, threads=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("yfinance download(%s) failed: %s", symbol, exc)
            return []
        if df is None or len(df) == 0:
            return []
        if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
            try:
                df = df.xs(symbol, axis=1, level=-1)
            except KeyError:
                df.columns = [c[0] for c in df.columns]
        out = []
        for idx, row in df.iterrows():
            c = row.get("Close")
            if c is None or (isinstance(c, float) and math.isnan(c)):
                continue
            out.append({"t": idx.strftime("%Y-%m-%d"), "o": float(row.get("Open", c)),
                        "h": float(row.get("High", c)), "l": float(row.get("Low", c)),
                        "c": float(c), "v": float(row.get("Volume", 0) or 0)})
        return out

    # -- minute --------------------------------------------------------------
    def minute(self, symbol: str, day: date) -> list[dict]:
        """Regular-session minute bars for one ET date (pages through
        next_page_token). Empty list when unavailable."""
        if not self.use_minute:
            return []
        cache = CACHE_DIR / f"{symbol}_1Min_{day}.json"
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except json.JSONDecodeError:
                cache.unlink()
        start = datetime.combine(day, es.REGULAR_OPEN, tzinfo=es.ET).astimezone(timezone.utc)
        end = datetime.combine(day, es.REGULAR_CLOSE, tzinfo=es.ET).astimezone(timezone.utc)
        params = {"timeframe": "1Min", "start": start.isoformat().replace("+00:00", "Z"),
                  "end": end.isoformat().replace("+00:00", "Z"), "limit": 10000,
                  "adjustment": "all", "feed": "iex"}
        bars: list[dict] = []
        token: Optional[str] = None
        try:
            for _ in range(20):
                p = dict(params)
                if token:
                    p["page_token"] = token
                d = self.client._request("GET", f"/v2/stocks/{symbol}/bars", params=p,
                                         base=self.client.data_base_url) or {}
                bars.extend({"t": b["t"], "c": b["c"], "v": b.get("v")} for b in (d.get("bars") or []))
                token = d.get("next_page_token")
                if not token:
                    break
        except Exception as exc:  # noqa: BLE001
            self.minute_failures += 1
            log.warning("minute bars %s %s failed: %s", symbol, day, exc)
            return []
        cache.write_text(json.dumps(bars))
        return bars


# ---------------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------------
def load_rows(since: date) -> Optional[list[dict]]:
    """Code-P acquired rows joined to the filing's published_at. None (with a
    printed reason) when the DB or table is missing."""
    db_path = pathlib.Path(CONFIG.db_path)
    if not db_path.exists():
        print(f"No database at {db_path} (data/ is local-only). Nothing to study.")
        return None
    with get_connection() as conn:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='insider_transactions'"
        ).fetchone()
        if not has_table:
            print("Table insider_transactions is missing. Run "
                  "scripts/refresh_insider_transactions.py first.")
            return None
        cols = {r[1] for r in conn.execute("PRAGMA table_info(insider_transactions)")}
        plan_col = "ix.is_10b5_1" if "is_10b5_1" in cols else "NULL"
        if "is_10b5_1" not in cols:
            print("NOTE: insider_transactions has no is_10b5_1 column; plan trades "
                  "cannot be excluded (treated as unknown).")
        rows = conn.execute(
            f"""SELECT ix.ticker, ix.transaction_code, ix.is_acquired, ix.shares,
                       ix.price, ix.role_score, {plan_col} AS is_10b5_1,
                       rs.published_at, rs.id AS signal_id
                FROM insider_transactions ix
                JOIN raw_signals rs ON rs.id = ix.signal_id
                WHERE ix.transaction_code = 'P' AND COALESCE(ix.is_acquired, 0) = 1
                  AND rs.published_at IS NOT NULL
                  AND substr(rs.published_at, 1, 10) >= ?
                  AND rs.source LIKE 'sec_edgar%'""",
            (since.isoformat(),),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _fmt(v, nd=2, width=8) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'nan':>{width}}"
    return f"{v:{width}.{nd}f}"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy scalars
        return _jsonable(obj.item())
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", type=date.fromisoformat, default=date(2025, 1, 1))
    ap.add_argument("--min-notional", type=float, default=25_000.0)
    ap.add_argument("--min-price", type=float, default=10.0)
    ap.add_argument("--horizons", default="1,5,20")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--latency-minutes", type=int, default=2)
    ap.add_argument("--out", default="results/event_study_form4.json")
    ap.add_argument("--families", type=int, default=1, help="Bonferroni divisor: number of "
                    "signal families / cuts examined before this one")
    ap.add_argument("--placebo-days", type=int, default=60, help="trading days around each "
                    "ticker's events used for its placebo pool")
    ap.add_argument("--no-minute", action="store_true", help="skip minute bars; enter at close")
    ap.add_argument("--senior-only", action="store_true")
    ap.add_argument("--include-10b5-1", action="store_true")
    ap.add_argument("--n-iter", type=int, default=5000)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    horizons = tuple(sorted({int(h) for h in args.horizons.split(",") if h.strip()}))
    if not horizons:
        print("--horizons must name at least one positive integer")
        return 2

    rows = load_rows(args.since)
    if rows is None:
        return 1
    print(f"insider_transactions code-P acquired rows since {args.since}: {len(rows)}")
    if not rows:
        print("No rows: nothing to study. (Is refresh_insider_transactions.py up to date?)")
        return 0

    bars = BarSource(use_minute=not args.no_minute)
    today = datetime.now(timezone.utc).astimezone(es.ET).date()
    start = args.since - timedelta(days=int(args.placebo_days * 1.6) + 10)
    bench_daily = bars.daily(args.benchmark, start, today)
    if not bench_daily:
        print(f"Could not load {args.benchmark} daily bars from {bars.source}; cannot proceed.")
        return 1
    calendar = es.trading_days_from_bars(bench_daily)
    d = max(calendar)
    for _ in range(400):  # extend with the rule-based calendar beyond the last bar
        d += timedelta(days=1)
        if es.is_trading_day(d):
            calendar.add(d)

    events = es.select_insider_events(
        rows, min_notional_usd=args.min_notional, min_price=args.min_price,
        exclude_10b5_1=not args.include_10b5_1, senior_only=args.senior_only,
        trading_days=calendar,
    )
    last_bar_day = max(es.trading_days_from_bars(bench_daily))
    max_h = max(horizons)
    events = [e for e in events if e.event_day <= last_bar_day]
    print(f"deduplicated events (one per ticker-day): {len(events)} over "
          f"{len({e.event_day for e in events})} distinct days, "
          f"{len({e.ticker for e in events})} tickers; price source={bars.source}, "
          f"minute entry={'on' if bars.use_minute else 'OFF (event-day close, optimistic for same-day filings)'}")
    if not events:
        print("No events survive the filters. Nothing to test.")
        return 0

    ticker_daily: dict[str, list[dict]] = {}
    for t in sorted({e.ticker for e in events}):
        tb = bars.daily(t, start, today)
        if tb:
            ticker_daily[t] = tb
        else:
            log.warning("no daily bars for %s; its events are dropped", t)

    per_event: list[dict] = []
    skipped_entry = 0
    bench_minute_cache: dict[date, list[dict]] = {}
    for ev in events:
        tb = ticker_daily.get(ev.ticker)
        if not tb:
            continue
        mins = bars.minute(ev.ticker, ev.event_day)
        entry = es.entry_price_for(ev, mins, tb, latency_minutes=args.latency_minutes)
        if entry is None:
            skipped_entry += 1
            continue
        entry_ts, entry_px = entry
        entry_mode = "minute" if mins else "close"
        if ev.event_day not in bench_minute_cache:
            bench_minute_cache[ev.event_day] = bars.minute(args.benchmark, ev.event_day) if mins else []
        bench_entry = es.entry_price_for(
            es.Event(ticker=args.benchmark, event_ts=ev.event_ts, event_day=ev.event_day,
                     notional=0.0, role_score=0),
            bench_minute_cache[ev.event_day], bench_daily, latency_minutes=args.latency_minutes,
        )
        if bench_entry is None:
            skipped_entry += 1
            continue
        b_ts, b_px = bench_entry
        raw = es.forward_returns(entry_px, entry_ts, tb, horizons)
        bench = es.forward_returns(b_px, b_ts, bench_daily, horizons)
        abn = es.abnormal_returns(raw, bench, beta=args.beta)
        net = {h: (None if a is None else es.net_of_cost(a, entry_px)) for h, a in abn.items()}
        per_event.append({
            "ticker": ev.ticker, "event_day": ev.event_day, "event_ts": ev.event_ts,
            "time_known": ev.time_known, "notional": ev.notional, "role_score": ev.role_score,
            "n_filings": ev.n_filings, "entry_ts": entry_ts, "entry_px": entry_px,
            "entry_mode": entry_mode, "bench_entry_px": b_px,
            "raw": raw, "bench": bench, "abnormal": abn, "net": net,
            "cost_pct": 100.0 * es._round_trip_cost_frac(entry_px),
        })
    print(f"events with an entry price: {len(per_event)} (no entry: {skipped_entry}; "
          f"minute-entry fills: {sum(1 for p in per_event if p['entry_mode'] == 'minute')})")

    event_days_by_ticker: dict[str, list[date]] = {}
    for p in per_event:
        event_days_by_ticker.setdefault(p["ticker"], []).append(p["event_day"])

    results: dict[str, dict] = {}
    hdr = (f"{'h':>3} {'n':>4} {'days':>4} {'mean%':>8} {'med%':>8} {'trim10%':>8} {'dropK%':>8} {'K':>3} "
           f"{'win%':>6} {'early%':>8} {'late%':>8} {'perm_p':>7} {'ci_lo':>7} {'ci_hi':>7}  verdict")
    print("\nNET of cost, market-adjusted (SPY, beta=%.1f), per deduplicated event" % args.beta)
    print(hdr)
    print("-" * len(hdr))
    for h in horizons:
        ers = [
            es.EventReturn(ticker=p["ticker"], event_day=p["event_day"], horizon=h,
                           raw=p["raw"][h], abnormal=p["abnormal"][h], net=p["net"][h],
                           entry_px=p["entry_px"])
            for p in per_event if p["net"].get(h) is not None
        ]
        pool = es.build_placebo_pool(ticker_daily, bench_daily, h, event_days_by_ticker,
                                     beta=args.beta, window_days=args.placebo_days)
        g = es.gauntlet(ers, pool, n_families_tested=args.families, min_days=MIN_DAYS,
                        n_iter=args.n_iter, seed=0)
        s, c = g["summary"], g["criteria"]
        if g["passed"]:
            verdict = "*** PASSES GAUNTLET ***"
        else:
            failed = [k for k, v in c.items() if not v["pass"]]
            verdict = "FAIL: " + ",".join(failed)
        print(f"{h:3d} {s['n']:4d} {g['n_days']:4d} {_fmt(s['mean'])} {_fmt(s['median'])} "
              f"{_fmt(s['trim10_mean'])} {_fmt(s['drop_top_k_mean'])} {s['drop_top_k']:3d} "
              f"{_fmt(100 * s['win_rate'], 1, 6)} {_fmt(g['early_mean'])} {_fmt(g['late_mean'])} "
              f"{_fmt(g['permutation']['p_value'], 4, 7)} {_fmt(g['bootstrap']['ci_low'], 2, 7)} "
              f"{_fmt(g['bootstrap']['ci_high'], 2, 7)}  {verdict}")
        if g["underpowered"]:
            print(f"    UNDERPOWERED at h={h}: only {g['n_days']} distinct event days "
                  f"(< {MIN_DAYS}). Nothing at this horizon is conclusive either way.")
        if g["permutation"]["n_excluded"]:
            print(f"    permutation: {g['permutation']['n_excluded']} events had no placebo "
                  f"pool (too few bars) and were excluded from the test")
        results[str(h)] = {"gauntlet": g, "placebo_pool_sizes": {t: len(v) for t, v in pool.items()}}

    print(f"\nBonferroni: alpha 0.05 / {args.families} families = {0.05 / max(1, args.families):.4f}. "
          "perm_p is the ticker-matched placebo p (pre-cost abnormal vs random non-event days "
          "of the same ticker). ci = day-cluster bootstrap 95% CI of the net mean.")
    if not bars.use_minute:
        print("Entry = event-day close for every event (no minute bars): same-day filings "
              "are filled with zero latency at the official close, which is OPTIMISTIC.")
    print("Returns are close-to-close at the horizon; intraday path (stops) is not modelled. "
          "IEX minute bars are a thin feed. The placebo pool shares the regime with the events.")

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: (v.isoformat() if isinstance(v, date) else v) for k, v in vars(args).items()},
        "price_source": bars.source,
        "minute_entry": bars.use_minute,
        "n_rows": len(rows),
        "n_events": len(events),
        "n_events_with_entry": len(per_event),
        "n_skipped_no_entry": skipped_entry,
        "horizons": {h: results[str(h)] for h in results},
        "events": per_event,
    }
    out.write_text(json.dumps(_jsonable(payload), indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
