"""Fast batch outcome resolver — recovers the June holdout for the OOS test (#1).

The per-row resolver is Alpaca-rate-limited to ~1 req/s => days for 135k rows.
This fetches each ticker's daily bars ONCE (~6,300 calls, not 135k) and resolves
ALL that ticker's outcomes from the cached series, applying the SAME clean hygiene
as outcomes/tracker: floor sub-$1 anchors and clamp/flag |ret|>_MAX as data_corrupt.
Split/dividend-adjusted via alpaca_client adjustment='all'. Matches the tracker's
calendar-offset convention (anchor + N calendar days -> first trading close on/after).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/recover_holdout_fast.py [--max-tickers N]
"""
import argparse
import pathlib
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from market_radar.execution.alpaca_client import AlpacaClient
from market_radar.outcomes.tracker import _MAX_ABS_RETURN_PCT, _MIN_ANCHOR_USD
from market_radar.storage import get_connection

WINDOWS = [("price_1d", "return_1d_pct", 1),
           ("price_5d", "return_5d_pct", 5),
           ("price_20d", "return_20d_pct", 20)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    client = AlpacaClient()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, ticker, price_at_flag, price_at_flag_ts FROM signal_outcomes "
            "WHERE return_5d_pct IS NULL AND price_at_flag IS NOT NULL "
            "AND price_at_flag_ts IS NOT NULL"
        ).fetchall()
    by_ticker = defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(dict(r))
    tickers = list(by_ticker)
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    print(f"[fast] {len(rows)} outcomes / {len(by_ticker)} tickers "
          f"(processing {len(tickers)})", flush=True)

    done = 0
    for ti, ticker in enumerate(tickers):
        outs = by_ticker[ticker]
        anchors = []
        for o in outs:
            try:
                anchors.append(datetime.strptime(o["price_at_flag_ts"], "%Y-%m-%dT%H:%M:%SZ"))
            except (TypeError, ValueError):
                pass
        if not anchors:
            continue
        start = (min(anchors) - timedelta(days=3)).strftime("%Y-%m-%d")
        end = (max(anchors) + timedelta(days=32)).strftime("%Y-%m-%d")
        try:
            bars = client.get_daily_bars(ticker, start=start, end=end, limit=80)
        except Exception:  # noqa: BLE001 — skip unpriceable ticker
            continue
        series = sorted((str(b.get("t") or "")[:10], float(b.get("c", 0) or 0))
                        for b in bars if float(b.get("c", 0) or 0) > 0 and b.get("t"))
        if not series:
            continue

        with get_connection() as conn:
            for o in outs:
                try:
                    adt = datetime.strptime(o["price_at_flag_ts"], "%Y-%m-%dT%H:%M:%SZ")
                except (TypeError, ValueError):
                    continue
                anchor_px = o["price_at_flag"]
                corrupt = 1 if (anchor_px is None or anchor_px < _MIN_ANCHOR_USD) else 0
                sets, params, got = [], [], False
                for price_col, ret_col, wd in WINDOWS:
                    target = (adt + timedelta(days=wd)).strftime("%Y-%m-%d")
                    close = next((c for (d, c) in series if d >= target), None)
                    if close is None:
                        continue
                    ret = None
                    if not corrupt and anchor_px:
                        ret = (close - anchor_px) / anchor_px * 100.0
                        if abs(ret) > _MAX_ABS_RETURN_PCT:
                            corrupt, ret = 1, None
                    sets += [f"{price_col}=?", f"{ret_col}=?"]
                    params += [close, ret]
                    got = True
                if not got:
                    continue
                sets.append("data_corrupt=CASE WHEN ?=1 THEN 1 ELSE COALESCE(data_corrupt,0) END")
                params.append(corrupt)
                sets.append("resolve_attempts=0")
                params.append(o["id"])
                conn.execute(f"UPDATE signal_outcomes SET {', '.join(sets)} WHERE id=?", params)
                done += 1
        if ti % 200 == 0:
            print(f"[fast] {ti}/{len(tickers)} tickers, {done} outcomes resolved", flush=True)
    print(f"[fast] DONE: {done} outcomes resolved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
