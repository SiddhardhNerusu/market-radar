# Insider lane — runbook

The whole trading system is now one long-only lane: large open-market insider
purchases (Form 4, code P), held for days with a broker-side stop. It exists to
answer one question honestly, on a forward record: **does this signal have an
edge a retail account can trade?** Until the promotion gate says yes it stays on
paper, sized small.

## What the lane does each minute

1. Snapshot account and positions from Alpaca. If that fails, nothing happens.
2. Day rollover: persist day-start equity.
3. Daily-loss kill switch: equity down `LANE_DAILY_LOSS_PCT` of the book since
   the day start → cancel all orders, flatten every position through the
   gateway (side from the live position sign), halt entries for the day.
4. Recover any `pending_submit` decision by asking the broker for its
   `client_order_id` (idempotent across crashes).
5. Reconcile the lane book: parent fills, bracket exits (stop/target), unfilled
   parents past `LANE_ENTRY_TTL_MINUTES`.
6. Time stop: positions open longer than `LANE_MAX_HOLD_CALENDAR_DAYS` are
   closed (legs cancelled first, then a gateway CLOSE).
7. Entries, only in regular hours, after the first 15 min and before the last 30:
   deduped Form 4 candidates from the last `LANE_FRESHNESS_HOURS` → IEX quote →
   spread check → ATR(14) → risk-sized bracket (`qty = 0.5% of book / (2×ATR)`)
   → gateway (long-only, hard / per-symbol / gross caps) → `lane_positions`.
   **Every qualifying candidate is written to `edge_ledger` whether or not it
   was traded.**

## Running

```bash
# the daemon must be running: it ingests SEC EDGAR, hydrates Form 4 bodies and
# parses them into insider_transactions every minute
PYTHONPATH=src .venv/bin/python scripts/run_daemon.py

# see what the lane would do right now, submit nothing
PYTHONPATH=src .venv/bin/python scripts/run_insider_lane.py --paper --dry-run --once

# paper trade (foreground)
PYTHONPATH=src .venv/bin/python scripts/run_insider_lane.py --paper

# or as launchd jobs (lane + nightly ledger resolve)
bash scripts/install_insider_lane_launchd.sh
```

Set `LANE_BOOK_EQUITY_USD` in `.env` to the capital you actually have.

## Reading the forward record

```bash
PYTHONPATH=src .venv/bin/python scripts/edge_ledger_update.py            # resolve 5d/20d, print gate
PYTHONPATH=src .venv/bin/python scripts/edge_ledger_update.py --horizon 20d --families 2
```

The promotion gate passes only when, net of the price-tier round-trip cost and
SPY-adjusted: ≥ 15 distinct event days over ≥ 2 weeks, median > 0, 10%-trimmed
mean > 0, mean after dropping the top 10% of winners > 0, early and late halves
both > 0, and a day-cluster bootstrap p < 0.05 / families. Nothing in the lane
sizes against the ledger; the gate is a human decision input.

## Historical check

```bash
PYTHONPATH=src .venv/bin/python scripts/event_study_form4.py --since 2025-01-01 --horizons 5,20 --families 3
```

Anchored on `published_at`, one event per (ticker, day), SPY-adjusted,
ticker-matched placebo permutation, Bonferroni in code. Needs the SQLite DB with
`insider_transactions` populated (`scripts/refresh_insider_transactions.py`).

## Kill switches

- `LANE_DRY_RUN=1` in `.env` and restart: decisions logged, nothing submitted.
- `launchctl unload ~/Library/LaunchAgents/com.marketradar.insiderlane.plist`.
- Positions keep their broker-side stop and target even with the lane stopped.

## What is deliberately gone

Options, crypto, shorts, the price-action scanner, RSS/Reddit/StockTwits
ingestion, Trading 212, the 4,857-line `live_trader.py`, Kelly sizing fed a
hard-coded probability, confluence/regime multipliers that the caps made
no-ops, the daily trailing profit-take, and the EOD flatten. Each either had
measured negative edge or did nothing.
