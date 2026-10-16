# Market Radar

[![tests](https://github.com/SiddhardhNerusu/market-radar/actions/workflows/tests.yml/badge.svg)](https://github.com/SiddhardhNerusu/market-radar/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11-blue)
![paper trading only](https://img.shields.io/badge/trading-paper%20only-orange)

An end-to-end quantitative research and automated paper-trading system, built independantly over about six weeks in 2026.
It ingests market-moving events from 80+ sources, classifies them with an LLM, scores them with a
calibrated ML ensemble, measures the real forward returns of every signal, and (on a broker **paper**
account only) sizes and executes trades under a rule-based risk manager.

**The headline result is a negative one, and I think it is the most valuable thing in the repo.**
After building the full pipeline I ran every signal family through a seven-test statistical gauntlet
(dedup, market-beta removal, transaction costs, out-of-sample split, outlier removal, permutation tests,
Bonferroni correction). The paper-trading bot lost money over 26 trading days. a system that handles money has to be able to tell you when it has no edge. 

> **This has never traded real money.** Every order in this repository was placed against an Alpaca
> paper account.

---

## What it does now (October 2026: the insider lane)

After the research below found no edge in any catalyst family, the system was
narrowed to the one candidate that was theory-backed and positive at the median,
built so that a forward record can settle it:

```
  SEC EDGAR Form 4 ──► sec_edgar / sec_hydrate ──► insider_transactions (every minute)
                                                          │
                    ┌─────────────────────────────────────▼──────────────────────────┐
                    │ INSIDER LANE  (src/market_radar/lane/, 60 s loop, paper only)   │
                    │  code-P open-market buys ≥ $25k, ≥ $10 names, no 10b5-1 plans   │
                    │  one bet per (ticker, event_day), anchored on published_at      │
                    │  ─► IEX quote, spread cap ─► ATR(14) ─► 0.5% of book at risk   │
                    │  ─► order gateway (long-only, hard/per-symbol/gross caps)       │
                    │  ─► BRACKET at the broker: 2×ATR stop, 2R target, GTC           │
                    │  ─► time stop 14 d, 3% daily-loss FLATTEN, ≤5 names, 5% heat   │
                    │  ─► edge_ledger row for EVERY qualifying event, traded or not   │
                    └────────────────────────────────────────────────────────────────┘
                                                          │
             scripts/edge_ledger_update.py ─► 5d/20d SPY-adjusted returns ─► promotion gate
             scripts/event_study_form4.py  ─► published_at-anchored historical event study
```

- `lane/trader.py` is ~470 lines and replaces the 4,857-line `live_trader.py`.
- Sizing is risk-based (`qty = risk$ / stop distance`), not Kelly fed a hard-coded p.
- The stop exists at the broker from the moment of fill. The old bot's configured
  ATR stop was never enforced during regular hours.
- Dedup and re-entry guards are database rows, not in-memory dicts lost on restart.
- Submits are idempotent via a deterministic `client_order_id` recovered from the broker.
- The promotion gate (`lane/promotion.py`) is code: ≥15 distinct days, median > 0,
  trimmed mean > 0, drop-top-10% > 0, early and late halves > 0, day-cluster
  bootstrap p < 0.05 / families. See `docs/INSIDER_LANE.md`.

Removed in the same change: options, crypto, shorts, the price-action scanner,
RSS/Reddit/StockTwits ingestion, Trading 212, and ~10k lines with them.

### Still in the repo (the research platform)
- **SEC EDGAR** parser for 8-K (item codes), 13D/G, S-1, Form 4 (10b5-1 detection) and 6-K,
  with body hydration and CIK→ticker lookup; ClinicalTrials.gov, FDA calendar, earnings
  calendar, FINRA short interest, Nasdaq halts, market movers, Alpaca news.
- **Near-duplicate collapse** with SimHash + LSH (`dedup/near_dup.py`).
- **LLM classification** (Claude Haiku, tool-use JSON) behind a deterministic SEC pre-classifier.
- **ML** (`ml/`): stacked HGB + LightGBM, walk-forward CV, isotonic calibration, split-conformal
  bands, learning-to-rank with a data-volume deploy gate, drift monitoring. The lane does
  not consume any of it; it is kept as the research track, with its known leakage
  (same-day bars in features, train/serve skew) documented in `MARKET_RADAR_ASSESSMENT.md`.
- **Fills-based P&L reconciliation** (`execution/pnl_reconcile.py`): FIFO over the broker's
  actual fills; the equity curve is the only honest P&L.
- **Single order gateway** (`execution/order_gateway.py`): every open and close goes through
  one choke point with reduce-only-by-live-sign, long-only and hard caps. It exists because
  of the TRDA incident described below.

## Results, honestly

### Paper trading (Alpaca paper account, 27 May → 22 Jun 2026)

| | |
|---|---|
| Trading days | 26 |
| Round-trips | 219 (69 wins / 136 losses) |
| Account equity | $100,000 → $91,958 (−8.0 %) |
| Sizing | capped to a $6.3 k–$10 k notional book to mimic a ~£5 k real account |

The bot lost money. About $2 k of the loss landed on 29 May, about $5.3 k on 10–11 June in the flip-short incident described above, and the remaining weeks were roughly flat. That is consistent with the research finding below, which is that the signals it was trading had no measurable edge. 

### The edge gauntlet

`scripts/edge_gauntlet.py` and its siblings (`edge_*.py`) test every signal family (by source, by LLM event type, by 8-K item code, insider purchases, 13D activism, dilution shorts, attention momentum, seasonality) against all seven of these bars at once:

1. De-duplicate to one bet per (ticker, event-day).
2. Day-demean returns to strip market beta.
3. Net of a realistic round-trip cost, bucketed by share price.
4. Out-of-sample: earliest 70 % of days train, latest 30 % test; must be positive in both.
5. Drop the top five winners; must still be positive.
6. Permutation test (2,000 within-day shuffles), p < 0.05.
7. At least 15 distinct event days.

**Nothing passed.** Several families looked significant on paper (8-K item codes 1.01, 3.01, 3.02 and 5.03 showed negative drift with permutation p between 0.02 and 0.04), but none survived Bonferroni correction across the 14 families tested, and the live-only subsets flipped sign or depended on a handful of outliers. A promising "large insider open-market purchase" signal turned out to be pseudo-replication: a 30-day window flag was counting one Form 4 as up to seven bets, which shrank the permutation null and manufactured a p of 0.003; with strict de-duplication it was p ≈ 0.075 and went negative after dropping five winners.

The full write-ups are in `docs/EDGE_HUNT_2026-06-16.md` and `docs/AUDIT_AND_RESEARCH_2026-06-15.md`. The system was correctly gated: with no proven edge, the ranker's deploy gate served the composite fallback and the retrain gate rejected every new model that did not beat the incumbent.

---

## Repository layout

```
src/market_radar/
├── ingestors/     SEC EDGAR, FDA, clinical trials, FINRA, halts, news, Reddit, StockTwits ...
├── dedup/         SimHash + LSH near-duplicate clustering
├── llm/           Claude classifier, prompts, spend tracking
├── scoring/       composite score, heuristics, impact tables, source weights
├── ml/            features, ensemble, training (walk-forward), conformal, ranker, drift
├── outcomes/      forward-return tracking at 1d / 5d / 20d
├── signals/       macro regime
├── risk/          rule-based risk manager
├── execution/     Alpaca client, sizer (ATR), order gateway, P&L reconcile
├── lane/          THE INSIDER LANE: config, candidates, sizing, ledger, promotion gate, trader
├── research/      published_at-anchored event study (pure functions)
├── backtest/      replay harness with cost model
├── notifications/ Telegram + macOS
├── storage/       SQLite access layer (WAL, schema loading)
└── daemon.py      APScheduler job graph
scripts/           run_insider_lane.py, edge_ledger_update.py, event_study_form4.py, backfills, edge_*.py
tests/             pytest suite (runs without a database or broker keys)
dashboard/         Flask server + HTML
sql/               schema files
launchd/           macOS job definitions
docs/              audits, research notes, runbooks
```

## Running it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements-ci.txt for tests only
cp .env.example .env                     # fill in keys; leave blank to run the free/heuristic path
python scripts/init_db.py
PYTHONPATH=src python scripts/run_daemon.py
PYTHONPATH=src python scripts/run_insider_lane.py --paper --dry-run --once   # what would it do
PYTHONPATH=src python scripts/run_insider_lane.py --paper                    # paper account only
PYTHONPATH=src python scripts/edge_ledger_update.py                          # forward record + gate
python dashboard/server.py               # http://localhost:8765
pytest
```

The daemon, trader and dashboard all read `data/market_radar.db`; the database, model artefacts, logs and `.env` are git-ignored.

## Stack

Python 3.11 · SQLite (WAL) · scikit-learn · LightGBM · APScheduler · Flask · Anthropic SDK · Alpaca REST · yfinance · feedparser · pytest · GitHub Actions


