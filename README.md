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
Bonferroni correction). The paper-trading bot lost money over 26 trading days. Both
facts are documented below. 

> **This has never traded real money.** Every order in this repository was placed against an Alpaca
> paper account.

---

## What it does

```
                         ┌──────────────────────────────────────────────────────────┐
                         │                 SIGNAL DAEMON  (launchd, 24/7)           │
  SEC EDGAR 8-K/13D/S-1/ │                                                          │
  Form 4/6-K ──────────► │  ingestors/  ─► raw_signals ─► dedup/ (SimHash + LSH)    │
  FDA, clinical trials,  │       │                              │                   │
  earnings, FINRA short  │       ▼                              ▼                   │
  interest, halts ─────► │  llm/  Claude Haiku classifier   scoring/  composite     │
  news RSS, Alpaca news, │  (event type, sentiment,          (source weight,        │
  Reddit, StockTwits ──► │   factual?, tool-use JSON)         corroboration,        │
                         │       │                            anti-pump heuristics) │
  1-min bars (85 stocks) │       ▼                              │                   │
  ──► price-action ────► │  ml/   stacked HGB + LightGBM ─► isotonic calibration    │
       scanner           │        walk-forward CV        ─► split-conformal bands   │
                         │        ─► selective gate  ─► signal_scores               │
                         │                                      │                   │
                         │  outcomes/  snapshot price at flag, re-check at          │
                         │             1d / 5d / 20d ─► signal_outcomes             │
                         └──────────────────────────────┬───────────────────────────┘
                                                        │ SQLite (WAL)
                         ┌──────────────────────────────▼───────────────────────────┐
                         │                 LIVE TRADER  (30 s loop, paper only)     │
                         │  snapshot broker state ─► reconcile orders/positions     │
                         │  ─► macro regime gate (VIX, SPY trend, yield curve)      │
                         │  ─► candidates ─► conformal gate ─► ATR stops            │
                         │  ─► fractional-Kelly sizing ─► 8-rule risk manager       │
                         │  ─► single order gateway ─► Alpaca (stock bracket /      │
                         │      vertical debit spread / crypto)                     │
                         │  ─► fills-based FIFO P&L reconciliation ─► Telegram      │
                         └──────────────────────────────────────────────────────────┘
                                                        │
                         Flask dashboard (localhost) ◄──┘  edge-research scripts (scripts/edge_*.py)
```

### Scale of the data it has processed

| | |
|---|---|
| Raw signals ingested | 1.24 M |
| Distinct sources | 81 |
| Signals scored | 711 k |
| Resolved 5-day forward outcomes | 426 k across 10.8 k tickers |
| LLM classifications | 10 k |
| Trader decisions logged (placed / blocked / no-quote, with reason) | 32 k |

---

## The interesting parts

### Signal ingestion
- **SEC EDGAR** parser for 8-K (with item-code extraction), 13D/G, S-1, Form 4 (insider trades, 10b5-1 plan detection) and 6-K foreign-issuer filings, with body hydration and CIK→ticker lookup.
- **ClinicalTrials.gov**, FDA calendar, earnings calendar, FINRA bi-monthly short interest, Nasdaq trading halts, market-movers scanner, RSS news wires, Alpaca news firehose, Reddit and StockTwits.
- **Near-duplicate collapse** with SimHash fingerprints and LSH-banded union-find clustering (`dedup/near_dup.py`), because exact hashing left ~93 % of re-syndicated stories looking "distinct".
- **Price-action scanner** over 1-minute bars: opening-range breakout, VWAP cross, Donchian breakout, RSI extremes, volume spikes, gap-and-go.

### LLM classification
- Claude Haiku with tool-use (structured JSON output) tags every signal with event type, sentiment, and whether it is factual or speculative. A deterministic pre-classifier handles SEC form types so the LLM is only paid for ambiguous text. Daily spend is tracked and capped.

### Machine learning
- **Stacked ensemble** (`ml/ensemble.py`): HistGradientBoosting + LightGBM base learners with a logistic meta-learner on out-of-fold predictions. An earlier logistic base learner was removed after it scored below random on the mixed-scale feature matrix.
- **Walk-forward validation** (`ml/train.py`): expanding-window `TimeSeriesSplit`, median fold AUC as the published metric, fold variance as a regime-fragility signal. Isotonic calibration on the final model. Latest deployed 5-day model: validation AUC 0.72 on a 745-row slice, so the error bars are wide.
- **Split-conformal prediction** (`ml/conformal.py`): distribution-free intervals around calibrated probabilities; the **selective gate** (`ml/selective_gate.py`) only lets a trade through when p is high *and* the interval is narrow.
- **Learning-to-rank** (`ml/ranker.py`): a day-demeaned LGBMRanker that asks "which bets outrank the others today" instead of "will this bet win", with a hard data-volume deploy gate that provably refuses to promote a model until enough distinct clean days exist.
- **Drift monitoring**, weekly automated retrain with an AUC gate (a new model only deploys if it beats the incumbent; `model_history.md` is the audit log of every retrain decision).

### Risk, sizing and execution
- **Fractional Kelly sizing** (`execution/sizer.py`): quarter-Kelly from calibrated p and the ATR-derived reward:risk, capped by a max position percentage, with fail-closed guards.
- **8-rule risk manager** (`risk/manager.py`, `execution/live_risk.py`): emergency stop, daily loss kill-switch, monthly drawdown halt, model-drift block, minimum calibrated probability, daily trade cap, gross exposure cap, per-ticker / per-sector / pile-on concentration caps. The live variant reads exposure from the broker, not the local database.
- **Macro regime gate** (`signals/macro_regime.py`): VIX, SPY trend, 10y–2y spread and breadth combine into halt / shrink / boost multipliers.
- **Options** (`execution/options/`): OCC symbol math, multi-leg order submission, vertical debit-spread construction (ATM long, ~0.25-delta short, 10–21 DTE), liquidity and earnings filters, asymmetric Kelly contract sizing, and exit rules at 30 % of max gain / 50 % of max loss / 1-DTE / max-hold.
- **Single order gateway** (`execution/order_gateway.py`): every path that can short, flip or oversize goes through one choke point. This exists because of an incident on the paper account where a long that had been flipped short kept "selling to close", doubling from 63 to 16,128 shares on a ~$6k book. The fix was a structural invariant, not a patch at the symptom site, plus regression tests.
- **Fills-based P&L reconciliation** (`execution/pnl_reconcile.py`): the broker's actual fills are matched FIFO into realized P&L, deterministically, after the per-trade ledger was found to capture only a fraction of real losses. Idempotent rebuilds; the equity curve is treated as the only honest P&L.
- Other production fixes worth reading: DST handling via `zoneinfo`, idempotent submits via a `pending_submit` row, orphan-position adoption, stale-quote refetch before bracket submission, cancel-all-before-flatten at end of day.

### Operations
- Nine `launchd` jobs (daemon, trader, health check, drift check, weekly retrain, P&L reconcile, digest, backup, keep-awake) with install scripts.
- Local Flask dashboard with signal feed, per-ticker drill-down, measured hit rate per signal class, and bot status.
- Telegram alerts with de-duplication.

---

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
├── signals/       price-action scanner, macro regime, confluence
├── risk/          rule-based risk manager
├── execution/     Alpaca client, sizer, live trader, order gateway, P&L reconcile, options/
├── backtest/      replay harness with cost model
├── notifications/ Telegram + macOS
├── storage/       SQLite access layer (WAL, schema loading)
└── daemon.py      APScheduler job graph
scripts/           CLI tools, backfills, and the edge_*.py research gauntlet
tests/             103 pytest tests (run without a database or broker keys)
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
PYTHONPATH=src python scripts/run_live_trader.py --paper   # paper account only
python dashboard/server.py               # http://localhost:8765
pytest                                   # 103 tests
```

The daemon, trader and dashboard all read `data/market_radar.db`; the database, model artefacts, logs and `.env` are git-ignored.

## Stack

Python 3.11 · SQLite (WAL) · scikit-learn · LightGBM · APScheduler · Flask · Anthropic SDK · Alpaca REST · yfinance · feedparser · pytest · GitHub Actions


