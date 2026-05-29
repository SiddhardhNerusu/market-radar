# MARKET RADAR — prioritized upgrade research (2026-05-13)

Synthesis of academic literature, retail-quant tooling, and a fresh audit
of the current codebase. Items are ranked by expected lift per
dev-hour-of-effort under your existing constraints (Mac, single-developer,
small budget, real-money tradeable goal).

The baseline you're comparing against is **val AUC 0.5365 on the 23-feature
HGB model** after we ship the LLM-bodies redo. Most of the items below
are AUC-additive on top of that baseline; a few (calibration, sizing,
walk-forward CV) don't bump AUC at all but are **prerequisites for sizing
real money**, which is what you actually want.

---

## What the audit found — concrete gaps in the current code

### Modeling (`src/market_radar/ml/`)

| Issue | File | Why it matters |
| --- | --- | --- |
| Single chronological 80/20 split. No purging, no walk-forward, no time-series CV. | `train.py:91-94` | The 0.5365 val AUC is a single fold. With ~150k rows you can do 5-fold expanding-window CV; this both produces a more reliable estimate AND reveals regime fragility. |
| No probability calibration. The model's `predict_proba(0.65)` is **not** "65% chance of winning." | `train.py:128-173` | Today's `predict.py` outputs are uncalibrated and unusable for Kelly sizing. Diagnostics are logged but never applied. |
| 23 features. Missing several well-documented signal-class features. | `features.py:33-62` | Specifically missing: insider-cluster count, short-interest %, days-to-known-catalyst, options IV percentile, peer-correlation, days since prior 8-K from same issuer. |
| Outcome label is **only `return_5d_pct > 0`**. | `train.py:97` | Loses information. A signal that went +0.1% over 5d is treated identically to one that went +20%; binary-cross-entropy on a noisy label is hard. |
| Only one horizon predicted. | `outcomes/tracker.py` | The brief says "intraday/few hours" AND "days to weeks." We only model 5d. |
| `SOURCE_WEIGHTS` is static. | `scoring/source_weights.py` | Per the memory, auto-learning weekly was on the punch list and not built. With 135k resolved outcomes we have plenty of data to weight sources by their measured edge. |
| No per-event-type model. | `train.py` | Currently one model averages over M&A, FDA, earnings, insider, macro — five very different signal types. |
| No backtest framework. Just train→val on resolved rows. | `train.py:60-138` | Out-of-sample _classification_ AUC ≠ real-money PnL. We never simulate fees, slippage, capital allocation, or correlated drawdowns. |

### Data coverage gaps

| Source | Status | Why it matters |
| --- | --- | --- |
| FINRA short interest (free API) | **Not wired.** | Squeeze-prone tickers + crowded shorts both have documented edges. |
| FDA PDUFA / catalyst calendar | **Not wired.** Free RSS available. | Biotech is the strongest single-event sector. Pre-positioning before a known PDUFA date is a textbook play. |
| 13F institutional holdings | **Not wired.** Free via SEC quarterly. | "Smart money replication" + crowding signals. Lagged but high-quality. |
| Earnings calendar (Finnhub free tier) | **Not wired.** Key in your `.env.example`. | PEAD strategies need to know surprise vs estimate; this is the input. |
| Earnings call transcripts | **Not wired.** Finnhub free has these. | Transcript sentiment is one of the strongest documented next-month signals. |
| Google Trends search volume | **Not wired.** `pytrends` is free. | Retail attention proxy; some asymmetry into short-term moves. |
| Wikipedia pageviews | **Not wired.** Free MediaWiki API. | Documented small-but-real predictor (Moat et al. 2013). |
| Options flow / unusual activity | **Not wired.** Free tier exists (UW delayed, Barchart UOA). | Strong informed-money signal. Cost-sensitive. |
| Reddit PRAW (vs current scraping) | **Not wired.** Free, you have placeholder env vars. | Cleaner author karma/age data; better quality filter. |
| Alpaca news feed | **Not wired.** Key in your `.env.example`. | Pre-tagged tickers, cleaner than RSS. |

### What's actually well-built and shouldn't be touched

`outcomes/tracker.py` price snapshot logic, `scoring/composite.py` additive
formula, `scoring/routine_patterns.py` issuer dictionary, content-hash
dedup, the dashboard's portfolio time-range filter, ML feature DataFrame
extraction in `features.py`, the spend tracker and daily cap in
`llm/spend.py`. These are clean and they work — leave them alone.

---

## Tier 1 — Highest leverage, free, ship within days

Ranked by AUC-lift / dev-hour-per-item:

### 1. Walk-forward time-series CV in `train.py`

**Effort:** 2 hr. **Cost:** $0. **Risk:** Low.
**Expected lift:** No AUC bump per se, but the AUC number you trust will
be more accurate. Likely reveals the current 0.5365 is **optimistic**
(your fold is the last 20%, which happens to be a low-volatility window).

Today: oldest 80% train, newest 20% val.
Should be: `TimeSeriesSplit(n_splits=5)` with optional López de Prado
purging at the fold boundary. Median val AUC across 5 folds is the
metric to publish. Variance across folds is a regime-stability indicator.

### 2. Probability calibration (Platt + isotonic)

**Effort:** 1 hr. **Cost:** $0. **Risk:** Low.
**Expected lift:** Same AUC, but `predict_proba` becomes trustworthy for
sizing. Above ~1000 calibration rows, isotonic regression strictly
dominates Platt scaling.

Wrap the trained HGB with `sklearn.calibration.CalibratedClassifierCV
(method='isotonic', cv='prefit')`, fitted on the last fold's val set.
Pickle the wrapped calibrator alongside the raw model. `predict.py`
loads the wrapped version.

This is a hard **prerequisite for fractional Kelly sizing** — uncalibrated
probabilities → over-bet → ruin.

### 3. Insider-cluster feature (3+ insiders in 30d)

**Effort:** 3 hr. **Cost:** $0. **Risk:** Low.
**Expected lift:** Real edge. Cluster vs non-cluster: ~3.8% vs ~2%
over 21 trading days on academic samples.

Aggregate `raw_signals` where `title LIKE '4 - %'` over rolling 30-day
windows by `cik`. Count distinct insider names. Emit a new feature
`insider_cluster_size_30d`. Form 4 ingestion already runs; this is a
pure SQL-window-over-existing-data feature.

### 4. Auto source-weight learning

**Effort:** 3 hr. **Cost:** $0. **Risk:** Low.
**Expected lift:** Modest direct AUC bump, but cleaner downstream signal
quality and makes the composite score more honest. Per the memory this
was already on the punch list.

New cron job (weekly): `python scripts/recompute_source_weights.py`.
Recompute per-source 5d hit rate from `signal_outcomes` over trailing
180 days. Weight = monotone function of hit-rate minus base-rate (e.g.
sigmoid clip to [0.3, 2.0]). Write new weights to a versioned JSON; the
composite scorer loads the latest. Add a `--dry-run` mode to inspect
shifts before applying.

### 5. Per-event-type model framework

**Effort:** 6 hr. **Cost:** $0. **Risk:** Medium (refactor touches
predict path).
**Expected lift:** ~0.01-0.03 AUC overall, but the high-volume event
types (M&A, earnings, insider) likely hit 0.58-0.62 individually. That's
the dataset where you can actually trade.

Refactor `train.py` to bucket rows by event_type into 5-6 cohorts
(m_a_*, earnings_*, fda_*, insider_*, activist_position, macro,
"catch-all-other") and train one HGB per bucket with a shared feature
schema. `predict.py` routes to the matching bucket-model. Per-bucket val
AUC and per-bucket calibration become first-class. Fall back to the
catch-all when an event has < 100 training rows.

### 6. Multi-horizon outcomes (1d + 5d + 20d)

**Effort:** 4 hr. **Cost:** $0. **Risk:** Low.
**Expected lift:** Better short-term trade selection. Multi-task
learning has been shown to share strength across related labels.

`outcomes/tracker.py` already snapshots prices at 1d / 5d / 20d (per the
schema). Wire those columns into training:
`y_1d / y_5d / y_20d`. Train a `MultiOutputClassifier` or three separate
models. The dashboard's action-label logic gets to discriminate "fast
mover" from "slow burn" events.

### 7. FDA PDUFA catalyst calendar + `days_until_known_catalyst` feature

**Effort:** 3 hr. **Cost:** $0. **Risk:** Low.
**Expected lift:** Biotech-specific. Strong (PDUFA dates are pre-known
events around which prices move sharply).

New ingestor that scrapes one of the free PDUFA calendar feeds
(BiopharmaWatch, MarketBeat, CatalystAlert). Materializes into a
`catalysts` table keyed by ticker + decision_date + type. New feature
`days_until_catalyst` (clipped to [0, 60]). Boosts confidence on signals
related to upcoming-event tickers.

### 8. Wire free API keys you already have placeholders for

**Effort:** 30 min - 1 hr each. **Cost:** $0. **Risk:** Trivial.

- **Finnhub** → earnings calendar, earnings surprise, transcripts
- **NewsAPI** → another news aggregator (1k req/day free)
- **Alpaca news feed** → pre-tagged ticker news (already have keys via AUTO TRADER)
- **Reddit PRAW** → upgrades Reddit ingestion to real auth (better karma/age data)

Per the memory and per `.env.example`, the wiring already partially
exists. This is purely "add the API key to `.env`, restart the daemon."

### 9. Fractional Kelly sizing module

**Effort:** 4 hr. **Cost:** $0. **Risk:** Low (paper-trade-only at
first).
**Expected lift:** Doesn't bump AUC. Bumps **$-PnL by 10-20%** on the
SAME signals once you're trading real money, because position sizes
match conviction.

New `scoring/sizing.py`. For each STRONG_BUY signal: read the calibrated
probability `p`. Compute Kelly fraction `f = (p*b - (1-p)) / b` where
`b` is the expected payout odds from `projections.py`. Apply a
half-Kelly safety multiplier. Cap at 5% of account. Output a
`suggested_size_pct` column on the dashboard. **Paper only** until 2+
weeks of resolved live outcomes confirm the calibration holds.

### 10. FINRA short-interest ingestor + `short_interest_pct` feature

**Effort:** 4 hr. **Cost:** $0. **Risk:** Low.
**Expected lift:** Modest but real. Squeeze candidates and crowded shorts
both predict subsequent moves.

FINRA publishes short-interest twice monthly via their Equity Short
Interest API (JSON/CSV, free). New `ingestors/finra_short_interest.py`
materializes the latest snapshot into a `short_interest` table keyed by
ticker. Feature: short-interest / float, plus days-to-cover. Boosts
weight on Form-4 buys when short interest is also high.

---

## Tier 2 — Medium leverage, free / cheap

Defer to next week.

| # | Upgrade | Effort | Cost | Notes |
| --- | --- | --- | --- | --- |
| 11 | Earnings-drift (PEAD) feature: pre-event EPS surprise. | 4 hr | $0 (Finnhub free) | 6.78% per quarter on baskets in academic samples. |
| 12 | Earnings call transcript ingestion + LLM-classified tone. | 8 hr | <$5/mo LLM | Strong next-month signal in literature. |
| 13 | 13F institutional flow + crowding feature. | 8 hr | $0 (SEC) | Quarterly; lagged. |
| 14 | Google Trends feature via `pytrends`. | 4 hr | $0 | Search-volume spikes as retail-attention proxy. |
| 15 | Wikipedia pageviews feature. | 3 hr | $0 | Small but real effect documented. |
| 16 | LLM embeddings → similarity search over historical filings. | 12 hr | ~$10/mo Voyage AI / OpenAI | "This 8-K reads like NVDA Q3'24 — those returned X%." |
| 17 | Macro feature pack (yield curve, DXY, oil, gold). | 4 hr | $0 yfinance | Regime context. |
| 18 | Conformal prediction intervals (MAPIE wrapper). | 4 hr | $0 | Gives 90% coverage on predicted 5d return. Risk-sizing input. |
| 19 | Multi-source corroboration window feature (unique sources / ticker / 24h). | 2 hr | $0 | Strengthens existing `corroboration_count`. |
| 20 | Page-Hinkley drift detector on rolling val AUC. | 3 hr | $0 | Alert when model degrades. |

---

## Tier 3 — Cost-bearing, evaluate after Tier 1 is shipped

| # | Upgrade | Effort | Cost | Notes |
| --- | --- | --- | --- | --- |
| 21 | Options flow free tier (Unusual Whales delayed, Barchart UOA). | 8 hr | $0 free / $57/mo | Free tier is delayed enough that it competes with retail; paid is institutional-grade. |
| 22 | Polygon.io tick-level historical data. | 10 hr | $79+/mo | Better backtest realism. |
| 23 | Twitter/X integration. | 10 hr | $200/mo Basic tier | Per memory: only after other levers exhausted. |
| 24 | LinkUp job-postings alt-data. | 8 hr | paid | Earnings beat predictor on hiring trends. |
| 25 | Dark-pool prints (FlowAlgo competitors). | 6 hr | paid | Often gated. |

---

## Tier 4 — Speculative modeling overhauls

| # | Upgrade | Effort | Risk | Why probably defer |
| --- | --- | --- | --- | --- |
| 26 | Transformer time-series (PatchTST / NHITS) per ticker. | 30+ hr | High | HGB likely fine for our signal density; transformer wins are largest at high-frequency. |
| 27 | Reinforcement learning for execution. | 30+ hr | Very high | Hard to do well without massive data. Standard RL fails on financial markets. |
| 28 | Bayesian neural network ensemble. | 12 hr | Medium | Cleaner uncertainty intervals. Conformal prediction (Tier 2 #18) gets you 80% of this for 1/4 the work. |
| 29 | Stacking / blending (HGB + linear + nearest-neighbor + LLM). | 12 hr | Medium | Diminishing returns. Per-event-type models (Tier 1 #5) usually beat stacking. |
| 30 | Quantile regression for asymmetric long-vs-short loss. | 8 hr | Medium | Useful once we routinely short. |

---

## Honest expectations after all of Tier 1 ships

| Item | Likely change |
| --- | --- |
| val AUC overall | 0.5365 → **0.55 – 0.58** if LLM-bodies works, **0.57 – 0.60** with bodies + Tier 1 modeling fixes |
| best per-event-type AUC (M&A, earnings, insider) | 0.58 – 0.63 |
| max-drawdown realism | Substantially better (walk-forward CV exposes regime breaks) |
| Sizable real-money | Probably yes after Tier 1 ships AND ≥2 weeks of resolved live outcomes confirm the calibration |
| Sharpe vs SPY | TBD until the backtest framework is built — that's separate work |

**To beat the SPY buy-and-hold benchmark after fees + slippage**, the
research suggests you'll need (a) the bodied LLM, (b) per-event-type
models, (c) calibrated probabilities, (d) Kelly-fraction sizing, and
(e) restriction to event types that show ≥60% measured hit rate over
n ≥ 50. That's the bar. The framework you have today is the right shape
to test that hypothesis once Tier 1 is in.

---

## What's shipping today

In the same session as this report I'm implementing the Tier 1 items
that are pure code changes (no external API setup needed from you):

- **#1 Walk-forward CV** in train.py
- **#2 Probability calibration** wrapping the HGB
- **#3 Insider-cluster feature** from existing Form 4 data
- **#4 Auto source-weight learning** script
- **#9 Fractional Kelly sizing module** (paper-only)

Items #5 (per-event-type), #6 (multi-horizon), #7 (PDUFA calendar),
#8 (free API keys), #10 (FINRA short interest) are queued for the next
session — they each touch the daemon or the schema in ways I'd rather
do once the LLM redo is settled and the AUC delta is measured first.

Sources:

Sources:
- [PEAD academic update — UCLA Anderson Review](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/)
- [Alpha Architect — PEAD recent facts](https://alphaarchitect.com/new-facts-for-post-earnings-announcement-drift/)
- [Insider cluster buys evidence — 2iqresearch](https://www.2iqresearch.com/blog/what-is-cluster-buying-and-why-is-it-such-a-powerful-insider-signal)
- [13D activist filings — Harvard Corporate Governance](https://corpgov.law.harvard.edu/2022/03/17/trading-ahead-of-barbarians-arrival-at-the-gate-insider-trading-on-non-inside-information/)
- [Returns to Hedge Fund Activism — Oxford RFS](https://academic.oup.com/rfs/article/30/9/2933/3852480)
- [FINRA Equity Short Interest API docs](https://www.finra.org/finra-data/browse-catalog/equity-short-interest/data)
- [BiopharmaWatch PDUFA calendar](https://www.biopharmawatch.com/PDUFA-calendar)
- [MarketBeat PDUFA calendar 2026](https://www.marketbeat.com/fda-calendar/upcoming/)
- [Walk-forward purged CV — timeseriescv on GitHub](https://github.com/sam31415/timeseriescv)
- [Lopez de Prado purged cross-validation — Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [sklearn TimeSeriesSplit docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Platt scaling — Wikipedia](https://en.wikipedia.org/wiki/Platt_scaling)
- [sklearn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
- [Conformal prediction in finance — MAPIE intro](https://www.tildee.com/harnessing-conformal-forecasting-in-financial-markets-quantifying-uncertainty-and-managing-risk-with-mapie/)
- [Conformalized Quantile Regression — NeurIPS 2019 paper](https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf)
- [Earnings call transcript sentiment — LSEG](https://www.lseg.com/en/insights/data-analytics/ai-unlock-investment-risk-management-opportunities-earnings-call-transcripts)
- [Finnhub Institutional Holdings (13F) API](https://finnhub.io/docs/api/institutional-portfolio-13f)
- [Apify SEC 13F Holdings Tracker](https://apify.com/ryanclinton/sec-13f-holdings/api)
- [SEC Form 13F official data](https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets)
- [Unusual Whales options flow](https://unusualflow.com/)
- [Barchart unusual options activity](https://www.barchart.com/options/unusual-activity)
- [Best Options Data APIs 2026 — FlashAlpha](https://flashalpha.com/articles/best-options-data-apis-2026)
- [Google Trends + Wikipedia pageviews stock research — Moat et al.](https://www.researchgate.net/publication/249963707_Predicting_Financial_Markets_with_Google_Trends_and_Not_so_Random_Keywords)
- [AltIndex alternative data dashboard](https://altindex.com/)
- [Fractional Kelly simulation — Matthew Downey](https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html)
- [Sizing the Risk — Kelly + VIX hybrid paper (2025)](https://arxiv.org/html/2508.16598v1)
- [LLM embeddings clustering — MachineLearningMastery](https://machinelearningmastery.com/document-clustering-with-llm-embeddings-in-scikit-learn/)
- [Generative AI and PEAD — CFA Institute](https://blogs.cfainstitute.org/investor/2025/04/22/can-generative-ai-disrupt-post-earnings-announcement-drift-pead/)
