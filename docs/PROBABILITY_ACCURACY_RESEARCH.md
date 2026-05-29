# Deep research — paths to higher probability accuracy on stock moves (2026-05-13)

Focused follow-up to `SYSTEM_UPGRADE_RESEARCH.md`. Where the first report
mapped *which features* to add, this one drills into: **how to reach
genuinely high-confidence (≥ 60-70 %) probability estimates on a useful
subset of signals**, plus everything new from the literature on news /
forums / past-data / graph-analysis that wasn't already in the system.

The honest math from the research: with public-data inputs, your *overall*
hit rate has a ceiling around 0.58-0.62. To exceed that, you don't try to
make all predictions more accurate — you find subsets where the model is
*confidently* right, and trade only those. That's the strategy you can
actually verify post-hoc with calibration plots, and it's what gets you
to "≥ 60 % measured hit rate over n ≥ 50" — your stated bar for a real-
money signal class.

---

## Part 1 — News & wire-service edge

### Source-quality findings

| Source | Latency vs Bloomberg | Cost | Has tag/ticker? | Free? |
| --- | --- | --- | --- | --- |
| **Bloomberg Terminal** | reference (0s) | $2 665/mo | Yes | No |
| **Benzinga Pro** | 5-15 minutes **faster than Bloomberg** on stock news per their own data | $166/mo | Yes | $99 entry tier |
| **Reuters** | comparable | included in Eikon | Yes | RSS free, tagged feed paid |
| **Briefing.com In Play** | similar to wires | free RSS | No tags (HTML) | Yes |
| **PR Newswire / Business Wire RSS** | source-of-truth for press releases | free | Yes (ticker in URL) | Yes |
| **SEC EDGAR RSS** | 1-5 min after filing | free | Indirect (via CIK) | Yes |
| **GlobeNewswire / AccessWire** | similar to PR Newswire | free RSS | Yes | Yes |
| **Alpaca News API** | sub-second to seconds | free with paper account | **Pre-tagged with tickers** | Yes |

**Key insight**: Benzinga Pro is the price/speed sweet spot for retail
that wants to compete with institutional latency. Their published
delivery beats Bloomberg by 5-15 min on retail-relevant tickers. For a
market-radar-style system that's the right paid upgrade *if* AUC
warrants it after Tier 1 ships.

### News features the current system is missing

1. **Novelty score** — Academic research (EPJ Data Science, 90M articles)
   shows stock volatility + volume only respond to *novel and topical*
   news. Repeat coverage of the same story has progressively less impact.
   Implementation: compute cosine similarity of each new article's title
   embedding against all articles within a 24h trailing window for the
   same ticker. Score `1 - max_similarity` is the novelty.

2. **Wire-service hierarchy** — first-mover edge. Same story breaking on
   PR Newswire vs. Benzinga vs. retail outlets has dramatically different
   impact decay. Currently `SOURCE_WEIGHTS` captures *credibility*; it
   does not capture *first-reporter* status. Add a per-ticker, per-event
   first-reporter detection that boosts weight on the source that
   originated the story.

3. **Topic decay model** — first-story-detection research shows novelty
   scores decay exponentially over a measurable time constant. The
   research-paper model uses ~0.14 daily decay for media cycles. Wire
   this into `corroboration_count` so a 4-hour-old story counts less
   than a fresh one even if it's accumulated more sources.

4. **Named-entity recognition vs cashtags** — current ticker extraction
   uses `$AAPL`-style cashtags. Add a curated company-name → ticker map
   for the top 2 000 US tickers so "Apple announces" / "Microsoft
   shareholders" parse correctly. Estimated +30 % raw signal volume from
   non-cashtag mentions.

5. **FinBERT sentiment vs. heuristic sentiment** — published research
   (2025) reports FinBERT classification accuracy on financial
   sentiment at 57-58 %, not great alone, but its disagreement with the
   LLM-classified sentiment is itself a useful feature. Run FinBERT
   locally (it's open-source, ~400MB, runs on CPU) and add a
   `finbert_sentiment_minus_llm` divergence feature.

### Implementation gap → priority

| Feature | Effort | Expected accuracy lift on its subset |
| --- | --- | --- |
| Novelty score (cosine sim over title embeddings) | 4 hr | +2-4 % AUC on news-driven signals |
| Wire-service first-reporter detection | 3 hr | +1-3 % AUC on M&A / earnings events |
| Topic decay weighting in corroboration count | 1 hr | +0.5-1 % overall |
| Name→ticker map (top 2 000 companies) | 2 hr | +30 % raw signal volume, ~+1 % AUC |
| FinBERT divergence as ML feature | 3 hr | +1 % AUC, also disagreement-is-info |

---

## Part 2 — Forums & social-sentiment edge

### What the research actually shows

**StockTwits > Twitter for financial signal.** A 2023 PeerJ study on
StockTwits + FinBERT achieved 76.65 % directional accuracy on next-day
moves — significantly higher than Twitter-based equivalents. Twitter
sentiment is too noisy because the platform is general-purpose;
StockTwits is finance-only by construction. The current system already
ingests StockTwits trending but does **not** classify each message via a
finance-tuned sentiment model — heuristic sentiment is doing the work.

**Conditional predictability.** Polarity on its own doesn't predict
next-day returns. But polarity *conditioned on a sudden volume spike of
posts* (mentions × 5 vs trailing average) predicts abnormal returns
robustly. So the right feature is `polarity × log(posts_24h /
posts_avg)`, not `polarity` alone.

**Reddit subreddit hierarchy** — academic GARCH analysis (2025) shows
Reddit sentiment has **stronger immediate impact** than Twitter and a
shorter half-life. The implication: Reddit is a momentum signal, not a
reversal signal, and the cleanest read is on the first 6-12 hours after
a post peaks. Subreddits with the highest signal-to-noise (per multiple
studies):

| Subreddit | Signal quality | Reason |
| --- | --- | --- |
| r/SecurityAnalysis | High | Older, more moderated, fundamentals-focused |
| r/ValueInvesting | High | Long-form analysis, less noise |
| r/investing | Medium-high | Volume + moderation |
| r/StockMarket | Medium | Broad |
| r/wallstreetbets | Low-medium for signal, **high for short-squeeze and meme momentum** specifically |
| r/pennystocks | Low | Pump-prone — current system already de-rates correctly |
| r/Shortsqueeze | Low | Self-fulfilling-prophecy noise, except as a contrarian feature |

You're already weighting these correctly. The improvement is to use
*the volume spike × FinBERT-classified message tone*, not just the
post count.

**Discord / Telegram channels**: research is unanimous — claimed
accuracy (82-96 %) is unverifiable and 70 % of users following them
lose money. Skip as a feature unless you can find a channel with a
verifiable independent track record.

### Specific social-data features to add

1. **Per-post FinBERT sentiment** on all StockTwits + Reddit
   ingestion. Roll up to ticker-hour buckets. Surface as
   `social_sentiment_finbert_24h` (signed) and
   `social_sentiment_magnitude_24h` (unsigned, for the LLM-style
   confidence-as-features model).

2. **Posting volume z-score** — `posts_in_24h / posts_avg_28d`. Feature
   only fires when ≥ 3σ above mean. Multiplicative with sentiment.

3. **Author quality weighting** — a small minority of posters drive
   most of the signal. Per existing literature, accounts with > 2 years
   of activity + > 1 000 karma carry meaningfully more weight than new
   accounts. Currently the system has `author_quality` but it's
   under-utilized. Plug in Reddit account-age and karma via PRAW once
   the user adds the API key.

4. **Bullish/bearish ratio** — the cleanest binary signal: among
   non-neutral classified posts on a ticker in 24h, what fraction are
   bullish? Conditional on volume spike, this is one of the most
   reliable retail-trader features in the literature.

5. **Discord/Telegram leakage** — *don't* try to ingest these. Use the
   downstream effect instead: Discord-driven moves usually correlate
   with sudden surges on r/wallstreetbets in the same 30-min window.
   That's already captured by feature (2).

---

## Part 3 — Past data & technical analysis

### What chart patterns actually do (academic evidence)

Patterns with statistically significant evidence (Tsai 2012, Vasiliou
et al. 2015, Bulkowski compendium, multiple meta-analyses):

| Pattern | Reported success rate | Bias |
| --- | --- | --- |
| Head & Shoulders (with volume confirmation) | 85 % | Top |
| Inverse Head & Shoulders | 83 % | Bottom |
| Double Bottom | 78 % to target | Bottom |
| Double Top | 75 % to target | Top |
| Symmetrical Triangle (with-trend break) | 76 % | Continuation |
| Bullish Flag (with volume) | 91 % | Continuation |
| Bullish Pennant | 67 % | Continuation |
| Bearish Pennant | 71 % | Continuation |

**The huge caveat** that comes up in every paper: volume confirmation.
Successful breakouts require *volume > 1.5 × 30-day average*. Without
that, false-break rate balloons.

The current system has `volume_ratio_5d_20d` but doesn't detect chart
patterns explicitly. Adding chart-pattern recognition would be a
multi-month project (need decent OHLC tick data + a pattern-matching
engine). **Cost/benefit-weak** unless you commit to it as a primary
input.

A cheaper path: borrow the *outputs* of pattern recognition via
**TA-Lib** indicator combinations that proxy the same patterns.
Quantified-Strategies research (2025) shows:

- **MACD + RSI (with volume filter)**: 73 % win rate over 235 trades,
  +0.88 % avg gain per winner, after costs.
- **Stochastic RSI + MACD**: 73 % win rate.
- **MACD alone**: 54-60 % — too noisy.
- **RSI extremes (< 30, > 70) with mean-reversion filter**: ~91 %
  win rate on the *opposite-direction* trade (mean-reversion), but
  only ~25 % signal frequency.

Indicators to add as ML features (all free, computed from existing OHLC
data via TA-Lib):

| Feature | Why |
| --- | --- |
| `rsi_14` | Mean-reversion proxy |
| `macd_signal` (binary: above/below signal line) | Trend filter |
| `macd_histogram` (signed magnitude) | Trend acceleration |
| `bb_position_20d` (where in Bollinger band, 0-1) | Mean-reversion bound |
| `atr_14` (Average True Range) | Volatility-adjusted position-sizing input |
| `adx_14` | Trend-strength filter |
| `vwap_distance_pct` | Distance from VWAP — institutional benchmark |
| `n_day_breakout_flag` (close > max(prior 20d close)) | Trend continuation |

Implementation: extend `market_features.py` to compute these at signal
date from the already-cached yfinance history. Adds ~8 features. The
catch: per the research these only work *in trending markets*. In
chop they generate noise. So they'd ideally be conditional on a
regime-detection feature (volatility z-score + ADX threshold).

### Order-book microstructure (intraday only — out of scope today)

Order-book imbalance research is real and strong: 64 % directional
accuracy on the next trade in academic papers, with R² up to 10.5 % on
5-second returns. **But** it requires Level-2 tick data which is
$80-200/mo (Polygon.io, Databento). Not free, not relevant unless you
move to intraday trading. Document as "Tier-3 paid path; revisit when
intraday is on the menu."

### Calendar / seasonality effects

- **Monday underperformance** (the "weekend effect"): real but small
  (≈ -0.05 % vs other days on average), often arbitraged away.
- **Turn-of-month effect**: last + first 4 trading days of each month
  outperform middle by ~0.4 % on average per multiple studies.
- **December / January** seasonality (tax-loss harvesting + Santa
  rally): real, persistent.
- **FOMC days**: equities drift up between meetings, drop on dovish
  surprises, rip on hawkish surprises that were over-priced. Big-bar
  trading window.
- **CPI / NFP releases**: large repricings (often > 0.3 %) only when
  surprise > +/-0.3 % vs consensus. Below that threshold, modest
  intraday range.

Calendar-feature additions (all free):

- `day_of_month` (1-31), `is_turn_of_month` (binary)
- `days_until_fomc` (-90..+90, computed against the FOMC calendar)
- `days_until_cpi_release` (-30..+30)
- `days_until_nfp_release` (-30..+30)

---

## Part 4 — Graph / network analysis

### Supply-chain peer effects — the strongest free graph signal

Multiple 2025 papers (FS-GCLSTM, HSGNN, TFT-GNN) on supply-chain
graphs show:

- Companies linked by supplier-customer relationships exhibit **delayed
  information diffusion**. When NVDA moves on AI news, TSMC moves with
  a measurable lag.
- A Temporal Graph Convolutional LSTM on Eurostoxx 600 + S&P 500 with
  LSEG value-chain data **beat baseline on annualised return AND
  Sharpe AND Sortino across both markets**.
- TFT-GNN research found GNN-derived relational features were
  consistently weighted **higher than RSI/MACD** by the trained model.

This is the most defensible graph-edge in public research. But: LSEG
value-chain data is paid. Free substitute: **SEC 10-K "Customers" /
"Suppliers" sections + earnings-call mentions** of named companies.
Both available in filings you already ingest. Build a directed graph
of company → company mentions over a trailing 4-quarter window.

### Free graph data sources

| Source | Edge type | Cost |
| --- | --- | --- |
| **10-K Customers section** (Item 1) | C → S (customer → supplier) | Free, already ingested |
| **Earnings-call transcripts** (named-company mentions) | Generic "discusses" | Free with Finnhub |
| **13F co-ownership clusters** | Joint institutional ownership | Free (SEC) |
| **News article co-mentions** | Generic "appears together" | Free, computed from existing ingest |
| **GICS sector / industry hierarchy** | Static taxonomy | Free (Wikipedia/SEC) |
| **ETF holdings overlap** | Membership in same ETF | Free (issuer websites) |
| **Crypto-equity correlations** (RIOT/MARA ↔ BTC) | Sector-specific known link | Free |

### Specific graph features to add

1. **Sector-momentum feature**: median 5d return of the GICS-industry
   peers of the signal's ticker over the past 5 days. Captures "tide
   lifting all boats" effect.

2. **Co-mention graph**: for every ticker, maintain a list of "most
   co-mentioned" peers in news + filings over the trailing 90 days.
   Compute a peer-co-move feature: average 5d return of the top 5
   most-co-mentioned peers.

3. **ETF-overlap crowding** — for each ticker, compute the % of ETFs
   it's a top-10 holding in. High overlap = more passive-flow
   sensitivity. Defensive risk feature.

4. **News-graph "ripple" detection** — if a ticker's *immediate
   neighbours* in the co-mention graph are seeing unusual flagging
   volume in the past 24h *while the ticker itself isn't yet*, that's
   a leading-indicator setup. Worth scoring.

5. **Crypto-equity sympathetic moves** — BTC 5d return as a feature
   for the crypto-sensitive ticker subset (RIOT, MARA, COIN, MSTR,
   MARA, HUT, etc.). Identifiable from a hardcoded list of 20-30
   crypto-correlated tickers.

Implementation effort: 2-3 days for all five. None of them require
paid data.

---

## Part 5 — Probability calibration & selective prediction

This is where you actually get to **high confidence**. The literature
calls it *selective classification* — instead of asking "what's the
probability?", you ask "is this signal in the subset where the model
is confidently right?"

### Why this matters more than another feature

A model with overall AUC 0.55 can have a *high-confidence subset*
where AUC > 0.75 — but only if you know how to identify that subset.
The framework:

1. **Calibrate** so `predict_proba(0.7)` actually corresponds to
   70 % hit rate. (We shipped isotonic calibration today.)
2. **Restrict to the calibrated-high-confidence region** (e.g. only
   `p >= 0.62` and `interval_width <= 0.15`). Per academic results,
   this typically retains 5-20 % of signals but raises hit rate by
   8-15 percentage points.
3. **Conditional features** that fire only when present strengthen the
   conviction further. Examples below.

### Specific selective-prediction features

1. **Model-ensemble agreement**. Train 3 models (sklearn HGB +
   LightGBM + Logistic) on the same features. Use the variance of
   their predictions as a confidence feature. Agreement boosts
   conviction; disagreement halves position size.

2. **Conformal interval width** (already shipped). Wider interval =
   model is less sure = smaller position. The threshold for "trade-
   able" should be calibrated: in the historical data, what's the
   widest interval that still gives ≥ 60 % hit rate?

3. **Cross-source corroboration** (Tier 2 #19 shipped). Signals
   confirmed by 3+ independent sources have measurably higher hit
   rates in news-based trading literature.

4. **Insider-cluster + activist + earnings beat alignment**. Multiple
   structural events on the same ticker within a 14-day window is the
   single strongest "high confidence" composite per Harvard's 13D
   research (12.09 % avg insider gain when insiders buy ahead of
   activist filings).

5. **Earnings-Whisper alignment**. EarningsWhispers.com data shows:
   - Beating whisper number → +1.8 % avg on day, 60 % up
   - Beating consensus but missing whisper → -0.3 % avg, 55 % down

   **There is no edge to beating consensus alone — only to beating
   whisper.** This is empirically true for 25+ years per their own
   data. Adding whisper-number alignment as a feature (Estimize API
   or Earnings Whispers scrape) would be one of the single highest-
   leverage adds for earnings events.

### Threshold optimization

After Tier 1 ships and we have ≥ 2 weeks of calibrated outcomes:

1. Plot calibration table per event-bucket. Find the probability
   threshold above which actual hit rate ≥ 60 %.
2. Plot conformal interval-width quantile. Find the interval width
   below which actual hit rate ≥ 60 %.
3. **Trade only signals passing both gates.** Document the expected
   signal volume per week, the expected average return, and the
   per-event drawdown floor.

Selective conformal risk control (2025 NeurIPS paper) formalises this
exact pattern. Worth implementing as a wrapper around the existing
conformal predictor once calibration is verified.

---

## Part 6 — Specific high-edge data sources we should add

Beyond what shipped today, ranked by expected confidence-region lift:

| Source | Cost | Why it matters | Hours |
| --- | --- | --- | --- |
| **Earnings Whispers number** | Free (scrape) | 70 % more accurate than consensus per 25y data | 4 |
| **Estimize crowd estimates** | Free tier (limited) | 64 % accuracy on ≥ 4-estimate stocks | 3 |
| **Insider role + transaction code** | Free (from existing Form 4 XML) | CEO buy ≠ Director award; P/S/A/M codes are predictive | 2 |
| **Fails-to-deliver** (FTD) | Free (SEC fails-to-deliver list) | Squeeze indicator orthogonal to short interest | 3 |
| **TA-Lib indicator pack** (RSI/MACD/BB/ATR/ADX/VWAP) | Free (pip install ta-lib) | 73 % win rate when combined w/ volume | 4 |
| **Sector / industry peer momentum** | Free (computed from existing data) | Confirms tide-lifts-boats | 2 |
| **FOMC / CPI / NFP calendar features** | Free (federalreserve.gov / BLS) | Captures binary-event windows | 2 |
| **Crypto-equity correlation feature** | Free (computed from BTC + 20 tickers) | RIOT/MARA/COIN/MSTR are BTC-dependent | 1 |
| **FinBERT sentiment (local)** | Free (HuggingFace) | Per-message tone scoring, no API cost | 3 |
| **Novelty score on news titles** | Free (sentence-transformers local) | Old news is priced; novel news moves | 4 |
| **10-K customer/supplier graph** | Free (parsing existing filings) | Supply-chain peer effects, GNN-grade signal | 8 |
| **Alpaca news API** | Free with paper account | Pre-tagged tickers, sub-second latency | 1 |
| **Benzinga free tier RSS** | Free | Stock-news latency edge | 1 |

### Worth paying for *only after Tier 1 verified*

| Source | Cost | When |
| --- | --- | --- |
| **Benzinga Pro** | $99-166/mo | Once you've shown selective prediction beats SPY |
| **Polygon.io Stocks** | $79/mo | Once intraday trading is on the menu |
| **Unusual Whales** | $57/mo | After options-flow Tier 2 ships and proves out |
| **LSEG value-chain data** | paid | Only if GNN-on-supply-chain becomes a primary input |
| **Ortex** (short-interest premium) | paid | Better short-squeeze prediction than FINRA bimonthly |

---

## Part 7 — Modeling techniques for higher hit rate

Beyond what was in the first report:

### Ensemble stacking
2025 literature: a blending ensemble (LSTM + GRU) reduced MSE by 57 %
and improved precision/recall by 40-50 percentage points vs single
LSTM. For our tabular HGB setup, the analogous move is:

- Train HGB + LightGBM + Logistic Regression on identical features
- Train a meta-learner (logistic) on their out-of-fold predictions
- Stacked prediction = meta-learner output

Expected lift: +0.01-0.02 AUC on the global model, **larger lift on the
high-confidence subset** because disagreement among base models is
itself a useful confidence feature.

### Quantile regression for asymmetric loss
Long-only trading rewards getting upside right; shorting rewards
getting downside right. A model trained with symmetric log-loss
treats both the same. Quantile regression at the 10th / 50th / 90th
percentile gives you signed-direction probabilities that are
natively asymmetric. Useful for ranking long-vs-short candidates.

### Embedding-based similarity
The LLM embeddings idea from the first report — vector-search over
historical filings/articles to find "this 8-K looks like prior 8-Ks
that returned X %" — is the most powerful single addition that hasn't
shipped. Cost is ~$5-10/mo at Voyage AI rates for the full backfill.

### SHAP-based feature attribution
Use SHAP to rank features by their contribution at the prediction
level. Highly informative for *why* a signal scored high and lets
you drop features that aren't pulling weight. Free (`pip install shap`),
adds interpretability to the dashboard.

---

## Part 8 — What this means for the system

### After everything in this report ships:

- **Overall** val AUC: 0.5365 → likely 0.58-0.62 (modest lift)
- **High-confidence subset** (≥ 5-15 % of signals): hit rate 0.65-0.75
- **Top selective subset** (1-3 % of signals): hit rate ≥ 0.80
  — these are the only ones you should size real-money positions on

### Recommended next-three-week roadmap

**Week 1** (after LLM bodies redo proves out):
- Add the 8 TA-Lib indicator features
- Add the 4 calendar features (FOMC / CPI / NFP / day-of-month)
- Add the 3 graph features (sector momentum, co-mention peer move,
  crypto-equity sympathetic)
- Wire FinBERT for per-message sentiment on StockTwits + Reddit

**Week 2**:
- Earnings Whisper alignment feature
- Novelty score on news titles (sentence-transformers local)
- Stacked ensemble (HGB + LightGBM + Logistic + meta-learner)
- Selective conformal risk control wrapper

**Week 3**:
- LLM embeddings → similarity search over filings (paid: ~$10/mo)
- Per-event-type calibration thresholds → trade-only gates
- Walk-forward backtest with fees, slippage, position sizing
- Compare to SPY buy-and-hold benchmark on actual paper-trade outcomes

### What this won't do

It still won't let you trade *every* signal. The framework that comes
out of all this work delivers high confidence on a *small subset*. You
trade 10-30 ideas per month with real money, not 100. That's the
trade-off public-data-only quant systems carry, no matter how clever
the modeling.

The goal isn't 90 % accuracy on every prediction. The goal is 65-75 %
verified hit rate on the small, identifiable subset where the model is
genuinely confident — and the discipline to skip everything else.

---

## Sources

Sources:
- [Benzinga Pro real-time news feed](https://www.benzinga.com/pro/blog/real-time-stock-news-feed)
- [Liberated Stock Trader — top 10 financial news services 2026](https://www.liberatedstocktrader.com/top-10-best-financial-stock-market-news-sources/)
- [Reuters / WallstreetZen comparison](https://www.wallstreetzen.com/blog/bloomberg-terminal-alternatives/)
- [StockTwits FinBERT prediction — PeerJ Computer Science](https://peerj.com/articles/cs-1403/)
- [Reddit vs Twitter sentiment volatility study (2025)](https://www.researchgate.net/publication/396206198_Analyzing_the_Impact_of_Reddit_and_Twitter_Sentiment_on_Short-Term_Stock_Volatility)
- [Conditional polarity → abnormal returns](https://link.springer.com/article/10.1007/s42521-023-00102-z)
- [Bullkowski / Tsai / Vasiliou chart-pattern meta-analyses](https://samuraitradingacademy.com/7-best-price-action-patterns/)
- [Strike — 55 chart patterns](https://www.strike.money/technical-analysis/chart-patterns)
- [MACD + RSI 73% win-rate backtest](https://www.quantifiedstrategies.com/macd-and-rsi-strategy/)
- [Order-book imbalance review — MDPI](https://www.mdpi.com/2227-7390/10/8/1234)
- [10.5% R² on 5-second returns — order-book deep learning paper](https://www.sciencedirect.com/science/article/pii/S0169207024000062)
- [Free dark-pool data — Meridian](https://meridianfin.io/knowledge/free-dark-pool-data)
- [Implied vol skew, IV percentile, IV crush — Spotgamma](https://support.spotgamma.com/hc/en-us/articles/15214218424595-Implied-Volatility-IV-Explained-What-It-Is-and-How-to-Use-It)
- [FOMC December 2025 projections](https://www.federalreserve.gov/monetarypolicy/fomcprojtabl20251210.htm)
- [Whisper number 70% accuracy — EarningsWhispers about page](https://www.earningswhispers.com/about-whispers)
- [Estimize crowd estimates 64% accuracy — TechCrunch profile](https://techcrunch.com/2012/04/16/estimize-challenges/)
- [Novelty + topicality in business news — EPJ Data Science](https://link.springer.com/article/10.1140/epjds/s13688-017-0123-7)
- [First-story-detection novelty decay paper](https://www.academia.edu/112063523/Counteracting_Novelty_Decay_in_First_Story_Detection)
- [Stock reaction to news drift + reversal](https://www.researchgate.net/publication/317672162_Stock_Price_Reaction_to_News_and_No-news_Drift_and_Reversal_After_Headlines)
- [Discord/Telegram trading signals — Hashtag Investing review](https://www.hashtaginvesting.com/blog/best-telegram-trading-groups)
- [ExtractAlpha — 5 best alt-data sources for hedge funds](https://extractalpha.com/2025/07/07/5-best-alternative-data-sources-for-hedge-funds/)
- [Lowenstein Sandler 2025 alt-data adoption report — VertData summary](https://vertdata.com/blog/alternative-data-hedge-funds-guide)
- [FS-GCLSTM supply chain — arXiv 2303.09406](https://arxiv.org/abs/2303.09406)
- [Heterogeneous GNN stock prediction — Springer 2025](https://www.sciencedirect.com/science/article/pii/S0167923625001290)
- [TFT-GNN hybrid model — MDPI 2025](https://www.mdpi.com/2673-9909/5/4/176)
- [Ensemble blending 57% MSE reduction — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8446482/)
- [Stacking vs boosting vs blending — Springer Big Data review](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00299-5)
- [FinBERT financial sentiment — arXiv 2306.02136](https://arxiv.org/abs/2306.02136)
- [FinBERT-enhanced sentiment with SHAP — MDPI 2025](https://www.mdpi.com/2227-7390/13/17/2747)
- [News + sentiment ML investigation — MDPI JRFM 2025](https://www.mdpi.com/1911-8074/18/8/412)
- [SelectLLM selective prediction OpenReview](https://openreview.net/forum?id=JJPAy8mvrQ)
- [Selective Conformal Risk Control — arXiv 2512.12844](https://arxiv.org/html/2512.12844v1)
- [Know When to Abstain — arXiv 2505.15008](https://arxiv.org/html/2505.15008)
- [Increase Alpha — AI-driven trading framework arXiv 2509.16707](https://arxiv.org/html/2509.16707v1)
