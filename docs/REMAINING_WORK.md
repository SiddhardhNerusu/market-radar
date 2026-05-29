# MARKET RADAR — what's left to do

Honest punch list of everything not yet built or finished, ranked by
expected impact / dollar value. Use this to pick your next CLI prompt.

Last updated 2026-05-14 after the LLM body redo.

---

## ✅ DONE (for reference, so you don't worry about these)

- 60-feature global model with walk-forward CV + isotonic calibration
- Multi-horizon (1d / 5d / 20d) + per-event-type bucket models
- Graph-only "predict from network alone" model
- Stacked ensemble (HGB + LightGBM)
- Conformal prediction intervals
- Risk manager with 7 hard rules (kill-switch, caps, drift coupling)
- T212 currency normalization (UK pence → USD)
- Real-time Telegram + macOS notifications (~1.2s latency)
- Live feed dashboard tab with auto-refresh
- BUY / SELL columns + click-to-expand detail panel
- Graph-only ranked view (predicted to rise / fall)
- Daily DB backup cron (3am, 14-day rotation)
- Weekly auto-retrain cron (Sunday 4am)
- Drift detector with Telegram alerts (every 6h)
- Daemon health-check (5-min interval, Telegram if dies)
- 749 insider transactions parsed with proper share counts
- 6,832 SEC bodies fetched
- LLM body redo with proper event_type taxonomy
- Cheat sheet documentation

---

## TIER 1 — DO RIGHT AFTER THE REDO FINISHES (this week)

These build directly on what the redo just produced.

### 1. Calibration refit on new labels  *(30 min, $0)*
**Why:** The screening showed predicted=0.56 → actual=0.73 — model is
under-confident. The redo retrained the model already, but the
isotonic calibrator was fit on the old broken-label data. Re-fitting
isotonic on the fresh properly-labeled validation set will fix that.

**How:** Wrap into a CLI prompt that loads the new model, re-fits the
calibration map on the most recent 2000 resolved signals with new
labels, saves the new wrapped model.

### 2. Per-event-type hit-rate dashboard panel  *(2 hr, $0)*
**Why:** The redo just trained per-event-type bucket models (m_a,
earnings, fda, insider). You need a UI panel that shows the MEASURED
hit rate for each bucket as outcomes resolve. "earnings_beat" might
be 62% accurate while "m_a_rumor" might be 48%. You only trade the
buckets that show real edge.

**How:** New `/api/edge` endpoint that joins llm_classifications with
signal_outcomes, groups by event_type, returns hit_rate + n. Dashboard
panel renders a sortable table.

### 3. Paper-trade tracking endpoint  *(3 hr, $0)*
**Why:** Right now paper-trading is a manual Google Sheet. Could
build a `/api/paper-trade` POST endpoint that records a decision
(BUY/SELL/SKIP) per signal, then auto-computes 1d / 5d / 20d returns
from yfinance. After 30 entries, calculates aggregate hit rate.

**How:** New table `paper_trades` with (signal_id, decision, entry_price,
1d_return, 5d_return, 20d_return). Dashboard adds "Record decision"
button to each pick card.

### 4. Lower the SELL threshold if too quiet  *(5 min, $0)*
**Why:** The redo may have shifted the probability distribution. After
running for ~24h, check how many alerts you actually got. If less than
3/day, lower the buy threshold to 0.62 / sell threshold to 0.35.

**How:** Edit .env, restart daemon. Or via CLI prompt.

---

## TIER 2 — NEXT WEEK (free, high impact)

### 5. News article body fetching  *(4 hr, $0)*
**Why:** Most news ingestors store only titles. Bodies would let the
LLM extract much richer event_type + structured fields (deal size,
analyst firm, etc.) from news the same way it does for SEC filings.
~+0.01-0.02 AUC.

**How:** Extend `ingestors/rss_news.py` with a body-fetcher similar to
the SEC body fetcher. Be polite about rate limits per source.

### 6. Reverse-lookup endpoint  *(2 hr, $0)*
**Why:** "Show me all past signals for AAPL and what happened" —
useful for manual research before deciding to trade.

**How:** `/api/history/<ticker>?days=90` returns signal_scores joined
to signal_outcomes for one ticker.

### 7. Per-ticker portfolio simulation  *(4 hr, $0)*
**Why:** Given the alerts we generate, what would a $10k portfolio
have returned over the last 30 days if you'd taken every BUY signal
at half-Kelly sizing?

**How:** New script `scripts/simulate_portfolio.py` walks through past
alerts, applies sizing logic, computes cumulative P&L vs SPY.

### 8. Notification snooze + watchlist  *(2 hr, $0)*
**Why:** Sometimes you want quiet hours (sleeping) or only want alerts
for tickers you actively track.

**How:** Add `NOTIFY_QUIET_HOURS_UTC` to .env (e.g. "00-06"). Add
`watchlist.json` for user-priority tickers — alerts on these always
fire even during cooldown.

### 9. Sector-momentum dashboard panel  *(2 hr, $0)*
**Why:** Quick visual: which sectors are running hot vs cold today.
Helps you understand the macro backdrop your individual signals are
sitting in.

**How:** Compute sector-median 5d return from SECTOR_PEERS map +
yfinance. Show heatmap or bar list on dashboard.

### 10. FINRA short interest fix  *(1 hr, $0)*
**Why:** Currently returns 0 rows; their JSON API shape is finicky.
Squeeze candidates are a real signal class.

**How:** Inspect their current API response shape, adjust the parser
in `refresh_short_interest.py`.

### 11. Reddit PRAW auth upgrade  *(1 hr, $0)*
**Why:** Currently public scrape. PRAW auth gives author karma + age
+ better rate limits. Per project_market_radar_phase2 — the keys are
already in your env.example.

**How:** Drop in PRAW client where current Reddit ingestor does
public json scrape. Use the existing `REDDIT_CLIENT_ID` /
`REDDIT_CLIENT_SECRET` env vars.

---

## TIER 3 — NEXT MONTH (free, polish)

### 12. Earnings call transcript LLM tone  *(3 hr build + ~$5-10/mo)*
**Why:** Per LSEG 2025 research, transcript-tone signals predict
next-month outperformance for high-sentiment stocks. ~+0.01-0.02 AUC
on earnings-related signals.

**How:** Finnhub `/stock/transcripts` (free tier) → store text →
classify via Anthropic Haiku for tone (-1 to +1) → feature
`transcript_tone_last_quarter`.

### 13. LLM embeddings for filing similarity  *(12 hr + ~$15 one-off + $2/mo)*
**Why:** "This 8-K reads like prior 8-Ks that returned +5% on average
in 5 days." Vector similarity over filings unlocks pattern-matching
the model itself can't do.

**How:** Voyage AI embeddings ($0.10/1M tokens) on every filing body →
FAISS or pgvector index → nearest-5-neighbours lookup at score time →
feature `similar_filings_avg_5d_return`.

### 14. SEC-EDGAR direct 13F parser  *(8 hr, $0)*
**Why:** Currently relies on Finnhub paid tier for 13F flow. Could
parse 13F-HR filings directly from EDGAR (free) and compute
quarter-over-quarter institutional buyers minus sellers per ticker.

**How:** New ingestor that downloads 13F XML, parses information
table, aggregates per quarter end.

### 15. Walk-forward live simulation  *(4 hr, $0)*
**Why:** Real PnL with fees + slippage modelling, vs the current val
AUC which doesn't reflect dollar outcomes.

**How:** Backtest framework that replays the daemon's decisions
through history, applies T212 fee schedule, calculates Sharpe / max
drawdown / win rate.

### 16. A/B testing framework  *(6 hr, $0)*
**Why:** Compare deployed model vs candidate model on live signals
before promoting the candidate.

**How:** Train both models on retrain. Score every live signal with
both. Track 5d outcomes per model. Promote candidate only if it beats
deployed by ≥ 0.005 AUC over n ≥ 200.

### 17. Insider cluster definition refinement  *(2 hr, $0)*
**Why:** Current `insider_cluster_size_30d` counts unique insiders in
30d. The research-backed definition is "3+ DISTINCT insiders BUYING
(transaction_code=P) in 30d." Tighter signal.

**How:** Adjust the compute_insider_clusters logic in features.py to
only count P (Purchase) transactions, then retrain.

### 18. Quantile regression for asymmetric loss  *(8 hr, $0)*
**Why:** Long-only rewards getting upside right; shorting rewards
getting downside right. Current symmetric log-loss treats both the
same. Quantile regression at p10 / p50 / p90 gives asymmetric
direction probabilities.

**How:** Sklearn GradientBoostingRegressor with `loss='quantile'`,
train three (10/50/90) per horizon.

---

## TIER 4 — PAID DATA (evaluate after observation)

### 19. Options flow (Unusual Whales) — $57-97/mo
Unusual options activity is a strong informed-money signal. Per literature,
60-70% directional accuracy on next-week move when filtered right.

### 20. Polygon.io tick-level data — $79/mo
Better backtest realism + intraday signal capacity. Order-book imbalance
features become possible. Not relevant until you trade intraday.

### 21. Ortex premium short-interest — $69/mo
Higher-frequency short-interest data than free FINRA bimonthly.
Better squeeze-prediction.

### 22. Estimize buy-side estimates — $50/mo
Crowd estimates 64% accurate on ≥4-estimate stocks. Free tier is
limited; paid unlocks more coverage.

### 23. EarningsWhispers paid API — $50/mo OR headless browser scrape
Whisper-number alignment is the single highest-leverage earnings
feature (70% more accurate than consensus per 25y data). Currently
deferred because their site is JS-only.

### 24. Twitter/X Basic tier — $200/mo
Per memory: defer unless other levers exhausted. Reddit + StockTwits
already cover most social-retail signal.

### 25. Voyage AI / OpenAI embeddings — ~$10-20/mo
Required for item #13 (filing similarity).

### 26. Anthropic ongoing — $30-50/mo
Already paying for it. Covers daemon's live LLM classifications +
weekly transcript tone if you add item #12.

---

## TIER 5 — DEFER UNLESS PROBLEM ARISES

### 27. Auto-start daemon on Mac reboot
You said you'd keep Mac on 24/7. If you ever restart it, you'll need
to manually `nohup python scripts/run_daemon.py &` again. If that
becomes annoying, add the launchd plist.

### 28. T212 CFD adapter
No T212 CFD API exists. You execute manually in the app. Unchanged.

### 29. SMS / WhatsApp alerts
Telegram covers this well enough.

### 30. Mobile-optimized dashboard
Current dashboard is responsive; works on phone. Native iOS/Android
app is overkill.

### 31. Crypto on-chain data integration
For RIOT/MARA/COIN, the BTC 5d return feature already captures most
of the sympathy. Tracking on-chain hash rate / fundamentals is
diminishing returns.

### 32. Tax-loss harvesting / wash-sale rule
US-specific. Not relevant for UK trader using T212.

### 33. Transformer time-series models (PatchTST / NHITS)
30+ hr build. HGB is fine for our signal density. Transformer wins
are largest at high-frequency, which isn't your use case.

### 34. Reinforcement learning for execution
30+ hr build. Standard RL fails on financial markets. Don't bother.

### 35. Bayesian neural network ensemble
Conformal prediction (already shipped) gives 80% of the benefit at
1/4 the work.

---

## HOW TO PRIORITIZE WHEN YOU COME BACK

The bar for "should I build this next" should be:

1. **Has the redo's val_auc landed?** If it's ≥ 0.61, the model is
   solid enough to focus on observation + items #2 (edge dashboard)
   and #3 (paper-trade endpoint). If < 0.60, prioritize #1
   (calibration refit) and #17 (insider cluster refinement).

2. **Have you observed 2 weeks of paper-trade outcomes?** If not,
   build #2 and #3, then WAIT. Don't add more features.

3. **Does hit rate justify spending?** After 2 weeks of paper data:
   - hit_rate ≥ 60% → start real money on the gated subset; items
     #12 (transcript tone) and #13 (LLM embeddings) become worthwhile
   - hit_rate 50-60% → polish items #5 (news bodies), #10 (FINRA),
     #17 (insider refinement) first
   - hit_rate < 50% → debug calibration; don't spend more on data

4. **Avoid feature-bloat.** Every new feature adds dimensionality and
   risk of overfitting. The current 60-feature model is already on the
   edge. Better to fix calibration than add 5 more features.
