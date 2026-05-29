# Still to do — what we haven't built (yet) and why

A frank inventory. Each item is either: (a) blocked on a paid data
source we haven't subscribed to, (b) waiting for the LLM redo to prove
out, (c) too engineering-heavy for one session, or (d) we've decided
the cost / benefit isn't favorable until earlier items are validated.

---

## 1. Saved for last — depend on Anthropic balance + LLM redo working

| Item | Status | Cost | When |
| --- | --- | --- | --- |
| LLM embeddings for SEC filing similarity (Voyage AI) | Designed, not built | ~$15 one-off + $2/mo | After bodied LLM produces real AUC lift |
| Earnings-call transcript ingestion + LLM tone scoring | Designed, not built | < $5/mo | Same gate |

Both are in the upgrade research as Tier 2 items and have real expected
lift. We aren't building them today because they share infrastructure
with the LLM body pipeline that's still re-running.

---

## 2. Blocked on paid data subscriptions you haven't bought

| Item | Cost / source | Why it matters | Wired? |
| --- | --- | --- | --- |
| Real-time options flow + unusual options activity | $57/mo Unusual Whales / $79/mo Polygon | Strong informed-money signal; 60-70% directional in literature | `.env` placeholder only |
| Dark-pool prints (premium feed) | $50-200/mo | Block-trade detection; not the FINRA aggregate we get free | Free aggregate path only |
| Level-2 / tick-by-tick order book | $79/mo Polygon, $$$+ Databento | Order-book imbalance features (64% next-tick accuracy in papers) | Not wired — intraday-only signal |
| Tick-level historical price data | $79+/mo | Better backtest realism; slippage modeling | Not wired |
| Ortex short interest (vs FINRA's bi-monthly) | $69/mo | Higher-frequency squeeze detection | FINRA only |
| Estimize buy-side estimates (API) | $50/mo | Crowd estimates 64% accurate on ≥4-estimate stocks | Not wired |
| Twitter / X firehose | $200/mo Basic | Speed + reach; but Reddit + StockTwits cover the same signal class | Not wired |
| LinkUp job postings | Enterprise | Earnings-beat predictor on hiring trends | Not wired |
| LSEG / Refinitiv value-chain data | Enterprise | Cleaner supply-chain graph for GNN-grade signals | Free 10-K path works |
| Bloomberg Terminal | $2,665/mo | Institutional reference | Not worth it for retail |

All paid placeholders are in `.env.example` (commented out, with the
exact cost listed). Uncomment + add the key when you decide a particular
source is worth paying for. The consumers (refresh scripts / feature
attaches) will activate automatically.

---

## 3. Designed but not implemented — engineering hours not invested

| Item | Hours | Why deferred |
| --- | --- | --- |
| True Graph Neural Network (PyTorch + torch-geometric) on supply-chain graph | 30+ hr | Our pragmatic alternative — extract graph-derived features + train a separate HGB on them only — captures most of the same signal at 1% of the build cost. Revisit if the simple version proves out. |
| Transformer time-series models (PatchTST / NHITS) per ticker | 30+ hr | HGB is likely fine for our signal density. Transformer wins are largest at high-frequency. |
| Reinforcement learning for execution | 30+ hr | Hard to do well without massive data; standard RL fails on financial markets in published literature. |
| Bayesian neural network ensemble | 12 hr | Conformal prediction (shipped) gets you 80% of the uncertainty benefits at 1/4 the work. |
| Quantile regression for asymmetric long-vs-short loss | 8 hr | Useful once shorting routinely; defer until long-only edge is verified. |
| Live walk-forward backtest framework with fees, slippage, position sizing | 8 hr | The validation AUC is good enough to gate paper deployment; the full PnL backtest belongs after Tier 1 is verified live. |
| Twitter / X scraper (free tier — read-only) | 4 hr | We're skipping social-graph expansion in favor of more carefully weighting Reddit + StockTwits. |
| Full SEC-EDGAR-direct 13F parser (fallback when no Finnhub key) | 8 hr | The 13F path is currently Finnhub-only; users without a Finnhub key can still ingest everything else. |

---

## 4. Data quirks we know about but haven't fully solved

- **FINRA Equity Short Interest API** returned 0 rows on our default
  empty-body GET in this session. The endpoint is reachable, but the
  request shape may need a body or different filter combo. Live test
  on a Mac with full network access may yield results — the script is
  correct in structure but un-verified end-to-end.
- **EarningsWhispers** doesn't publish an official API. Our scraper
  pattern-matches their HTML, which is brittle by definition. If they
  redesign the page, the script needs a parser refresh.
- **Wikipedia pageviews z-score** needs ≥13 weekly buckets to be
  meaningful. First few runs report `None`. After ~3 months of
  accumulated history, the feature lights up.
- **Co-mention graph** is computed from the last 90 days of signals.
  If you wipe `raw_signals` or run a fresh DB, the graph is sparse
  until the daemon has ingested for a few weeks.
- **Form 4 ownership.xml parsing** assumes the modern XML schema. Very
  old Form 4 filings (pre-2003) use SGML and won't parse. Not relevant
  for our 6-month window.

---

## 5. What I'm explicitly NOT recommending

| Item | Why skip |
| --- | --- |
| Discord / Telegram signal channels | 70% of users following these lose money per published research. No verifiable independent track record exists. Adding this as a feature is unlikely to improve the system. |
| Generic "AI trading bot" wrappers | The ones reviewed in the literature don't beat baseline. Build your own with clear, observable inputs. |
| Sentiment-only models (no other features) | Sentiment alone is too noisy — its predictive power is conditional on volume spikes + corroboration. The current system handles this correctly. |
| Penny-stock pump signals | Pump-prone subreddits already get de-rated to 2.0-3.0 weight. We don't trade these; we measure them as anti-signals. |
| Day-of-week effect arbitrage | Real but tiny (~5bp), often arbitraged away. Captured as a feature for completeness but not a primary edge. |

---

## 6. Next-3-week priorities (after LLM redo proves out)

In order of expected impact per hour:

1. **Verify body-fetched LLM redo lifts AUC**. This is the single gate
   for everything below. If it doesn't, the next step is investigating
   the LLM-output-quality issue, not adding more features.
2. **Run all the free refreshers** built this session — fails-to-deliver,
   FDA catalysts, FINRA short interest, attention data, insider
   transactions, novelty/FinBERT. Each feeds an existing feature slot.
3. **Top up Finnhub** (free key — no money) to unlock earnings PEAD,
   13F flow, calendar.
4. **Retrain with `--all`** — global + multi-horizon + per-event-type
   + graph-only + ensemble. Measure each variant's val AUC.
5. **Compute SHAP** on the deployed model. Drop or down-weight any
   feature that's pulling zero weight.
6. **Calibrate the selective gate** — find the (p, interval_width)
   thresholds that retain at least 5% of signals at ≥60% hit rate.
7. **Activate only the high-confidence-subset signals on the dashboard**.
   Document the expected weekly volume + expected return.
8. **Watch for 2 weeks**, then make the paid-source decisions.

Anything below position #7 is premature optimization.

---

## 7. Genuine "we just can't" items

| Item | Reason |
| --- | --- |
| Beat institutional hedge funds on speed at zero cost | Their latency is real and their data fees buy real edge. Our angle is event-selection on free data, not microsecond execution. |
| Predict completely surprising events (war, pandemic) | These events have no precedent in our training set by definition. The system will be wrong-footed; the risk-management framework (kill-switch, daily-loss cap, position concentration limits) is what protects against this. |
| Get to 90% hit rate overall | Public-data ceiling is ~58-62% AUC. The road to high confidence runs through *selective prediction* on a small subset, not improved overall accuracy. |
| Trade options or futures from this system | The current outputs target equity directional prediction. Options strategies need IV-surface modeling and dealer-positioning intel we don't yet have. |
| Run a multi-strategy portfolio (long + short + pairs) | Today the framework is long-event-driven. Pairs / market-neutral / vol-arb each need their own model. Pick one strategy and validate it before expanding. |

---

## 8. What the codebase is missing in operational ergonomics

- **Notification routing** — beyond macOS desktop notifications, no
  integration with Slack / Telegram / SMS. Easy to add when needed.
- **Live performance dashboard panel** — current dashboard is descriptive;
  there's no "this week's P&L vs SPY" panel. 2 hr to add.
- **Auto-retraining schedule** — train_ml runs ad-hoc today. Wiring
  the weekly cron is a 10-line systemd/launchd unit.
- **Data-quality dashboard** — daemon health is tracked but there's
  no automatic "X source has gone stale for 24h" alert routing.
- **Model rollback** — joblib files are versioned, but no UI for
  rolling back to a prior model if a new one underperforms live.
