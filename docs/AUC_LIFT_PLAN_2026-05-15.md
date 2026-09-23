# MARKET RADAR — AUC Lift Plan

**Generated:** 2026-05-15
**Current state:** main 5d val_auc 0.6002 (median across 5 walk-forward folds, std 0.0306)
**Target:** main 5d val_auc 0.62+ with fold std <0.020, top-bucket val_auc 0.65+
**Budget for this push:** $25 LLM round + ~10 days of dev iterations

---

## Why the AUC isn't moving as much as expected

Today's audit of `src/market_radar/ml/` revealed five things the headline number hides:

1. **LLM-classified rows have a LOWER 5d win rate (42.6%) than heuristic-only rows (45.9%).** This is the loudest signal in the data. The LLM is either reading noisier signals, mislabeling them, or being routed to the wrong rows. Until this is fixed, throwing more LLM budget at the problem makes it *worse*, not better. **This is the single highest-priority finding.**

2. **Train/val AUC gap is 0.16** (0.71 train, 0.60 val) despite aggressive regularization. That's not a tuning problem — it's almost certainly subtle label leakage from one of the temporal features (weekly_offset, composite_score-with-corroboration, or backward market context computed at an off-by-one timestamp).

3. **Fold-to-fold AUC swings 7 percentage points** (0.6221 → 0.5487). The model is regime-fragile. No regime features exist in the current 60-feature set.

4. **The binary >0% label is too noisy.** 45.7% positives, most clustered near zero. The tradeable edge lives in the |return| > 2% tails, but the model is being trained to predict sign, not magnitude.

5. **Only 25% of raw_signals have body text populated.** LLM coverage is capped at 3.9% as a result. The SEC body fetcher built last week needs to be run at scale before another LLM round is worth doing.

---

## Ranked plan, with expected lift and cost

| # | Workstream                          | Expected lift to main val_auc | Cost          | Risk         |
|---|-------------------------------------|-------------------------------|---------------|--------------|
| 1 | Audit & fix LLM-row underperformance| +0.005 to +0.020              | $0            | Low          |
| 2 | Hunt label leakage (train/val gap)  | +0.005 to +0.015 OR -0.02     | $0            | Med (might reveal we've been over-stating AUC) |
| 3 | Regime features                     | +0.005 to +0.010              | $0            | Low          |
| 4 | Pre-event price action features     | +0.005 to +0.015              | $0            | Low          |
| 5 | Drop / audit sparse features        | +0.000 to +0.005              | $0            | Low          |
| 6 | Triple-barrier + magnitude labels   | +0.000 main, +0.03-0.08 on tradeable bucket | $0 | Med (changes downstream API) |
| 7 | Targeted $25 LLM round + structured extraction | +0.005 to +0.015 main, +0.02-0.04 per-bucket | $25 | Low (verify_llm_sample.py gate) |

**Sequencing logic:** Do 1–2 first because they cost nothing AND they may reveal we've been measuring AUC wrong. There is no point spending another $25 on LLM until we know the LLM rows aren't actively hurting us. Once 1–2 are settled, 3–5 are quick adds. 6 is a structural change worth its own session. 7 happens only after 1 lands.

---

## Phase 1 — Diagnose before spending (this weekend, $0)

### Prompt 1A: Audit why LLM-labeled rows underperform

```
Hard rules:
- Read-only analysis. Do not modify any tables, do not retrain, do not call LLM API.
- Back up data/market_radar.db before any future write step.

Context:
- 7,042 / 181,786 raw_signals have an llm_classifications row.
- LLM-classified rows show 42.6% 5d positive-return rate vs 45.9% for heuristic-only rows.
- This is counterintuitive — LLM was supposed to add signal, not subtract it.

Steps:
1. SQL: for each event_type from llm_classifications, compute count, 5d win rate, mean return_5d_pct, median composite_score. Save to outputs/llm_audit_event_type.csv.
2. SQL: same breakdown but grouped by source (sec_edgar_backfill_*, sec_edgar live, rss_*, reddit_*). Save to outputs/llm_audit_source.csv.
3. Compare LLM-row composite_score distribution vs non-LLM-row. Are LLM rows getting selected because they look more important to the filter, but actually are 8-K boilerplate?
4. Open src/market_radar/llm/filter.py and report the exact rules: min composite, min body length, source allowlist, event_type allowlist. Quote the lines.
5. Pull 20 random llm_classifications rows, joined with raw_signals (title + first 500 chars of body) and signal_outcomes (return_5d_pct). Format as a table I can scan. I want to see WHAT the LLM is classifying.

Backup:
- Read-only, no backup needed.

Success:
- CSVs produced. Filter rules quoted. 20-sample table printed.
- You hypothesise WHY LLM rows underperform (likely candidates: filter selects routine filings, LLM mislabels event_type, LLM-row training subset is regime-skewed, or LLM-row outcomes are calculated with a different timestamp).

Failure:
- If SQL errors, stop and report the schema mismatch. Do not guess columns.

Status report:
- Report the CSVs, the 20-row sample, your hypothesis, and a one-line recommendation for the fix.
```

### Prompt 1B: Hunt label leakage in the 0.16 train/val AUC gap

```
Hard rules:
- Read code first, then run training with permutation tests. No model deployment.
- Do not change defaults in scripts/train_ml.py — make all changes in a new scripts/train_ml_leakage_audit.py I can throw away.

Context:
- Current model trains to 0.7083 AUC, validates at 0.6002. Gap is 0.16 despite l2=1.0, min_samples_leaf=200, max_leaf_nodes=15.
- Walk-forward CV is correct (TimeSeriesSplit n=5). So the gap is feature-level leakage, not split leakage.
- Suspects in order: weekly_offset, composite_score (uses corroboration_count which might include posts that arrive AFTER the anchor), market_context features (spy_5d_return etc. — verify they're computed STRICTLY from data available at published_at, not the snapshot timestamp).

Steps:
1. For each of the 60 features in features.py, write one line stating: "computed strictly from data available at raw_signals.published_at? YES/NO/UNSURE." Confirm by reading the feature-build code.
2. Pick the top 5 features by HGB feature_importance. For each, run a permutation test: shuffle ONLY that column on val data, retrain calibration, re-measure val_auc. Drop in AUC should be proportional to importance — if a feature has 30% importance but shuffling it only drops AUC 0.005, it's probably leaking via correlation with another feature.
3. Specifically retrain WITHOUT composite_score and WITHOUT weekly_offset (one experiment each) and report val_auc + train/val gap.
4. Specifically retrain on data where I FORCE the market_context lookback timestamp to be (published_at - 1 day) instead of whatever's current. Report val_auc + train/val gap.

Backup:
- Copy data/market_radar.db to data/market_radar.db.bak_2026-05-15_leakage before run.

Success:
- Feature-by-feature timestamp audit table.
- Permutation importance table (compare to declared feature_importance).
- Two retrain experiments (no composite, no weekly_offset) with reported metrics.
- Verdict: leakage source identified, OR "no leakage found, gap is genuine overfit, recommend stronger regularization".

Failure:
- If training crashes, stop and report. Don't silently degrade.

Status report:
- Top 3 leakage suspects ranked. One concrete fix recommendation. Estimated val_auc impact of the fix.
```

---

## Phase 2 — Cheap structural wins (next week, $0)

### Prompt 2A: Add regime + pre-event features

```
Hard rules:
- Backup data/market_radar.db before any write.
- All new features must declare in code "computed_at: published_at-strict" with a unit test that fails if any feature value uses post-published_at data.
- Do not remove existing features in this prompt — purely additive.

Context:
- Fold-to-fold val_auc swings 7 points across walk-forward (0.6221, 0.6002, 0.6059, 0.5640, 0.5487). No regime features exist.
- Pre-event price action also absent — current market context is symbol-level lookback, not anchor-day mechanics.

Steps:
1. Add the following features to src/market_radar/ml/features.py (or features/market.py if that's the right module — confirm before editing):
   REGIME (5):
   - vix_regime: 0/1/2 tertile of trailing-60d VIX
   - spy_trend: 1 if SPY close > SPY 50d > SPY 200d, -1 if reverse, 0 otherwise
   - sector_rotation: 20d % spread of XLK minus XLU
   - realized_vol_regime: 0/1/2 tertile of trailing-60d SPY realized vol
   - return_skew_20d: skewness of SPY daily returns last 20d
   PRE-EVENT (4):
   - gap_at_signal_pct: (open[signal_day] - close[signal_day-1]) / close[signal_day-1] * 100
   - return_3d_pre_signal_pct: 3-day return ending at close[signal_day-1]
   - volume_zscore_5d_at_signal: (vol[signal_day] - mean_vol_20d) / std_vol_20d
   - intraday_range_signal_day_pct: (high[signal_day] - low[signal_day]) / close[signal_day-1] * 100
2. Build a one-row unit test per feature confirming the timestamp constraint.
3. Retrain. Report per-fold val_auc, fold std, train/val gap, calibration ECE.

Backup:
- cp data/market_radar.db data/market_radar.db.bak_2026-05-15_regime_features

Success:
- 9 new features wired. Tests pass. Retrain completes.
- Compare to baseline (current main 0.6002, std 0.0306, gap 0.1596).

Failure:
- If new feature computation requires data we don't have (e.g. intraday OHLCV not in cache), STOP. Report the gap. Don't fake it with daily data.

Status report:
- Before/after table: median val_auc, fold std, train/val gap, top 10 feature importances. One-line "worth keeping" verdict per new feature.
```

### Prompt 2B: Drop sparse features that don't pull weight

```
Hard rules:
- Read-only audit first; deletion is a separate confirmed step.

Context:
- Several features in features.py have ~96% null rates (gtrends_zscore, wikipedia_pageviews_zscore, eps_surprise_pct, etc.). Tree models tolerate nulls but the splits add noise.

Steps:
1. For each of the 60 features, report null rate on the training set.
2. For each feature with null rate > 70%: train a leave-one-out model dropping ONLY that feature and report val_auc delta.
3. Recommend keep/drop for each. Drop only if val_auc delta is non-negative AND null rate > 70%.

Success: a table of {feature, null_rate, val_auc_without_it, keep_or_drop}.
Status: I'll confirm before any deletion lands.
```

---

## Phase 3 — Structural change (next week+, $0)

### Prompt 3A: Triple-barrier + magnitude-bucket labels

```
Hard rules:
- This is a LABEL change. Old binary model stays in place behind a feature flag; new label model is parallel until validated.
- Do not change the decision API on the dashboard until per-bucket calibration is verified.

Context:
- Current label: return_5d > 0. Result: model learns to predict sign of near-zero noise.
- Goal: a magnitude model that predicts P(|return_5d| > 2%) AND a separate model that predicts P(return > +2%) conditional on a move firing. Decisions get triggered only when magnitude AND direction both fire above threshold.

Steps:
1. Add triple-barrier label generator in src/market_radar/ml/labels.py. Barriers: TP at +3 ATR(14), SL at -1.5 ATR(14), timeout at 5 trading days. Label = which barrier hit first (+1, -1, 0).
2. Train three parallel models:
   a. p_move = P(|return_5d| > 2%)
   b. p_up_given_move = P(return > +2% | |return| > 2%)
   c. existing binary >0 model (keep as baseline)
3. Report val_auc for each. Also report the "tradeable subset" metric: on rows where p_move > 0.55, what's the win rate of p_up_given_move > 0.55? Compare to current STRONG_BUY hit rate.

Success: tradeable-subset win rate exceeds 60% on enough rows (n>=50) per event bucket.
```

---

## Phase 4 — Targeted LLM round ($25, only after Phase 1 fixes land)

### Prompt 4A: Plan the targeted LLM round

```
Hard rules:
- Do not call the LLM API in this prompt. This is planning only.
- The actual run goes through scripts/verify_llm_sample.py FIRST (10-sample $0.05 dry-run with manual sign-off).

Context:
- $25 budget. ~6,000 classifications at current Haiku 4.5 pricing if bodies average 3,000 input tokens.
- Phase 1A audit will reveal which event_type x source combos are highest-EV for additional LLM coverage.

Steps:
1. Read outputs/llm_audit_event_type.csv and outputs/llm_audit_source.csv from Phase 1A.
2. Compute EV per row class as: (outcome_variance * (1 - current_llm_coverage_in_class)). Rank classes.
3. Propose the top-N classes that fit in $25, with row count and expected coverage uplift per class.
4. Propose a STRUCTURED EXTRACTION schema (not just event_type) — fields we want the LLM to pull. Candidates per bucket:
   - M&A: deal_value_usd, deal_premium_pct, friendly_or_hostile, cash_vs_stock_pct
   - Earnings: eps_surprise_pct, revenue_surprise_pct, guidance_change_direction, guidance_change_pct
   - FDA: drug_name, phase, indication, primary_endpoint_met
   - Insider: role, transaction_type, shares, value_usd, cluster_size_30d
   - Activist: filer_name, stake_pct, intent_summary
5. Show me what the new prompt would look like (don't run it).

Success: a written run plan with sample size, cost, expected coverage uplift, structured schema, and the new prompt. I approve before any spend.
```

### Prompt 4B: Execute the LLM round (after 4A approval)

```
Hard rules:
- MANDATORY: python3 scripts/verify_llm_sample.py --n 10 --live MUST run and return PASSED before the full backfill.
- Hard daily cap: $25. Soft per-batch cap: $5 (so I can pull the plug between batches).
- Backup data/market_radar.db before run.
- Print full LLM input + output for the first 5 calls of each new event bucket so we catch empty-body or hallucination bugs at $0.05 not $5.

Steps: standard run_llm_backfill.py flow (see run_llm_backfill.py --help), with the new structured-extraction prompt from 4A.

Success: 6,000+ new classifications, structured fields populated, $25 cap not breached, daemon llm_classify resumed.
Failure: if verify_llm_sample.py reports >50% "other" or any structured field missing >70% of the time, STOP and tell me. Do not push through.

Status: report spend, count, top event_type distribution, top extracted_fields fill rate, and re-run scripts/train_ml.py with the new features. Compare main val_auc + per-bucket val_auc to today's 0.6002 / bucket numbers.
```

---

## Phase 5 — Verification (always, after every phase)

### Prompt 5: Re-train + sanity-check predictions

```
Hard rules:
- Always after a feature/label change.
- Backup data/market_radar.db (model artefacts go to data/models/, those are kept automatically).

Steps:
1. python3 scripts/train_ml.py
2. Print: median val_auc, std across folds, train/val gap, calibration ECE per bin, top 15 feature importances.
3. Pull 20 random predictions from the latest model where model_p_5d > 0.55. For each, print: title, event_type, composite_score, model_p_5d, the top 5 contributing features (via SHAP if available, else permutation).
4. Sanity check: do the top contributing features make domain sense? Or is the model leaning on source-name shortcuts and weekly_offset?

Success: AUC moved in the expected direction OR you have a clean reason it didn't. Sample predictions look domain-sensible.
Failure: AUC dropped > 0.005 vs prior best → roll back. AUC moved as expected but features look like leakage → flag for Phase 1B reopen.
```

---

## What success looks like after all phases

- Main 5d val_auc: 0.6002 → 0.625+ (median across folds)
- Fold std: 0.0306 → <0.020
- Train/val gap: 0.1596 → <0.10
- Top bucket val_auc: m_a 0.5876 → 0.64+, earnings 0.5790 → 0.62+
- Tradeable-subset win rate (when both magnitude and direction fire): ≥ 60% on n ≥ 50 per bucket
- Calibration: ECE < 0.03 across all probability bins

If we hit 4 of 5 of those, the system is genuinely ready for sized real-money positions. If we hit 0–2 of them, the public-data ceiling (~0.58–0.62 per the original expectation) is the real cap and the next investment is paid point-in-time data, not more model tweaks.

---

## What I'm NOT proposing right now

- More LLM budget beyond the $25 cap (we don't know yet if LLM rows even help)
- Switching off sklearn HistGradientBoosting to LightGBM/XGBoost (libomp pain already documented; sklearn HGB is competitive with the regularization in place)
- Deep learning / transformer over signals (would need 10x the data we have)
- Twitter/X integration ($200/mo)
- Options flow / IV surface data (not in budget; revisit only if AUC hits ceiling on free data)
