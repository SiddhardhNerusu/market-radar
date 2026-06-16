# MARKET RADAR — Catalyst/Signal Ingestion: C+ → Honest A/A+ Blueprint

**Date:** 2026-06-15
**Branch:** `rebuild/overhaul-2026-06-15`
**DB:** `data/market_radar.db` (1.38 GB)
**Python:** `.venv/bin/python` (`PYTHONPATH=src`)
**Scope:** This blueprint is about **engineering quality** of catalyst/signal *ingestion* — how cleanly, completely, quickly, and measurably raw signals enter the system and get labelled. It is **NOT** a claim of trading edge. Where a grade depends on accumulating more market days (not on writing code), the dimension is explicitly flagged **data-gated** and capped at A-/A.

> **Honesty contract:** every "A" below is earned by an *objective, re-runnable acceptance criterion* an independent re-audit can execute against the code or DB. No grade is asserted. Items that cannot be lifted by code alone are not promised — they are flagged and their ceiling is stated.

---

## 0. What was already fixed this session (verified — build ON, do not redo)

All five verified present in code + DB this session:

| Fixed item | Verification |
|---|---|
| Beta-neutral, deduped, cost-aware eval harness + label hygiene | `scripts/edge_screen_v2.py` (115 lines) day-demeans + nets cost; `tracker.py:29-35` `_MIN_ANCHOR_USD=1.0` / `_MAX_ABS_RETURN_PCT=600` flag `data_corrupt`. |
| 8-K Item-code parser → `event_subtype` | `ingestors/sec_item_codes.py` (76 lines); **2,730** `llm_classifications` rows carry `event_subtype` (DB query). |
| Corroboration **de-syndication** | `composite.py:409` `COUNT(DISTINCT COALESCE(rs.content_hash, 'id:'||rs.id))`. |
| `activist_position` split (13D-init bullish, amends blocked) | `scoring/heuristics.py`. |
| reconcile-gap Telegram alert | `daemon_health_alerts` table exists; `live_trader.py`. |

**Correction to the audit's storage claim:** `event_subtype` lives on **`llm_classifications`**, not `raw_signals` (`PRAGMA table_info` confirmed). The 8-K subtype→event_type reconciliation (Item B2 below) therefore reads/writes `llm_classifications.event_subtype`, not a `raw_signals` column. Any design step referencing `raw_signals.event_subtype` must be retargeted.

---

## 1. Honest current per-dimension grade — and why the overall is C+

All grades and figures below were re-verified against the live DB and source files this session.

| # | Dimension | Grade | Hard evidence (this session) |
|---|---|---|---|
| a | **Coverage / breadth** | B− | 80 sources live. **Form-4 parser exists but is backfill-only**: `parse_form4_xml` is called **only** in `scripts/refresh_insider_transactions.py:77` — never in `ingestors/` or `daemon.py` (grep confirmed). `short_interest`=**0 rows**; `earnings_whispers`/`institutional_holdings` empty. No clinicaltrials.gov primary source. |
| b | **Latency** | C+ | `base.py:107` opens `with get_connection() as conn:` and calls `self.parse(entry)` at line 110 **inside** that block; `sec_edgar.py:152-160` runs `SecBodyFetcher.fetch_body()` (synchronous HTTP, 15-30s timeouts) inside `parse()`. The SQLite writer is held across the network call. SEC poll interval = **300s** (`daemon.py:61`). |
| c | **Classification accuracy** | C+ | **37.5%** of `llm_classifications` are `event_type='other'` (3,215/8,566). **8-K: 40.8% other** (2,657/6,521). One 26-way enum prompt; `prompt.py:14` literally says *"when in doubt, choose 'other'"*. `classifier.py:96` is a plain `messages.create` with **no** `output_config`, followed by manual code-fence stripping (`:130-136`) + `json.loads` reject path (`:137-141`). |
| d | **Dedup / corroboration** | A− | Exact `content_hash` dedup + de-syndicated corroboration both real (`composite.py:409`). **But** exact-prefix hash only — **no** SimHash/MinHash anywhere (grep clean). Near-dups (one filer's 663 424B2s) mint 663 distinct hashes → defeat corroboration + training dedup. |
| e | **Scoring predictiveness** | C+ | `composite.py:22-27` is a hand-tuned additive constant stack. Only **19** distinct clean 5d-resolved days / **23** 1d-days (DB). edge_screen_v2: composite non-/inversely-predictive once day-demeaned; `other` bucket = 69% of bets at net −0.67%. |
| f | **Measurement / outcome coverage** | C+ | `market_movers`=**1,348/1,348** `published_at` NULL; `nasdaq_halts`=**809/809** NULL (DB). Bang sources have short-horizon clean outcomes (alpaca_news 2,171 / movers 72 / halts 40 clean_5d) but **0 fully_resolved** (age artifact: ingest started 2026-06-04/06-08). Sub-$1 floor (`tracker.py:34`) nukes ~43% of movers into `data_corrupt`. |
| g | **Reliability** | B | `daemon_health` is current-state only (PK source) — no time-series; FINRA refresh failed *silently* (inserted=0, no alert) for a month. |

### Why the overall is C+ (not B, not C)
The system is **architecturally sound and instrumented** (dedup at A−, reliability at B, an honest eval harness already built), which floors it above C. But three load-bearing dimensions are simultaneously weak and **mutually reinforcing**:

- **(c) 37.5% `other`** poisons **(e) scoring** (the 69% `other` bucket *is* the net-negative slice the gate selects) and **(f) measurement** (mislabelled events can't be measured against the right outcome window).
- **(b) latency** holds the SQLite writer across HTTP, which both delays scoring and starves WAL checkpoints (back-pressure on every other writer).
- **(c)'s** single biggest contributor is a **data-corruption bug**, not a model limit: `sec_edgar.py:129` `form = raw_entry.get("form","")` reads the *feed-query filter* (set at `:104`/`:122`), not the entry's real form. **463** rows are classified as Form-4 while their title is `424B%` (debt prospectus) — confirmed `SELECT COUNT(*) ... json_extract(raw_payload,'$.form')='4' AND title LIKE '424B%'` = 463 over `llm_classifications`, 6,999 over all `raw_signals`.

A system whose largest accuracy defect is a one-line tag bug, whose latency defect is a writer-held-across-HTTP anti-pattern, and whose scoring is un-fit constants is correctly graded **C+**: real foundations, three avoidable defects dragging the core.

---

## 2. Achievable overall grade — and the honest engineering-vs-data split

**Achievable overall with the proposed work: A− (engineering), trending to A as market days accumulate.**

The honest split — which sub-dimensions are *engineering-fixable to A/A+ now* vs *data-gated*:

### Fixable to A / A+ with code ALONE (no waiting on the market)
- **(c) Classification → A.** Form-tag bug fix + form-routed structured extraction + deterministic Form-4/8-K maps + `output_config.format`. The 37.5%→<12% target is a *labelling-pipeline* outcome, fully under code control. The only residual that is data-gated is "are the labels *correct* beyond the 300-row gold set" — and that residual is bounded by the gold-set macro-F1 criterion, so the **engineering** ceiling is a clean A.
- **(b) Latency → A.** Stop holding the writer across HTTP; out-of-band hydrate daemon job; defer LLM on un-hydrated stubs; drop poll to 60s. All wiring of existing components. Median <5 min / p90 <8 min is a pure-engineering criterion.
- **(a) Coverage → A.** Form-4 live wiring + new fields (10b5-1, notional, role); FINRA `consolidatedShortInterest` (verified HTTP 200 live this session); ClinicalTrials.gov v2 (verified HTTP 200 live). All "named source ingested + populating + verifiable" — engineering criteria.
- **(d) Dedup → A.** SimHash + LSH bands + `dup_cluster_id`, make every consumer cluster-aware. Pure code.
- **(f) Measurement — the `published_at`=0%-NULL and price-tier parts → A.** Parsing a timestamp at ingest and splitting the price floor from the corruption flag are pure engineering.
- **(g) Reliability → A.** `daemon_health_history` append-only table + zero-insert alert (reuse the reconcile-gap path). Pure code.

### DATA-GATED — code can build the *machinery* to A-, but the *edge claim* needs more market days
- **(e) Scoring predictiveness — A is data-gated.** With only **19** distinct clean 5d-days, a learned ranker **cannot** be validated for persistent OOS skill regardless of code quality. The *engineering* deliverable (LGBMRanker on day-demeaned excess return, purged day-walk-forward, monotone constraints, a hard distinct-day deploy gate, fallback to the hand-tuned composite) is fully buildable to **A−** now. The gate then refuses to promote until ~40+ days exist. **A is earned later by the market, not by this PR.**
- **(f) Measurement — full bang-source resolution is partly calendar-gated.** `fully_resolved>0` for halts/movers needs ~20 trading days from the 2026-06-08 ingest start (≈2026-07-08). The *code* (timestamp + price-tier) lands now; the 20d outcomes accrue on a clock.

**Bottom line:** Six of seven dimensions reach A by code alone. Scoring's machinery reaches A−; its A is honestly deferred to data. Overall **A− now, A when ~40 clean days exist** — and that is the *only* honest framing.

---

## 3. Ranked build plan (impact × effort, highest first)

Ranking principle: **fix data-corruption and writer-blocking first** (they poison everything downstream and are cheap), then unlock measurement, then the learned scorer (gated), then dedup, then reliability. Every item has file-level steps and an objective, re-auditable acceptance criterion.

---

### ① Form-tag corruption fix + backfill — **GATES EVERYTHING**
**Impact: Very High · Effort: ~0.5 day · Data-gated: No**

Single highest-ROI item. `sec_edgar.py:129` reads the feed-query filter as the form. 463 debt prospectuses (`424B%`) are classified as Form-4; ~380 of them fall to `other`. Every downstream dimension (c, e, f) inherits this poison.

**Steps**
1. `ingestors/sec_edgar.py:128-129` — derive the real form from `entry.title`'s leading token (e.g. `"424B2 - ..."` vs `"4 - ..."`) or `entry.category`/`entry.tags`, **not** from `raw_entry["form"]` (which is the `type=` filter set at `:104`/`:122`). Use the parsed form for `raw_payload.form` and the `FORM_TYPES` lookup.
2. One-off `scripts/backfill_form_tags.py` over `raw_signals` (mirror `analyze_8k_items.py`): for `source='sec_edgar'`, recompute `raw_payload.form` from the `title` prefix; UPDATE in batches.
3. Re-queue the corrected rows for re-classification (clear/flag their `llm_classifications` so the form router re-runs).

**Acceptance criterion (re-auditable)**
- `SELECT COUNT(*) FROM raw_signals WHERE source='sec_edgar' AND json_extract(raw_payload,'$.form')='4' AND title LIKE '424B%'` = **0** (baseline this session: 6,999 in `raw_signals`, 463 in `llm_classifications`).
- After re-classification, the count of `llm_classifications.event_type='other'` rows whose joined title is `424B%` and form-tag was `'4'` drops to 0.

---

### ② Stop holding the SQLite writer across HTTP + out-of-band hydrate
**Impact: Very High · Effort: ~1 day · Data-gated: No**

`base.py:107` holds `get_connection()` across `parse()` (`:110`), and `parse()` runs `fetch_body()` synchronously (`sec_edgar.py:154`). One slow EDGAR doc back-pressures every writer.

**Steps**
1. `ingestors/sec_edgar.py` — default `fetch_bodies=False` on the **live daemon** path so `parse()` writes a stub row (`body=summary` RSS metadata, NO network). Keep the body-fetch code, gate it off in the poll path. Confirm `base.py:107-147` performs **zero** HTTP inside the open `with get_connection()` block.
2. New `ingestors/sec_hydrate.py` `hydrate_sec_bodies(batch_size=15)` extracted from the inner loop of `scripts/fetch_sec_bodies.py`: `SELECT ... WHERE source='sec_edgar' AND (body IS NULL OR body LIKE '<b>Filed:</b>%') ORDER BY published_at DESC LIMIT 15`, fetch via the shared cached `SecBodyFetcher` (module-global `_RATE_LOCK`, `sec_body_fetcher.py:247`), **then** open a short per-row connection to UPDATE `raw_signals.body` — connection opened *after* fetch returns.
3. `daemon.py` — register `_job_sec_hydrate` on a ~20s `IntervalTrigger` via `_safe()` + `max_instances=1` + stagger.
4. `llm/classifier.py` candidate WHERE — add `AND NOT (rs.source='sec_edgar' AND (rs.body IS NULL OR rs.body LIKE '<b>Filed:</b>%'))` so the LLM never burns on a stub; `composite.rescore_with_classification` corrects once hydrated.
5. `daemon.py:61` — drop `sec_edgar_seconds` 300 → 60.
6. Add a `sec_edgar` latency panel (trailing-24h median + p90 of `(scored_at − published_at)`).

**Acceptance criteria (re-auditable)**
- **Writer-never-held-across-HTTP (binary):** with the daemon's `SecEdgarIngestor(fetch_bodies=False)`, the call graph of `base.poll()` (`:107-147`) reaches no `requests.*` call. Equivalently, assert no SQLite write-connection is open on the thread inside `SecBodyFetcher._get_with_retry` except when called by the hydrate job.
- **Latency:** trailing-24h **median** publish→score for live `source='sec_edgar'` (excluding `*_backfill_*`) **< 5 min** AND **p90 < 8 min**, via `SELECT (julianday(ss.scored_at)−julianday(rs.published_at))*1440 ...`.
- **Hydration completeness:** ≥95% of live SEC 8-K/Form-4/13D rows from the trailing 24h have non-stub body within 5 min of ingest.
- **No LLM waste on stubs:** zero `llm_classifications` rows joined to an un-hydrated SEC stub.
- **SEC rate compliance:** across a full RTH day with 60s poll + 20s hydrate live, daemon logs show zero sustained 429/403 from www.sec.gov.

---

### ③ Form-routed structured extraction (the 38%-other fix)
**Impact: Very High · Effort: ~1.5-2 days · Data-gated: No (engineering ceiling = A; label-correctness beyond gold set is bounded, not promised)**

Depends on ① (clean form tags) and benefits from ② (hydrated bodies). Replace the one mega-prompt + free-text JSON with a form router + schema-constrained output + deterministic maps.

**Steps**
1. **Schema-constrained output** — `classifier.py:96`: switch `messages.create` to pass `output_config={"format":{"type":"json_schema","schema":...}}` (verified GA on `claude-haiku-4-5` this session via docs). Define a base schema (`event_type` enum + sentiment/magnitude/factual/confidence) and per-form variants narrowing the enum. **Delete** the code-fence stripping + `json.loads` reject path (`:130-141`); keep one defensive `try/except` for the refusal `stop_reason` only.
2. **Form router** — `classify_one()` dispatches on `raw_payload.form`: `'4'/'4/A'` → deterministic Form-4 parser (no LLM); `'8-K'/'8-K/A'` → item-code-conditioned extractor; `'SC 13D'/'SC 13D/A'` → 13D Item-4 intent; `'425'` → M&A; else/news → generic schema-constrained call.
3. **Form-4 deterministic extractor (no LLM)** — wire `sec_form4_parser.parse_form4_xml` at classify time. Map `P→insider_buy`, `S→insider_sell` (open-market), `A/M/F`→`insider_routine`. Emit `extracted_fields {code, role, notional_usd, is_routine, is_10b5_1}`. (8,721 stored Form-4 bodies contain `transactionCode`, 8,762 contain `aff10b5One` — DB-verified.)
4. **8-K subtype→event_type map** — feed `sec_item_codes.dominant_subtype` into the prompt as a strong prior AND apply a deterministic map for unambiguous cases: `officer_or_director_change→leadership_change`, `results_of_operations→earnings_announcement`, `notice_of_delisting→delisting`, `non_reliance_restatement→restatement`, `bankruptcy_or_receivership→bankruptcy`, `completion_of_acquisition→m_a_announcement`, `unregistered_equity_sale→dilution`. Read/write `llm_classifications.event_subtype` (the real column).
5. **Confidence gate** — add `event_type='unclassified_low_confidence'` (vs `'other'`=confidently no-catalyst) when `confidence < 0.55` (informed by the 0.66-on-other / 0.84-on-non-other split). Wire it so it does NOT pass the trade gate (`composite.rescore_with_classification`) and is re-queueable, not burned.
6. **Gold set + harness** — `scripts/eval_classification.py`: sample ≥300 filings/news stratified by form+source, store spot-checked labels in a new `gold_classifications` table, report overall + per-form `other` rate + macro-F1. Mirror `analyze_8k_items.py`.

**Acceptance criteria (re-auditable)**
- Overall `other` rate (excluding `unclassified_low_confidence`) **< 12%** on a re-classified sample ≥1,000 rows (baseline 37.5%).
- **8-K `other` < 15%** (baseline 40.8%); every 8-K whose dominant item code is in the deterministic map gets a non-`other` event_type (0 exceptions).
- 0 rows where form=`'4'` but title `424B%` (inherited from ①); on genuine Form-4 rows, ≥95% get insider_buy/sell/routine with non-null `extracted_fields.code` + `.role`.
- **100%** of stored classifications come from `output_config.format` — verified by **zero** `json.loads`-parse-failure log lines over a full re-classification run (structurally eliminates the historical 96%-other incident).
- On the ≥300-row gold set, macro-F1 over top-12 event types **≥ 0.80**; the low-confidence gate captures ≥70% of would-be-misclassified rows into `unclassified_low_confidence`.

---

### ④ Bang-source `published_at` + price-tier (unlock measurement)
**Impact: High · Effort: ~1 day · Data-gated: Partly (full 20d resolution is calendar-gated)**

**Steps**
1. `halts.py:151` — build ISO `published_at` from `HaltDate`+`HaltTime` (ET→UTC) = the t0 event time. `market_movers.py:152` — set `published_at = datetime.now(timezone.utc)` explicitly (honest: detection IS the event for a screener), not `None`. Unit test asserts non-NULL.
2. `outcomes/tracker.py` — split the price floor from the corruption flag: add a `price_tier` column (`penny`|`small`|`std`) via `db._migrate_columns`; compute returns normally for anchors in `[0.10, 1.0)`; keep `data_corrupt` only for sub-$0.10 anchors and `|return|>_MAX_ABS`. `edge_screen_v2.py` can still FILTER on `price_tier` so trading conclusions stay liquidity-honest.

**Acceptance criteria (re-auditable)**
- `SELECT SUM(published_at IS NULL) FROM raw_signals WHERE source IN ('nasdaq_halts','market_movers') AND ingested_at > <deploy>` = **0** (baseline 809 + 1,348 = 100% NULL).
- After resolver re-run, each of nasdaq_halts/market_movers/alpaca_news has ≥50 clean (non-`data_corrupt`) resolved 1d outcomes; movers sub-$1 recovered rows (under `price_tier='penny'`) raise its clean_1d sample by ≥40%.
- **[Calendar-gated]** by ~2026-07-08 (20 trading days after 2026-06-08 ingest start), nasdaq_halts + market_movers each show `fully_resolved>0` and ≥20 clean_20d outcomes. (Today 0 = age artifact, not a bug.)

---

### ⑤ High-alpha structured sources + fields
**Impact: High · Effort: ~2.5 days to green on 1-4 · Data-gated: No (engineering = sources ingested + verifiable)**

Order by ROI: Form-4 fields → macro-tagging bug → FINRA → ClinicalTrials → (stretch) 13D Item-4.

**Steps**
1. **Form-4 fields** — `sec_form4_parser.py:107`: add `is_10b5_1 = (_extract(xml,'aff10b5One') or '0') in {'1','true','True'}` and per-transaction `notional = shares*price`. `ALTER TABLE insider_transactions ADD COLUMN is_10b5_1 INTEGER; ADD COLUMN notional REAL;` (idempotent in `refresh_insider_transactions.py`). Add features `insider_opportunistic_buy_notional_30d` (sum P-buys, role≥2, not 10b5-1) and `insider_routine_share` in `external_features.py` (~`:411`); register names in `features.py`. Widen the `NOT EXISTS` backfill guard (`:65`) to re-parse rows missing `is_10b5_1`. (Current `insider_transactions` = 1,463 rows, lacks both new columns — DB-verified.)
2. **Macro-gnews tagging bug** — `rss_news.py`: add `is_macro: bool` to `FeedSpec`, set True on `gnews_fed_macro`; in `parse()`, if macro, keep only tickers with `confidence ≥ 0.9` (drop the 0.55 ALL_CAPS body path). Tag `event_type='macro'`. Cleanup: `DELETE FROM signal_tickers WHERE signal_id IN (SELECT id FROM raw_signals WHERE source='gnews_fed_macro') AND confidence < 0.9`.
3. **FINRA short_interest** — `refresh_short_interest.py:44`: change `equityShortInterest` (OTC) → `consolidatedShortInterest` (exchange-listed). Replace GET+client-filter with a POST using `compareFilters` EQUAL on `settlementDate` (partition key). **Verified live this session:** POST to `consolidatedShortInterest` with `{"compareFilters":[{"fieldName":"settlementDate","compareType":"EQUAL","fieldValue":"2026-05-15"}]}` → HTTP **200**, returns `symbolCode/currentShortPositionQuantity/averageDailyVolumeQuantity/daysToCoverQuantity/settlementDate/marketClassCode`. Wire `attach_short_interest` to read `days_to_cover` point-in-time (settlement ≤ published_at).
4. **ClinicalTrials.gov v2** — new `scripts/refresh_clinical_trials.py` (mirror `refresh_fda_catalysts.py`): GET `clinicaltrials.gov/api/v2/studies` (verified HTTP 200 live), page via `nextPageToken`, extract NCT/phase/overallStatus/completionDate/sponsor, map sponsor→ticker via `CIK_LOOKUP` (skip unmatched), write to `catalysts` (`source='clinicaltrials_v2'`). `attach_catalyst_features` lights up `days_until_catalyst` for free.
5. **(Stretch) 13D Item-4 intent** — small keyword parser → `event_subtype`. Defer unless 1-4 land green.

**Acceptance criteria (re-auditable)**
- **Form-4:** `SELECT COUNT(*) FROM insider_transactions WHERE is_10b5_1 IS NOT NULL` = total row count; `... WHERE notional > 0` ≥ 90. `features.py` FEATURE_NAMES contains `insider_opportunistic_buy_notional_30d`; a train run emits non-zero for ≥20 distinct tickers.
- **Macro:** `SELECT COUNT(*) FROM signal_tickers st JOIN raw_signals rs ON st.signal_id=rs.id WHERE rs.source='gnews_fed_macro' AND st.confidence<0.9` = **0** (baseline: 1,033 contaminated signals / 1,189 tags — DB-verified this session).
- **FINRA:** `SELECT COUNT(*) FROM short_interest` > 3,000 AND `MAX(report_date)` ≥ `'2026-05-15'` AND `COUNT(*) WHERE days_to_cover IS NOT NULL` > 2,000 (baseline 0).
- **ClinicalTrials:** `SELECT COUNT(*) FROM catalysts WHERE source='clinicaltrials_v2'` > 50, every row non-null ticker + future/recent decision_date; ≥10 emit non-null `days_until_catalyst`.

---

### ⑥ Near-dup clustering (SimHash + LSH)
**Impact: Medium-High · Effort: ~3-4 days · Data-gated: No**

`content_hash_for` (`db.py:278`) is SHA over the first 200 chars — one differing char mints a new hash. No SimHash/MinHash anywhere (grep clean). Inflates corroboration + pollutes training.

**Steps**
1. `storage/near_dup.py` — 64-bit SimHash over 3-gram word shingles of normalized title+body (pure-stdlib, no new dep). Add `raw_signals.simhash64` + `dup_cluster_id` via `_migrate_columns`.
2. `insert_raw_signal` — after `content_hash`, banded-LSH lookup (4×16-bit bands, indexed) over last 48h within Hamming ≤3; set `dup_cluster_id` to the matched cluster else mint `= this row id`. Always insert (cluster, don't drop).
3. Make consumers cluster-aware: `composite.py:409` `COUNT(DISTINCT COALESCE(dup_cluster_id, content_hash, 'id:'||rs.id))`; training selection + `edge_screen_v2.py` dedup on `dup_cluster_id`; `graph_features.py:144` co-mention on `dup_cluster_id`.
4. `scripts/backfill_near_dup.py` — one-shot backfill, batched; re-run corroboration on affected scored rows.

**Acceptance criteria (re-auditable)**
- On a trailing-14-day window, collapsed share `distinct dup_cluster_id / total_rows` rises from ~6.5% (exact-hash) to ≥18% (measured near-dup ceiling ~20.5%); the single-filer 424B2 case collapses from 663 distinct stories to ≤5 clusters.
- Regression test: insert 10 near-identical PRs (Hamming ≤3) for one ticker → `corroboration_count == 1`, not 10.

---

### ⑦ Learned beta-neutral ranker (machinery now, edge data-gated)
**Impact: High (if data) · Effort: ~4-6 days · Data-gated: YES — A is deferred to ~40 days**

Build the ranker + gate + fallback now; the gate refuses promotion until the market provides enough days. **This item's "A" is the only one honestly capped on data.**

**Steps**
1. `ml/train.py` — `_build_cs_target()`: `target = return_5d_pct − day_mean(return_5d_pct)` (the day-demean already in `edge_screen_v2.py:62-65`). Keep the raw-return HGB untouched (Kelly path); the ranker is a separate artifact.
2. `ml/rank_train.py` — `LGBMRanker(objective='lambdarank', n_estimators≤150, num_leaves≤15, min_child_samples≥200, lambda_l2≥1.0, max_depth≤4)`, one query-group per scored-day. Reuse `features.py:extract_features` but **drop `composite_score`** (circular). Monotone constraints: `+1` sentiment_magnitude/unique_sources_24h/insider_cluster/news_novelty/eps_surprise; `−1` anti_pump + a numeric "routine-ness" feature.
3. **Purged day-walk-forward** — folds over sorted DISTINCT DAYS (not rows) with a 5-day embargo; per-day Spearman RankIC + top-vs-bottom quintile demeaned spread; publish median RankIC + day-to-day std.
4. **Deploy gate** — env `RANK_MIN_DAYS=40`, `RANK_MIN_OOS_RANKIC=0.03`, `RANK_MAX_RANKIC_STD` (mirror `train.py:340-351`). Refuse promotion unless distinct OOS days ≥40 AND median RankIC > floor AND sign stable early/late. Until then serve the hand-tuned composite.
5. **Dual-write + fallback** — add `signal_scores.rank_score` + `rank_model_version`; `daemon.py` `rank_predict_pending()` after `score_pending`/`predict_pending`; switch consumers (`live_trader.py:1429/1450-1452/3299/3316`, `notifier.py:48-50`, `llm/filter.py:10-11`, `action_labels.py`) to `COALESCE(rank_score, composite_score)`. Keep `composite_score` persisted for audit.
6. **Harness** — extend `edge_screen_v2.py` / add `rank_screen.py` for OOS RankIC + cost-net quintile spread + early/late split; nightly retrain hook logs RankIC to `model_history.md`.

**Acceptance criteria (re-auditable)**
- **[Data-gate, the honest-A bar when data is thin]** If distinct clean OOS days < `RANK_MIN_DAYS` (today 19 for 5d / 23 for 1d), the gate BLOCKS promotion, the system provably serves the composite fallback, and scripts log `data-gated: N/40 days`. **Acceptance here = the gate + harness + fallback exist and the threshold is set — NOT a live edge.**
- **[When data exists]** On held-out days, deciling `rank_score` yields monotone non-decreasing mean day-demeaned forward return (Spearman ρ of 10 decile-means ≥ 0.9); median per-day OOS RankIC ≥ +0.03 with sign stable across early/late.
- **Non-regression:** `rank_train.py` runs end-to-end on the current 19-day DB and either deploys a ranker meeting the bars OR cleanly declines, leaving every consumer reading the unchanged `composite_score` (verified by a test analogous to `test_scoring.py`).

---

### ⑧ Reliability: health history + zero-insert alert
**Impact: Medium · Effort: ~0.5 day · Data-gated: No**

**Steps**
1. Add append-only `daemon_health_history(source, checked_at, success)` for per-source 7-day uptime SLOs (current `daemon_health` is PK-source, no time-series).
2. Add a "zero-insert on a source that should always insert" alert for `refresh_*.py` jobs, reusing the reconcile-gap Telegram path (the FINRA refresh failed silently for a month).

**Acceptance criteria (re-auditable)**
- `daemon_health_history` exists and supports a per-source 7-day uptime query.
- A deliberately broken refresh job (forced inserted=0) fires a Telegram alert within one schedule interval.

---

### Build-order summary

| Order | Item | Impact | Effort | Data-gated | Why this slot |
|---|---|---|---|---|---|
| ① | Form-tag corruption fix | Very High | 0.5d | No | Poisons (c)(e)(f); cheapest fix |
| ② | Writer-not-held-across-HTTP + hydrate | Very High | 1d | No | Unblocks latency + every writer |
| ③ | Form-routed structured extraction | Very High | 1.5-2d | No | Kills the 38%-other root cause |
| ④ | Bang-source published_at + price-tier | High | 1d | Partly (calendar) | Unlocks measurement |
| ⑤ | New structured sources/fields | High | 2.5d | No | Coverage to A |
| ⑥ | Near-dup SimHash/LSH | Med-High | 3-4d | No | Fixes corroboration + training dedup |
| ⑦ | Learned beta-neutral ranker | High | 4-6d | **Yes** | Machinery now, edge later |
| ⑧ | Health history + zero-insert alert | Med | 0.5d | No | Closes the silent-failure gap |

Engineering total to lift all non-data-gated dimensions to A/A−: **~2-2.5 weeks**.

---

## 4. The honest ceiling: what A+ ingestion does and does NOT buy

**What A/A+ ingestion DOES buy (toward the goal):**
- A clean, fast, complete, **measurable** signal substrate: <12% `other`, <5 min median latency, near-dups collapsed, bang sources timestamped, opportunistic-insider/short-interest/clinical-trial fields populated, every source's outcomes resolvable against the *right* event window.
- It removes the *engineering* excuses for poor edge measurement. After this work, if the system finds no edge, that is an honest empirical finding about the market — not an artifact of a tag bug, a held writer, a 38%-`other` bucket, or NULL timestamps.
- It makes the eval harness's verdicts **trustworthy**: day-demeaned RankIC on clean, deduped, correctly-labelled, correctly-windowed data.

**What A+ ingestion does NOT buy (the hard ceiling — stated plainly):**
- **It does NOT create a trading edge.** A clean pipeline that ingests perfectly can still feed signals with zero alpha. Ingestion quality is *necessary, not sufficient*.
- **It does NOT shorten the calendar.** The scoring model (⑦) and full bang-source resolution (④) are gated on **distinct market days** — today 19 (5d) / 23 (1d) vs the ~40-day floor. No amount of code manufactures independent cross-sections. The ranker's deploy gate will *correctly refuse to promote* until the days exist, and that refusal is a feature, not a failure.
- **It does NOT validate label correctness beyond the gold set.** Classification reaches an engineering A (schema-guaranteed, form-routed, <12% other), but "is every label *true*" beyond the ≥300-row gold set remains a sampling claim bounded by macro-F1 ≥0.80 — not a universal guarantee.
- **It does NOT survive the cost model on its own.** edge_screen_v2 already nets `_round_trip_cost_frac`; an A-grade pipeline can still produce signals whose gross edge is real but net-of-cost negative. That verdict is the harness's job, and it stays honest only because the pipeline beneath it is clean.

**Therefore the honest claim after this work is:** *"MARKET RADAR's catalyst/signal ingestion is engineering-grade A− (six of seven dimensions at A by code; scoring machinery at A− with a data-gated path to A). It is the cleanest, fastest, most-measurable substrate the available data supports — and it makes any future edge claim honest, because the edge question is now answered by the market and the cost-aware harness, not masked by ingestion defects."* Nothing here asserts profitability; everything here is checkable by the acceptance criteria above.
