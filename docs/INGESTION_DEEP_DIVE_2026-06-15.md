# MARKET RADAR — Ingestion Deep-Dive & Rethink

**Date:** 2026-06-15
**Branch:** `rebuild/overhaul-2026-06-15`
**DB:** `data/market_radar.db` (1.36 GB snapshot, 2026-06-15 18:12)
**Scope:** catalyst/signal ingestion as "the moat" — find the highest-yield rethink and the single most exploitable signal in the data we already have, separating real cost-surviving out-of-sample (OOS) edge from in-sample / momentum / multiple-testing artifacts.

Every number below was re-verified against the live DB on 2026-06-15. Where the upstream investigation disagrees with the current DB, the current DB wins and the discrepancy is flagged.

---

## 1. Honest verdict on the current ingestion

**Grade: C / C+.** The plumbing is good. The *evaluation substrate* and a few scoring wires are what make every downstream verdict garbage-in.

### What is genuinely solid (keep)
- **Failure isolation & operational hardening.** `poll()` never raises, per-row try/except, daemon_health logging, cached ingestor instances (TCP pools persist), staggered startup, WAL mode, FD bump. One bad source cannot crash the daemon.
- **Coverage breadth.** SEC 8-K/6-K/4/13D/13G/S-1/DEF-14A, ~50 RSS+Google-News feeds, Alpaca firehose, Nasdaq halts, market movers. The net is cast across the whole market, not a watchlist.
- **Cost model is honest and correct.** `_round_trip_cost_frac` (replay.py) is price-bucketed **400 bps sub-$1 / 300 / 200 / 120 / 50 / 20 bps liquid** — verified by running it. The docstring openly states the old flat 5 bps was "1-2 orders of magnitude too low." This is the single most intellectually honest piece of the system and it does real work: it is *why* microcap catalyst chasing does not net out.
- **`projections.py`** reports empirical historical distributions (median/P25/P75/hit-rate) instead of inventing numbers — the right primitive to build on.

### The real problems (the moat leaks here)

**(A) The "clean" universe is not 24 days — it is 7, and 3 of them are one market rally.** Verified clean scored-day distribution:

| scored day | n (clean 5d-resolved) | regime |
|---|---|---|
| 2026-05-12 | 140,978 | down (–1.0% all-source) |
| 2026-05-13 | 2,462 | down |
| 2026-05-14 | 1,084 | down |
| 2026-05-15 | 1,183 | down |
| 2026-05-27 | 10,763 | **rally (+8.8%)** |
| 2026-05-28 | 17,468 | **rally (+8.8%)** |
| 2026-05-29 | 2,922 | rally (+3.9%) |

80% of clean weight is one back-fill-rescored day (05-12); the only positive returns live in the 3-day late-May rally. Any source over-sampled on the rally days looks like alpha. **This single fact explains every "lead" in the brief.**

**(B) Pseudo-replication.** `stocktwits_trending` re-emits the same hot ticker every poll. Verified: 35,565 clean rows collapse to **1,673 distinct (day,ticker) bets — 21.3× inflation**. DELL alone on 2026-05-28 = **1,339 rows / 1,339 distinct external_id / 1,255 distinct content_hash**. Every win-rate and mean computed on raw rows weights by *polling frequency*, not information.

**(C) Backfill look-ahead contaminates the SEC "leads."** Verified `sec_edgar_backfill_8-k`: **avg latency 329 days, avg price_at_flag $805,224, max $31,200,000,000/share.** Returns are measured at *scoring time*, not filing time, and the `price_at_flag>=1` clean rule does **not** catch a $31B price. The headline `sc_13d +4.21%` is from a single in-sample batch with ~595-day latency.

**(D) Two live scoring wires are mis-specified.**
- `min_source_weight = 7.0` is the production gate (replay.py:98). Verified the `source_weight>=7` subset is **net –1.30% (all), –1.37% ex-rally, 40.6% win** — the gate selects a net-*negative* slice. Higher curated credibility ≠ predictiveness.
- `activist_position` is force-labeled **sentiment = 0.0 (neutral)** on all 2,155+ rows (`signal_class = t1|activist_position|neutral|factual|...`) **AND** is listed in `blocked_event_types` (replay.py:104). A SC 13D activist initiation — unambiguously bullish in 15 yrs of event studies — is both neutralized and screened out. Own-goal (low-magnitude, but free to fix).

> **Correction to the upstream investigation:** it claimed `model_p_5d` maxes at 0.558 on non-backfill data, so the `p_buy_min>=0.65` gate "fires on 0 rows." **This is no longer true.** The model was retrained and deployed 2026-06-04 (val_auc 0.7214); on the current DB `model_p_5d` reaches 1.0 and **20,358 rows clear 0.65** (concentrated on live days 06-01..06-15). The gate is live, not dead. (The `source_weight>=7` mis-specification, however, still holds.)

**(E) Highest-density free feature is collected and thrown away.** 8-K **Item codes** are present in **6,142** filing bodies and trivially parseable — verified regex hits: Item 9.01, 7.01, 5.07, 8.01, 1.01, 5.02, 2.03, 3.02... Yet `llm_classifications.event_subtype` is **NULL on all 8,560 rows** and every 8-K collapses to one `material_event`/`other` bucket. The taxonomy that separates an M&A agreement from a routine exhibit is sitting unparsed.

**(F) The purpose-built "bang" ingestors are still unmeasurable.** Verified: `nasdaq_halts` 802 scored / **0 clean**, `market_movers` 1,335 / **0 clean**, `alpaca_news` 6,953 / **0 clean**. The sources literally designed to catch the edge cannot be validated or refuted. (Live `sec_edgar`, by contrast, now has **1,956 clean resolved outcomes** — up from 0 at audit time — so the resolver *does* work; halts/movers are gated out earlier, likely by sub-$1 / NULL-published_at.)

---

## 2. Rethought ingestion architecture

The moat is **not a new data source and not raw speed** (HFT owns speed). It is **(i) an evaluation layer beta cannot fool, (ii) correct counting, and (iii) turning the free structured SEC stream into a hard event taxonomy.** Ordered by yield:

### 2.1 Evaluation layer — make beta unable to masquerade as alpha (do this FIRST)
- **Score and train on day-demeaned / cross-sectional abnormal return** = `return − same-day all-signal mean` (or `− SPY`, once `benchmark_prices` is populated; it is currently empty). Raw-return labels teach the model "it was an up day." Verified effect: this flips stocktwits from "+4.16% star" to negative.
- **Collapse to one observation per (ticker, scored-day) / content_hash before any stat.** Kills the 21× pseudo-replication.
- **Segregate backfill from live by latency.** Any row with `published_at→scored_at > 48h` is historical; never pool it with live rows for training or edge claims, and re-fetch `price_at_flag` at true `published_at` or drop it.
- **Every leaderboard shows per-day n + day-demeaned stat beside the raw stat.** Any edge that collapses under demeaning is auto-flagged "regime, not signal."
- **Tighten the clean rule** to `data_corrupt=0 AND price_at_flag BETWEEN 1 AND 2000 AND ABS(return_5d_pct) <= 100` and backfill `data_corrupt=1` on out-of-range rows.

### 2.2 Sources — add / fix / drop
- **FIX (highest yield): 8-K Item-code parser.** Regex `Item\s+(\d\.\d\d)` over the already-fetched body → `event_subtype`. Score 1.01 (material agreement), 2.02/9.01 (results), 4.02 (restatement, bearish), 5.02 (exec change), 8.01 (other) **separately**. Free, already-collected, structured, low-false-positive. Bias toward low-attention (small, less-covered) names where the literature finds the drift.
- **FIX: SEC EDGAR latency.** Live `sec_edgar` averages **79.5 min** publish→score (verified) — far better than gnews (3-5 h) but worse than the ~7-min ideal. Cause is the synchronous body-fetch inside the poll loop holding the write connection. Decouple: ingest a stub row in <1 min, hydrate+rescore body out-of-band.
- **FIX: activist label.** Split `activist_position` → `activist_position_new` (SC 13D, allow, bullish bias) vs `_amend`/13G (block/passive). Stop forcing sentiment=0.0. Remove `activist_position` from `blocked_event_types`. **Caveat:** justified by external event studies, *not* by the in-sample number — do not size on it.
- **FIX: Form-4 capture.** Persist transaction_code (P/S/A), role_score, notional, and a routine-vs-opportunistic flag (calendar-regularity of the filer) so the literature's only durable insider signal — *clustered opportunistic officer BUYS* — can be isolated. Currently everything is one direction-blind `insider_transaction`.
- **ADD (cheap, schema already exists): FINRA bi-monthly short-interest** → empty `short_interest` table, as a slow overlay factor (not a trigger). **ADD: clinicaltrials.gov v2** catalyst calendar for biotech binaries.
- **DROP / deprioritize:** per-ticker tagging of macro gnews feeds (china_tariffs, layoffs, fed_macro — 49-71% corrupt, the "MSN" publisher-suffix penny-stock bug); price_action breakouts as long signals (net-negative day-demeaned); FTD, USAspending, paid options flow (poor latency/cost/evidence for a solo dev).

### 2.3 Dedup & corroboration redo
- Dedup at flag time: one signal per (ticker, day) / content_hash. Add MinHash/embedding near-dup clustering so paraphrased republications collapse.
- Redefine corroboration as **count of distinct PUBLISHERS** (map reuters_*_google→reuters, gnews_*→google_news, etc.) **AND ≥2 distinct content_hashes** — not count-of-feeds. Verified: the current `corroboration_count` is ~95% stocktwits self-corroboration and has **zero** predictive power once collapsed to events.

### 2.4 Classification / extraction upgrade
- Replace the one 26-way mega-prompt with **form-specialized extractors** (13D Item-4 intent, 8-K Item-code, earnings-surprise) using constrained/tool-call JSON. 38% of current classifications come back `other` — wasted budget.
- Add an allowed-event-type coercion (unknown → `other`) and add the missing `EVENT_IMPACT`/`EVENT_BIAS` entries for valid labels (`earnings_announcement`, `product_launch`, etc.) that currently silently flatten to impact 1.0.
- Bridge the backfill LLM corrections into `signal_scores` (or stop classifying backfill) — `signal_scores.event_type` agrees with `llm_classifications` only 44.9% of the time. **But** label hygiene here unlocks ~no P&L (the rescued buckets are net-negative anyway); do it for cleanliness, not alpha.

### 2.5 Latency / velocity
- Emit a **per-ticker rolling signal-velocity / attention-acceleration** feature (count of prior same-ticker signals in 24h, causal/prior-only). It is the honest version of "corroboration." See §3 for why it is a *feature to test*, not a green-lit edge.
- Capture **price at signal genesis** + a "minutes-since-first-mention" field. The few signals with raw strength had already moved +5-13% by day 1 — `price_at_flag` is a fictional entry.

### 2.6 Scoring made predictive, not hand-tuned
- Retire the additive hand-picked composite (`source_weight/3.33 + EVENT_IMPACT + 0.5*corr − 2.0*anti_pump…`). Verified non-monotonic / inversely predictive at the top once day-demeaned (band 8-10 are the *worst* relative to their own day).
- Fit a logistic/GBM on `event_type + sentiment + price-bucket` against the **day-demeaned** target, validated on held-out *dates*. Drop `source_weight` and `anti_pump_flag` as positive terms (both anti- or zero-calibrated). Wire `author_quality` in only if it predicts day-demeaned return, else delete it (currently computed and unused).

---

## 3. The single most exploitable signal in the data we already have

**Honest answer: there is no validated, cost-surviving, out-of-sample LONG edge in the current data.** Every candidate is a regime / pseudo-replication / backfill artifact. The *least-dead* candidate, and the one worth instrumenting forward, is **signal velocity / swarm** — but it does **not** clear the bar today.

### Velocity/swarm (causal, deduped, liquid) — VERIFIED numbers
Reconstructed independently: ≥16 prior same-ticker signals in a causal 24h window (bisect on publish times, no look-ahead), one trade per (ticker, day), live sources only, net of `_round_trip_cost_frac`:

| cut | n | net 5d | win% | median |
|---|---|---|---|---|
| ALL days | 641 | **+2.67%** | 48.8% | **−0.18%** |
| **EX-RALLY (drop 05-27/28/29)** | 177 | **−3.18%** | 39.0% | **−2.03%** |

**Why it is the most-defensible candidate yet still FRAGILE→ARTIFACT:** it *does* dodge the worst traps — it deduplicates to episodes, is causal/no-look-ahead, concentrates in liquid $50+ names (cost ~20 bps, not a microcap cost trap), and survives leave-top-5-tickers-out. **But** its entire positive mean is the rally days; ex-rally it is net-negative with a coin-flip-minus win rate and a negative all-days median. The data-miner's headline "+4.06% / 54.5%" was restricted to the rally regime only.

### Why every other "lead" fails (verified)
| lead | verdict | verified kill |
|---|---|---|
| **stocktwits_trending** | ARTIFACT | dedup ALL +2.87% → **EX-RALLY −2.31%, 38.7% win**; raw rows EX-RALLY **−7.69%**. 21.3× pseudo-replicated. Day-clustered t≈0.08. |
| **sc_13d / activist** | FRAGILE | only positive SEC family, but 595-day backfill latency, prices to $7.5M, NORD +209% double-counted, median net ≈0, CI touches zero, reverses by 20d. Label fix is real; the trade is not. |
| **signal_scores feature mine** | ARTIFACT | every cut (corroboration, px≥50, source_weight) goes negative ex-rally. `source_weight≥7` net −1.37%. |
| **corroboration ≥3** | ARTIFACT | +9.4% alpha but 24% is DELL's earnings gap; ex-top-10-megacap → −4.30%. PEAD on a handful of reactors in a rally. |
| **8-K backfill firehose** | net-negative | −1.35% net, 94% undifferentiated `other` — *because Item codes aren't parsed* (see §2.2). |

### The one time-stable, OOS-robust effect found
**S-1 / S-1A registrations: net −4 to −11% across 23/23 published-months (t ≈ −7.7).** A high-confidence **AVOID-LONG / short-watch overlay** ("never go long into a fresh registration"), not a tradeable long (names are illiquid).

---

## 4. Ranked opportunity list (potential × feasibility)

| # | Opportunity | Potential | Feasibility | Verified status | Concrete steps |
|---|---|---|---|---|---|
| 1 | **Recover the June stocktwits holdout** | High (only true OOS test available) | High (data already captured) | VERIFIED: 105,397 rows scored 05-30..06-08, **98,198 have price_at_flag, 100% return_5d NULL, 100% resolve_attempts=0**, 2,172 tickers; all 5d windows now closed | Re-run `outcomes/tracker.py` over those score_ids (or backfill prices). Yields ~14× the entire current clean live sample. Then re-test velocity + corroboration-on-liquid-earnings-reactors day-demeaned. |
| 2 | **Day-demean + dedup the whole evaluation layer** | High (makes every verdict trustworthy) | High (SQL + one harness) | VERIFIED to flip stocktwits sign | Add `return − same-day mean` as the scored target; collapse to (ticker,day)/content_hash; show per-day n + demeaned stat on every leaderboard; exclude/re-time backfill. |
| 3 | **8-K Item-code parser → event_subtype** | High (turns 1 mushy bucket into a hard taxonomy; free) | High (regex over fetched body) | VERIFIED: codes in 6,142 bodies, event_subtype 100% NULL | `re.findall(r'Item\s+(\d\.\d\d)')` at ingest → `event_subtype`; score 1.01/2.02/5.02/8.01/4.02 separately; bias to low-attention names. |
| 4 | **Fix the source_weight≥7 gate** | Medium (stops trading a net-negative slice) | High (one constant) | VERIFIED: subset net −1.30% | Drop `min_source_weight` as a hard gate or recalibrate from day-demeaned return per source. |
| 5 | **Async SEC body-hydration** | Medium (79.5→~3 min) | Medium (worker refactor) | VERIFIED: live latency 79.5 min, body-fetch in poll loop | Ingest stub in <1 min, hydrate+rescore out-of-band; never hold the SQLite writer across HTTP. |
| 6 | **Fix activist_position label + split 13D/13G** | Low magnitude, high conviction (15 yr OOS) | High (label map) | VERIFIED: sentiment hardcoded 0.0, in blocked_event_types | Split new vs amend; bullish bias on 13D; un-block. Track forward — do NOT size on in-sample number. |
| 7 | **Form-4 buy/role/size/cluster capture** | Medium (durable literature edge) | Medium (parser fields) | VERIFIED: fields absent; insider_transaction net −0.84% | Persist code P/S, role_score, notional, routine flag; isolate clustered opportunistic officer BUYS on a 1-12mo horizon. |
| 8 | **Resolve halts/movers outcomes + gate to price≥1 at ingest** | Medium (untested moat sources) | Medium | VERIFIED: 0 clean outcomes on all three bang ingestors | Find why resolver skips them (sub-$1 / NULL published_at); parse HaltDate/HaltTime into published_at; let 5d windows mature. |
| 9 | **Velocity feature, instrument forward** | Medium (best candidate, unproven) | High (rolling counter) | VERIFIED: ex-rally net −3.18% | Emit causal per-ticker 24h velocity; re-test on holdout (#1) day-demeaned before trusting. |
| 10 | **S-1/S-1A avoid-long overlay** | Low (defensive only) | High | VERIFIED: 23/23 months negative, t≈−7.7 | Hard "no long into fresh registration" filter. |

---

## 5. Does any of this realistically move us toward the income goal?

**Not yet — and honesty requires saying so plainly.** The current data **cannot certify an edge exists**; it can only certify that the evaluation can't yet detect one. The prior "no edge" verdict stands, for tighter reasons: 7 clean days, 3 of them one rally, 21× pseudo-replication, and backfill look-ahead. With 81 sources × ~40 event_types × ~5 price buckets × ~8 features × 7 day-slices, *any* 5/7-day winner is expected by chance.

**Realistic ceiling for a solo dev:** a small, cost-surviving, day-demeaned drift edge in **liquid (>$10, ideally >$50) names** where cost is 20-50 bps — most headline return is beta. Sub-$5 microcap catalyst chasing is structurally a cost sink (200-400 bps round-trip vs negative medians). Treat any surviving edge as a positive-skew *tail* strategy with strict sizing and a low win rate, never "the mean is my per-trade expectation."

**Data volume needed to prove anything:**
- **Minimum bar:** ≥15-20 *independent* scored days spanning ≥2-3 market regimes (we have 7, with 3 dominant). The single fastest way to get there is **recovering the 97.6k-row holdout (opportunity #1)** — it instantly multiplies the clean live sample ~14× and gives the only genuine OOS test.
- **Promotion gate for any signal:** must survive (1) dedup to (ticker,day), (2) day-demean/market-excess, (3) net of `_round_trip_cost_frac`, (4) a per-day bootstrap CI excluding zero across ≥15 distinct days, (5) a walk-forward fit-early/test-late split. Nothing clears this bar today.

**The honest moat is process, not a magic feed:** correct counting + beta-neutral evaluation + free structured SEC item-code arbitrage + accumulating clean low-latency days. Build that, recover the holdout, and *then* ask whether velocity or item-conditioned 8-K drift is real. Until then, the work is "instrument the pipeline to detect a real edge," not "deploy the edge we found."
