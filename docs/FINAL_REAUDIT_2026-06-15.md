# MARKET RADAR — Final Re-Audit (Capstone)

**Date:** 2026-06-15
**Branch:** `rebuild/overhaul-2026-06-15` (UNCOMMITTED — 13 modified, 5 untracked incl. `order_gateway.py`, `scripts/{edge_screen,quarantine_corrupt_outcomes,db_maintenance}.py`, `tests/`)
**Live DB:** `data/market_radar.db` (1.36 GB)
**Tests:** `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` → **28 passed in 0.44s** (15 gateway / 4 config / 7 label-hygiene / 2 P&L)

This is the capstone re-audit. Every grade below is verified against actual code (`file:line`), live DB queries, and live test/tool runs — not prior summaries. The headline: **the rebuild succeeded at making the system SAFE and HONEST, but it did NOT find a tradeable edge, and it introduced one genuine four-figure accounting regression of its own.**

---

## 1. Scorecard — Baseline → Final

| # | Area | Original baseline | Final grade | Direction |
|---|------|:---:|:---:|:---:|
| 1 | Risk gate / order submission / safety | **D** | **A** | ▲▲▲ |
| 2 | P/L accounting truth | **F** | **B** | ▲▲▲ (with a NEW high-sev gap) |
| 3 | ML / edge measurement | **F** | **A** | ▲▲▲ (with a NEW high-sev gap) |
| 4 | Backtest realism | **F** | **B** | ▲▲▲ |
| 5 | Catalyst / signal ingestion | **D** | **C** | ▲ |
| 6 | Architecture / reliability | **D** | **B** | ▲▲ |
| 7 | Whole-system integrity | **B** | **A−** | ▲▲ |

**Net:** Five of seven areas moved up two-or-three letter grades. The two "A" technical areas (P/L truth, ML) each still carry exactly one **high-severity** gap that prevents a clean bill of health — these are the gating items for any money decision.

---

## 2. Empirical EDGE Verdict — **NO EDGE FOUND**

This is the most important finding and it is unambiguous across three independent lines of evidence run against the live data today:

1. **Honest ML AUC is ~0.55 (coin-flip).** A fresh `train_and_save()` against the live DB yields median `val_auc = 0.5676`, `std = 0.135`, `fold_aucs = [0.527, 0.568, 0.389, 0.620, 0.760]` — one fold *below* 0.5. The P1 stability gate (`train.py:347-351`, `ML_MAX_FOLD_STD=0.05`) correctly **refuses to deploy** it (`version=None`, pointer unchanged).

2. **Backtest is net-negative after honest costs.** `scripts/backtest.py --preset new_gate` → **282 trades, 47.2% hit, total −$188, −3.0% return, Sharpe −5.04.** All four presets are net-negative (old_gate, new_gate, new_gate_options, aggressive). The old→new correction (price-bucketed round-trip costs `replay.py:61-88` + day-1 stop look-ahead fix `replay.py:327-340`) flips the same 939-candidate gate from **+1.31% to −0.006%** — i.e. the "edge" was entirely an artifact of the dishonest flat 5 bps and look-ahead.

3. **Out-of-sample screen finds nothing real.** `scripts/edge_screen.py` reports "no edge" / "overfit" for every event type with a tradeable OOS sample. The lone `*** SURVIVES ***` (macro) is a **same-day-leakage artifact**: the clean dataset spans only **7 calendar days** (2026-05-12 … 2026-05-29) with **80% of rows on 2026-05-12 alone**, and the row-index split puts 2026-05-12 in *both* the in-sample and out-of-sample windows. It is noise, not survival.

**Why no edge:** this is a **data-volume limit, not a code bug.** Only ~10,315 clean, non-"other" labelled training rows exist, dominated by a handful of backfill days. The harness is now honest enough to *say so* rather than inflate a number. That is the correct outcome — but it means there is currently nothing to trade on.

---

## 3. Funding Go / No-Go — **NO-GO for both £5k and £10k**

**Decision: DO NOT fund live trading at £5k or £10k right now.**

The recommendation is driven by the *measurement* result, not the *safety* result. The safety rebuild is genuinely excellent (Area 1 = A; the TRDA death-spiral is provably impossible — see §4). But you do not deploy capital into a strategy that has **no demonstrated edge**, a **net-negative backtest across every preset**, and a **measured ML AUC of 0.55**. Funding now would be funding a coin-flip minus transaction costs — a structurally guaranteed slow bleed.

Two additional **hard blockers** independently justify the No-Go even if an edge existed:

- **Blocker A (P/L truth, HIGH):** The bot's own daily ledger reads **net +$64.09** while the true realized figure including options is **−$1,050.66**. You cannot risk-manage capital on a ledger that masks a four-figure loss. (See §5, gap 2.A.)
- **Blocker B (ML deployment, HIGH):** The currently-LIVE model (`current.json` → `sklearn_hgb_calibrated_20260615T040122Z`, `val_auc_std = 0.169`, one fold = 0.476) is a **pre-gate model the new gate would reject**, and `predict.py` re-serves it with no quality re-check. The Kelly sizing currently runs off a model the system itself would refuse to deploy today. (See §5, gap 3.A.)

**Path to a future GO** (in order): (1) fix the two high-sev gaps; (2) accrue enough *temporally-disjoint* resolved outcomes (target: ≥30–40 distinct trading days, not 7) to run a real frozen-OOS test; (3) demonstrate a positive expectancy after honest costs on that out-of-sample window with a deployed, gated model; (4) *then* fund a paper-to-small-live ramp (start well below £5k). Until step 3 shows daylight, the honest answer is **the system is safe to run but has nothing to trade.**

---

## 4. What the Rebuild Got RIGHT (verified)

- **P0 order gateway is a real single choke-point.** Only `order_gateway.py:248` calls `alpaca.submit_simple_order` for routed paths; the 5 remaining direct calls in `live_trader.py` (3180/3322/4162/4172/4317) are all documented reduce-only / broker-managed crypto/option-leg exceptions. `submit_bracket_order` (the old TRDA vehicle) has **zero live callers.**
- **TRDA death-spiral is provably impossible.** `plan_order()` blocks the TRDA short (`stock SHORT blocked long-only`) AND a $90,317 long (`exceeds absolute hard cap $2,500`, env-independent). The CLOSE path derives side/qty from the *live* position sign and clamps to `|position|`; `test_trda_death_spiral_cannot_recur` drives 10 stale "sell" closes on a −63 short and proves total BUYs == 63, ends flat, never a SELL. Gateway **fails closed** when positions are unreadable.
- **Config coherence + hard caps at import.** `config.py:221` `sys.exit(2)` on `reserve≥gross`, `hard_order>gross`, `daily_loss≥gross`, plus a `$2000` sanity ceiling. Live config (gross 6000 / hard 2500 / reserve 2500 / daily_loss 600) is coherent.
- **All-session equity circuit breaker** (`live_trader.py:964-1009`) reads Alpaca `equity − last_equity` directly (never the DB P&L column), persists the halt flag, and restores it on restart. Option losses ARE seen by the kill-switch via equity even though they are missing from the ledger.
- **No runtime ALTER TABLE.** The only `ALTER TABLE` in source is the idempotent startup migrator (`db.py:153`); `live_trader.py:393` is a comment. Verified.
- **Label hygiene is real.** Live DB: `data_corrupt` 380,867 clean / 20,833 flagged; clean `return_5d_pct` min −99.51% / max 541.83% / avg 0.626% (NOT the baseline 211% avg / 3,002,400% max). **Zero** clean rows with `|return|>600%`. The quarantine script has since been re-run: **0 corrupt rows now carry a non-null return** (the latent inconsistency several auditors flagged is currently clean).
- **ML deploy gates fire for real**, backtest costs are honest, the T212 job is flag-gated (`daemon.py:174`), and the P2 blocklist is data-driven (every blocked event_type is net-negative OOS).

---

## 5. Remaining Gaps — Prioritized

### HIGH severity (gating for funding)

**5.A — Option-spread realized P/L is omitted from `bot_daily_pnl` by every write path (NEW regression introduced by the rebuild).**
The P0 fix removed the equity-delta clobber but never routed option P/L into `_update_daily_pnl`. Verified: an `awk` scan of `_book_option_spread_pnl` shows **0** calls to `_update_daily_pnl`, and its justifying comment (`live_trader.py:~3550`: "`_reconcile_realized_pnl` owns the daily total (authoritative Alpaca equity-delta...)") is now **stale** — the rebuild made `_reconcile_realized_pnl` write ONLY `equity_delta_intraday_usd`. Live data:
- `SUM(bot_daily_pnl.realized_pnl_usd)` = **+$64.09**, byte-identical to stock-only `SUM(bot_orders.realized_pnl_usd)` = **+$64.09**.
- `SUM(bot_option_spreads.realized_pnl_usd)` = **−$1,114.75** across 39 spreads — in **zero** daily rows.
- True realized incl. options = **−$1,050.66.** The ledger again reads net-positive while masking a four-figure loss — the *same class* of failure that earned the original F.

**Fix:** call `_update_daily_pnl(conn, realized)` inside `_book_option_spread_pnl`'s booking transaction (guarded by `realized_pnl_usd IS NULL` to prevent double-booking); delete the stale comment; add a `test_pnl_truth` case for an option close. Re-grade only after `SUM(bot_daily_pnl)` reconciles to `SUM(bot_orders)+SUM(bot_option_spreads)` by Eastern date.

**5.B — The deployed model is a pre-gate model the new gate would reject.**
`current.json` → `sklearn_hgb_calibrated_20260615T040122Z` with persisted `val_auc=0.733`, **`val_auc_std=0.169`**, `fold_aucs=[0.672, 0.476, 0.817, 0.733, 0.928]` (one fold below coin-flip). Deployed before the std gate existed (gate is uncommitted). `predict.py.is_stale()` is a pure mtime/pointer check with **no quality re-gate**, so this std-0.169 model still serves `predict_proba` for Kelly sizing, and `only_replace_if_better` blocks the honest 0.57 model for being "lower."

**Fix:** on loading `current.json`, re-validate persisted `val_auc_std` against `ML_MAX_FOLD_STD` and `val_auc` against `ML_MIN_DEPLOY_AUC`; if it fails today's bar, fall back to no-ML / neutral prior. Quarantine the 0.169-std pointer now.

**5.C — TRDA loss + 32 unbooked exits never backfilled.**
The baseline remediation (`docs/AUDIT_AND_RESEARCH_2026-06-15.md:124`) to rebuild `bot_daily_pnl` history was NOT performed. Both TRDA rows (`bot_orders` id 180 filled / id 171 canceled) still have `realized_pnl_usd = NULL`; **32 filled orders** remain NULL. The historical ledger matches stock-only because the going-forward clobber was removed, not because anything was reconciled. **Fix:** a one-time reconciliation script (sibling to `quarantine_corrupt_outcomes.py`) rebuilding `bot_daily_pnl` from `bot_orders + bot_option_spreads` by Eastern exit date, backfilling TRDA from Alpaca activities. Until then, treat the Alpaca equity curve as the only authoritative P/L.

**5.D — OOS split is structurally incapable of proving an edge, and the tool doesn't hard-stop.**
`edge_screen.py` splits by row index, landing inside 2026-05-12, so IS and OOS share a calendar day and "macro SURVIVES" is same-day leakage. **Fix:** split by *distinct day* (or week); refuse any SURVIVES verdict when IS/OOS span < N distinct days or share a calendar day; print an `OOS INVALID: dataset spans 7 days` banner.

### MEDIUM severity

- **5.E — `live_trader.py` is a 4,793-line god object** (baseline 4,691, **+102** this branch — the dock number is now stale). Only the gateway was extracted; the named `ExitManager`/`PnLLedger`/`Reconciler` subsystems do **not** exist as classes (grep returns nothing). 59 inline methods remain.
- **5.F — DB VACUUM/ANALYZE never run on the live 1.36 GB DB.** Conclusive: `sqlite_stat*` tables are entirely **absent** (`db_maintenance.py` runs ANALYZE immediately before VACUUM, so their absence proves the script has never executed here); `freelist_count=0`; `integrity_check=ok`. The tool is built and correct — it just needs to run in a market-closed window.
- **5.G — Zero direct unit tests for `backtest/replay.py`.** No test pins the cost-bucket boundaries, the day-1 stop firing, the `data_corrupt` filter, or a golden positive→negative flip. A regression (restoring flat 5 bps, dropping the day-1 stop) would pass the 28-test suite silently.
- **5.H — Ingestion corroboration still double-counts syndication.** `composite.py:406` counts `DISTINCT rs.source`, not `content_hash`/publisher-domain, even though `content_hash` exists and is indexed. Live DB: 8,060 content-hashes span ≥2 sources (one wire story = 5 "corroborating sources"). `min_corroboration` defaults off, so harmless today — but raising it for "precision" would *lower* precision. ~22% of the firehose (111,473 rows) is Google-News-family feeds, none collapsed.
- **5.I — SEC body-fetch still synchronous in the poll loop** (`sec_edgar.py:152-154`); head-of-line latency on a slow EDGAR fetch.

### LOW severity

- **5.J — `dashboard/api.py` corrupt-exclusion is partly implicit.** `_measured_edge()` and the `by_source` aggregate rely on `return IS NOT NULL` rather than an explicit `COALESCE(data_corrupt,0)=0` (verified: only lines 409 and 428 carry the explicit filter). Correct today (corrupt ⇒ NULL invariant holds) but fragile. Add the explicit filter for defense-in-depth.
- **5.K — Crash-survivable broker-side stops are partial.** The trailing_stop is armed only when `market_open` and can be cooled off 300s after a failure; a daemon crash in the pre-arm/extended-hours/backoff window leaves a stock position with no broker-side exit (the gateway prevents a runaway but won't *exit* a loser). Arm a reducing GTC stop at entry regardless of session.
- **5.L — Stale `live_trader.py.bak`** (142 KB, contains the OLD runtime ALTER at line 255) sits untracked in the execution package. Not imported, harmless to execution, but audit-confusing cruft. Delete it.
- **5.M — `weekly_digest.py` headline aggregate** reads `bot_daily_pnl` and inherits the option understatement; self-corrects once 5.A lands.
- **5.N — `ticker_extractor.py:155-156`** 0.55 all-caps body path survives (baseline said drop it); low impact behind blocklist + model gate.
- **5.O — Untested crypto-open submit payload**; `plan_order` now stamps `position_intent='buy_to_open'` on crypto opens (a behavioral change vs pre-rebuild, on a branch whose gateway has never run live — 0 "gateway" lines in `daemon.log`). Confirm against Alpaca paper before first live crypto entry; add a full-submit crypto test.

---

## 6. Prioritized Roadmap (P-next)

1. **[HIGH] Route option P/L into `bot_daily_pnl`** (5.A) + delete stale comment + add P/L-truth test. *Gating for funding.*
2. **[HIGH] Re-gate the deployed model on load / quarantine the 0.169-std pointer** (5.B). *Gating for funding.*
3. **[HIGH] One-time ledger reconciliation + TRDA/32-NULL backfill** (5.C). *Gating for funding.*
4. **[HIGH] Frozen-OOS calibration:** split `edge_screen` by distinct day, hard-stop on degenerate windows, banner the 7-day data immaturity (5.D). Re-run when ≥30 distinct trading days of clean outcomes exist.
5. **[MED] Run `scripts/db_maintenance.py`** once in a closed window (VACUUM + ANALYZE the 1.36 GB DB) (5.F). Operational, not code.
6. **[MED] `tests/test_backtest_realism.py`** — cost-bucket boundaries, day-1 stop fires, corrupt filter, golden negative-expectancy fixture (5.G).
7. **[MED] Decompose `live_trader.py`** — extract `ExitManager` / `PnLLedger` / `Reconciler` (5.E); arrest the file's growth.
8. **[MED] Ingestion dedup** — rewrite `_corroboration_count` to `DISTINCT content_hash`/publisher-domain; collapse Google-News feeds; decouple SEC body fetch (5.H, 5.I).
9. **[MED→future] DSR / PBO** (Deflated Sharpe Ratio, Probability of Backtest Overfitting) before ever calling a backtest result an edge.
10. **[future] Intraday-bar backtest** — replace the optimistic "final 5d return as TP/SL proxy" (already disclosed as an upper bound in `replay.py:31-36`) with intraday bars once data exists; add a **drift detector** for live model degradation.
11. **[LOW] Cleanup** — explicit api.py corrupt filters (5.J), crash-survivable GTC stops (5.K), delete `.bak` (5.L), crypto-submit test (5.O).

---

## 7. Honest Overall Verdict

The rebuild is a **genuine, verifiable success at its primary objective: making MARKET RADAR safe and honest.** The TRDA death-spiral that motivated the whole effort is now provably impossible (single choke-point, long-only guard, hard caps, reduce-only closes, fail-closed positions — all test-covered). The labels are clean, the backtest costs are honest, the ML gates actually refuse to deploy, and the config refuses to start when risk parameters are incoherent. Architecture and reliability moved from D to B. **28/28 tests pass.**

But the capstone has to be honest about two things the rebuild did *not* do:

1. **It found no edge.** The honest answer — 0.55 AUC, net-negative across every backtest preset, no real out-of-sample survivor on 7 days of data — is the *correct* answer, and arriving at it honestly is itself a win over the dishonest baseline. But it means there is **nothing to fund yet.**
2. **It introduced one regression of its own** (option P/L missing from the daily ledger, 5.A) that re-creates the exact *class* of accounting-truth failure that earned the original F — the ledger reads +$64 while the truth is −$1,051.

**Overall: A−.** Safety: excellent. Measurement honesty: excellent. Remaining blockers: two HIGH P/L/ML gaps and the data-maturity ceiling on edge. **The system is now safe to run, but it is not yet safe to fund — and it has nothing to trade on.** Fix the three HIGH P/L/ML items, run the VACUUM, accrue real out-of-sample history, then re-audit the edge before any capital decision.
