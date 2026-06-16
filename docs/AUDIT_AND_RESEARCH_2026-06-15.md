# MARKET RADAR — Full Audit, Research & Rebuild Report

**Date:** 2026-06-15
**Account:** Alpaca PAPER `PA3GBM9C6UJE`
**Scope:** Code audit (6 areas) + adversarial verification + external research, reconciled against Alpaca's own records and the live SQLite DB.
**Status of system:** PAPER. Must remain PAPER. Not safe to trade real money in its current state.

---

## 1. Executive Summary

The bot started with $100,000 on 2026-05-26 and now holds **$91,992 in equity — a net loss of $8,008** over three weeks. That alone is not the alarming part. The alarming part is that **the bot cannot see its own loss**: its trade ledger (`bot_orders.realized_pnl_usd`) sums to **+$64.09** across 146 closed trades, while its daily table (`bot_daily_pnl`) sums to **-$1,878.50**, and the broker says **-$8,008**. Three numbers, none of which agree, and the one the bot reports as "realized P/L" is actually an account-equity delta that structurally omits its single largest loss.

That largest loss was the **TRDA death spiral** on 2026-06-10: an exit loop tried to "close" a position that had already flipped short, and because the close side was keyed off the original trade intent rather than the live position sign, every "sell to close" *added* to the short. It doubled every ~2 minutes — 63 → 126 → … → **16,128 shares short (~$90,300, ~14x the intended $6,300 book)** — then could not be covered after hours. Net **~-$6,112 in one session**. The single most important risk guardrail, `RISK_MAX_GROSS_EXPOSURE_USD=6000`, was never consulted because it is only checked on new entries, never on exits.

The deeper finding: **there is no demonstrated edge.** Resolved 1-day win rate is **49.7% — a coin flip.** The outcome data the model trains and is graded on is corrupted (average return 213%, max 3,000,000%+). The "frozen" 0.72 AUC is a deploy-gate artifact, not a real metric, and even at face value it is measured against those corrupted labels.

**Verdict:** capital preservation — a stated hard constraint — is not met. Fix safety and accounting first. Then prove an edge on clean data. Only then talk about money.

---

## 2. Ground-Truth P/L Reconciliation

Three numbers, verified this session against Alpaca and the live DB (`data/market_radar.db`):

| Source | Value | What it actually is |
|---|---|---|
| **Alpaca true equity change** | **-$8,008** | The truth. $100,000 → $91,992 (snapshots: 100000.00 → 91994.23 ≈ -$8,005.77). |
| `bot_daily_pnl` realized sum | **-$1,878.50** | An account-**equity delta**, overwritten every loop. Not a trade sum. Under-reports the loss by ~$6,100. |
| `bot_orders` realized sum (146 trades) | **+$64.09** | A genuine but **incomplete** sum of bracket-matched closed trades. Blind to the TRDA loss entirely. |

**Why they disagree:**

- `bot_orders.realized_pnl_usd` only books P/L when a *bracket child leg* fills and matches a parent bracket row. The TRDA orders were extended-hours `sell_to_open`/`buy_to_close` **market** orders (not brackets), and the covers were **canceled** (market closed). Only 2 TRDA rows exist in `bot_orders`, **both with `realized_pnl_usd = NULL`**. The ~$6,112 loss never produced a single booked closed-trade row. So +$64.09 is "correct" for the trades it saw — it just never saw the one that mattered.
- `bot_daily_pnl.realized_pnl_usd` is overwritten every 30s loop by `_reconcile_realized_pnl` with `(account.equity - account.last_equity) - unrealized_pl` — an equity delta, not a trade sum (see §4). This is why it shows **+$840.93 on 2026-06-11 with 0 wins / 3 losses** (impossible for real P/L) and **positive P/L on 06-13/14/15 with `trades_count = 0`** (pure mark-to-market drift on the 6 held longs).

**What is actually true:** the account is down **$8,008**. Treat the Alpaca equity curve (from `bot_account_snapshots`) as the only authoritative P/L until the ledger is rebuilt. Everything else the bot reports about its own profitability is fiction.

Current book: cash $89,803, long market value ~$2,189 across 6 small catalyst longs (AES, ASTS, IVR, PLCE, TRIN, WBD, ~$365 each). Margin ON (multiplier=4), shorting enabled, options level 3. The bot *sizes* off `LIVE_OVERRIDE_EQUITY_USD=6300`, but the real account has $90k+ buying power — so the only guardrails that matter are in the bot's own Python, and they failed.

---

## 3. Critical Safety Post-Mortem: The TRDA Stacking Bug

### Root cause (CONFIRMED against code + git history)

The pre-fix stock exit poller chose its close side from the **decision direction**, not the live position sign:

```python
# pre-fix (git show 5d3291a^:.../live_trader.py, ~line 2837)
close_side = "sell" if direction == "buy" else "buy"
qty = abs(float(p.qty))   # re-read from the LIVE (growing) position every 30s loop
```

A long `buy` decision whose position had flipped short kept emitting **sell** orders sized to the absolute live quantity. Each "sell to close" *opened more short*, and because `qty` was re-read from the now-larger position every loop, it doubled: 63 → 126 → 252 → … → **16,128 shares short**. All orders were `extended_hours`. The covers (~25 market `buy_to_close(16128)` between 21:13–21:33) were all **canceled** because regular market was closed and extended-hours/market-order mismatch — so it could not flatten. Net ~**-$6,112**. `client_order_id` prefixes `mr-sx-*` (exit) and `mr-refill-*` (refill) are both present.

**The gross-exposure cap never fired** because `RISK_MAX_GROSS_EXPOSURE_USD=6000` is enforced *only* in `RiskManager.evaluate()` (Rule 6, `risk/manager.py:204-249`), which is called *only* on the entry path (`live_trader.py:1654`, `:2127`). Every exit/refill/flatten submit calls `self.alpaca.submit_simple_order` directly and bypasses risk entirely. ~17 doubling submits passed through **zero** aggregate gross gate.

### Exact fix that landed (and what it does and does not cover)

Commit `5d3291a` ("CRITICAL: stock exit poller close by ACTUAL position sign", 2026-06-11 01:24 — the day after the incident) rewrote the exit poller:

```python
# current (live_trader.py:2917-2920)
pos_qty   = float(p.qty)
close_side = "sell" if pos_qty > 0 else "buy"   # keyed off LIVE sign
qty        = abs(pos_qty)
# plus sign-mismatch forced flatten (live_trader.py:2900-2901)
if (direction == "buy") != (float(p.qty) > 0):
    exit_reason = exit_reason or "sign_mismatch_flatten"
```

A flipped-short stock position now receives a **reducing buy** that drives toward flat — the exact doubling cannot recur in the stock exit path. Defense-in-depth also landed: a fail-closed "no positions snapshot ⇒ no new opens" guard (`:1031-1038`) and a **rogue-orphan flatten** for any position larger than the account (`:631-652`, which explicitly cites the 16,128-share TRDA short).

**Adversarial verdict: PARTIAL.** The specific recurrence is blocked, but the *class* of bug is not:
- The **refill path** (`_reconcile_stock_fills`, `:2691-2705`) still re-submits by **stale decision direction** without re-reading the live position sign — the un-patched sibling of the exact same defect. It runs every loop *before* the exit poller.
- There is still **no aggregate gross-exposure check on any exit/refill/flatten submit.**
- The crypto exit/flip paths still key off decision direction (`:3069`, `:3219`) — harmless only because Alpaca crypto is spot-only and cannot short.
- **No regression test** locks the fix in.

### Required gross-exposure gate design (the real fix)

A symptom-site patch is not enough. The invariant must move into a **single submit choke-point** that *every* path (entry, exit, refill, EOD flatten, loss-stop) must call. Before each `POST /orders`:

1. **Reduce-only enforcement.** Reject any non-reduce-only order that would not move `|position|` toward zero. Derive side and qty from the **live position sign at submit time**, never from a stored decision. Pass Alpaca's `reduce_only=true` on all closes so the *broker* makes it structurally impossible for a "close" to open or flip a position. (The option-spread paths already pass explicit `position_intent` — `:3667`, `:4063`, `:4380` — proving the team knows the pattern; it was simply never extended to stock closes.)
2. **Aggregate gross gate.** Re-fetch live positions, compute summed `|market_value|`, reject if `(gross + this notional) > RISK_MAX_GROSS_EXPOSURE_USD` unless reduce-only.
3. **Absolute hard-dollar cap** independent of equity/env multipliers, plus a **per-symbol absolute share/notional ceiling** (`intended_qty * small_factor`).
4. **Long-only assertion.** Reject stock `direction == "sell"` unless an explicit, off-by-default `LIVE_ALLOW_STOCK_SHORTS` flag is set (mirror the crypto guard at `:1827`); disable shorting on the Alpaca account until a short edge is proven. Catalyst-only is **not** long-only today — `_pick_direction` returns "sell" for stocks in three branches (`:4596`, `:4630`, `:4646`), and shorts were the TRDA vehicle.
5. **Cover-order clock branching.** RTH → market; valid extended session → marketable limit `extended_hours=True`; CLOSED → queue a session-valid/GTC order and **verify acceptance** (the TRDA covers all canceled because they were market orders submitted after the session closed).

---

## 4. The P/L Reporting Bug

### Mechanism (CONFIRMED — adversarial verdict: confirmed)

`bot_daily_pnl.realized_pnl_usd` has **two writers fighting over one column** every 30s loop:

- `_update_daily_pnl` (`live_trader.py:1229-1250`) — the *correct* writer: `realized_pnl_usd = realized_pnl_usd + excluded` (incremental), and bumps `trades_count`/`wins`/`losses`.
- `_reconcile_realized_pnl` (`live_trader.py:550-589`) — runs unconditionally every loop (`:863`) and **absolutely overwrites** the column:

```python
alpaca_intraday = float(account.equity - account.last_equity)   # equity delta
current_unreal  = sum(float(p.unrealized_pl) for p in positions)
true_realized   = alpaca_intraday - current_unreal
# ... SET realized_pnl_usd = ?   <- OVERWRITE, not +=  (line 580-582)
```

Because reconcile is the **last writer each loop**, the equity-delta value wins and the genuine trade-sum is clobbered. `last_equity` is **yesterday's close**, so `(equity - last_equity)` is an intraday equity change — not a closed-trade sum. This conflates unrealized swings, overnight resets, and crypto off-hours drift into a column named "realized."

**Live-DB fingerprints that prove it:**
- 06-13/14/15: positive realized with `trades_count = 0` (impossible for a trade sum).
- 06-11: +$840.93 with `wins=0, losses=3` (three losers cannot sum to +$841).
- The `tp_peak_usd` column shows 841.24 (06-12) and 682.12 (06-10/11) — the phantom +$840 equity swing armed the trailing take-profit and froze the bot for a day (the code even carries a "corrupt-lock detection" band-aid at `:372-385`).

**Why it is dangerous:** this corrupted column feeds the three most safety-critical consumers:
- the **daily-loss kill-switch** (`risk/manager.py:138-149`, `:347-375`) — so the $600 cap reads a number that can show +$840 on a catastrophic week and never tripped on the -$6,112 day;
- the **weekly digest** (`scripts/weekly_digest.py:61-66, 146, 165-168`) — "Total realized" and "above/below £150/day" are computed off the lie;
- the **dashboard** (`dashboard/server.py:312-316`) — which *also* uses UTC `date('now')` against US/Eastern `trading_date` keys, so 00:00–04:00 UTC reads the wrong day's row.

### Fix

1. **Make `_update_daily_pnl` the sole writer** of `realized_pnl_usd`/`wins`/`losses`/`trades_count`. Delete the overwrite at `:580-582`.
2. **Book realized P/L from Alpaca account activities (per-fill realized lots)**, not bracket-child matching — so non-bracket exits (extended-hours, refill, manual flatten) are first-class closing paths. Flag any filled order with `realized_pnl_usd IS NULL` after close as an **UNBOOKED EXIT** and alert. Backfill the TRDA loss; rebuild `bot_daily_pnl` history from corrected `bot_orders` + option spreads grouped by Eastern exit date.
3. **Move the equity delta to its own clearly-named column** (`equity_delta_intraday_usd`), computed from **today's first equity snapshot** (not `last_equity`). Point the kill-switch and trailing TP/loss-stop at *this* figure, never the realized column.
4. **Add a daily reconcile alert**: if `|Alpaca equity Δ − ledger realized − unrealized|` exceeds tolerance, alarm. That gap is exactly what hid the TRDA loss; surfacing it makes a future blind spot visible within a day.

---

## 5. Edge Credibility: Is There Any Real Edge?

**No demonstrated, validated out-of-sample edge exists.** (Adversarial verdict on the headline: confirmed; one supporting detail corrected below.)

### The signals are a coin flip
- Resolved **1-day win rate = 49.7%** (0.4971 over 187,034 rows; 46.6% on the fully-resolved subset). Verified this session.
- The live **calibration is inverted** in the band the bot trades: signals scored 0.52–0.62 (the `RISK_MIN_CALIBRATED_P=0.52` trade zone, n=104,171) have a **45.0% forward win rate and -0.38% avg 5d return — they lose money** — while the "avoid" bucket scored <0.45 returns **+2.04%**. The bot buys exactly the signals that lose. Only the tiny `≥0.62` bucket (n=318) is positive (+3.35%), and it is too small and likely dominated by illiquid penny-stock M&A pops.

### The outcome labels are corrupted
- `signal_outcomes` raw: AVG `return_1d` = 212.9%, AVG `return_5d` = 211.6%, MAX = **3,012,400%**. Median return is **0.0%** — i.e. the true central tendency is *no movement*, and 99.9% of the summed return mass comes from ~674 outlier rows.
- Two root causes: (a) division by an **unfloored sub-penny `price_at_flag`** (e.g. INRE anchor $0.0004 → +3,002,400%) in both `outcomes/tracker.py:219-223` and `backfill/prices.py:163-166`; (b) **split-unadjusted prices** (`adjustment="raw"` in `alpaca_client.py:416`, `auto_adjust=False` in `prices.py:92`) — a reverse split manufactures phantom returns, and reverse splits cluster in exactly the low-priced names this strategy chases.
- **Correction to the brief's framing:** training does *not* literally learn on 213% labels — `train.py:480-487` filters `ABS(return_5d_pct) < 200`, `event_type != 'other'`, and drops backfill rows, leaving ~11,260 rows averaging +1.19% / 51.5% win. The corruption discredits every *measurement* derived from the column (dashboard, drift, hit-rates) and admits 50–200% artifacts into labels, but the model is not trained directly on the 3,000,000% rows.

### The "frozen 0.7214 AUC" is mis-diagnosed
**Correction (adversarial verdict: partial).** It is **not** a cached/no-op metric. It is a **deploy-gate artifact**: `train.py:301-310` (`only_replace_if_better`: skip deploy if new median val_auc < prev − 0.005) rejected every fresh candidate because their AUCs (0.6837 / 0.6493 / 0.6955 on 06-11/12/13) never beat the stale 06-04 model's 0.7214, so `current.json` stayed pinned and `scheduled_retrain.py` logged the unchanged pointer for both "old" and "new." Fresh models **do** train daily on growing data with different, unstable AUCs. On 06-15 a 0.7332 model finally beat it and deployed.

But the 0.72 is untrustworthy regardless:
- Trained on only **3,730 rows / 745 val rows**, fold AUC std **0.0495** — above the code's own 0.03 fragility threshold (`train.py:227`).
- The current 06-15 model's folds span **0.476–0.928** (std 0.169, one fold *below chance*) — not reproducible edge.
- All AUCs are computed against the corrupted/coin-flip labels, so even the number itself is meaningless.
- Live predictions largely ran on a **>1-month-stale May-13 model** (153k of the predictions), because the deploy gate pinned an old pointer for ~10 days.

### Leakage risks
- **Walk-forward is not chronologically valid:** `scored_at = '2026-05-12'` holds ~151,504 of ~400k rows (one backfill), so 38% of data shares one timestamp. `TimeSeriesSplit` folds straddle that boundary with **no purge/embargo** — the "López de Prado no-leakage" claim in the docstring is false in practice.
- **Calibration is fit on the leaky last fold** and never validated on a fresh holdout — which is why it is empirically inverted live; `predict_proba` is not a trustworthy probability, breaking any Kelly/selective-gate sizing built on it.
- **~60% of the 60 features are constant defaults** (`short_interest`=0 rows, `institutional_holdings`=0, `earnings_whispers`=0, `catalysts`=99). `days_until_catalyst` defaults to 60 — the headline "catalyst" feature is essentially absent.
- **Survivorship bias**: backtesting on currently-listed small-caps silently drops halted/diluted/delisted names — exactly the disasters this universe is most exposed to.
- The **drift detector is inert**: `model_drift_observations` has 0 rows (requires ≥50 resolved outcomes per exact model_version, which daily retrains never accumulate), so the only automated degradation guard has never fired.

### What a trustworthy validation harness requires
1. **Clean the labels at source.** Floor `price_at_flag` (reject < $1, ideally < $5 for this universe), use split/dividend-adjusted closes (Alpaca `adjustment="all"`, yfinance `auto_adjust=True`), clamp `|daily return|` to a sane band, flag out-of-band rows corrupt, and re-resolve history. Nothing downstream is trustworthy until this lands.
2. **Time-honest validation:** Combinatorial Purged Cross-Validation with embargo ≥ max label horizon (≥5 days for 5-day labels); cap/de-dup the 151k-row May-12 block; refuse deploy when fold-AUC std > 0.05. Expect AUC to fall toward ~0.5.
3. **Convert AUC → cost-inclusive net P/L.** AUC is a classification metric that ignores spread/slippage/borrow. Re-run with a realistic small-cap cost model (≥3% round-trip) and *double-cost* stress test.
4. **Multiple-testing discipline.** Log **N** (every feature set / lookback / threshold / model tried). Compute the **Deflated Sharpe Ratio** (deploy only if DSR > 0.95; reject < 0.80) and **PBO via CSCV** (reject if near 0.5). MinBTL: with ~5 years of data, ~45 independent trials before a worthless strategy is expected to show Sharpe 1.0; with ~2 years, ~7. Count effective *independent* bets (down-weight overlapping labels), not raw trade count.
5. **A frozen final OOS block** touched exactly once, with reliability (Brier/ECE) measured, gating deployment on calibration sanity — not median-fold AUC.

A ~3%/day target implies an annualized Sharpe of roughly **6–16**. No real strategy sustains that. **Any backtest or paper curve that appears to hit it should be treated as proof of leakage/overfitting, not validation.**

---

## 6. Per-Area Code-Quality Grades

| Area | Grade | Top findings |
|---|---|---|
| **P/L accounting & reporting** | **F** | Equity-delta overwrites the trade ledger every loop; three contradictory numbers; ledger blind to the TRDA loss (both rows `NULL`); the corrupt column feeds the kill-switch, digest, and dashboard. |
| **ML / edge** | **F** | No OOS edge; trade band (0.52–0.62) is negative-EV; labels corrupted (avg 213%); "frozen" AUC is a deploy-gate artifact; degenerate walk-forward (38% of rows in one timestamp); ~60% of features are constants; live model >1 month stale. |
| **Backtest / validation** | **F** | Look-ahead sim reads realized future return and back-fits a bracket; ~5bps flat cost (off by 1–2 orders of magnitude); a *separate* reimplementation of the gate that imports zero live-execution code — could never surface the TRDA bug; drift table empty; options payoff fabricated from constants. |
| **Risk gate / sizing / order submission** | **D** | TRDA root cause reproducible; gross cap is entry-only; catalyst-only ≠ long-only (shorts reachable); refill path is the un-patched sibling; all sizing hinges on one env var; symptom patched, systemic invariants (broker reduce-only, submit-time gross gate, long-only assertion) still missing. |
| **Execution / costs / PDT / after-hours** | **D** | Exit/refill/flatten bypass risk entirely; all entries/exits are **market orders on sub-$5 microcaps** (`LIVE_MIN_STOCK_PRICE=1.0`); extended-hours entries have **no server-side stop** (poller is the only stop → sleeping Mac = naked); covers auto-cancel after session close; PDT awareness off with zero round-trip accounting. |
| **Catalyst / signal ingestion** | **D** | SEC body fetch is synchronous in the poll loop (multi-second-to-minute latency) and the freshest source (news) isn't traded while the traded source (SEC) isn't fresh; `catalysts` table (99 rows) is a manual FDA scraper disconnected from the 500k firehose; **corroboration double-counts syndication** (+2.0 composite points from one wire story across ~30 Google-News feeds); body all-caps ticker extraction mis-tags real-but-wrong symbols; composite score is hand-tuned constants, not predictive. |
| **Architecture / reliability** | **D** | `live_trader.py` is a 4,691-line god object (552-line method, 236-line `run_once`, 64 swallow-and-continue blocks); stale docstring claims crash-safe brackets that don't exist; 1.35GB SQLite hit every 30s by ~15 jobs with no retention/VACUUM; **zero real automated tests**; live-money T212 credentials polled on a timer for a dead, non-trading path; runtime `ALTER TABLE` in the hot path. |

**Genuine strengths to preserve:** the TRDA sign-fix itself; fail-closed instincts throughout (no-snapshot ⇒ no opens, rogue-orphan flatten, `_effective_equity` → 0 if override unset); idempotent entry path (pending-submit row + `UNIQUE(score_id)`); the daemon's job isolation (`_safe()`); option-spread paths that correctly pass `position_intent`; honest forensic comments documenting past incidents; the disciplined trailing-feature anchoring in `market_features`/`graph_features`; and the team's demonstrated willingness to cut measured-negative-edge sources (StockTwits disabled).

---

## 7. The Honest Verdict on the £150/day Goal

### The math
£150/day on ~£5k is **~3%/day**. Compounded over 252 trading days that is **(1.03)^252 ≈ 1,718× — £5,000 → ~£8.6M in one year** (~171,000%). Even taken as flat, non-compounded income, £150/day is a **~756% simple annual return** on £5k.

For calibration:
- **Renaissance Medallion** (best track record in history): ~66% gross / ~39% net **per year** = ~0.13–0.20% per **day**. The goal is **15–23× Medallion, every day**.
- The **top 500 day traders** in the definitive Taiwan study earned ~0.38%/day net — the goal is **~8× that**, and they are the <1% who reliably profit.
- Average CTA: ~3% per **year**. The goal is ~one year of an average CTA in a single day.
- Base rates: ~1–3% of active day traders are reliably net-profitable over 1y+; 82% of FCA-sampled CFD accounts lose; 97% of persistent Brazilian day traders lost; ~90–95% fail even a 10%/month prop evaluation.

**3%/day is not an aggressive-but-reachable target — it is physically/statistically unreachable as a sustained average.** Worse, the position sizing required to *attempt* it is itself the ruin mechanism: to make £150 on £5k daily you must risk a large fraction of the book per trade, and a normal (mathematically certain) losing streak then produces a terminal drawdown. The target and the blow-up are the same lever.

### The PDT constraint — corrected
**Correction (adversarial verdict: partial).** The brief's premise that PDT structurally caps a sub-$25k 30-second loop is **no longer true.** Effective **June 4, 2026** (11 days before this report), SEC-approved amendments to FINRA Rule 4210 **eliminated the entire PDT framework** — the $25,000 minimum *and* the 4-trades-in-5-days count. Alpaca implemented it on June 4. The trade-count wall is gone. The new binding constraint is a real-time **Intraday Margin Deficit** (de-minimis trigger = lower of $1,000 or 5% of equity ≈ **$315** on a $6.3k book); repeated unmet deficits can freeze the account 90 days. Keep equity ≥ $2,000 for full margin. **For this specific bot, PDT was never the binding guardrail anyway** — the real account has $90k+ buying power, margin, and shorting, so the only guardrails that ever mattered were the bot's own (failed) Python checks.

UK-specific drag that *does* persist: Alpaca is not FCA-regulated (SIPC only, no FSCS), USD-only funding (FX cost), US-equity gains are CGT-able for a UK resident (no spread-bet shelter), and US-situs estate-tax exposure above ~$60k.

### The credible ladder
Reframe from a daily-income target to a **risk-survival-first compounding target**, judged on Sharpe and max drawdown over 6–12 months, not daily £.

| Stage | What it is | Credible daily £ |
|---|---|---|
| **0 — Now (paper)** | Fix safety + accounting, clean labels, prove edge on clean OOS data with realistic costs. The £5k book is a **research instrument, not an income engine.** | £0 (do not risk capital) |
| **1 — Edge proven** | 100+ independent bets, positive net-of-cost expectancy, Sharpe > 1 net of modelled slippage, DSR > 0.95. Go live **tiny** (£500–£1,000), withdraw don't compound. | ~£0–£5 (break-even-after-costs is a *good* first-6-month result) |
| **2 — Modest scale on own capital** | At a strong-but-realistic ~0.1–0.3%/day net (~15–50%/yr, top few percent of retail), on the full ~£5k. | **~£5–£15/day** |
| **3 — Rent size via equity prop** | Convert a *proven* edge into prop buying power (e.g. Trade The Pool: real US stocks, small-cap shorts, $5k–$200k, 70/30 split, ~$50–$1,475 fee). Personal downside capped at the fee + daily-loss limit. | **~£50–£150/day** *if* the edge survives prop drawdown/consistency rules — a top-decile outcome, not a baseline |

The honest bottom line: **£150/day is an income figure tied to capital, not a return earnable on £5k.** At a realistic professional ~3–5%/month it requires roughly **£75k–£150k of working capital**. The fastest legitimate route to that size is *not* compounding £5k (the math and fixed costs make it a multi-year-to-impossible grind) but **renting size on a proven edge** via equity prop — and that only works *after* the edge is proven. Prop consistency rules (cap your best day at ~30% of total profit) actively penalise a lumpy catalyst strategy, so design for per-day profit-throttling explicitly.

---

## 8. Best-Practice Synthesis (from the research)

**Which catalyst edges actually work for retail:**
- **Speed edges are hopeless** for seconds-of-latency public data: 8-K/earnings reactions happen in **milliseconds** on native feeds; FDA/biotech outcomes are fully priced by day +1 with **no exploitable drift**; LULD halt resumptions *increase* small-cap volatility with no directional edge; short squeezes are variance, not a forecastable edge. **Hard-blocklist all of these** (react-to-fresh-8K, hold-into-binary, buy-halt-resumption, chase-high-SI) with comments citing why.
- **Slow drift edges are the only retail-defensible ones**: **PEAD long-leg** and **opportunistic/cluster insider buying (Form 4)**, traded as **multi-day holds (5–60 days)**, never same-second reactions. But both are strongest exactly in small/illiquid names where **70–100% of the gross edge is eaten by costs** (Chordia et al.), and PEAD "may have disappeared" in recent decades. Merger arb is real but ~4%/yr net with crash-correlated tail risk — incompatible with a daily target on a concentrated £5k book.

**Microstructure (why size is a trap):**
- Sub-$10 small-caps: round-trip frictions realistically **100–300bps+** (spread + impact + slippage). A $0.50 spread on a $5 stock is a 10% round trip.
- Market impact follows the **square-root law** `I ≈ 0.84·σ·√(Q/V)` — cost grows with the square root of participation, so the same signal that's cheap at £100 clips becomes ruinous as the book grows. The book size that makes the edge real is the size at which £150/day is unextractable.
- ~94% of LULD halts are Tier-2 (small-cap) names — the bot's exact universe.
- **Alpaca paper fills do not model small-cap slippage/spread/halts** (randomized partials) — paper success is an upper bound, near-certainly an artifact, not validation.

**Sizing & risk control:**
- Per-trade risk **0.5–1% of current equity** (0.5% for catalyst small-caps given slippage). Size via ATR: `shares = floor((equity·risk%)/(ATR_mult·ATR))`, ATR_mult ~1.5–2×, with a per-name liquidity clamp (≤1–2% of ADV).
- **Fractional Kelly ≤ 1/4**, or just a flat 1% rule — never a backtest-derived Kelly off a noisy edge estimate (full Kelly → ~60% avg drawdown).
- **Hard pre-trade portfolio-heat gate at ~5–6%** summed open risk (the single most important aggregate control for an order-spamming bot). Cap per-theme/sector so correlated catalyst names aggregate as one bet. ~4–5 concurrent names max.
- **Daily-loss kill-switch at ~3%** (≈3× per-trade risk) that **flattens and blocks**, plus stacked weekly (~6%) / monthly (~10–12%) limits and drawdown-tiered de-risking (halve size past ~6–8% drawdown, pause past ~10–12%).

**Validation:** purged+embargoed CPCV, DSR/PBO with logged N, cost-inclusive net P/L, frozen OOS block, effective-bet counting. Confidence is a function of **independent bets, not calendar time** — even a genuine Sharpe-1 strategy needs ~1,000 daily bets (~4 years) to clear t=2.

---

## 9. Staged Rebuild Plan

> **P0 must be complete before any further trading — paper or real.** Every item references real files.

### Phase P0 — Safety & Correctness (blocks all trading)
**Goal:** make capital preservation a structural invariant and the books trustworthy.
1. **Single submit choke-point.** Route entry/exit/refill/EOD/loss-stop through one `submit_order()` (no path calls `self.alpaca.submit_*` directly). It re-fetches live positions and enforces: reduce-only on all closes (side+qty from live sign), aggregate gross ≤ `RISK_MAX_GROSS_EXPOSURE_USD`, an absolute hard-dollar cap, and a per-symbol share/notional ceiling. Add `reduce_only` to `alpaca_client.submit_simple_order` (`:522-565`). Fixes the entry-only gap (`risk/manager.py:204-249`) and the un-patched refill path (`live_trader.py:2691-2705`).
2. **Long-only assertion.** Reject stock `direction == "sell"` in `_process_stock_candidate` unless `LIVE_ALLOW_STOCK_SHORTS` is explicitly set (mirror crypto guard `:1827`); disable shorting on the Alpaca account config.
3. **Collapse P/L to one source of truth.** Delete the equity-delta overwrite (`live_trader.py:580-582`); make `_update_daily_pnl` (`:1229`) the sole writer of realized/wins/losses/trades. Book realized P/L from **Alpaca account activities (per-fill lots)**, not bracket-child matching (`_maybe_realize_pnl` `:1139-1205`); flag `realized_pnl_usd IS NULL` filled-after-close orders as UNBOOKED EXIT + alert.
4. **Real intraday kill-switch.** New `equity_delta_intraday_usd` column off **today's first snapshot** (not `last_equity`); point the daily-loss cap (`risk/manager.py:138-149`) and trailing TP/loss-stop (`live_trader.py:3847,3930,3942`) at it; make it fire in **extended hours** and **flatten** (not just block). Add a column-independent absolute-equity-drop circuit breaker.
5. **Crash-survivable stops.** Submit a server-side stop/bracket atomically with each RTH entry; arm the stop the instant an extended-hours limit fills. Set `LIVE_ALLOW_AFTER_HOURS=0` until that exists. Fix the stale crash-safe docstring (`:22-29`). Cover-order clock branching with acceptance verification.
6. **Startup config validation.** Enforce `gross > reserve`, `gross ≤ hard cap`, `daily-loss < gross`, override within a sane band (1k–20k); refuse to start on inconsistency (defaults today are incoherent — reserve 2500 > gross 2000, `config.py:75-76,87`).
7. **Regression tests + FakeAlpaca.** Lock the invariants: exit-always-reduces-toward-flat (TRDA), gross/short-cap rejection, long/short P/L sign, single-writer daily P/L, idempotent retries. Wire to a pre-commit/CI gate.

### Phase P1 — Trustworthy Measurement & Edge Validation
**Goal:** be able to honestly answer "is there an edge?" before risking anything.
1. **Clean labels at source** (`outcomes/tracker.py:219-223`, `backfill/prices.py:163-166`): floor `price_at_flag` (< $1, prefer < $5), split/dividend-adjusted closes (`alpaca_client.py:416` → `adjustment="all"`; `prices.py:92` → `auto_adjust=True`), clamp returns, flag corrupt, re-resolve history, remove the `ABS<200` band-aid (`train.py:487`).
2. **Time-honest validation:** CPCV with embargo ≥ max horizon; cap/de-dup the 151k-row May-12 block; refuse deploy when fold-AUC std > 0.05 (`train.py:189-218`). Recompute AUC; expect ~0.5.
3. **Frozen OOS block** + calibration on a dedicated later holdout (fix `train.py:243-280`); gate deploy on Brier/ECE + observed-vs-predicted win rate, not median-fold AUC. Log **N**; compute **DSR** (deploy > 0.95) and **PBO**.
4. **Fix accounting consumers:** rebuild `weekly_digest.py:61-66` and `dashboard/server.py:312-316` to read the corrected ledger; fix dashboard UTC→US/Eastern date; add a "ledger realized vs Alpaca equity Δ vs unexplained gap" line to both.
5. **Realistic backtest:** unify the gate/sizer/risk into shared pure functions called by *both* `live_trader` and `replay.py` (kill the parallel reimplementation, `replay.py:194-245`); drive replay through a simulated broker enforcing the *same* gross/reduce-only logic; add intraday-bar cost model (spread, square-root impact, borrow, halts, partials). Stop reading the realized future return as the exit (`replay.py:259`).
6. **Make drift fire:** track per model *family*, persist partial observations, alert on N days of zero (`drift.py:98-106`); feed cleaned labels.

### Phase P2 — Strategy / Edge Construction
**Goal:** build around the only retail-defensible edges, on clean data, net of costs.
1. **Hard-blocklist the hopeless catalysts** (react-to-fresh-8K initial move, hold-into-FDA/PDUFA, buy-halt-resumption, chase-high-SI) with citing comments.
2. **Build the two slow-drift edges as multi-day holds** (5–60 days): PEAD long-leg + opportunistic/cluster Form-4 insider buys (Cohen-Malloy-Pomorski filter); prefer disclosure-after-momentum.
3. **Fix ingestion that corrupts signal quality:** corroboration over distinct content-hash/publisher-domain not feed source_id (`composite.py:214-219,396-416`); collapse ~30 Google-News topical feeds per publisher; disambiguate `catalysts` (rename to `fda_calendar` or build a real catalyst-events table); decouple SEC body fetch from the poll loop with high-water-mark pagination; harden body ticker extraction (drop the 0.55 all-caps path, `ticker_extractor.py:82-84`).
4. **Hard liquidity filters + limit-only execution:** min ADV ≥ $2–5M, spread cap ≤ 0.5–0.75%, reject sub-$3 tickers; marketable-limit orders with max-slippage cap; participation ≤ 1% of ADV; record per-trade slippage. Replace the heuristic composite gate with a properly cross-validated model probability (heuristic as transparent prior only).
5. **Sizing/heat gates as code:** 0.5–1% per-trade ATR sizing off live equity; ≤1/4 Kelly; hard 5–6% portfolio-heat gate; per-theme/sector caps; ≤4–5 concurrent names. Test by trying to over-allocate and confirming refusal.

### Phase P3 — Scale (only after a proven, cost-aware track record)
**Goal:** convert a proven edge into income without ruin.
1. **Tiny live pilot** (£500–£1,000), withdraw don't compound; size and gate off *live* measured slippage, not paper.
2. **Pre-registered live-evaluation protocol:** define the number of independent bets and the Probabilistic-Sharpe threshold the live record must clear before adding capital; treat short paper runs as statistically indistinguishable from noise.
3. **Equity prop funding** (real US stocks, small-cap shorts) once the edge survives realistic costs; backtest against the firm's exact daily-loss/consistency rules first; add a per-day profit-throttle so no day exceeds ~30% of trailing total profit.
4. **Architecture & ops hardening:** decompose `live_trader.py` into Gateway/ExitManager/PnLLedger/Reconciler/RiskGate/CandidateSource/Orchestrator; add DB retention + VACUUM and split telemetry from accounting; one versioned migration runner (remove runtime `ALTER TABLE`, `:359-364`); delete or flag-gate the dead live-money T212 path (`daemon.py:51,167-170`).
5. **Scale via breadth, not bravado:** multiple funded accounts on a proven system; budget a 20–30% annual account-failure rate; re-underwrite the whole plan if live Sharpe target isn't held for 3+ consecutive months.

---

*Prepared from the verified session ground truth, the six-area code audit, six external research briefings, and four adversarial verdicts. All P/L figures cross-checked against the live `data/market_radar.db` and Alpaca account `PA3GBM9C6UJE`.*
