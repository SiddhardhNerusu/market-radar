# MARKET RADAR — Deployed-State Audit (2026-06-16)

**Commit:** `201f63c` on branch `rebuild/overhaul-2026-06-15`
**Account:** PAPER Alpaca `PA3GBM9C6UJE` (ACTIVE)
**Synthesizer verdict:** Safe, truthful, and genuinely UP — but running in a deliberately degraded (ML-off) mode, with one HIGH-severity data-pipeline regression and one HIGH operational risk that must be fixed. **No proven edge. No profitability is certified. Keep it on paper.**

This is the new STANDARD/baseline. Graded on safe + truthful + correct + actually-running — NOT on profit.

---

## 1. IS IT UP AND RUNNING? — YES (verified live)

| Evidence | Confirmed |
|---|---|
| Daemon PID 97637 + live_trader PID 97639 both up since `Tue Jun 16 14:56:27 2026` (`ps`) | yes |
| Both running on committed HEAD `201f63c`; zero uncommitted src drift; no `.py` newer than process start | yes |
| Paper Alpaca ACTIVE, equity **$92,062**, market open (`get_clock` is_open=true) | yes |
| Exactly **6 long positions** (AES, ASTS, IVR, PLCE, TRIN, VNCE) — matches DB cap of 6 | yes |
| All positions `side:long` despite `shorting_enabled:true` → long-only gate holding live | yes |
| Every position has a **live broker-side GTC trailing-stop** (`trail_percent:15`, `sell_to_close`, status `new`, expires 2026-09-14), placed 13:30Z. qty_available=0 on all 6 → fully reserved by exits. Crash-survivable. | yes |
| Ingestion writing fresh rows post-restart: sec_edgar / nasdaq_halts / alpaca_news / market_movers / rss all `last_success_at` within the last hour, `consecutive_errors=0` | yes |
| daemon_health_history actively populated post-restart (newest `2026-06-16T14:11:39Z`, 3777 rows) | yes |
| 85 tests pass (`pytest tests/ -q` → 85 passed) | yes |

**Caveat on "running":** "live and running" is TRUE. "Actively opening new trades" is currently **FALSE by design** — zero new orders since the 13:56Z restart (`get_orders(after=13:56Z)` = `[]`). Two compounding correct causes: (a) ML gate off → most candidates score p=0.500; (b) account is AT the 6-position cap. Entries resume only when a slot frees AND a candidate qualifies via a bypass.

---

## 2. HONEST GRADES (the new standard) — safe / correct / operational, NOT profit

| Area | Status | Grade | Verified live? |
|---|---|---|---|
| Live deployment health | WARN | **B** | yes |
| Safety / capital-preservation invariants | PASS | **A−** | yes |
| Regression / correctness / integration | PASS | **A−** | yes |
| Going-forward data integrity | WARN | **B** | yes |
| Best-practice alignment | PASS | **A−** | yes |
| Deployed rebuild (overall) | WARN | **B+** | yes |

**Composite: B+.** Safety/honesty posture is A-grade and verified holding live. The downgrades are all operational/data-pipeline, not safety: the ML edge path is off (correct but degraded), the SEC classifier is frozen (regression), and the launchd stdout log is unrotated (durability risk).

---

## 3. REGRESSIONS / BREAKAGE TO FIX — ranked

### R1 — HIGH — SEC classifier has made ZERO forward progress since restart (data regression)
The deterministic/LLM event_type refinement is frozen. `llm_classifications` written since 13:56Z = **0**; max `classified_at` stuck at `2026-06-16T13:49:10Z` (pre-restart). Live log shows `classify_pending: candidates=30 filtered=30 classified=0` for 15+ consecutive cycles, still firing at 15:10. Root cause = head-of-line block: the candidate query orders by composite_score DESC, the top 30 are all formless tier-1 price_action rows that `should_classify` rejects every cycle (`if not ok: continue` runs BEFORE the deterministic SEC router), so the loop never advances and ~9k SEC rows starve behind the head — including fresh 8-Ks whose event_type should be deterministically routed but stay at the coarse heuristic label.
**Impact:** daily monitoring reading `signal_scores.event_type` on fresh rows reads a LESS-precise label than the rebuild intends. Not a safety issue.
**Fix:** run the free deterministic SEC item-code router BEFORE `should_classify`, and add forward-progress paging (cursor on signal_id) so a permanently-filtered head can't starve the tail.

### R2 — HIGH (operational durability) — launchd stdout log unrotated, 228MB and growing
`~/Library/Logs/MarketRadar/daemon.out.log` = **228MB** (same day as restart), NOT covered by the app's RotatingFileHandler (which caps a different file, `logs/daemon.log`). Every apscheduler job logs at INFO; 25k+ apscheduler lines. Left alone this fills the disk over a multi-day run.
**Fix:** silence `apscheduler.executors.default` to WARNING, drop the StreamHandler under launchd, or point StandardOutPath at a rotated sink / `/dev/null` (rotating logs/daemon.log already captures everything).

### R3 — MEDIUM — daemon CPU-saturated; scheduler shedding work
Daemon at **99.7% CPU** (one full core, confirmed via `ps`). apscheduler "maximum number of running instances reached (1)" fired 143× in the last 5000 lines — the 30s job (incl. scoring) overruns its interval. Non-fatal (next tick picks it up) but scoring cadence degrades to actual runtime, not the configured 30s.
**Fix:** profile the 30s job, raise its interval or set coalesce/max_instances, and/or move scoring to its own executor pool.

### R4 — MEDIUM — no health/freshness alert for the classifier stall
`daemon_health` tracks only per-source poll/success/errors. A multi-hour `classified=0` stall (R1) raises NO alert; all sources show green while the event_type pipeline is frozen. Daily monitoring would not catch it.
**Fix:** add a check that alerts when `classify_pending` reports classified=0 for N cycles while candidates>0, or when the unclassified-2d backlog exceeds a threshold.

### R5 — LOW — macOS notification digest failing 100% (sent=0/failed=N every cycle)
Headless process → osascript banners don't post. Confirmed live: `notifications: candidates=3 sent=0 failed=3` at 15:11. The row is INSERTed into `notifications_sent` BEFORE the ok-check, so failures are marked "sent" and never retried. The realtime telegram+macos path worked earlier (01:00), so primary alerting isn't necessarily dead — but the digest/macos channel is.
**Fix:** disable the macos digest channel in a headless deploy (or route via the working realtime path); do not write `notifications_sent` until after a successful send.

### R6 — LOW — pytest collects a non-test helper
`scripts/test_t212_connection.py::test_account` errors during full collection (`92 passed, 1 error`). Unrelated to the trading system (T212 connectivity helper, fixture not found). `pytest tests/ -q` is clean at 85.
**Fix:** rename the helper or exclude `scripts/` from collection.

### R7 — LOW — crypto candidates sized+rejected every loop; misleading log
Every UNI/ETH/BCH/AAVE candidate runs full quote+sizing only to reject at `qty 0.00 < min 0.0 (notional $0)`. Correct and safe (crypto hard-disabled, cap=$0 in `.env`), but wasteful per-loop API calls + misleading message (true reason is zero-edge at p=0.5, not a min-qty floor).
**Fix:** short-circuit crypto before quote+sizing when cap=0; fix the log string.

### R8 — LOW — disabled ingestors look stalled in daemon_health
`reddit_public_aggregate` (last success 18d) and `stocktwits_trending` (8d) show `consecutive_errors=0` but stale timestamps. Both are intentionally disabled (documented in daemon.py), but the leftover rows create a false "silently broken ingestor" signal for monitoring.
**Fix:** purge or explicitly mark disabled sources in daemon_health.

### Not-yet-done from FINAL_REAUDIT (operational, not code, not safety-gating)
- **5.C historical ledger backfill:** TRDA rows (bot_orders id 171, 180) still NULL realized; 68 filled rows NULL. Going-forward ledger reconciles and the circuit breaker reads Alpaca equity directly, so no live decision rides the NULL history — but historical P&L is still partly fictional.
- **5.F VACUUM/ANALYZE:** never run — no `sqlite_stat*` tables exist on the 1.38GB DB. Run `scripts/db_maintenance.py` once in a closed window.

---

## 4. HONEST CAVEATS (restated)

- **NO PROVEN EDGE.** EDGE_HUNT: "There is no real edge in this data." Best Form-4 candidate dies on strict dedup (p=0.075 NS). Every other candidate is artifact/untestable. The `*** persists ***` on activist_position is +0.124% netA — noise-level, survives-the-test, NOT a proven edge.
- **ML is OFF in production.** The deployed model fails its own load-time quality re-gate every cycle (`val_auc_std 0.1691 > 0.05`, one fold 0.476 — worse than coin-flip). `model_p_5d` fill rate collapsed from 100% (6235/6235 on 06-15) to **0.3% (7/2239 on 06-16)**. This is CORRECT defensive behavior — the model is genuinely untrustworthy. Do NOT set `LIVE_ALLOW_UNGATED_MODEL=1`. But it means the bot trades ONLY via news-catalyst / price-action bypasses right now.
- **DATA-LIMITED.** One ~5-week regime. 250–365%/yr to hit any income goal is arithmetically impossible. ~90% of retail algos lose year one.
- **PAPER-ONLY.** Do not fund. Accrue forward OOS days. Hold the promotion gate (≥15 new distinct non-backfill days, both halves positive, drop-top-5 positive, permutation p<0.05) before any tiny-live pilot. Options stay frozen (real ~21% RT cost vs ~0.5% measured edge).
- The now-correct ledger (2026-06-16 realized −$7.26 with equity_delta +$42.62 stored separately, no clobber) is the rebuild's main win — it is TRUTHFUL bookkeeping, NOT evidence of edge.

---

## 5. DAILY-MONITORING READINESS — partial

**Can you trust the daily numbers?** The **P&L / trade ledger: YES** (truthful, single-writer, reconciles). The **event_type / classification labels on fresh rows: NO** until R1 is fixed — they're frozen at the coarse heuristic and no alert fires (R4). Ingestion freshness and broker reconciliation: YES.

**Daily check should watch (3–5):**
1. **`model_p_5d` fill-rate** — if it stays ~0%, ML is gated off (expected today); if it suddenly jumps to ~100%, a model passed the bar OR the gate was bypassed — investigate which.
2. **`classify_pending classified=N`** — must be >0 with a falling unclassified-2d backlog. Currently STUCK at 0 (R1). Treat 0 for hours as a red flag.
3. **`daemon.out.log` size** — must not approach disk limits. 228MB today and unrotated (R2).
4. **Broker reconciliation** — open positions count and equity in DB must equal Alpaca; all positions must carry a live GTC trailing stop (qty_available=0). Daily-loss kill switch state.
5. **New-order activity vs the 6-position cap** — zero entries is currently NORMAL (cap + ML off); distinguish "intentionally idle" from "silently broken" before assuming the bot is fine.

---

## 6. BOTTOM LINE

The rebuild is genuinely deployed and the safety/honesty posture is real and verified holding LIVE: long-only gate enforced, every position protected by a crash-survivable broker-side trailing stop, the order gateway enforces reduce-only/long-only/hard-cap (directly closing the 2026-06-10 TRDA short-spiral hole), the ML model is correctly refused because it's untrustworthy, the P&L ledger is now truthful, and the system honestly states it has no edge. That is an A-grade safety/honesty baseline.

But it is **not flawless and not a profit machine.** Two HIGH items need fixing before the daily numbers are fully trustworthy: the SEC classifier is frozen with no alert (data regression), and the launchd stdout log will fill the disk over a multi-day run. The ML edge path is off by design, so the bot is in a degraded bypass-only mode. There is no proven edge, the data is thin, and this must stay paper-only. Confirm what's solid (safety, truthfulness, it's up), fix what isn't (R1, R2), and do not read any of this as evidence the system makes money — it doesn't claim to, and neither should anyone reading the dashboard.
