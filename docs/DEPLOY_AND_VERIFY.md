# Deploy & Verify — MARKET RADAR rebuild (branch `rebuild/overhaul-2026-06-15`)

Everything built this session is on the branch and is **uncommitted + undeployed**.
The live daemon (PID 743) is running pre-fix code, so until you do steps 1–3 the
fixes are NOT live. Nothing here touches real money; the live *trader* stays
paused on purpose. Run from the project root.

## 0. Review (recommended before committing)
```bash
cd ~/Documents/Claude/Projects/MARKET\ RADAR
git status                       # see new files + modified
git diff --stat                  # scope of changes
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q   # must be all green (85+)
```

## 1. Commit (YOUR action — Claude does not commit)
```bash
git add -A && git commit          # review the message; this is your call
```

## 2. Restart the data daemon so it loads the new code
The daemon does ingestion/scoring only — NOT trading. Safe to restart.
```bash
launchctl unload ~/Library/LaunchAgents/com.marketradar.daemon.plist
launchctl load   ~/Library/LaunchAgents/com.marketradar.daemon.plist
# init_db() runs on start and applies the new columns (dup_cluster_id, is_10b5_1, etc.)
```
Leave **com.marketradar.livetrader** OFF — no live trading until an edge is proven.

## 3. One-time data backfills (idempotent; safe to re-run)
```bash
PYTHONPATH=src .venv/bin/python scripts/refresh_insider_transactions.py   # populates is_10b5_1
# already run this session (safe to re-run): backfill_form_tags, classify_sec_deterministic,
# quarantine_corrupt_outcomes, backfill_near_dup
```

## 4. Verify the fixes are LIVE (after ~30 min of daemon uptime)
```bash
# (a) latency: publish->score should trend < 5 min on NEW live SEC rows
sqlite3 data/market_radar.db "SELECT ROUND(AVG((julianday(ss.scored_at)-julianday(rs.published_at))*1440),1) AS med_min FROM signal_scores ss JOIN raw_signals rs ON rs.id=ss.signal_id WHERE rs.source='sec_edgar' AND rs.published_at>=strftime('%Y-%m-%dT%H:%M:%SZ',datetime('now','-1 day'));"
# (b) hydrate job running: grep the daemon log
grep -c sec_hydrate ~/Library/Logs/MarketRadar/daemon.out.log
# (c) classification 'other' (should stay <12% on sec)
sqlite3 data/market_radar.db "SELECT ROUND(100.0*SUM(CASE WHEN COALESCE(event_type,'other')='other' THEN 1 ELSE 0 END)/COUNT(*),1) FROM signal_scores ss JOIN raw_signals rs ON rs.id=ss.signal_id WHERE rs.source='sec_edgar';"
# (d) new sources producing
sqlite3 data/market_radar.db "SELECT (SELECT COUNT(*) FROM short_interest) finra, (SELECT COUNT(*) FROM catalysts WHERE source='clinicaltrials_v2') ctgov;"
# (e) reliability: stalled-source detector + uptime
PYTHONPATH=src .venv/bin/python -c "from market_radar.storage import get_connection; from market_radar.storage.db import source_uptime_7d; import json; [print(r) for r in source_uptime_7d(next(get_connection().__enter__().__class__ and get_connection().__enter__() for _ in [0]))]" 2>/dev/null || echo "use the dashboard /bot/status"
```

## 5. The edge question — gated on DATA, not code
The trader stays paused. As clean trading days accumulate (need ~40 distinct;
have ~19), periodically:
```bash
PYTHONPATH=src .venv/bin/python scripts/recover_holdout_fast.py   # recover the OOS holdout (~100 min, rate-limited)
PYTHONPATH=src .venv/bin/python scripts/edge_screen_v2.py         # beta-neutral, deduped, cost-aware OOS verdict
```
Fund real money ONLY if a signal clears: positive day-demeaned net-of-cost edge,
persistent out-of-sample across ≥40 distinct days, DSR > 0.95. Nothing clears it today.

## What's still open (honest)
- **Deploy-gated:** latency, new-source population, is_10b5_1 — all become real after steps 2–3.
- **Data-gated (no code fixes this):** scoring A (~40 clean days), measurement A (~2026-07-08 for bang-source 20d windows).
- **Low-value remainders:** macro-gnews tag cleanup (root parser unlocated), gold-set macro-F1 harness (needs a labelled set), opportunistic-buy ML feature (data-gated).
