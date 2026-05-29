# MARKET RADAR — daily cheat sheet

Practical reference. All commands assume you're at the project root.

```bash
cd "$HOME/Documents/Claude/Projects/MARKET RADAR"
source .venv/bin/activate
```

---

## DAILY HEALTH CHECK (run when you wake up + once at night)

```bash
# 1. Is the daemon alive?
ps -p $(cat logs/daemon.pid) -o pid,stat,etime,command

# 2. Is the dashboard responding?
curl -s http://127.0.0.1:8765/health
# expect: {"ok":true,"service":"market-radar-dashboard"}

# 3. Signals scored in last hour (sign of life)
sqlite3 data/market_radar.db "SELECT COUNT(*) FROM signal_scores WHERE scored_at >= datetime('now','-1 hour');"
# > 0 during US market hours = healthy

# 4. launchd jobs registered (backup, retrain, drift)
launchctl list | grep com.marketradar
# expect 3 lines

# 5. Recent daemon errors
tail -30 $(ls -t logs/daemon-*.log | head -1) | grep -iE "error|traceback" | head -5
# empty = healthy

# 6. Notifications fired today
sqlite3 data/market_radar.db "SELECT COUNT(*) FROM notifications_sent WHERE sent_at >= datetime('now','start of day');"

# 7. Latest deployed model AUC
cat data/models/current.json | python -m json.tool
```

---

## DAEMON CONTROL

```bash
# === Start (if it's not running) ===
nohup python scripts/run_daemon.py > "logs/daemon-$(date +%Y%m%d-%H%M).log" 2>&1 &
echo $! > logs/daemon.pid
sleep 5
ps -p $(cat logs/daemon.pid) -o pid,stat,etime,command

# === Stop ===
pkill -f run_daemon.py
sleep 3
ps aux | grep run_daemon | grep -v grep   # should be empty

# === Restart (e.g. after editing .env) ===
pkill -f run_daemon.py
sleep 3
nohup python scripts/run_daemon.py > "logs/daemon-$(date +%H%M).log" 2>&1 &
echo $! > logs/daemon.pid
sleep 5
ps -p $(cat logs/daemon.pid) -o pid,stat,etime,command

# === Watch live log ===
tail -f $(ls -t logs/daemon-*.log | head -1)
# Ctrl-C to exit
```

---

## DASHBOARD

Open in browser:
- `http://127.0.0.1:8765/` — main BUY/SELL view + live feed
- `http://127.0.0.1:8765/graph` — graph-only ranked view (long/short candidates from network features alone)

These only work while the daemon is running.

---

## TEST NOTIFICATIONS (verify Telegram + macOS still work)

```bash
python - <<'PY'
import sys; sys.path.insert(0,'src')
from market_radar.notifications.notifier import Notifier, AlertCandidate
from datetime import datetime, timezone
n = Notifier()
r = n.notify_if_eligible(AlertCandidate(
    signal_id=999_999, ticker="TEST", direction="buy", action="STRONG_BUY",
    p=0.80, title="Manual test alert",
    source="manual_test",
    ingested_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
))
print(f"sent={r.sent}  channels={r.channels_sent}  latency={r.latency_seconds}s")
PY
```
You should physically see:
- A Telegram message on your phone
- A macOS banner notification on your Mac

---

## WHERE FILES LIVE

```
~/Documents/Claude/Projects/MARKET RADAR/
├── .env                      ← API keys, thresholds, config
├── data/
│   ├── market_radar.db       ← the database
│   ├── backups/              ← nightly auto-backups (14 daily kept)
│   ├── models/               ← trained models + pointers
│   └── sec_body_cache/       ← downloaded SEC filing texts
├── logs/
│   ├── daemon-*.log          ← daemon stdout/stderr
│   ├── launchd-*.log         ← scheduled job output
│   └── daemon.pid            ← current daemon process ID
├── scripts/                  ← all the .py runners
├── src/market_radar/         ← the application code
├── dashboard/                ← the web UI
└── docs/                     ← all the docs (you are here)
```

---

## TUNING (edit .env, then restart daemon)

```bash
open -t .env   # or: nano .env
```

Common knobs:
| Setting | Default | Effect |
| --- | --- | --- |
| `NOTIFY_BUY_THRESHOLD` | 0.65 | Lower → more BUY alerts |
| `NOTIFY_SELL_THRESHOLD` | 0.30 | Higher → more SELL alerts |
| `NOTIFY_PER_TICKER_COOLDOWN_MIN` | 30 | Per-ticker no-spam window |
| `NOTIFY_MAX_DAILY` | 20 | Daily total cap |
| `RISK_DAILY_LOSS_CAP_USD` | 200 | Daily loss kill-switch |
| `RISK_MAX_POSITION_PCT` | 5 | Max % of account in one ticker |
| `RISK_EMERGENCY_STOP` | 0 | Set to 1 to block all gates immediately |

After editing `.env`: **restart the daemon** (see DAEMON CONTROL above).

---

## ON-DEMAND OPS

```bash
# Manual DB backup right now
sqlite3 data/market_radar.db ".backup 'data/backups/market_radar.manual-$(date +%Y%m%d-%H%M).db'"

# Manual retrain (takes 20-30 min)
python scripts/train_ml.py --all

# Manual drift check
python scripts/check_model_drift.py --window 500

# Auto-update source weights from measured outcomes
python scripts/recompute_source_weights.py --apply

# Refresh any data source manually
python scripts/refresh_fda_catalysts.py
python scripts/refresh_attention_data.py --top-tickers 100
python scripts/refresh_earnings_data.py --top-tickers 100   # needs Finnhub key
python scripts/refresh_insider_transactions.py
python scripts/refresh_novelty_finbert.py
python scripts/refresh_fails_to_deliver.py
```

---

## PAPER-TRADE TRACKING SHEET

Open a Google Sheet with these columns:

| Date | Time (UTC) | Ticker | Direction | Model p% | Entry price | 1d return | 5d return | Hit (1/0) | Notes |

For each Telegram alert:
1. Add a row immediately when it fires (timestamp + ticker + p%)
2. Look up entry price in T212 app or yfinance
3. Come back in 1-5 days and fill in the returns
4. Mark "Hit = 1" if the direction was right (buy went up / sell went down)

After 30 rows, calculate hit rate:
- ≥ 60% → real edge, worth sizing real money
- 50-60% → marginal, watch longer
- < 50% → no edge, don't trade

---

## TROUBLESHOOTING

### No alerts in 24 hours
1. Check daemon is alive: `ps -p $(cat logs/daemon.pid)`
2. If dead → restart it (DAEMON CONTROL section)
3. Check thresholds in .env aren't too strict
4. Send test notification (above)

### Dashboard shows "loading…" forever
- Daemon dead → restart it
- OR daemon alive but predict.py stuck → restart it

### Mac sleeps and everything stops
- System Settings → Battery → Power Adapter → "Prevent automatic sleeping when display is off" = ON
- OR run `caffeinate -d &` in a terminal (keeps Mac awake while terminal is open)

### Telegram silent but macOS banners work
- `curl "https://api.telegram.org/bot<TOKEN>/getMe"` → should return bot info
- If 401: token is wrong → re-grab from BotFather
- If 200: token works → check chat_id is right

### Disk filling up
```bash
du -sh data/backups/
# If > 5GB, rotate manually:
ls -t data/backups/market_radar.daily-*.db | tail -n +14 | xargs rm
```

### "Database is locked" error
- Daemon + another script writing at the same time
- Stop the daemon, run the script, restart the daemon
- DB has WAL mode so this is rare but possible

### Database corrupt ("disk image malformed")
```bash
# Recovery — keep this command handy
cp data/market_radar.db data/market_radar.broken.db
sqlite3 data/market_radar.db ".recover" > /tmp/recover.sql
mv data/market_radar.db data/market_radar.old.db
sqlite3 data/market_radar.db < /tmp/recover.sql
sqlite3 data/market_radar.db "PRAGMA integrity_check;"
```

---

## ANNUAL HOUSEKEEPING

Once a year, refresh the FOMC calendar in
`src/market_radar/ml/calendar_features.py` (hardcoded dates run through
2027 currently). When the Fed publishes 2028 meetings, add them to
`FOMC_DATES`.

Once a month, glance at the SHAP top-10 features (visible at
`/api/risk/status` after running `python scripts/explain_model.py`) —
if a feature drops to mean(|SHAP|) ≈ 0 for 4+ weeks, consider removing
it from FEATURE_NAMES.
