# Live Trading — Operational Guide

Auto-trader for the MARKET RADAR platform. Runs as a persistent local
process, consumes scored signals, routes through stock brackets or options
spreads, enforces a 7-rule risk gate, and reports via Telegram + a live
dashboard endpoint.

> **Honest framing.** No bot can guarantee a fixed daily P&L. Realistic
> outcome for the strategy implemented here, on a £5k account, is
> ~£20–£200/day on good weeks and net break-even to -£100/day on bad
> weeks, averaging out to ~15-40%/year IF the measured edge holds in live
> conditions. **Always paper-trade for 2-4 weeks before live money.**

---

## Architecture (current state)

```
   signal sources (57 active)
        ↓
   raw_signals → LLM classify → composite score
        ↓
   ML predictor (HistGB + isotonic + conformal)
        ↓   fills model_p_1d, model_p_5d, model_p_20d
   signal_scores
        ↓
   LiveTrader main loop (every 45s):
   ┌───────────────────────────────────────────────────┐
   │ 1. Snapshot Alpaca account + positions + orders   │
   │ 2. Reconcile stock bracket fills → bot_orders     │
   │ 3. Poll open options spreads → exit at TP/SL/time │
   │ 4. EOD flatten — close stocks before market close │
   │ 5. Macro regime gate — halt on PANIC              │
   │ 6. Fetch candidates (model_p ≥ 0.65 or ≤ 0.35,    │
   │    factual=1, source_weight ≥ 7, anti-pump = 0,   │
   │    event_type NOT IN blocked list)                │
   │ 7. For each candidate:                             │
   │    a. Direction pick (sentiment + regime veto)    │
   │    b. PDT guard (US stocks only, < $25k account)  │
   │    c. Selective conformal gate                    │
   │    d. ATR + entry quote                           │
   │    e. Confluence multiplier (insider, catalyst,   │
   │       short interest, FTD, attention)             │
   │    f. Kelly-fractional sizing                     │
   │    g. 7-rule risk manager                         │
   │    h. Route: options whitelist → spread path,     │
   │       else → stock bracket path                   │
   │    i. Submit + persist + Telegram alert           │
   └───────────────────────────────────────────────────┘
```

---

## Strategy: 1-day max holds + daily income focus

The default config is tuned for **same-day or overnight max holds**, NOT
multi-day swing trades. Key changes from a swing setup:

| Parameter | Day-trade default | Why |
|---|---|---|
| `signal_horizon` | `1d` | Bot trades on model_p_1d (1-day move probability), not 5d |
| `stock_sl_atr_mult` | `0.75` | Tight stop = quick resolution |
| `stock_tp_atr_mult` | `1.25` | Quick TP = same R:R 1.67 but faster |
| `eod_flatten_minutes_before_close` | `5` | Force-close all stock positions before close — no overnight stock holds |
| `options_target_dte` | `7` | Short DTE for fast P&L |
| `options_max_hold_hours` | `24` | Spreads close within 24h regardless of TP/SL |
| `options_time_stop_dte` | `1` | Force-close spreads at 1-DTE (no pin risk) |
| `pdt_enforce` | `true` | Stops opening stock day-trades after 2/3 used in rolling 5d |

### The PDT reality

US accounts under $25k face **Pattern Day Trader rules**: max 3 day-trades
per rolling 5 days. Exceed → 90-day restriction. The bot tracks this and
**blocks new stock day-trades once near the limit** (default safety margin = 1).

**To dodge PDT** the bot routes daily income through:
1. **Crypto** (no PDT, 24/7 — 6 pairs: BTC/ETH/LTC/AVAX/LINK/DOGE)
2. **Options held overnight** — not a "day-trade" under SEC definition
3. **Stock day-trades sparingly** — used only on highest-conviction setups

If you fund the account to $25k+, set `LIVE_PDT_ENFORCE=0` to disable.

---

## Setup

### 1. Add Alpaca keys to `.env`

Get free **paper-trading** API keys at https://alpaca.markets → Paper → API Keys.
Then in `.env`:

```bash
ALPACA_API_KEY=PK_your_key
ALPACA_API_SECRET=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # default; live is api.alpaca.markets

# CRITICAL for paper trading: cap the bot's sizing to your real capital
# Alpaca paper starts at $100k of fake money — without this the bot will
# size for $100k and your paper P&L won't translate to your real account.
LIVE_OVERRIDE_EQUITY_USD=6300        # ~£5,000

# Optional: enable Telegram trade alerts
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### 2. Initialize the DB

```bash
python scripts/init_db.py
```

Creates 7 `bot_*` tables: decisions, orders, daily_pnl, account snapshots,
option decisions, option spreads, option legs.

### 3. Verify Alpaca connection

```bash
python scripts/test_alpaca_connection.py
```

### 4. Run integration tests

```bash
python scripts/test_critical_fixes.py
```

All 16 should pass — verifies DST fix, P&L sign, race-safe reconciliation,
idempotency, day-trading pivot, confluence multipliers, regime, dedup.

---

## Running the bot

### Interactive (foreground)

```bash
# Paper trading, stock-only
python scripts/run_live_trader.py --paper

# Paper trading, stock + options on whitelist
python scripts/run_live_trader.py --paper --options

# Dry-run one loop — see what it WOULD do, no orders
python scripts/run_live_trader.py --paper --options --dry-run --once
```

### Background (auto-start on login, restart on crash)

```bash
bash scripts/install_livetrader_launchd.sh --options
# or without options:
bash scripts/install_livetrader_launchd.sh
```

Bot is now a launchd service. Auto-starts at login. Restarts on crash.
Logs:
- `~/Library/Logs/MarketRadar/livetrader.out.log`
- `~/Library/Logs/MarketRadar/livetrader.err.log`

### Status / monitoring

| Tool | Use case |
|---|---|
| `tail -f ~/Library/Logs/MarketRadar/livetrader.out.log` | Watch live decisions stream |
| `curl localhost:8765/bot/status \| jq` | JSON snapshot of P&L + regime + positions |
| Telegram alerts | Phone notification on every PLACED / FILLED / RISK_BLOCKED / REGIME_HALT |
| Alpaca paper dashboard | Source of truth for actual fills |

---

## Kill switches

In order of speed:

1. **Emergency stop (per-iteration)** — add to `.env` and bot picks it up next loop:
   ```bash
   RISK_EMERGENCY_STOP=1
   ```

2. **Halt + flatten everything immediately:**
   ```bash
   python -c "from market_radar.execution import AlpacaClient; \
       c = AlpacaClient(); c.cancel_all_orders(); \
       [c.close_position(p.symbol) for p in c.get_positions()]"
   ```

3. **Unload the launchd job (stops future restarts):**
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.marketradar.livetrader.plist
   ```

---

## Going LIVE (real money)

Only after **≥ 2 weeks of profitable paper trading** with:

- Win rate ≥ 55%
- Positive average daily P&L
- Max drawdown < 8% of equity
- At least 30+ trades (statistically meaningful)
- Performance holds in BOTH up + down regime weeks

Then:

```bash
# In .env:
ALPACA_BASE_URL=https://api.alpaca.markets    # remove "paper-"
# Remove LIVE_OVERRIDE_EQUITY_USD or set it equal to your real account size

# Confirm + start
python scripts/run_live_trader.py --live --i-understand --options
```

Both `--live` and `--i-understand` flags are required.

---

## Known limitations / caveats

These are **honest** — don't gloss over them.

| Issue | Impact | Mitigation |
|---|---|---|
| **Options backtest uses theoretical mid-price fills** | Live spreads cost more (slippage 5-15%) | Bot uses limit orders at mid; real fills may slip 2-5% |
| **Short interest data is FINRA-delayed (twice/month)** | Squeeze setup detection lags | Real-time alternatives cost $$$ (Ortex $69/mo); deferred |
| **Crypto has no leverage on Alpaca** | Daily income from crypto capped by raw price moves | Inherent constraint; offset by 24/7 trading |
| **Paper account is `$100k` by default** | Sizing would be unrealistic | `LIVE_OVERRIDE_EQUITY_USD=6300` caps it to £5k |
| **ML model val_auc ≈ 0.60** | Real edge but not a magic predictor | Combined with selective gate + conformal → 75% measured hit rate at high p |
| **Macro regime fetch uses yfinance (free)** | Occasional 30s lag or stale data | Cached 15min in process; fallback = neutral |
| **Slippage modeled at 5bps** | Real may be 5-25bps on illiquid names | Options whitelist restricted to mega-liquid names |
| **No tick-by-tick options data** | Bot uses Alpaca's OPRA quotes (free tier) | Adequate for spreads on SPY/QQQ; consider Polygon ($79/mo) for tighter exit timing |

---

## Continuous learning

The ML model retrains weekly via launchd:

```bash
launchctl list | grep com.marketradar.retrain
```

Or trigger manually:

```bash
python scripts/scheduled_retrain.py            # weekly cron uses this
python scripts/train_ml.py --multi-horizon     # interactive
```

The new model deploys atomically (pointer file swap) if its
walk-forward val_AUC matches/beats the previous model.

---

## Files

| Path | Purpose |
|---|---|
| `src/market_radar/execution/live_trader.py` | Main loop + all routing |
| `src/market_radar/execution/options/` | Options API client + spread builder + exit poller |
| `src/market_radar/signals/price_action.py` | Real-time bar-based signal generator |
| `src/market_radar/signals/macro_regime.py` | VIX + SPY + yield curve regime classifier |
| `src/market_radar/signals/confluence.py` | Alt-data sizing multipliers |
| `src/market_radar/risk/manager.py` | 7-rule fail-closed risk gate |
| `scripts/run_live_trader.py` | CLI entry point |
| `scripts/install_livetrader_launchd.sh` | Background install (auto-restart) |
| `scripts/test_critical_fixes.py` | Integration tests |
| `dashboard/server.py` | Flask + `/bot/status` JSON endpoint |
