-- MARKET RADAR — Insider lane schema (long-only, multi-day, Form 4 code-P buys)
-- Loaded alongside schema.sql by storage/db.py:init_db().
-- All timestamps are ISO 8601 UTC strings; *_day columns are US/Eastern trading dates.

-- ============================================================
-- lane_decisions: exactly ONE row per (ticker, event_day) the lane ever
-- considered. This is the strict one-bet-per-ticker-day dedup (the sticky
-- 30-day window that manufactured a false p=0.003 in the June research is
-- structurally impossible here), the re-entry guard (survives restarts,
-- unlike the old in-memory 300 s dict), and the idempotency record for the
-- submit (pending_submit row + deterministic client_order_id).
-- ============================================================
CREATE TABLE IF NOT EXISTS lane_decisions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker             TEXT NOT NULL,
    event_day          TEXT NOT NULL,            -- ET trading date the filing became actionable
    published_at       TEXT NOT NULL,            -- raw_signals.published_at of the Form 4
    signal_ids         TEXT,                     -- comma-joined raw_signals ids aggregated into this bet
    notional_usd       REAL,                     -- summed insider purchase notional that day
    role_score         INTEGER,
    -- 'placed' | 'pending_submit' | 'skipped' | 'blocked' | 'failed'
    outcome            TEXT NOT NULL,
    outcome_detail     TEXT,
    client_order_id    TEXT UNIQUE,
    parent_order_id    TEXT,
    decided_at         TEXT NOT NULL,
    UNIQUE (ticker, event_day)
);
CREATE INDEX IF NOT EXISTS idx_lane_decisions_ticker ON lane_decisions(ticker, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_lane_decisions_outcome ON lane_decisions(outcome, decided_at DESC);

-- ============================================================
-- lane_positions: the lane's own book. One row per placed entry.
-- Exits are broker-side (bracket stop + target, GTC) so a crash or a
-- sleeping machine never leaves a position naked; the lane only adds a
-- time stop and the daily-loss flatten.
-- ============================================================
CREATE TABLE IF NOT EXISTS lane_positions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id        INTEGER NOT NULL,
    ticker             TEXT NOT NULL,
    qty                REAL NOT NULL,
    entry_limit        REAL NOT NULL,
    stop_loss          REAL NOT NULL,
    take_profit        REAL NOT NULL,
    atr                REAL,
    risk_usd           REAL NOT NULL,            -- qty * (entry - stop): the $ at risk this bet adds to heat
    parent_order_id    TEXT NOT NULL,
    client_order_id    TEXT NOT NULL,
    -- 'submitted' | 'open' | 'closing' | 'closed' | 'canceled'
    status             TEXT NOT NULL,
    submitted_at       TEXT NOT NULL,
    filled_at          TEXT,
    filled_avg_price   REAL,
    closed_at          TEXT,
    exit_reason        TEXT,                     -- 'take_profit' | 'stop_loss' | 'time_stop' | 'daily_loss' | 'external' | 'unfilled'
    FOREIGN KEY (decision_id) REFERENCES lane_decisions(id)
);
CREATE INDEX IF NOT EXISTS idx_lane_positions_status ON lane_positions(status, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_lane_positions_ticker ON lane_positions(ticker, submitted_at DESC);

-- ============================================================
-- lane_state: tiny key/value store for state that must survive a restart
-- (day-start equity, halted-on date). Replaces the lazily created
-- hasattr() dicts in the old trader that were lost on every restart.
-- ============================================================
CREATE TABLE IF NOT EXISTS lane_state (
    key                TEXT PRIMARY KEY,
    value              TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

-- ============================================================
-- edge_ledger: the forward out-of-sample record. One row per
-- (candidate, ticker, event_day) that QUALIFIED, whether or not it was
-- traded. price_at_flag is the live ask at decision time (an actionable
-- price, not a stale close), bench_* is SPY at the same instants, so the
-- 5d / 20d abnormal returns are what a real buyer could have earned.
-- Append-only. The promotion gate (lane/promotion.py) reads this table
-- and nothing else.
-- ============================================================
CREATE TABLE IF NOT EXISTS edge_ledger (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate          TEXT NOT NULL,            -- e.g. 'insider_P_ge10_notional_ge25k'
    ticker             TEXT NOT NULL,
    event_day          TEXT NOT NULL,
    published_at       TEXT NOT NULL,
    notional_usd       REAL,
    role_score         INTEGER,
    price_at_flag      REAL,
    price_at_flag_ts   TEXT,
    bench_at_flag      REAL,
    price_5d           REAL,
    bench_5d           REAL,
    ret_5d_pct         REAL,                     -- raw, close of 5th trading day after event_day
    abn_5d_pct         REAL,                     -- ret_5d - bench_5d
    price_20d          REAL,
    bench_20d          REAL,
    ret_20d_pct        REAL,
    abn_20d_pct        REAL,
    cost_frac          REAL,                     -- round-trip cost assumed for this price tier
    traded             INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL,
    resolved_at        TEXT,
    UNIQUE (candidate, ticker, event_day)
);
CREATE INDEX IF NOT EXISTS idx_edge_ledger_candidate ON edge_ledger(candidate, event_day);
