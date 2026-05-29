-- MARKET RADAR — Execution schema (bot order book + decisions)
-- Loaded alongside schema.sql by storage/db.py:init_db().
-- All timestamps are ISO 8601 UTC strings.

-- ============================================================
-- bot_decisions: one row per signal the live trader looked at.
-- We persist EVERY decision (acted or skipped) so the audit trail
-- is complete and we can backtest the gate's selectivity.
-- ============================================================
CREATE TABLE IF NOT EXISTS bot_decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    score_id          INTEGER NOT NULL,
    ticker            TEXT NOT NULL,
    direction         TEXT NOT NULL CHECK (direction IN ('buy','sell')),
    model_p           REAL NOT NULL,
    composite_score   REAL NOT NULL,
    interval_width    REAL,
    base_disagreement REAL,

    -- Gate verdict
    gate_passed       INTEGER NOT NULL,  -- 0/1
    gate_reason       TEXT,
    risk_passed       INTEGER NOT NULL,  -- 0/1
    risk_reason       TEXT,
    risk_blocking_rule TEXT,

    -- Sizing (USD)
    account_equity    REAL,
    size_pct          REAL,
    notional_usd      REAL,
    qty               REAL,
    entry_estimate    REAL,
    stop_loss         REAL,
    take_profit       REAL,
    atr               REAL,

    -- Outcome of decision (one of: 'placed', 'gate_blocked', 'risk_blocked',
    -- 'execution_failed', 'duplicate_position', 'no_quote', 'closed_market')
    outcome           TEXT NOT NULL,
    outcome_detail    TEXT,
    alpaca_order_id   TEXT,
    decided_at        TEXT NOT NULL,

    FOREIGN KEY (score_id) REFERENCES signal_scores(id) ON DELETE CASCADE,
    UNIQUE(score_id)  -- one decision per signal
);

CREATE INDEX IF NOT EXISTS idx_bot_decisions_ticker
    ON bot_decisions(ticker, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_decisions_outcome
    ON bot_decisions(outcome, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_decisions_placed
    ON bot_decisions(decided_at DESC) WHERE outcome = 'placed';

-- ============================================================
-- bot_orders: orders submitted to Alpaca, with their lifecycle.
-- Reconciled against Alpaca's order history every loop iteration.
-- ============================================================
CREATE TABLE IF NOT EXISTS bot_orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     INTEGER,
    alpaca_order_id TEXT NOT NULL UNIQUE,
    client_order_id TEXT NOT NULL UNIQUE,
    ticker          TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('buy','sell')),
    order_class     TEXT NOT NULL,     -- 'bracket', 'simple', 'oco'
    qty             REAL NOT NULL,
    submitted_price REAL,              -- limit/marketable price
    stop_loss       REAL,
    take_profit     REAL,

    -- Lifecycle
    status          TEXT NOT NULL,     -- 'new','accepted','filled','partially_filled','canceled','expired','rejected'
    filled_qty      REAL DEFAULT 0,
    filled_avg_price REAL,
    submitted_at    TEXT NOT NULL,
    filled_at       TEXT,
    canceled_at     TEXT,

    -- Realized P&L when the position closes (sum of TP/SL fill - entry fill, signed by direction)
    realized_pnl_usd REAL,
    pnl_pct          REAL,
    exit_reason      TEXT,             -- 'take_profit','stop_loss','manual','timeout','market_close'

    FOREIGN KEY (decision_id) REFERENCES bot_decisions(id)
);

CREATE INDEX IF NOT EXISTS idx_bot_orders_status
    ON bot_orders(status, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_orders_ticker
    ON bot_orders(ticker, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_orders_filled
    ON bot_orders(filled_at DESC) WHERE filled_at IS NOT NULL;

-- ============================================================
-- bot_daily_pnl: pre-aggregated daily P&L for fast risk-cap checks.
-- Updated on every fill reconciliation.
-- ============================================================
CREATE TABLE IF NOT EXISTS bot_daily_pnl (
    trading_date    TEXT PRIMARY KEY,  -- 'YYYY-MM-DD' in US/Eastern
    realized_pnl_usd REAL NOT NULL DEFAULT 0,
    trades_count    INTEGER NOT NULL DEFAULT 0,
    wins            INTEGER NOT NULL DEFAULT 0,
    losses          INTEGER NOT NULL DEFAULT 0,
    largest_win     REAL DEFAULT 0,
    largest_loss    REAL DEFAULT 0,
    starting_equity REAL,
    ending_equity   REAL,
    updated_at      TEXT NOT NULL
);

-- ============================================================
-- bot_account_snapshots: equity curve, captured every loop iteration.
-- Used for performance reporting and drawdown calc.
-- ============================================================
CREATE TABLE IF NOT EXISTS bot_account_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at     TEXT NOT NULL,
    equity_usd      REAL NOT NULL,
    cash_usd        REAL NOT NULL,
    buying_power_usd REAL NOT NULL,
    long_market_value REAL,
    open_positions  INTEGER,
    open_orders     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_bot_snapshots_time
    ON bot_account_snapshots(snapshot_at DESC);
