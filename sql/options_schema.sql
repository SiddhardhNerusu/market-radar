-- MARKET RADAR — Options spread execution schema
-- Loaded alongside execution_schema.sql by storage/db.py:init_db().
-- Captures multi-leg option spread orders + their lifecycle.

-- ============================================================
-- bot_option_decisions: one row per signal we evaluated for an option spread.
-- Mirrors bot_decisions but for the options path.
-- ============================================================
CREATE TABLE IF NOT EXISTS bot_option_decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    score_id            INTEGER NOT NULL,
    underlying          TEXT NOT NULL,
    direction           TEXT NOT NULL CHECK (direction IN ('buy','sell')),
    model_p             REAL NOT NULL,
    strategy            TEXT NOT NULL,        -- 'bull_call_debit' | 'bear_put_debit'
    long_strike         REAL,
    short_strike        REAL,
    expiration_date     TEXT,
    width_usd           REAL,
    debit_per_spread    REAL,
    max_loss_per_spread REAL,
    max_gain_per_spread REAL,
    contracts           INTEGER,
    total_debit_usd     REAL,
    kelly_raw           REAL,
    size_pct            REAL,

    -- Gate verdict
    gate_passed         INTEGER NOT NULL,
    gate_reason         TEXT,
    risk_passed         INTEGER NOT NULL,
    risk_reason         TEXT,

    outcome             TEXT NOT NULL,        -- 'placed','rejected','no_chain','wide_spread','no_quote','dry_run'
    outcome_detail      TEXT,
    alpaca_order_id     TEXT,
    decided_at          TEXT NOT NULL,

    FOREIGN KEY (score_id) REFERENCES signal_scores(id) ON DELETE CASCADE,
    UNIQUE(score_id)
);

CREATE INDEX IF NOT EXISTS idx_bot_option_decisions_underlying
    ON bot_option_decisions(underlying, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_option_decisions_outcome
    ON bot_option_decisions(outcome, decided_at DESC);

-- ============================================================
-- bot_option_spreads: a placed multi-leg spread (parent).
-- ============================================================
CREATE TABLE IF NOT EXISTS bot_option_spreads (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id         INTEGER,
    alpaca_order_id     TEXT NOT NULL UNIQUE,
    client_order_id     TEXT NOT NULL UNIQUE,
    underlying          TEXT NOT NULL,
    strategy            TEXT NOT NULL,
    direction           TEXT NOT NULL,
    long_strike         REAL NOT NULL,
    short_strike        REAL NOT NULL,
    expiration_date     TEXT NOT NULL,
    contracts           INTEGER NOT NULL,
    entry_debit_usd     REAL,                   -- per-spread debit at fill
    total_debit_usd     REAL,                   -- contracts * entry_debit * 100
    max_loss_usd        REAL,
    max_gain_usd        REAL,

    status              TEXT NOT NULL,          -- 'new','filled','partially_filled','canceled','expired'
    submitted_at        TEXT NOT NULL,
    filled_at           TEXT,

    -- Exit tracking
    closed_at           TEXT,
    exit_credit_usd     REAL,                   -- per-spread credit at close
    realized_pnl_usd    REAL,
    pnl_pct             REAL,
    exit_reason         TEXT,                   -- 'take_profit','stop_loss','time_stop','expiration','manual'

    FOREIGN KEY (decision_id) REFERENCES bot_option_decisions(id)
);

CREATE INDEX IF NOT EXISTS idx_bot_option_spreads_status
    ON bot_option_spreads(status, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_option_spreads_underlying
    ON bot_option_spreads(underlying, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_option_spreads_open
    ON bot_option_spreads(closed_at) WHERE closed_at IS NULL;

-- ============================================================
-- bot_option_legs: individual legs of each spread (child).
-- Multi-leg orders fill at the spread level but each leg has its own
-- contract symbol + fill price. We persist both for accurate exit pricing.
-- ============================================================
CREATE TABLE IF NOT EXISTS bot_option_legs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    spread_id           INTEGER NOT NULL,
    role                TEXT NOT NULL CHECK (role IN ('long','short')),
    contract_symbol     TEXT NOT NULL,          -- OCC-style: AAPL241220C00150000
    option_type         TEXT NOT NULL CHECK (option_type IN ('call','put')),
    strike              REAL NOT NULL,
    expiration_date     TEXT NOT NULL,
    side                TEXT NOT NULL CHECK (side IN ('buy','sell')),
    ratio_qty           INTEGER NOT NULL DEFAULT 1,
    filled_qty          REAL DEFAULT 0,
    filled_avg_price    REAL,                   -- premium paid/received per contract

    FOREIGN KEY (spread_id) REFERENCES bot_option_spreads(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_bot_option_legs_spread ON bot_option_legs(spread_id);
