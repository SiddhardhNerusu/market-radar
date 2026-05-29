-- MARKET RADAR — SQLite schema
-- All timestamps are ISO 8601 UTC strings ("2026-05-12T11:37:00Z").
-- Run via: python scripts/init_db.py
--
-- NOTE: journal_mode and foreign_keys are configured at connection time
-- (see storage/db.py), not here, so this script runs on filesystems that
-- don't support WAL (network mounts, the Cowork sandbox, etc.).

-- ============================================================
-- raw_signals: every piece of content ingested from any source
-- ============================================================
CREATE TABLE IF NOT EXISTS raw_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,   -- 'sec_edgar', 'yahoo_rss', 'reddit_wallstreetbets', 'stocktwits', 'finnhub_news', 'alpaca_news', etc.
    source_tier     INTEGER NOT NULL CHECK (source_tier IN (1, 2, 3, 4)),
    external_id     TEXT,            -- source-specific unique ID for dedup
    url             TEXT,
    title           TEXT,
    body            TEXT,
    author          TEXT,
    author_metadata TEXT,            -- JSON: karma, account age, post history
    raw_payload     TEXT,            -- JSON: full original payload for re-processing
    ingested_at     TEXT NOT NULL,
    published_at    TEXT,
    UNIQUE(source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_signals_ingested ON raw_signals(ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_signals_published ON raw_signals(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_signals_source ON raw_signals(source, ingested_at DESC);

-- ============================================================
-- signal_tickers: extracted tickers per signal (many-to-many)
-- ============================================================
CREATE TABLE IF NOT EXISTS signal_tickers (
    signal_id    INTEGER NOT NULL,
    ticker       TEXT NOT NULL,
    market       TEXT,                  -- 'US', 'LSE', 'XETRA', etc.
    asset_class  TEXT,                  -- 'large_cap', 'mid_cap', 'small_cap', 'penny', 'etf', 'adr'
    confidence   REAL DEFAULT 1.0,      -- 0-1 confidence this signal is about this ticker
    PRIMARY KEY (signal_id, ticker),
    FOREIGN KEY (signal_id) REFERENCES raw_signals(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_signal_tickers_ticker ON signal_tickers(ticker);
CREATE INDEX IF NOT EXISTS idx_signal_tickers_asset ON signal_tickers(asset_class);

-- ============================================================
-- signal_scores: scored output per (signal, ticker) pair
-- ============================================================
CREATE TABLE IF NOT EXISTS signal_scores (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id             INTEGER NOT NULL,
    ticker                TEXT NOT NULL,

    -- LLM-classified fields
    event_type            TEXT,        -- 'earnings_beat', 'earnings_miss', 'm_a_rumor', 'm_a_confirmed',
                                       -- 'fda_approval', 'fda_rejection', 'analyst_upgrade', 'analyst_downgrade',
                                       -- 'insider_buy', 'insider_sell', 'guidance_raise', 'guidance_cut',
                                       -- 'macro', 'sector_rotation', 'speculation', 'other'
    sentiment             REAL,        -- -1.0 (very bearish) to +1.0 (very bullish)
    sentiment_magnitude   REAL,        -- 0.0 (weak) to 1.0 (strong)
    factual               INTEGER,     -- 0 = speculation/rumor, 1 = factual/confirmed

    -- Algorithmic fields
    source_weight         REAL NOT NULL,  -- 1-10 by source tier
    corroboration_count   INTEGER DEFAULT 0,  -- # independent signals on same ticker in last 4h
    author_quality        REAL DEFAULT 1.0,  -- 0-1 for social, 1 for institutional
    anti_pump_flag        INTEGER DEFAULT 0,  -- 1 = flagged as likely pump

    -- Final composite
    composite_score       REAL NOT NULL,  -- 0-10
    signal_class          TEXT,           -- bucket key for outcome aggregation, e.g.
                                          -- 'tier1_factual_bullish_high_corroboration'
    scored_at             TEXT NOT NULL,

    -- ML model output (populated by the predict job; null until model exists)
    model_p_5d            REAL,           -- predicted probability of positive 5d return
    model_version         TEXT,           -- which model produced model_p_5d

    FOREIGN KEY (signal_id) REFERENCES raw_signals(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_signal_scores_composite ON signal_scores(composite_score DESC, scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_signal_scores_ticker ON signal_scores(ticker, scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_signal_scores_class ON signal_scores(signal_class);
CREATE INDEX IF NOT EXISTS idx_signal_scores_event ON signal_scores(event_type);

-- ============================================================
-- signal_outcomes: tracked price returns after flag time
-- This is the edge-measurement table. Every scored signal gets a row.
-- ============================================================
CREATE TABLE IF NOT EXISTS signal_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    score_id        INTEGER NOT NULL UNIQUE,
    ticker          TEXT NOT NULL,

    price_at_flag   REAL,
    price_at_flag_ts TEXT,

    price_1d        REAL,
    price_1d_ts     TEXT,
    return_1d_pct   REAL,    -- computed; null until 1d has passed

    price_5d        REAL,
    price_5d_ts     TEXT,
    return_5d_pct   REAL,

    price_20d       REAL,
    price_20d_ts    TEXT,
    return_20d_pct  REAL,

    fully_resolved  INTEGER DEFAULT 0,  -- 1 when 20d snapshot taken

    FOREIGN KEY (score_id) REFERENCES signal_scores(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_outcomes_resolved ON signal_outcomes(fully_resolved, ticker);
CREATE INDEX IF NOT EXISTS idx_outcomes_ticker ON signal_outcomes(ticker);

-- ============================================================
-- notifications_sent: dedup macOS notifications
-- ============================================================
CREATE TABLE IF NOT EXISTS notifications_sent (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    score_id  INTEGER NOT NULL,
    channel   TEXT NOT NULL,   -- 'macos_notification', 'email', etc.
    sent_at   TEXT NOT NULL,
    FOREIGN KEY (score_id) REFERENCES signal_scores(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notifications_score ON notifications_sent(score_id);

-- ============================================================
-- T212 portfolio data
-- ============================================================
CREATE TABLE IF NOT EXISTS t212_account_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at   TEXT NOT NULL,
    account_type  TEXT NOT NULL CHECK (account_type IN ('invest', 'isa')),
    cash          REAL,
    total_value   REAL,
    invested      REAL,
    pnl           REAL,
    raw_payload   TEXT
);

CREATE INDEX IF NOT EXISTS idx_t212_account_snapshots ON t212_account_snapshots(account_type, snapshot_at DESC);

CREATE TABLE IF NOT EXISTS t212_positions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at    TEXT NOT NULL,
    account_type   TEXT NOT NULL CHECK (account_type IN ('invest', 'isa')),
    ticker         TEXT NOT NULL,
    quantity       REAL NOT NULL,
    average_price  REAL NOT NULL,
    current_price  REAL,
    pnl_pct        REAL,
    pnl_value      REAL,
    raw_payload    TEXT
);

CREATE INDEX IF NOT EXISTS idx_t212_positions_snap ON t212_positions(account_type, snapshot_at DESC);
CREATE INDEX IF NOT EXISTS idx_t212_positions_ticker ON t212_positions(ticker, snapshot_at DESC);

CREATE TABLE IF NOT EXISTS t212_orders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      TEXT NOT NULL,
    account_type  TEXT NOT NULL CHECK (account_type IN ('invest', 'isa')),
    ticker        TEXT NOT NULL,
    side          TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity      REAL NOT NULL,
    price         REAL,
    order_type    TEXT,
    status        TEXT,
    placed_at     TEXT NOT NULL,
    filled_at     TEXT,
    raw_payload   TEXT,
    UNIQUE(order_id, account_type)
);

CREATE INDEX IF NOT EXISTS idx_t212_orders_ticker ON t212_orders(ticker, placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_t212_orders_status ON t212_orders(status, placed_at DESC);

-- ============================================================
-- Equity curve points (computed periodically from T212 snapshots)
-- ============================================================
CREATE TABLE IF NOT EXISTS equity_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          TEXT NOT NULL,    -- YYYY-MM-DD
    account_type  TEXT NOT NULL CHECK (account_type IN ('invest', 'isa', 'combined')),
    total_value   REAL NOT NULL,
    cash          REAL,
    invested      REAL,
    UNIQUE(date, account_type)
);

CREATE INDEX IF NOT EXISTS idx_equity_history ON equity_history(account_type, date DESC);

-- ============================================================
-- Benchmark prices (SPY, QQQ, etc.) for comparison charts
-- ============================================================
CREATE TABLE IF NOT EXISTS benchmark_prices (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    close   REAL NOT NULL,
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_prices ON benchmark_prices(ticker, date DESC);

-- ============================================================
-- Daemon health: track last successful poll per source
-- ============================================================
CREATE TABLE IF NOT EXISTS daemon_health (
    source            TEXT PRIMARY KEY,
    last_poll_at      TEXT,
    last_success_at   TEXT,
    last_error        TEXT,
    consecutive_errors INTEGER DEFAULT 0
);
