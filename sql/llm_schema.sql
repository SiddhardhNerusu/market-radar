-- Tables for LLM enrichment (applied via _migrate_columns in storage/db.py).
-- These are layered on top of signal_scores — not a replacement.

-- Per-signal LLM classification result. One row per (signal_id, ticker)
-- where we ran the classifier.
CREATE TABLE IF NOT EXISTS llm_classifications (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id         INTEGER NOT NULL,
    ticker            TEXT NOT NULL,

    -- Refined event_type (from LLM, more accurate than heuristic)
    event_type        TEXT,
    event_subtype     TEXT,        -- e.g. "earnings_beat_with_raised_guidance"

    -- Refined sentiment (-1..+1) and magnitude (0..1)
    sentiment         REAL,
    sentiment_magnitude REAL,

    -- factual = 1, speculation = 0
    factual           INTEGER,

    -- LLM-extracted structured fields (JSON; varies by event type)
    -- e.g. {"insider_role": "CEO", "buy_size_usd": 2300000, "is_buy": true}
    --      {"deal_size_usd": 5000000000, "acquirer": "NVDA", "target": "ARM"}
    --      {"drug_name": "FOO-123", "indication": "lung cancer", "decision": "approved"}
    extracted_fields  TEXT,        -- JSON

    -- LLM's confidence in its own classification (0..1)
    confidence        REAL,

    -- Model used + cost tracking
    model             TEXT,        -- e.g. "claude-haiku-4-5-20251001"
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    cost_usd          REAL,

    classified_at     TEXT NOT NULL,

    FOREIGN KEY (signal_id) REFERENCES raw_signals(id) ON DELETE CASCADE,
    UNIQUE(signal_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_llm_classifications_event
    ON llm_classifications(event_type, sentiment);
CREATE INDEX IF NOT EXISTS idx_llm_classifications_signal
    ON llm_classifications(signal_id);

-- Daily LLM cost tracking — enforces the hard daily-spend cap
CREATE TABLE IF NOT EXISTS llm_spend_daily (
    date              TEXT PRIMARY KEY,    -- YYYY-MM-DD UTC
    total_cost_usd    REAL NOT NULL DEFAULT 0,
    total_calls       INTEGER NOT NULL DEFAULT 0,
    last_updated_at   TEXT NOT NULL
);
