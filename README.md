# MARKET RADAR

A 24/7 signals research dashboard with a Trading 212 portfolio view. Scrapes verified market-moving sources, scores each signal for credibility and impact, and **measures the actual edge** of every signal class against subsequent price action.

You make every trade decision yourself. This tool does not auto-execute trades.

## What it does

- **Ingest** market signals continuously from tiered sources:
  - **Tier 1 (factual):** SEC EDGAR filings (8-K, 13D/G, S-1, Form 4 insider trades), earnings calendar, FDA calendar, Alpaca news API.
  - **Tier 2 (mainstream news):** Yahoo Finance, MarketWatch, CNBC, Reuters/Bloomberg via Google News, Seeking Alpha, Benzinga, Finnhub free.
  - **Tier 3 (social):** Reddit (WSB, stocks, investing, SecurityAnalysis, options, biotechplays) filtered by karma/age, StockTwits public stream.
- **Score** every signal: source weight + multi-source corroboration in a 4h window + LLM event/sentiment/factual classification + author quality + anti-pump heuristics → composite 0–10.
- **Track outcomes** for every flagged signal: snapshot price at flag time and re-check 1d/5d/20d returns. The dashboard surfaces measured hit rate per signal class. *This is the entire point — measured edge, not asserted probability.*
- **Display** in a Cowork artifact dashboard: T212 portfolio (Invest + ISA), live signal feed, per-ticker drill-downs, edge metrics, and a stage-and-confirm trade panel.
- **Notify** via macOS notification when a signal clears a high composite threshold.

## Architecture

```
                +--------------------+
   sources ---> | scraper daemon     | ---> SQLite DB <--- T212 client
                | (launchd, 24/7)    |          ^
                +--------------------+          |
                                                |
                                        Cowork artifact dashboard
```

Scraper daemon runs as a `launchd` job on your Mac — auto-starts at login, restarts on crash, logs to `~/Library/Logs/MarketRadar/`. The Cowork artifact is a thin UI that reads from the local SQLite database via the workspace bash bridge.

## Project layout

```
MARKET RADAR/
├── src/market_radar/
│   ├── ingestors/      # Tier 1/2/3 fetchers
│   ├── scoring/        # composite score + LLM classification
│   ├── outcomes/       # post-flag price tracking
│   ├── storage/        # SQLite access layer
│   ├── t212/           # Trading 212 API client (Invest + ISA)
│   └── notifications/  # macOS notifications
├── dashboard/          # Cowork artifact HTML
├── sql/schema.sql      # DB schema
├── scripts/            # CLI tools (init_db, run_daemon, install_launchd)
├── launchd/            # com.marketradar.daemon.plist
├── data/               # SQLite DB lives here (gitignored)
└── logs/               # daemon logs (gitignored)
```

## Setup

Coming soon — populated after scaffolding is complete.

## Status

- [x] Project scaffold
- [ ] Tier 1 ingestors (SEC EDGAR, earnings, FDA)
- [ ] Tier 2 ingestors (financial news RSS + Alpaca)
- [ ] Tier 3 ingestors (Reddit + StockTwits)
- [ ] Scoring + verification layer
- [ ] Outcome tracker
- [ ] launchd daemon install
- [ ] T212 client (blocked on API key)
- [ ] Cowork artifact dashboard
- [ ] End-to-end verification
