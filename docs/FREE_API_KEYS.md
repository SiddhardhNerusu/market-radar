# Free API keys to grab (no cost, big impact)

Each of these unlocks an existing code path that is already wired but
inactive without the key. Adding the key to `.env` and restarting the
daemon (or running the relevant refresh script) is all that's needed.

## 1. Finnhub — earnings, 13F, transcripts

Free tier: 60 calls/minute, no card required.

1. Sign up at https://finnhub.io/register
2. Copy your API key from the dashboard
3. Add to `.env`:

```
FINNHUB_API_KEY=xxxxxxxxxxxxxxxx
```

Activates:
- `scripts/refresh_earnings_data.py` — earnings calendar, EPS/revenue
  surprises (PEAD feature)
- `scripts/refresh_13f_flow.py` — institutional flow per ticker
- Future: earnings-call transcripts via `/stock/transcripts` (Tier 2 #12)

## 2. NewsAPI — additional news aggregator

Free tier: 1,000 requests/day. Useful as a corroboration source.

1. Sign up at https://newsapi.org/register
2. Add to `.env`:

```
NEWSAPI_KEY=xxxxxxxxxxxxxxxx
```

Currently the key is loaded but no ingestor consumes it. A consumer
ingestor (analogous to `rss_news.py`) is a 1-2 hour follow-up.

## 3. Alpaca news feed — pre-tagged ticker news

Free with Alpaca paper account. You may already have these in your
AUTO TRADER project's `.env`.

1. https://alpaca.markets/ → sign up for paper trading account
2. Generate API keys from dashboard
3. Add to MARKET RADAR `.env`:

```
ALPACA_API_KEY=xxxxxxxxxxxxxxxx
ALPACA_API_SECRET=xxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Currently loaded into config but no ingestor consumes it yet — adding
the Alpaca news stream is a small follow-up (Alpaca's news API maps
each story to ticker symbols, which cuts down on ticker-resolution
noise vs RSS scraping).

## 4. Reddit PRAW — auth'd Reddit ingestion

Free. Upgrades the existing `reddit_public.py` ingestor from public
JSON scraping to authenticated PRAW (better rate limits, author
karma/age data for the anti-pump filter).

1. https://www.reddit.com/prefs/apps → create a "script" app
2. Note the client_id (under the app name) and client_secret
3. Add to `.env`:

```
REDDIT_CLIENT_ID=xxxxxxxxxxxxxxxx
REDDIT_CLIENT_SECRET=xxxxxxxxxxxxxxxx
REDDIT_USER_AGENT=market-radar/0.1 by yourname
```

The existing `reddit_public.py` will detect the credentials and switch
to PRAW mode automatically (the credentials are already read from
`CONFIG.reddit_client_id` and `CONFIG.reddit_client_secret`).

## After you add any of these

Restart the daemon (so it reloads `.env`):

```bash
# foreground
# Ctrl-C the running daemon, then:
python scripts/run_daemon.py
```

Or run the relevant one-off refresh:

```bash
python scripts/refresh_earnings_data.py --top-tickers 100   # needs Finnhub
python scripts/refresh_13f_flow.py --top-tickers 100        # needs Finnhub
```

## Verification

After adding a key, this command shows what's set:

```bash
python -c "
import sys; sys.path.insert(0,'src')
from market_radar.config import CONFIG
print('Finnhub:', 'SET' if CONFIG.finnhub_api_key else 'EMPTY')
print('NewsAPI:', 'SET' if CONFIG.newsapi_key else 'EMPTY')
print('Alpaca: ', 'SET' if CONFIG.alpaca_api_key else 'EMPTY')
print('Reddit: ', 'SET' if CONFIG.has_reddit else 'EMPTY')
"
```
