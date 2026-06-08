"""Source weight registry.

Source weight is the base credibility score (0–10) of a source. Tier 1
(verified factual) sources sit at 9+, Tier 2 (mainstream news) at 6–8.5,
Tier 3 (social) at 2–5. Unknown sources get a low fallback so we never
over-rate something we haven't curated.

Keep this list aligned with the ingestor registries in ``ingestors/``.
"""
from __future__ import annotations


SOURCE_WEIGHTS: dict[str, float] = {
    # Tier 1 — factual, verified
    "sec_edgar": 9.5,
    "nasdaq_halts": 8.0,   # exchange trading-halt feed — authoritative, real-time

    # Tier 2 — mainstream financial news
    "bloomberg_google": 8.5,
    "reuters_google": 8.0,
    "reuters_business_google": 8.0,
    "reuters_world_google": 8.0,
    "wsj_google": 8.0,
    "alpaca_news": 7.5,
    "marketwatch_topstories": 7.5,
    "marketwatch_realtime": 7.5,
    "marketwatch_marketpulse": 7.5,
    "cnbc_topnews": 7.0,
    "cnbc_business": 7.0,
    "cnbc_markets": 7.0,
    "cnbc_earnings": 7.5,
    "yahoo_finance_news": 6.5,
    "benzinga_news": 6.0,
    "investing_general": 6.0,
    "investing_stockmarket": 6.0,
    "finnhub_news": 6.5,
    "seekingalpha_news": 6.5,
    "zacks_news": 6.5,
    # Topical Google News queries
    "gnews_ma_topic": 7.0,
    "gnews_earnings_topic": 7.0,
    "gnews_fda_topic": 7.5,
    "gnews_buyback_topic": 6.5,
    "gnews_fed_macro": 8.0,
    "gnews_analyst_topic": 6.5,
    "gnews_insider_topic": 7.0,
    "gnews_ipo_topic": 6.5,
    "gnews_layoffs_restructure": 6.5,
    "gnews_lawsuit_regulatory": 7.0,
    "gnews_short_squeeze": 6.0,
    "gnews_clinical_trial": 7.0,
    "gnews_guidance": 7.0,
    "gnews_ceo_change": 7.0,
    "gnews_china_tariffs": 7.5,
    "gnews_dividends": 6.5,
    "gnews_split_spinoff": 6.5,
    "gnews_short_seller": 7.5,
    # Retail-oriented outlets
    "streetinsider_general": 7.0,
    "thestreet_news": 6.5,
    "motley_fool": 5.5,
    "investorplace": 5.5,
    "marketbeat_ratings": 6.5,
    "forbes_markets": 6.0,
    "bi_markets": 6.0,
    "briefing_in_play": 7.0,

    # Tier 3 — social
    "stocktwits_trending": 4.0,
    "reddit_securityanalysis": 5.0,
    "reddit_valueinvesting": 4.5,
    "reddit_investing": 4.0,
    "reddit_stocks": 4.0,
    "reddit_stockmarket": 4.0,
    "reddit_options": 3.5,
    "reddit_thetagang": 4.0,
    "reddit_swingtrading": 3.5,
    "reddit_daytrading": 3.0,
    "reddit_biotechplays": 3.5,
    "reddit_biotechnology": 4.0,
    "reddit_spacs": 3.0,
    "reddit_dividends": 4.0,
    "reddit_economics": 4.0,
    "reddit_finance": 4.0,
    "reddit_energystocks": 3.5,
    "reddit_realestateinvesting": 3.5,
    "reddit_wallstreetbets": 3.0,
    "reddit_pennystocks": 2.0,
    "reddit_shortsqueeze": 2.0,
}

# Fallback if a source we haven't catalogued lands in the DB. Deliberately
# low — keeps unknown sources from generating high-confidence signals.
UNKNOWN_SOURCE_WEIGHT = 3.0


def weight_for(source: str) -> float:
    # Historical backfill rows carry source names like
    # "sec_edgar_backfill_8-k" — same credibility as live SEC.
    if source.startswith("sec_edgar_backfill_") or source == "sec_edgar":
        return SOURCE_WEIGHTS["sec_edgar"]
    return SOURCE_WEIGHTS.get(source, UNKNOWN_SOURCE_WEIGHT)
