"""Human-readable translations used by the dashboard API.

Plain-English source names and event-type labels. Keep this in the
dashboard layer (not the scorer) so it's purely a presentation concern.
"""
from __future__ import annotations

from typing import Optional


SOURCE_DISPLAY: dict[str, str] = {
    # Tier 1 — SEC EDGAR live + backfill
    "sec_edgar": "SEC filing",

    # Tier 2 — news wires
    "bloomberg_google": "Bloomberg",
    "reuters_google": "Reuters",
    "reuters_business_google": "Reuters (biz)",
    "wsj_google": "WSJ",
    "alpaca_news": "Alpaca News",
    "marketwatch_topstories": "MarketWatch — Top",
    "marketwatch_realtime": "MarketWatch — Real-Time",
    "marketwatch_marketpulse": "MarketWatch — Pulse",
    "cnbc_topnews": "CNBC — Top News",
    "cnbc_business": "CNBC — Business",
    "cnbc_markets": "CNBC — Markets",
    "cnbc_earnings": "CNBC — Earnings",
    "yahoo_finance_news": "Yahoo Finance",
    "benzinga_news": "Benzinga",
    "investing_general": "Investing.com",
    "investing_stockmarket": "Investing.com — Stocks",
    "finnhub_news": "Finnhub",
    "seekingalpha_news": "Seeking Alpha",
    "zacks_news": "Zacks",
    "reuters_world_google": "Reuters (6h)",

    # Topical Google News queries — these aggregate across many outlets
    "gnews_ma_topic":            "Google News — M&A",
    "gnews_earnings_topic":      "Google News — Earnings",
    "gnews_fda_topic":           "Google News — FDA / biotech",
    "gnews_buyback_topic":       "Google News — Buyback",
    "gnews_fed_macro":           "Google News — Fed/macro",
    "gnews_analyst_topic":       "Google News — Analyst",
    "gnews_insider_topic":       "Google News — Insider",
    "gnews_ipo_topic":           "Google News — IPO",
    "gnews_layoffs_restructure": "Google News — Layoffs",
    "gnews_lawsuit_regulatory":  "Google News — Legal",
    "gnews_short_squeeze":       "Google News — Short squeeze",
    "gnews_clinical_trial":      "Google News — Clinical trial",
    "gnews_guidance":            "Google News — Guidance",
    "gnews_ceo_change":          "Google News — CEO change",
    "gnews_china_tariffs":       "Google News — China/tariffs",
    "gnews_dividends":           "Google News — Dividend events",
    "gnews_split_spinoff":       "Google News — Split/spinoff",
    "gnews_short_seller":        "Google News — Short seller report",
    "streetinsider_general":     "StreetInsider",
    "thestreet_news":            "TheStreet",
    "motley_fool":               "Motley Fool",
    "investorplace":             "InvestorPlace",
    "marketbeat_ratings":        "MarketBeat",
    "forbes_markets":            "Forbes — Markets",
    "bi_markets":                "Business Insider",
    "briefing_in_play":          "Briefing.com",

    # Tier 3 — social
    "stocktwits_trending": "StockTwits",
    "reddit_securityanalysis": "r/SecurityAnalysis",
    "reddit_valueinvesting": "r/ValueInvesting",
    "reddit_investing": "r/investing",
    "reddit_stocks": "r/stocks",
    "reddit_stockmarket": "r/StockMarket",
    "reddit_options": "r/options",
    "reddit_thetagang": "r/thetagang",
    "reddit_swingtrading": "r/swingtrading",
    "reddit_daytrading": "r/Daytrading",
    "reddit_biotechplays": "r/biotechplays",
    "reddit_biotechnology": "r/Biotechnology",
    "reddit_spacs": "r/SPACs",
    "reddit_dividends": "r/dividends",
    "reddit_economics": "r/economics",
    "reddit_finance": "r/finance",
    "reddit_energystocks": "r/energystocks",
    "reddit_realestateinvesting": "r/realestateinvesting",
    "reddit_wallstreetbets": "r/wallstreetbets",
    "reddit_pennystocks": "r/pennystocks",
    "reddit_shortsqueeze": "r/Shortsqueeze",
}


EVENT_DISPLAY: dict[str, str] = {
    "m_a_announcement":   "M&A deal announced",
    "m_a_rumor":          "M&A talks rumored",
    "fda_approval":       "FDA approval",
    "fda_rejection":      "FDA rejection",
    "earnings_beat":      "Beat earnings",
    "earnings_miss":      "Missed earnings",
    "guidance_raise":     "Raised guidance",
    "guidance_cut":       "Cut guidance",
    "analyst_upgrade":    "Analyst upgrade",
    "analyst_downgrade":  "Analyst downgrade",
    "insider_buy":        "Insider buying",
    "insider_sell":       "Insider selling",
    "insider_transaction": "Insider transaction",
    "activist_position":  "Activist stake (13D)",
    "passive_5pct_stake": "Passive 5%+ stake (13G)",
    "ipo_registration":   "IPO registration",
    "ipo_registration_amend": "IPO filing amendment",
    "macro":              "Macro / Fed news",
    "lawsuit":            "Lawsuit / legal",
    "leadership_change":  "Leadership change",
    "buyback":            "Stock buyback",
    "dividend":           "Dividend news",
    "material_event":     "Material event (8-K)",
    "material_event_amend": "Material event amendment",
    "proxy_statement":    "Proxy / annual meeting",
    "routine_prospectus": "Routine prospectus",
    "routine_proxy":      "Routine proxy",
    "speculation":        "Speculation",
    "other":              "General signal",
}


def display_source(source: Optional[str]) -> str:
    if not source:
        return "—"
    # Backfilled sources: "sec_edgar_backfill_8-k" → "SEC filing (historical)"
    if source.startswith("sec_edgar_backfill_"):
        form = source.removeprefix("sec_edgar_backfill_").upper().replace("_", " ")
        return f"SEC filing ({form}, historical)"
    return SOURCE_DISPLAY.get(source, source)


def display_event(event_type: Optional[str]) -> str:
    if not event_type:
        return "—"
    return EVENT_DISPLAY.get(event_type, event_type.replace("_", " ").title())
