"""Nasdaq Trader trading-halt feed — the real-time "this stock is banging" detector.

A LULD (Limit-Up/Limit-Down) volatility halt fires when a stock moves too far,
too fast — which is exactly the signature of the catalyst "bangs" we hunt
(INHD +2000%, VERU +159%, etc.). A halt is the market screaming that a name is
moving NOW. The Nasdaq Trader feed covers halts across ALL US exchanges (not
just Nasdaq) — volatility pauses, news halts (T1/T2), and regulatory halts.

This ingestor records every halt into ``raw_signals`` so the scoring + LLM +
notification pipeline sees it. A halt alone is not a directional trade (you
literally can't trade during a halt), but it (a) surfaces names that are moving
violently so a paired news catalyst gets corroboration, and (b) is the seed for
trading the resumption. Tier 1 — it is authoritative exchange data, a hard fact.

Feed: https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts (public, no auth).
Each <item> carries ndaq:-namespaced fields (IssueSymbol, ReasonCode, HaltTime,
ResumptionTradeTime, ...). We parse by local tag name so a namespace-URI change
doesn't break us.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Optional

import requests

from .base import Ingestor, ParsedSignal, TickerMention

log = logging.getLogger(__name__)

FEED_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"

# Halt reason codes → human description (drives LLM classification + readability).
# https://www.nasdaqtrader.com/Trader.aspx?id=TradeHaltCodes
REASON_CODES = {
    "T1": "news pending",
    "T2": "news released",
    "T3": "news and resumption times",
    "T5": "single-stock volatility trading pause",
    "T6": "extraordinary market activity",
    "T8": "ETF halt",
    "T12": "additional information requested by exchange",
    "H4": "non-compliance",
    "H9": "not current in regulatory filings",
    "H10": "SEC trading suspension",
    "H11": "regulatory concern",
    "LUDP": "volatility (LULD) trading pause",
    "LUDS": "volatility (LULD) trading pause - straddle",
    "MWC1": "market-wide circuit breaker (level 1)",
    "MWC2": "market-wide circuit breaker (level 2)",
    "MWC3": "market-wide circuit breaker (level 3)",
    "M": "volatility trading pause",
    "D": "listing deficiency",
}


class NasdaqHaltsIngestor(Ingestor):
    """Nasdaq Trader trading-halt RSS → ``raw_signals``. Tier 1 (exchange fact)."""

    source = "nasdaq_halts"
    source_tier = 1

    def __init__(self, *, timeout: float = 15.0) -> None:
        super().__init__()
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "market-radar/1.0 (catalyst halt monitor)",
                "Accept": "application/rss+xml, application/xml, text/xml",
            }
        )

    # ------------------------------------------------------------------

    def fetch(self) -> Iterable[dict[str, Any]]:
        try:
            resp = self._session.get(FEED_URL, timeout=self.timeout)
        except requests.RequestException as exc:
            log.warning("[%s] fetch failed: %s", self.source, exc)
            return
        if resp.status_code >= 400:
            log.warning("[%s] HTTP %d", self.source, resp.status_code)
            return

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            log.warning("[%s] XML parse error: %s", self.source, exc)
            return

        items = 0
        # Items live under channel/item; iterate any <item> regardless of namespace.
        for item in root.iter():
            if item.tag.split("}")[-1] != "item":
                continue
            fields: dict[str, str] = {}
            for child in item:
                local = child.tag.split("}")[-1]
                if child.text and child.text.strip():
                    fields[local] = child.text.strip()
            if fields:
                items += 1
                yield fields

        log.info("[%s] fetched %d halt rows", self.source, items)

    # ------------------------------------------------------------------

    def parse(self, raw_entry: dict[str, Any]) -> Optional[ParsedSignal]:
        symbol = (raw_entry.get("IssueSymbol") or "").strip().upper()
        if not symbol:
            return None  # market-wide circuit-breaker rows have no symbol — skip

        reason_code = (raw_entry.get("ReasonCode") or "").strip().upper()
        reason = REASON_CODES.get(reason_code, reason_code or "halted")
        issue_name = (raw_entry.get("IssueName") or "").strip()
        halt_date = raw_entry.get("HaltDate") or ""
        halt_time = raw_entry.get("HaltTime") or ""
        resume_date = raw_entry.get("ResumptionDate") or ""
        resume_trade = raw_entry.get("ResumptionTradeTime") or ""
        pause_px = raw_entry.get("PauseThresholdPrice") or ""

        # external_id: one row per (symbol, halt timestamp) so re-polls dedup but a
        # genuinely new halt on the same name later still inserts.
        external_id = f"{self.source}|{symbol}|{halt_date}|{halt_time}|{reason_code}"

        title = f"TRADING HALT: {symbol} — {reason}"
        if issue_name:
            title += f" ({issue_name})"

        body_parts = [f"{symbol} halted: {reason} [{reason_code}]."]
        if halt_date or halt_time:
            body_parts.append(f"Halted {halt_date} {halt_time} ET.")
        if resume_date or resume_trade:
            body_parts.append(f"Resumes {resume_date} {resume_trade} ET.")
        if pause_px:
            body_parts.append(f"Pause threshold price ${pause_px}.")
        body = " ".join(body_parts)

        return ParsedSignal(
            external_id=external_id,
            title=title,
            body=body,
            url=FEED_URL,
            author="nasdaqtrader",
            author_metadata={"reason_code": reason_code, "issue_name": issue_name},
            # HaltDate is mm/dd/yyyy + HaltTime hh:mm:ss ET — leave published_at None
            # rather than emit a malformed ISO; ingested_at captures arrival time.
            published_at=None,
            raw_payload=dict(raw_entry),
            tickers=[TickerMention(ticker=symbol, market="US", confidence=1.0)],
        )
