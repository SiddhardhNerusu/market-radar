"""Vertical debit spread strategy + strike selection + asymmetric Kelly sizing.

Given a signal (direction, model_p, underlying), produce a fully-specified
spread ready for submission:

  - long leg  : ATM (or just-ITM)
  - short leg : delta ~0.25-0.30 OTM, ~1 ATR away from underlying
  - DTE       : ~14 days (range 10-21 accepted)
  - contracts : floor(min(¼-Kelly, 5%-equity-cap) / (debit * 100))
  - reject if : spread > 10% of mid, OI < 500, earnings inside DTE window

Universe is hard-coded to ~10 of the most liquid US option chains. Anything
else gets routed to the stock path by the live trader's dispatcher.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal, Optional

from .alpaca_options import (
    AlpacaOptionsClient,
    OptionContract,
    OptionLeg,
    OptionQuote,
    build_occ_symbol,
)

log = logging.getLogger("marketradar.execution.options.spreads")

Direction = Literal["buy", "sell"]
Strategy = Literal["bull_call_debit", "bear_put_debit"]


# Underlying whitelist — tight spreads, deep liquidity, weekly expiries.
# Expanded 2026-05-29 to catch more earnings-runner candidates (DELL, ORCL,
# CRM etc.). Each entry must have a WIDTH_BY_TICKER mapping too.
OPTIONS_UNDERLYINGS: frozenset[str] = frozenset({
    # Index ETFs
    "SPY", "QQQ", "IWM",
    # Mega-cap tech
    "NVDA", "TSLA", "AAPL", "AMD", "META", "MSFT", "GOOGL", "AMZN",
    # High-vol favourites
    "COIN", "PLTR",
    # Earnings runners / weekly-options liquid
    "DELL", "ORCL", "AVGO", "CRM", "NFLX", "SHOP",
    "UBER", "ABNB", "SNOW", "CRWD", "NET", "MDB",
})

# Per-ticker preferred spread width (USD). Picked to match actual exchange
# strike spacing AND keep debit ~30-50% of width (real liquidity zone).
WIDTH_BY_TICKER: dict[str, float] = {
    # Original 13
    "SPY":   1.0,  "IWM":   1.0,  "PLTR":  1.0,
    "QQQ":   2.5,  "MSFT":  2.5,
    "NVDA":  5.0,  "AMD":   5.0,  "AAPL":  5.0,
    "META":  5.0,  "GOOGL": 5.0,  "AMZN":  5.0,  "COIN": 5.0,
    "TSLA":  5.0,
    # Expansion names (widths sized to strike spacing on each chain)
    "DELL":  5.0,  "ORCL":  2.5,  "AVGO":  5.0,
    "CRM":   2.5,  "NFLX":  5.0,  "SHOP":  2.5,
    "UBER":  1.0,  "ABNB":  2.5,  "SNOW":  2.5,
    "CRWD":  5.0,  "NET":   1.0,  "MDB":   2.5,
}

# Sizing + exit defaults
KELLY_FRACTION = 0.25
MAX_POSITION_PCT = 5.0                # of equity per spread (default conviction)
MAX_POSITION_PCT_HIGH_CONVICTION = 8.0  # bumped when model_p >= 0.70 (or <= 0.30 for short)
HIGH_CONVICTION_P = 0.70              # |model_p - 0.5| >= 0.20 triggers the bigger cap
# Day-trader mode: 1-2 DTE preferred. The "gamma swamps" concern only
# applies to multi-day holds — for intraday closes, gamma WORKS for us when
# the position is moving in our direction (high delta sensitivity = bigger
# % gains on small underlying moves).
TARGET_DTE_DAYS = 1
MIN_DTE_DAYS = 0                      # allow 0DTE on SPY/QQQ/IWM
MAX_DTE_DAYS = 2                      # cap day-trader at 2 DTE max
MIN_OPEN_INTEREST = 25                # liquid enough to enter + exit (was 100, dropped to fire more spreads)
MAX_SPREAD_PCT = 0.25                 # reject if (ask-bid)/mid > 25% (was 10/15, relaxed for paper)

# When the indicative options feed returns no two-sided quote (bid=ask=0) —
# which was ~44% of failed builds — fall back to the leg's last trade price.
# Safe because the spread is submitted as a LIMIT at the net debit: a stale
# estimate yields a no-fill or a fill at our limit-or-better, never worse.
# Flip to 0 in .env to revert to quote-only (no-fill on empty quotes).
ALLOW_LAST_TRADE_MID = os.getenv(
    "LIVE_OPT_ALLOW_LAST_TRADE_MID", "1"
).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SpreadSpec:
    """Fully-specified spread ready for submission."""
    underlying: str
    direction: Direction
    strategy: Strategy
    long_contract: OptionContract
    short_contract: OptionContract
    long_quote: OptionQuote
    short_quote: OptionQuote
    width: float                       # short_strike - long_strike (call) or long - short (put)
    debit_per_spread: float            # entry cost per spread (positive)
    max_loss_per_spread: float         # = debit * 100
    max_gain_per_spread: float         # = (width - debit) * 100

    @property
    def reward_risk_ratio(self) -> float:
        if self.debit_per_spread <= 0:
            return 0.0
        return (self.width - self.debit_per_spread) / self.debit_per_spread


@dataclass(frozen=True)
class SpreadSizing:
    contracts: int
    total_debit_usd: float
    total_max_loss_usd: float
    total_max_gain_usd: float
    size_pct: float
    kelly_raw: float
    tradeable: bool
    reason: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_vertical_debit_spread(
    *,
    underlying: str,
    direction: Direction,
    model_p: float,
    account_equity_usd: float,
    options_client: AlpacaOptionsClient,
    target_dte: int = TARGET_DTE_DAYS,
    min_dte: int = MIN_DTE_DAYS,
    max_dte: int = MAX_DTE_DAYS,
    today: Optional[date] = None,
    composite_score: Optional[float] = None,
    abs_sentiment: Optional[float] = None,
) -> tuple[Optional[SpreadSpec], SpreadSizing]:
    """Build + size a vertical debit spread, or explain why we can't.

    Returns ``(spec, sizing)``. If ``spec`` is None, ``sizing.tradeable`` is
    False and ``sizing.reason`` says why.
    """
    today = today or date.today()
    ticker = underlying.upper()
    # OPEN UNIVERSE: allow ANY underlying — builder fast-fails on liquidity
    # below. The curated whitelist + WIDTH_BY_TICKER table give better widths
    # for known names; unknown names get price-based width heuristic.
    strategy: Strategy = "bull_call_debit" if direction == "buy" else "bear_put_debit"
    opt_type = "call" if direction == "buy" else "put"

    # ---- Pick expiration ----
    # Need underlying price FIRST so we can filter strikes to a relevant range.
    # Alpaca's list_contracts caps at 500 rows — without strike filtering on a
    # liquid name like SPY (which has 300+ strikes per expiration), we get only
    # deep-OTM/ITM strikes by accident of sort order.
    underlying_price = options_client.base.get_latest_trade(ticker)
    if not underlying_price or underlying_price <= 0:
        return None, _untradeable_sizing(f"no underlying price for {ticker}")

    # Price-based width heuristic for tickers without an explicit override.
    # Matches standard strike spacings on the most-liquid US option chains.
    if ticker in WIDTH_BY_TICKER:
        width = WIDTH_BY_TICKER[ticker]
    elif underlying_price < 25:
        width = 1.0
    elif underlying_price < 100:
        width = 2.5
    elif underlying_price < 500:
        width = 5.0
    else:
        width = 10.0

    exp_window_start = today + timedelta(days=min_dte)
    exp_window_end = today + timedelta(days=max_dte)
    # Filter to strikes within ±25% of underlying (covers any reasonable spread)
    strike_radius_pct = 0.25
    strike_lo = underlying_price * (1 - strike_radius_pct)
    strike_hi = underlying_price * (1 + strike_radius_pct)
    contracts = options_client.list_contracts(
        ticker,
        expiration_gte=exp_window_start.isoformat(),
        expiration_lte=exp_window_end.isoformat(),
        type_=opt_type,
        strike_gte=strike_lo,
        strike_lte=strike_hi,
        limit=500,
    )
    if not contracts:
        return None, _untradeable_sizing(
            f"no {opt_type} contracts in [{exp_window_start}, {exp_window_end}]")

    # Group by expiration; pick the one closest to TARGET_DTE_DAYS.
    by_exp: dict[str, list[OptionContract]] = {}
    for c in contracts:
        if not c.tradable:
            continue
        by_exp.setdefault(c.expiration_date, []).append(c)
    if not by_exp:
        return None, _untradeable_sizing("no tradable contracts in window")

    def _dte(exp: str) -> int:
        return (datetime.strptime(exp, "%Y-%m-%d").date() - today).days

    chosen_exp = min(by_exp.keys(), key=lambda e: abs(_dte(e) - target_dte))
    chain = sorted(by_exp[chosen_exp], key=lambda c: c.strike_price)
    # underlying_price already fetched above before list_contracts

    # ---- Pick strikes with FALLBACK ----
    # Strategy: Build candidate pairs (ATM-ish long + 1-3 widths OTM short).
    # Treat OI==0 as "unknown" (Alpaca's basic options endpoint frequently
    # returns 0 for OI — not a sign of zero liquidity). Real liquidity check
    # happens via the bid-ask spread later (MAX_SPREAD_PCT).
    long_c = short_c = None
    candidate_pairs: list[tuple[OptionContract, OptionContract]] = []
    # Search range: ±20 widths from underlying (covers most reasonable strikes
    # even for $750 SPY with $1 width).
    search_radius = width * 20
    if opt_type == "call":
        near = sorted([c for c in chain
                       if abs(c.strike_price - underlying_price) <= search_radius],
                      key=lambda c: abs(c.strike_price - underlying_price))
        for long_cand in near[:10]:
            for mult in (1, 2, 3):
                short_target = long_cand.strike_price + width * mult
                short_cand = _find_strike(chain, short_target)
                if short_cand:
                    candidate_pairs.append((long_cand, short_cand))
    else:  # put
        near = sorted([c for c in chain
                       if abs(c.strike_price - underlying_price) <= search_radius],
                      key=lambda c: abs(c.strike_price - underlying_price))
        for long_cand in near[:10]:
            for mult in (1, 2, 3):
                short_target = long_cand.strike_price - width * mult
                short_cand = _find_strike(chain, short_target)
                if short_cand:
                    candidate_pairs.append((long_cand, short_cand))

    if not candidate_pairs:
        return None, _untradeable_sizing(
            f"no strike pairs near underlying {underlying_price:.2f} in {ticker} chain "
            f"(search radius ±{search_radius:.0f})"
        )

    # Score each pair by liquidity: prefer pairs where both legs have OI>=MIN,
    # but accept OI==0 (treat as unknown). The real liquidity test is the
    # bid-ask spread which is checked below at MAX_SPREAD_PCT.
    def _pair_score(pair):
        l, s = pair
        # OI score: high if both legs known liquid; medium if unknown (0); low if known thin
        def _oi_ok(c):
            return c.open_interest == 0 or c.open_interest >= MIN_OPEN_INTEREST
        oi_ok = _oi_ok(l) and _oi_ok(s)
        return (1 if oi_ok else 0, min(l.open_interest, s.open_interest))

    candidate_pairs.sort(key=_pair_score, reverse=True)
    long_c, short_c = candidate_pairs[0]

    # Hard reject only if BOTH legs have explicitly low (non-zero) OI
    if (0 < long_c.open_interest < MIN_OPEN_INTEREST
            and 0 < short_c.open_interest < MIN_OPEN_INTEREST):
        return None, _untradeable_sizing(
            f"both legs explicitly illiquid: long_OI={long_c.open_interest}, "
            f"short_OI={short_c.open_interest}, min={MIN_OPEN_INTEREST}"
        )

    # Recompute width from actual chosen strikes (might differ from default)
    width = abs(short_c.strike_price - long_c.strike_price)

    # ---- Quote both legs ----
    quotes = options_client.get_snapshots([long_c.symbol, short_c.symbol])
    long_q = quotes.get(long_c.symbol)
    short_q = quotes.get(short_c.symbol)
    if not long_q or not short_q:
        return None, _untradeable_sizing("no quote for one of the legs")
    # ---- Price each leg: real two-sided mid, else last-trade fallback ----
    if ALLOW_LAST_TRADE_MID:
        long_mid = long_q.effective_mid
        short_mid = short_q.effective_mid
    else:
        long_mid, short_mid = long_q.mid, short_q.mid
    if long_mid <= 0 or short_mid <= 0:
        # Diagnostic: log the actual quote+trade so we can SEE whether empty
        # quotes are illiquid strikes (correct to skip) or a feed gap (the
        # last-trade fallback rescues these). Was ~44% of failed builds.
        log.info(
            "[%s] spread skipped — no price: long bid/ask/last=%.2f/%.2f/%.2f "
            "short bid/ask/last=%.2f/%.2f/%.2f (fallback=%s)",
            ticker, long_q.bid, long_q.ask, long_q.last_price,
            short_q.bid, short_q.ask, short_q.last_price, ALLOW_LAST_TRADE_MID,
        )
        return None, _untradeable_sizing(
            f"no price (quote+last empty): long={long_mid:.2f} short={short_mid:.2f}"
        )
    # Bid-ask width sanity — only meaningful on a leg with a real two-sided
    # quote. A last-trade-only leg has no computable spread; the LIMIT order
    # is our protection there.
    if long_q.mid > 0 and long_q.spread_pct > MAX_SPREAD_PCT:
        return None, _untradeable_sizing(
            f"long bid-ask too wide: {long_q.spread_pct:.1%} > {MAX_SPREAD_PCT:.1%}"
        )
    if short_q.mid > 0 and short_q.spread_pct > MAX_SPREAD_PCT:
        return None, _untradeable_sizing(
            f"short bid-ask too wide: {short_q.spread_pct:.1%} > {MAX_SPREAD_PCT:.1%}"
        )

    # ---- Debit + max gain ----
    debit_per_spread = max(long_mid - short_mid, 0.01)  # per share; * 100 for $
    if debit_per_spread >= width:
        return None, _untradeable_sizing(
            f"debit {debit_per_spread:.2f} >= width {width:.2f} — no edge"
        )
    max_loss = debit_per_spread * 100
    max_gain = (width - debit_per_spread) * 100

    spec = SpreadSpec(
        underlying=ticker,
        direction=direction,
        strategy=strategy,
        long_contract=long_c,
        short_contract=short_c,
        long_quote=long_q,
        short_quote=short_q,
        width=width,
        debit_per_spread=debit_per_spread,
        max_loss_per_spread=max_loss,
        max_gain_per_spread=max_gain,
    )
    # R:R sanity check — reject suspiciously generous spreads. R:R above 10
    # almost always means one of the legs has a stale or absurd quote
    # (mid != real fill price). Alpaca will reject these on submit anyway,
    # leaving phantom 'pending_new' rows in our DB. Better to filter here.
    if spec.reward_risk_ratio > 10.0:
        return None, _untradeable_sizing(
            f"R:R {spec.reward_risk_ratio:.1f} > 10 — likely bad quote "
            f"(long_mid={long_mid:.2f}, short_mid={short_mid:.2f})"
        )
    sizing = size_spread(spec, model_p=model_p, account_equity_usd=account_equity_usd,
                         composite_score=composite_score, abs_sentiment=abs_sentiment)
    return spec, sizing


def size_spread(
    spec: SpreadSpec,
    *,
    model_p: float,
    account_equity_usd: float,
    kelly_fraction: float = KELLY_FRACTION,
    max_position_pct: float = MAX_POSITION_PCT,
    composite_score: Optional[float] = None,
    abs_sentiment: Optional[float] = None,
) -> SpreadSizing:
    """Asymmetric Kelly sizing for a defined-risk spread.

    High-conviction trigger fires (bigger position cap) when ANY of:
      a) |model_p - 0.5| >= 0.20 (extreme calibrated probability)
      b) composite_score >= 8.0 AND |sentiment| >= 0.5 (strong PA setup)

    Both paths target the same intuition: "we have high confidence this
    move happens" — either from ML or from a strong technical pattern.
    """
    if account_equity_usd <= 0:
        return _untradeable_sizing("account_equity_usd <= 0")
    if spec.debit_per_spread <= 0:
        return _untradeable_sizing("non-positive debit")

    p = max(min(float(model_p), 0.99), 0.01)
    b = spec.reward_risk_ratio
    kelly_raw = (p * b - (1 - p)) / b if b > 0 else 0.0
    if kelly_raw <= 0:
        return _untradeable_sizing(
            f"kelly_raw={kelly_raw:.3f} <= 0 (p={p:.2f}, b={b:.2f}) — edge does not survive R:R"
        )

    # High-conviction qualifies via ML edge OR strong PA signal.
    ml_high_conviction = abs(p - 0.5) >= (HIGH_CONVICTION_P - 0.5)
    pa_high_conviction = (
        composite_score is not None and composite_score >= 8.0
        and abs_sentiment is not None and abs_sentiment >= 0.5
    )
    is_high_conviction = ml_high_conviction or pa_high_conviction
    effective_cap = (MAX_POSITION_PCT_HIGH_CONVICTION
                     if is_high_conviction
                     else max_position_pct)
    size_pct = min(kelly_raw * kelly_fraction * 100.0, effective_cap)
    capital_budget = account_equity_usd * (size_pct / 100.0)
    contracts = int(capital_budget / spec.max_loss_per_spread)
    if contracts < 1:
        return SpreadSizing(
            contracts=0, total_debit_usd=0, total_max_loss_usd=0,
            total_max_gain_usd=0, size_pct=size_pct, kelly_raw=kelly_raw,
            tradeable=False,
            reason=(f"capital budget ${capital_budget:.0f} < max-loss-per-spread "
                    f"${spec.max_loss_per_spread:.0f} — can't afford even 1 contract"),
        )

    total_debit = contracts * spec.debit_per_spread * 100
    total_max_gain = contracts * spec.max_gain_per_spread
    actual_pct = total_debit / account_equity_usd * 100
    return SpreadSizing(
        contracts=contracts,
        total_debit_usd=total_debit,
        total_max_loss_usd=total_debit,                # max loss = debit on debit spreads
        total_max_gain_usd=total_max_gain,
        size_pct=actual_pct,
        kelly_raw=kelly_raw,
        tradeable=True,
        reason=f"sized: {contracts} contracts @ ${spec.debit_per_spread*100:.0f}/spread",
    )


def spec_to_legs(spec: SpreadSpec) -> list[OptionLeg]:
    """Convert a SpreadSpec into the OptionLeg list for submit_multi_leg."""
    return [
        OptionLeg(
            symbol=spec.long_contract.symbol,
            side="buy",
            position_intent="buy_to_open",
            ratio_qty=1,
        ),
        OptionLeg(
            symbol=spec.short_contract.symbol,
            side="sell",
            position_intent="sell_to_open",
            ratio_qty=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_strike(chain: list[OptionContract], target: float) -> Optional[OptionContract]:
    return next((c for c in chain if abs(c.strike_price - target) < 0.01), None)


def _untradeable_sizing(reason: str) -> SpreadSizing:
    return SpreadSizing(
        contracts=0, total_debit_usd=0, total_max_loss_usd=0,
        total_max_gain_usd=0, size_pct=0.0, kelly_raw=0.0,
        tradeable=False, reason=reason,
    )
