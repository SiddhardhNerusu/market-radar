"""Smoke tests for src/market_radar/risk/manager.py — exercises every
one of the seven rules with synthetic TradeProposals.

Run inline:
    python scripts/test_risk_manager.py

No DB writes.  Uses monkey-patched RiskManager helpers so the tests
don't depend on transient DB state.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.config import CONFIG
from market_radar.risk import RiskManager, TradeProposal, RiskDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def _mk_rm(*,
           today_pnl=0.0,
           today_trades=0,
           drift_hours=None,
           equity=10_000.0,
           gross=0.0,
           sector_pct=0.0) -> RiskManager:
    """Return a RiskManager with deterministic, monkey-patched helpers."""
    rm = RiskManager()
    rm._today_realized_pnl_usd = lambda: today_pnl  # type: ignore[assignment]
    rm._today_trade_count = lambda: today_trades  # type: ignore[assignment]
    rm._hours_since_last_drift_alert = lambda: drift_hours  # type: ignore[assignment]
    rm._latest_account_equity = lambda: equity  # type: ignore[assignment]
    rm._current_gross_exposure_usd = lambda: gross  # type: ignore[assignment]
    rm._current_sector_exposure_pct = (  # type: ignore[assignment]
        lambda sector, equity_usd: sector_pct
    )
    return rm


def _expect(case_name: str,
            decision: RiskDecision,
            expect_allowed: bool,
            expect_rule: str | None = None) -> bool:
    """Print pass/fail for one case.  Returns True if it matched."""
    ok = decision.allowed is expect_allowed
    if expect_rule is not None and decision.blocking_rule != expect_rule:
        ok = False
    color = GREEN if ok else RED
    status = "PASS" if ok else "FAIL"
    print(f"  {color}[{status}]{RESET} {case_name}")
    print(f"      → allowed={decision.allowed} rule={decision.blocking_rule}")
    print(f"      → reason: {decision.reason}")
    return ok


# A neutral, baseline-safe proposal that *should* pass when all gates are open.
def _safe_proposal() -> TradeProposal:
    return TradeProposal(
        ticker="AAPL", direction="buy", size_pct=2.0,
        calibrated_p=0.72, sector="tech",
    )


# ---------------------------------------------------------------------------
# Per-rule cases
# ---------------------------------------------------------------------------

def case_emergency_stop() -> list[bool]:
    print("\n[rule 1] emergency_stop")
    results: list[bool] = []
    saved = CONFIG.risk_emergency_stop
    try:
        # Monkey-patch via object.__setattr__ since CONFIG is a frozen dataclass.
        object.__setattr__(CONFIG, "risk_emergency_stop", True)
        d = _mk_rm().evaluate(_safe_proposal())
        results.append(_expect("with emergency_stop=1 → blocked",
                               d, False, "emergency_stop"))
        object.__setattr__(CONFIG, "risk_emergency_stop", False)
        d = _mk_rm().evaluate(_safe_proposal())
        results.append(_expect("with emergency_stop=0 → allowed",
                               d, True, None))
    finally:
        object.__setattr__(CONFIG, "risk_emergency_stop", saved)
    return results


def case_daily_loss_cap() -> list[bool]:
    print("\n[rule 2] daily_loss_cap")
    cap = CONFIG.risk_daily_loss_cap_usd
    results = []
    # Hit the cap exactly → block.
    d = _mk_rm(today_pnl=-cap).evaluate(_safe_proposal())
    results.append(_expect(f"pnl=-${cap:.0f} (== cap) → blocked",
                           d, False, "daily_loss_cap"))
    # Far below → block.
    d = _mk_rm(today_pnl=-(cap + 50)).evaluate(_safe_proposal())
    results.append(_expect(f"pnl=-${cap+50:.0f} → blocked",
                           d, False, "daily_loss_cap"))
    # Just above (less loss) → allowed.
    d = _mk_rm(today_pnl=-(cap - 1)).evaluate(_safe_proposal())
    results.append(_expect(f"pnl=-${cap-1:.0f} → allowed",
                           d, True, None))
    # Helper returning None → fail closed.
    rm = _mk_rm()
    rm._today_realized_pnl_usd = lambda: None  # type: ignore[assignment]
    d = rm.evaluate(_safe_proposal())
    results.append(_expect("pnl helper returns None → blocked (fail-closed)",
                           d, False, "daily_loss_cap"))
    return results


def case_drift_block() -> list[bool]:
    print("\n[rule 3] drift_block")
    window = CONFIG.risk_drift_block_hours
    results = []
    # No drift ever observed → allowed.
    d = _mk_rm(drift_hours=None).evaluate(_safe_proposal())
    results.append(_expect("drift_hours=None → allowed",
                           d, True, None))
    # Inside the window → blocked.
    d = _mk_rm(drift_hours=window - 0.5).evaluate(_safe_proposal())
    results.append(_expect(
        f"drift {window - 0.5:.1f}h ago (< {window}h) → blocked",
        d, False, "drift_block",
    ))
    # Outside the window → allowed.
    d = _mk_rm(drift_hours=window + 1).evaluate(_safe_proposal())
    results.append(_expect(
        f"drift {window + 1}h ago → allowed",
        d, True, None,
    ))
    return results


def case_min_p() -> list[bool]:
    print("\n[rule 4] min_calibrated_p")
    min_p = CONFIG.risk_min_calibrated_p
    results = []
    # Buy below threshold → block.
    d = _mk_rm().evaluate(TradeProposal(
        ticker="AAPL", direction="buy",
        size_pct=2.0, calibrated_p=min_p - 0.05, sector="tech",
    ))
    results.append(_expect(f"buy p={min_p-0.05:.2f} < {min_p:.2f} → blocked",
                           d, False, "min_p"))
    # Buy at threshold → allowed.
    d = _mk_rm().evaluate(TradeProposal(
        ticker="AAPL", direction="buy",
        size_pct=2.0, calibrated_p=min_p, sector="tech",
    ))
    results.append(_expect(f"buy p={min_p:.2f} (== min) → allowed",
                           d, True, None))
    # Sell above (1 - min_p) → block.
    sell_max = 1 - min_p
    d = _mk_rm().evaluate(TradeProposal(
        ticker="AAPL", direction="sell",
        size_pct=2.0, calibrated_p=sell_max + 0.05, sector="tech",
    ))
    results.append(_expect(
        f"sell p={sell_max+0.05:.2f} > {sell_max:.2f} → blocked",
        d, False, "min_p",
    ))
    # Sell below (1 - min_p) → allowed.
    d = _mk_rm().evaluate(TradeProposal(
        ticker="AAPL", direction="sell",
        size_pct=2.0, calibrated_p=sell_max - 0.05, sector="tech",
    ))
    results.append(_expect(f"sell p={sell_max-0.05:.2f} → allowed",
                           d, True, None))
    return results


def case_max_daily_trades() -> list[bool]:
    print("\n[rule 5] max_daily_trades")
    cap = CONFIG.risk_max_daily_trades
    results = []
    d = _mk_rm(today_trades=cap).evaluate(_safe_proposal())
    results.append(_expect(f"today_trades={cap} (== cap) → blocked",
                           d, False, "max_daily_trades"))
    d = _mk_rm(today_trades=cap + 5).evaluate(_safe_proposal())
    results.append(_expect(f"today_trades={cap+5} → blocked",
                           d, False, "max_daily_trades"))
    d = _mk_rm(today_trades=cap - 1).evaluate(_safe_proposal())
    results.append(_expect(f"today_trades={cap-1} → allowed",
                           d, True, None))
    return results


def case_max_gross_exposure() -> list[bool]:
    print("\n[rule 6] max_gross_exposure")
    gcap = CONFIG.risk_max_gross_exposure_usd
    results = []
    # Equity 10k, size 2% → +$200.  Set existing gross just below cap.
    equity = 10_000.0
    new_pos = equity * 0.02
    # Existing gross sits at (cap - new_pos + 50) so after_new > cap.
    d = _mk_rm(equity=equity, gross=gcap - new_pos + 50).evaluate(_safe_proposal())
    results.append(_expect("gross_after > cap → blocked",
                           d, False, "max_gross_exposure"))
    # Existing gross 0 → plenty of room.
    d = _mk_rm(equity=equity, gross=0.0).evaluate(_safe_proposal())
    results.append(_expect("gross_after well under cap → allowed",
                           d, True, None))
    # Equity helper returns None → fail closed.
    rm = _mk_rm()
    rm._latest_account_equity = lambda: None  # type: ignore[assignment]
    d = rm.evaluate(_safe_proposal())
    results.append(_expect("equity=None → blocked (fail-closed)",
                           d, False, "max_gross_exposure"))
    # Gross helper returns None → fail closed.
    rm = _mk_rm()
    rm._current_gross_exposure_usd = lambda: None  # type: ignore[assignment]
    d = rm.evaluate(_safe_proposal())
    results.append(_expect("gross=None → blocked (fail-closed)",
                           d, False, "max_gross_exposure"))
    return results


def case_max_position_pct() -> list[bool]:
    print("\n[rule 7a] max_position_pct")
    cap = CONFIG.risk_max_position_pct
    results = []
    d = _mk_rm().evaluate(TradeProposal(
        ticker="AAPL", direction="buy",
        size_pct=cap + 1, calibrated_p=0.72, sector="tech",
    ))
    results.append(_expect(f"size={cap+1}% > {cap}% cap → blocked",
                           d, False, "max_position_pct"))
    d = _mk_rm().evaluate(TradeProposal(
        ticker="AAPL", direction="buy",
        size_pct=cap - 0.5, calibrated_p=0.72, sector="tech",
    ))
    results.append(_expect(f"size={cap-0.5}% → allowed",
                           d, True, None))
    return results


def case_max_sector_pct() -> list[bool]:
    print("\n[rule 7b] max_sector_pct")
    cap = CONFIG.risk_max_sector_pct
    results = []
    # Existing sector exposure already at cap → adding any size triggers
    # the block.
    d = _mk_rm(sector_pct=cap).evaluate(_safe_proposal())
    results.append(_expect(f"sector already at {cap}% → blocked",
                           d, False, "max_sector_pct"))
    # Room available → allowed.
    d = _mk_rm(sector_pct=cap - 10).evaluate(_safe_proposal())
    results.append(_expect(f"sector at {cap-10}% → allowed",
                           d, True, None))
    return results


def case_input_validation() -> list[bool]:
    """Bonus: malformed proposals must fail closed."""
    print("\n[bonus] input validation (fail-closed on malformed proposal)")
    results = []
    # Empty ticker
    d = _mk_rm().evaluate(TradeProposal(
        ticker="", direction="buy", size_pct=2.0,
        calibrated_p=0.72, sector="tech",
    ))
    results.append(_expect("empty ticker → blocked", d, False, "input_validation"))
    # Bad direction
    d = _mk_rm().evaluate(TradeProposal(
        ticker="AAPL", direction="hodl", size_pct=2.0,
        calibrated_p=0.72, sector="tech",
    ))
    results.append(_expect("direction='hodl' → blocked",
                           d, False, "input_validation"))
    # p out of range
    d = _mk_rm().evaluate(TradeProposal(
        ticker="AAPL", direction="buy", size_pct=2.0,
        calibrated_p=1.5, sector="tech",
    ))
    results.append(_expect("p=1.5 → blocked", d, False, "input_validation"))
    # size_pct <= 0
    d = _mk_rm().evaluate(TradeProposal(
        ticker="AAPL", direction="buy", size_pct=0,
        calibrated_p=0.72, sector="tech",
    ))
    results.append(_expect("size=0 → blocked", d, False, "input_validation"))
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"{DIM}Risk manager smoke tests · "
          f"thresholds: daily_loss=${CONFIG.risk_daily_loss_cap_usd:.0f} "
          f"gross=${CONFIG.risk_max_gross_exposure_usd:.0f} "
          f"pos={CONFIG.risk_max_position_pct}% "
          f"sector={CONFIG.risk_max_sector_pct}% "
          f"trades/day={CONFIG.risk_max_daily_trades} "
          f"drift_block={CONFIG.risk_drift_block_hours}h "
          f"min_p={CONFIG.risk_min_calibrated_p}{RESET}")

    all_results: list[bool] = []
    for fn in (
        case_emergency_stop,
        case_daily_loss_cap,
        case_drift_block,
        case_min_p,
        case_max_daily_trades,
        case_max_gross_exposure,
        case_max_position_pct,
        case_max_sector_pct,
        case_input_validation,
    ):
        all_results.extend(fn())

    passed = sum(1 for r in all_results if r)
    total = len(all_results)
    print()
    print(f"=== {passed}/{total} cases passed ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
