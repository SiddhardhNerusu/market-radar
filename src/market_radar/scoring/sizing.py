"""Fractional Kelly position sizing.

Inputs:
  - ``p_calibrated`` — calibrated probability of a *winning* outcome.
    Must come from the calibrated model (see ``ml/train.py`` isotonic
    wrap); raw HGB scores are not probabilities and will over-size.
  - ``expected_win_pct`` — projected up-move conditional on winning
    (positive number, e.g. 0.045 for +4.5%).
  - ``expected_loss_pct`` — projected down-move conditional on losing
    (positive number, e.g. 0.030 for a -3.0% loss).

These should come from the comparable-setup projections in
``scoring/projections.py`` (median win / median loss across the matched
historical cohort), not from defaults — Kelly's edge comes from
*signal-specific* odds.

We multiply the full Kelly fraction by ``kelly_fraction`` (default 0.5 —
half-Kelly). Empirically half-Kelly captures ~75% of full-Kelly's growth
rate at half the volatility, and is robust to mis-estimated edge. Cap at
``max_size_pct`` of account equity (default 5%).

The module returns ``0.0`` (i.e. "don't trade") when the math says the
edge is non-positive — strictly safer than always sizing something.

This module is **paper-only** in its current state: the dashboard and
notification layer should display the suggested size, but the trade
executor must NOT auto-place orders sized by this until ≥2 weeks of
resolved live outcomes confirm the calibration holds at the size you'd
trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Defaults conservative enough for early live deployment
DEFAULT_KELLY_FRACTION = 0.5      # half-Kelly
DEFAULT_MAX_SIZE_PCT = 5.0        # never above 5% of account on one idea
DEFAULT_MIN_EDGE = 0.0            # full Kelly must be > 0 to size
DEFAULT_MIN_CALIBRATED_P = 0.50   # never size on p <= 50%


@dataclass(frozen=True)
class SizingResult:
    suggested_size_pct: float           # of account equity, 0..max_size_pct
    full_kelly: float                   # what unlimited Kelly would say
    half_kelly_or_frac: float           # full_kelly × kelly_fraction
    expected_growth_per_trade: float    # E[log(1+f*r)] proxy
    expected_value_pct: float           # p*win + (1-p)*(-loss)
    reason: str                         # why the size landed where it did


def suggested_size_pct(
    *,
    p_calibrated: float,
    expected_win_pct: float,
    expected_loss_pct: float,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_size_pct: float = DEFAULT_MAX_SIZE_PCT,
    min_calibrated_p: float = DEFAULT_MIN_CALIBRATED_P,
) -> SizingResult:
    """Compute a half-Kelly position size as % of account.

    Returns 0 (with ``reason``) if the edge is non-positive, the
    calibrated probability is too low, or the expected loss is
    non-positive (i.e. there's no downside scenario to size against).
    """
    p = float(p_calibrated)
    win = float(expected_win_pct)
    loss = float(expected_loss_pct)

    if p <= 0.0 or p >= 1.0:
        return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0,
                            "calibrated probability out of (0, 1)")
    if p < min_calibrated_p:
        return SizingResult(0.0, 0.0, 0.0, 0.0,
                            p * win - (1 - p) * loss,
                            f"p={p:.3f} < min {min_calibrated_p:.2f}")
    if win <= 0:
        return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0,
                            "expected win is non-positive")
    if loss <= 0:
        return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0,
                            "expected loss is non-positive (no downside model)")

    # Kelly with asymmetric odds:
    #   f* = (p*b - (1-p)) / b   where b = win / loss is the payoff ratio
    b = win / loss
    full_kelly = (p * b - (1 - p)) / b
    fractional = full_kelly * float(kelly_fraction)

    ev_pct = p * win - (1 - p) * loss

    if full_kelly <= DEFAULT_MIN_EDGE:
        return SizingResult(0.0, full_kelly, fractional, 0.0, ev_pct,
                            f"full Kelly = {full_kelly:.4f} (no edge)")
    if fractional <= 0:
        return SizingResult(0.0, full_kelly, fractional, 0.0, ev_pct,
                            "fractional Kelly <= 0 after applying multiplier")

    # Cap as a % of account equity
    sized_pct = min(fractional, max_size_pct / 100.0) * 100.0

    # Log-growth proxy: g ≈ p*ln(1 + f*win) + (1-p)*ln(1 - f*loss)
    # Useful for displaying "expected compound growth contribution".
    import math
    try:
        g = p * math.log(1 + fractional * win) + \
            (1 - p) * math.log(1 - fractional * loss)
    except ValueError:
        g = 0.0

    return SizingResult(
        suggested_size_pct=round(sized_pct, 3),
        full_kelly=round(full_kelly, 4),
        half_kelly_or_frac=round(fractional, 4),
        expected_growth_per_trade=round(g, 5),
        expected_value_pct=round(ev_pct, 4),
        reason=("capped at max_size_pct" if sized_pct >= max_size_pct - 1e-9
                else "sized at fractional Kelly"),
    )


# --- module-level convenience ---------------------------------------------

def _get(o, key, default=None):
    """Read ``key`` from either a dict or an object (attribute)."""
    if o is None:
        return default
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def size_from_projection(
    *,
    p_calibrated: float,
    projection,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_size_pct: float = DEFAULT_MAX_SIZE_PCT,
) -> SizingResult:
    """Convenience wrapper that pulls win/loss expectations from a
    ``scoring/projections.py`` distribution.

    Accepts either:
      - a ``ReturnDistribution`` dataclass (with ``.p25`` / ``.p75``
        attributes)
      - the dict form returned by ``projected_price_range`` (with
        ``"median_return_pct"`` etc.)
      - any dict carrying ``p25`` / ``p75`` (return %) OR
        ``median_win`` / ``median_loss`` keys.

    Returns a zero-size result with a reason if it can't find suitable
    win/loss numbers.
    """
    if projection is None:
        return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0,
                            "no projection available for this signal class")

    # Try several shapes in priority order
    win = (_get(projection, "median_win")
           or _get(projection, "p75")
           or _get(projection, "p75_return"))
    loss = (_get(projection, "median_loss")
            or _get(projection, "p25")
            or _get(projection, "p25_return"))
    if win is None or loss is None:
        return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0,
                            "projection missing win/loss (p25/p75)")

    # projections.py percentiles are signed % returns (e.g. p25=-3.2, p75=+4.5).
    # Convert to *magnitudes* for the Kelly math; also convert from % to
    # fraction (4.5 → 0.045) so the win/loss values match the canonical
    # suggested_size_pct interface.
    win_frac = float(win) / 100.0 if abs(float(win)) > 1.0 else float(win)
    loss_frac = float(loss) / 100.0 if abs(float(loss)) > 1.0 else float(loss)
    if win_frac < 0:
        return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0,
                            f"projection p75 is negative ({win_frac:.4f}) — no upside scenario")
    if loss_frac > 0:
        return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0,
                            f"projection p25 is positive ({loss_frac:.4f}) — no downside scenario")
    loss_pos = abs(loss_frac)

    return suggested_size_pct(
        p_calibrated=p_calibrated,
        expected_win_pct=win_frac,
        expected_loss_pct=loss_pos,
        kelly_fraction=kelly_fraction,
        max_size_pct=max_size_pct,
    )
