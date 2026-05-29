"""Conformal prediction wrapper — distribution-free coverage intervals
around the calibrated model's outputs.

Split-conformal pattern (the simplest exchangeable variant from
Vovk et al.):
  1. Hold out a calibration set ``C`` of size ``n_cal``.
  2. For each ``(x_i, y_i)`` in ``C`` compute a nonconformity score
     ``s_i = |y_i - p_hat_i|``.
  3. The (1 - alpha)-quantile ``q_hat`` of ``{s_i}`` is the half-width
     of the conformal interval.
  4. At inference, return ``[max(0, p̂ - q_hat), min(1, p̂ + q_hat)]``
     which contains the true ``y`` at least (1 - alpha) of the time
     under data exchangeability.

For time series exchangeability doesn't strictly hold; we mitigate by
keeping the calibration set chronologically adjacent to the validation
set used for isotonic calibration in ``train.py``. Coverage is
approximate but the empirical interval width is still a useful signal-
quality input for sizing decisions: wider intervals → less conviction
→ smaller Kelly fraction.

Usage::

    from market_radar.ml.conformal import ConformalPredictor
    cp = ConformalPredictor.fit_from_holdout(model, X_cal, y_cal, alpha=0.10)
    p_lo, p_hat, p_hi = cp.predict_interval(x_new)
    # interval width is `p_hi - p_lo`; use as an uncertainty input

The ``q_hat`` value is serialisable so we can pickle it alongside the
calibrated model and load it back without re-running fit.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class ConformalPredictor:
    """Split-conformal half-width for a binary classifier."""

    q_hat: float
    alpha: float
    n_cal: int

    @classmethod
    def fit_from_holdout(
        cls,
        model,
        X_cal,
        y_cal,
        *,
        alpha: float = 0.10,
    ) -> "ConformalPredictor":
        """Fit ``q_hat`` from a held-out calibration set.

        ``alpha=0.10`` corresponds to 90% coverage.
        """
        import numpy as np  # type: ignore
        probas = model.predict_proba(X_cal)[:, 1]
        y = np.asarray(y_cal).astype(float)
        scores = np.abs(y - probas)
        # The split-conformal quantile is ⌈(n+1)(1-alpha)⌉ / n. For large n
        # this is just the (1-alpha)-quantile of the scores.
        n = len(scores)
        q_level = min(1.0, (1 - alpha) * (n + 1) / max(n, 1))
        q_hat = float(np.quantile(scores, q_level, method="higher"))
        log.info("Conformal: n_cal=%d alpha=%.2f q_hat=%.4f (90%% interval ≈ ±%.3f)",
                 n, alpha, q_hat, q_hat)
        return cls(q_hat=q_hat, alpha=alpha, n_cal=n)

    def predict_interval(self, p_hat: float) -> tuple[float, float, float]:
        """Return ``(lo, p_hat, hi)`` — the conformal interval clamped to [0, 1]."""
        lo = max(0.0, float(p_hat) - self.q_hat)
        hi = min(1.0, float(p_hat) + self.q_hat)
        return (lo, float(p_hat), hi)

    def predict_intervals_batch(self, p_hats) -> list[tuple[float, float, float]]:
        return [self.predict_interval(p) for p in p_hats]

    def interval_width(self, p_hat: float) -> float:
        lo, _, hi = self.predict_interval(p_hat)
        return hi - lo

    # ---- persistence ------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({"q_hat": self.q_hat, "alpha": self.alpha,
                           "n_cal": self.n_cal})

    @classmethod
    def from_json(cls, s: str) -> "ConformalPredictor":
        d = json.loads(s)
        return cls(q_hat=float(d["q_hat"]), alpha=float(d["alpha"]),
                   n_cal=int(d["n_cal"]))

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> "ConformalPredictor":
        return cls.from_json(Path(path).read_text())
