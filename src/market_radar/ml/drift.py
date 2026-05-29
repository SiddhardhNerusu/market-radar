"""Model-drift detection — Page-Hinkley test on rolling val AUC.

Why this exists:
  Models can degrade over time as markets shift regime. We can't catch it
  by looking at backtest AUC (frozen) or by squinting at the dashboard.
  We need a programmatic check: after every batch of resolved outcomes,
  compute the model's rolling AUC and alert when it's decayed beyond a
  noise threshold.

The Page-Hinkley test maintains a running ``m_t`` of cumulative deviation
from the historical mean. When ``PH_t = max(0, ..., m_t)`` exceeds
``delta * lambda``, drift is declared. Standard parameterisation from
the time-series literature.

Storage: every observation goes to ``model_drift_observations`` so the
dashboard / future analyses can plot the AUC curve and PH stat over
time.

Use case:
  Call ``compute_and_record_drift()`` after every weekly retrain, OR
  from a cron job on the last 500 resolved outcomes.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..storage import get_connection

log = logging.getLogger(__name__)


@dataclass
class DriftResult:
    observed_at: str
    model_version: Optional[str]
    window_n: int
    rolling_auc: Optional[float]
    ph_stat: float
    alerted: bool
    reason: str


# Page-Hinkley parameters. ``delta`` is the magnitude tolerated before
# accumulating; ``lambda_`` is the alert threshold. Default values
# follow Page (1954) defaults for moderate sensitivity on noisy signals.
DEFAULT_DELTA = 0.005      # ~0.5pp AUC slop ignored
DEFAULT_LAMBDA = 0.04      # alert when cumulative drift > 4pp
DEFAULT_WINDOW = 500       # last N resolved outcomes


def compute_rolling_auc(window: int = DEFAULT_WINDOW) -> tuple[Optional[float], int, Optional[str]]:
    """Compute AUC of the deployed model's predictions on the most recent
    ``window`` resolved signals. Returns ``(auc, n, model_version)`` or
    ``(None, 0, None)`` if there's nothing to measure.
    """
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
    except ImportError:
        log.warning("sklearn unavailable — cannot compute rolling AUC")
        return None, 0, None

    # Load the currently-deployed model's version from the pointer
    # so drift only measures the model the bot actually uses.
    current_version = None
    try:
        import json
        from pathlib import Path
        pointer = Path(__file__).resolve().parents[3] / "data" / "models" / "current.json"
        if pointer.exists():
            current_version = json.loads(pointer.read_text()).get("version")
    except Exception:  # noqa: BLE001
        pass

    with get_connection() as conn:
        if current_version:
            rows = conn.execute(
                """
                SELECT ss.model_p_5d AS p, ss.model_version AS version,
                       so.return_5d_pct AS r
                FROM signal_scores ss
                JOIN signal_outcomes so ON so.score_id = ss.id
                WHERE so.return_5d_pct IS NOT NULL
                  AND ss.model_p_5d IS NOT NULL
                  AND ss.model_version = ?
                ORDER BY ss.id DESC
                LIMIT ?
                """,
                (current_version, window),
            ).fetchall()
        else:
            rows = []

    if len(rows) < 50:
        # Not enough resolved outcomes for the CURRENT model yet — drift
        # observation skipped. This is correct behaviour after a fresh
        # retrain: new model needs ~5 days for outcomes to resolve.
        log.info("Drift skipped: only %d resolved outcomes for model %s (need 50+)",
                 len(rows), current_version)
        return None, len(rows), current_version

    y = [1 if (r["r"] or 0) > 0 else 0 for r in rows]
    p = [float(r["p"]) for r in rows]
    if sum(y) == 0 or sum(y) == len(y):
        log.warning("Rolling window has no class variance (all positives or all negatives)")
        return None, len(rows), None
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError as exc:
        log.warning("AUC computation failed: %s", exc)
        return None, len(rows), None

    model_version = rows[0]["version"] if rows else None
    return auc, len(rows), model_version


def compute_and_record_drift(
    *,
    window: int = DEFAULT_WINDOW,
    delta: float = DEFAULT_DELTA,
    lambda_: float = DEFAULT_LAMBDA,
) -> DriftResult:
    """One-shot Page-Hinkley measurement + write to drift observations.

    Loads the historical mean of ``rolling_auc`` from
    ``model_drift_observations`` (excluding rows from a different model
    version — drift is per-model). Computes the latest rolling AUC,
    updates the PH cumulative statistic, and flags ``alerted`` if the PH
    stat crosses ``lambda_``.
    """
    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    auc, n, version = compute_rolling_auc(window=window)
    if auc is None:
        result = DriftResult(observed_at, version, n, None, 0.0, False,
                             f"too few rows (n={n})")
        return result

    with get_connection() as conn:
        # Previous observations for *this* model version. PH state is
        # per-model — drift only makes sense within a continuous model.
        prev = conn.execute(
            """
            SELECT rolling_auc, ph_stat FROM model_drift_observations
            WHERE model_version = ?
            ORDER BY id ASC
            """,
            (version,),
        ).fetchall()
        if not prev:
            mean_so_far = auc
            ph_prev = 0.0
        else:
            aucs = [float(r["rolling_auc"]) for r in prev if r["rolling_auc"] is not None]
            mean_so_far = sum(aucs) / len(aucs) if aucs else auc
            ph_prev = float(prev[-1]["ph_stat"] or 0.0)

        # Page-Hinkley accumulates *negative* deviations (drift = AUC
        # degrading below the mean). Centre on ``mean_so_far - delta``
        # so small noise doesn't accumulate.
        deviation = mean_so_far - auc - delta
        ph_stat = max(0.0, ph_prev + deviation)
        alerted = ph_stat > lambda_
        reason = (
            f"PH stat {ph_stat:.4f} > lambda {lambda_:.3f} — drift detected!"
            if alerted else
            f"PH stat {ph_stat:.4f} <= lambda {lambda_:.3f} — within tolerance"
        )

        conn.execute(
            """
            INSERT INTO model_drift_observations (
                observed_at, model_version, window_n, rolling_auc, ph_stat, alerted
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (observed_at, version, n, auc, ph_stat, 1 if alerted else 0),
        )

    log.info("Drift: model=%s n=%d rolling_auc=%.4f ph_stat=%.4f alerted=%s",
             version, n, auc, ph_stat, alerted)
    if alerted:
        log.warning("⚠ Model %s shows persistent drift below historical mean. "
                    "Consider retraining.", version)
        # Telegram coupling — never raise into the drift loop.
        try:
            from ..config import CONFIG
            if CONFIG.telegram_bot_token and CONFIG.telegram_chat_id:
                import requests
                msg = (
                    f"⚠️ <b>MARKET RADAR — MODEL DRIFT DETECTED</b>\n"
                    f"model={version}\n"
                    f"rolling_auc={auc:.4f}\n"
                    f"PH stat={ph_stat:.4f} (threshold {lambda_:.3f})\n"
                    f"Consider triggering a retrain."
                )
                requests.post(
                    f"https://api.telegram.org/bot{CONFIG.telegram_bot_token}/sendMessage",
                    json={
                        "chat_id": CONFIG.telegram_chat_id,
                        "text": msg,
                        "parse_mode": "HTML",
                    },
                    timeout=10,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Drift Telegram alert failed: %s", exc)
    return DriftResult(observed_at, version, n, auc, ph_stat, alerted, reason)
