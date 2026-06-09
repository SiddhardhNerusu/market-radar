"""Train the ML model on resolved signal outcomes.

Uses scikit-learn's ``HistGradientBoostingClassifier`` (histogram-binned
gradient boosted trees, same algorithm family as LightGBM but with no
native-library dependency).

Validation strategy (upgraded 2026-05-13):
  - ``TimeSeriesSplit`` expanding-window walk-forward CV with N folds.
    Each fold trains on a contiguous prefix and evaluates on the next
    chronological slice — the López de Prado "no leakage" pattern.
    Median val AUC across folds is the published metric; standard
    deviation across folds surfaces regime fragility.
  - Final deployed model: trained on all-but-last-fold data, then wrapped
    in ``CalibratedClassifierCV(method='isotonic', cv='prefit')`` fitted
    on the last fold. The wrapper is what gets saved; it exposes the same
    ``predict_proba`` interface so ``predict.py`` works unchanged.

Why calibrate: an uncalibrated ``predict_proba(0.65)`` is a model score,
not a probability. Half-Kelly sizing only works when the number actually
means "65% chance of winning." Above ~1k calibration rows isotonic
regression dominates Platt scaling, so we use isotonic.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import PROJECT_ROOT
from ..storage import get_connection
from .calendar_features import attach_calendar_features
from .conformal import ConformalPredictor
from .features import (FEATURE_NAMES, compute_corroboration_windows,
                       compute_insider_clusters, extract_features_df)
from .graph_features import (GRAPH_ONLY_FEATURE_NAMES, attach_graph_features,
                              build_co_mention_graph)
from .market_features import MarketFeatureCache
from .ta_features import attach_ta_features

log = logging.getLogger(__name__)


MODELS_DIR = PROJECT_ROOT / "data" / "models"
CURRENT_POINTER = MODELS_DIR / "current.json"
MIN_TRAINING_ROWS = 500
DEFAULT_N_SPLITS = 5

# Indexes (in FEATURE_NAMES) of categorical features. HistGradientBoosting
# supports native categorical handling via column indices.
CATEGORICAL_FEATURE_INDICES = [
    FEATURE_NAMES.index("event_type_idx"),
    FEATURE_NAMES.index("source_tier"),
    FEATURE_NAMES.index("day_of_week"),
]


@dataclass
class TrainResult:
    success: bool
    version: Optional[str]
    train_rows: int
    val_rows: int
    train_auc: Optional[float]
    val_auc: Optional[float]               # median across walk-forward folds
    val_accuracy: Optional[float]          # last fold
    val_positive_class_rate: Optional[float]
    model_path: Optional[str]
    reason: Optional[str] = None
    # Walk-forward CV diagnostics
    fold_aucs: list[float] = field(default_factory=list)
    val_auc_std: Optional[float] = None
    calibrated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _build_hgb() -> "HistGradientBoostingClassifier":  # type: ignore[name-defined]
    """Configure the base classifier. Same hyperparameters used for every
    walk-forward fold so per-fold metrics are comparable."""
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=15,            # was 31 — fewer, broader leaves
        min_samples_leaf=200,         # was 50 — predictions backed by 200+ samples
        l2_regularization=1.0,        # was 0.1 — penalise complex fits
        max_features=0.8,             # subsample features per tree → less overfit
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        categorical_features=CATEGORICAL_FEATURE_INDICES,
        random_state=42,
        verbose=0,
    )


def _print_calibration_table(val_pred, y_val) -> None:
    """Pre-calibration diagnostic: how close are raw model scores to actual
    hit rates? After we wrap with isotonic calibration on the last fold,
    the deployed model's predict_proba is the trustworthy number; this
    table is informational only.
    """
    bins = [(0.0, 0.40), (0.40, 0.50), (0.50, 0.55),
            (0.55, 0.60), (0.60, 0.65), (0.65, 1.00)]
    log.info("  Raw-score calibration (pre-isotonic):")
    for lo, hi in bins:
        mask = (val_pred >= lo) & (val_pred < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        actual_pos_rate = float(y_val[mask].mean())
        mean_pred = float(val_pred[mask].mean())
        tag = "OK " if abs(mean_pred - actual_pos_rate) < 0.03 else "OFF"
        log.info(
            "    %s p[%.2f, %.2f): n=%5d  predicted=%.3f  actual=%.3f  diff=%+.3f",
            tag, lo, hi, n, mean_pred, actual_pos_rate,
            actual_pos_rate - mean_pred,
        )


def train_and_save(
    *,
    min_rows: int = MIN_TRAINING_ROWS,
    n_splits: int = DEFAULT_N_SPLITS,
    only_replace_if_better: bool = True,
    calibrate: bool = True,
    _horizon: int = 5,
    _label_col: str = "return_5d_pct",
    _bucket: Optional[str] = None,
    _preloaded_rows: Optional[list[dict]] = None,
) -> TrainResult:
    """Train + save the model. Returns a structured result.

    Walk-forward strategy:
      - Sort rows chronologically.
      - ``TimeSeriesSplit(n_splits=N)`` gives N expanding-window folds.
      - Train an identically-configured HGB on each fold's train set,
        evaluate on its val set, record val AUC.
      - Published ``val_auc`` is the median across folds; ``val_auc_std``
        signals regime fragility.
      - For deployment: refit the HGB on all-but-last-fold data, wrap with
        ``CalibratedClassifierCV(method='isotonic', cv='prefit')`` on the
        last fold's val set. The wrapped object is what gets pickled.

    Hidden args (underscore-prefixed) drive the multi-horizon /
    per-event-type variants. Default values keep the existing
    "binary 5-day, all events" pipeline intact.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib  # type: ignore
    import numpy as np  # type: ignore
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore
    from sklearn.metrics import accuracy_score, roc_auc_score  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore

    if _preloaded_rows is not None:
        rows = list(_preloaded_rows)
    else:
        rows = _load_training_rows(require_horizon=_label_col)
        log.info("Loaded %d labelled rows for training (horizon=%dd label=%s)",
                 len(rows), _horizon, _label_col)
        if rows:
            _enrich_with_market_features(rows)
            # Tier 1 upgrade: insider-cluster context feature
            compute_insider_clusters(rows)
            # Tier 2 #19: multi-source corroboration window
            compute_corroboration_windows(rows)
            # Tier 1/2: external-data features (best-effort — defaults if absent)
            _enrich_with_external_features(rows)
            # Round-2 free batch: TA + calendar + graph features
            _enrich_with_ta_and_graph_features(rows)

    if len(rows) < min_rows:
        return TrainResult(
            success=False, version=None,
            train_rows=len(rows), val_rows=0,
            train_auc=None, val_auc=None, val_accuracy=None,
            val_positive_class_rate=None, model_path=None,
            reason=f"Not enough labelled rows: {len(rows)} < {min_rows}",
        )

    rows.sort(key=lambda r: r.get("scored_at") or "")
    X_all = extract_features_df(rows)
    # Dead-band: only a MEANINGFUL up-move is labelled 1, so the model isn't taught to
    # treat noise around zero as signal (previously +0.05% and +20% were both class 1).
    _deadband = float(os.getenv("ML_LABEL_DEADBAND_PCT", "0.2"))
    y_all = np.array([1 if (r.get(_label_col) or 0) > _deadband else 0 for r in rows])

    # --- Walk-forward CV ---
    n_splits = min(n_splits, max(2, len(rows) // 500))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_train_aucs: list[float] = []
    fold_val_aucs: list[float] = []
    last_train_idx = None
    last_val_idx = None

    log.info("Running walk-forward CV with %d folds on %d rows (%d features)…",
             n_splits, len(rows), len(FEATURE_NAMES))
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_all), 1):
        X_tr, X_va = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_va = y_all[train_idx], y_all[val_idx]
        clf = _build_hgb()
        clf.fit(X_tr, y_tr)
        tr_auc = float(roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1]))
        va_auc = float(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1]))
        fold_train_aucs.append(tr_auc)
        fold_val_aucs.append(va_auc)
        log.info("  fold %d/%d  train n=%6d  val n=%6d  train_auc=%.4f  val_auc=%.4f",
                 fold_i, n_splits, len(train_idx), len(val_idx), tr_auc, va_auc)
        last_train_idx = train_idx
        last_val_idx = val_idx

    median_val_auc = float(statistics.median(fold_val_aucs))
    val_auc_std = float(statistics.stdev(fold_val_aucs)) if len(fold_val_aucs) >= 2 else 0.0
    last_train_auc = fold_train_aucs[-1]

    log.info("Walk-forward summary: val AUC median=%.4f  std=%.4f  folds=%s",
             median_val_auc, val_auc_std,
             ", ".join(f"{a:.4f}" for a in fold_val_aucs))
    if val_auc_std > 0.03:
        log.warning("  ⚠ High fold-to-fold variance (std=%.3f) — model is "
                    "regime-sensitive; treat published AUC with care.",
                    val_auc_std)

    # --- Deployment model: base HGB trained on all-but-last-fold,
    #     calibrated on the last fold ---
    X_train_final = X_all.iloc[last_train_idx]
    y_train_final = y_all[last_train_idx]
    X_val_final = X_all.iloc[last_val_idx]
    y_val_final = y_all[last_val_idx]

    log.info("Refitting deployment base model on %d rows…", len(last_train_idx))
    base = _build_hgb()
    base.fit(X_train_final, y_train_final)

    val_pred_raw = base.predict_proba(X_val_final)[:, 1]
    _print_calibration_table(val_pred_raw, y_val_final)

    # Wrap with isotonic calibration on the last fold's val set. Above ~1k
    # rows isotonic dominates Platt. sklearn >= 1.6 removed cv='prefit'
    # in favor of FrozenEstimator; we handle both versions.
    calibration_method = "isotonic" if (calibrate and len(last_val_idx) >= 1000) else (
        "sigmoid" if calibrate else None
    )
    deployed_model = base
    calibrated = False
    if calibration_method is not None:
        log.info("Calibrating with method=%s on %d val rows…",
                 calibration_method, len(last_val_idx))
        try:
            try:
                from sklearn.frozen import FrozenEstimator  # type: ignore
                calib = CalibratedClassifierCV(
                    estimator=FrozenEstimator(base), method=calibration_method,
                )
            except ImportError:
                calib = CalibratedClassifierCV(
                    estimator=base, method=calibration_method, cv="prefit",
                )
            calib.fit(X_val_final, y_val_final)
            deployed_model = calib
            calibrated = True
            cal_pred = calib.predict_proba(X_val_final)[:, 1]
            _post_calibration_table(cal_pred, y_val_final)
        except Exception as exc:  # noqa: BLE001
            log.warning("Calibration failed: %s — deploying uncalibrated base",
                        exc)

    val_acc = float(accuracy_score(
        y_val_final,
        (deployed_model.predict_proba(X_val_final)[:, 1] >= 0.5).astype(int),
    ))
    val_pos_rate = float(y_val_final.mean())

    # Train/val gap (last fold) — overfit warning
    gap = last_train_auc - fold_val_aucs[-1]
    if gap > 0.10:
        log.warning(
            "  ⚠ Overfit warning: last-fold train AUC %.4f >> val AUC %.4f "
            "(gap %.3f). Predictions may be over-confident.",
            last_train_auc, fold_val_aucs[-1], gap,
        )
    else:
        log.info("  ✓ Last-fold train/val gap %.3f — model generalises reasonably", gap)

    deploy = True
    # Absolute floor: never deploy a model barely better than a coin flip, even if it
    # beats the previous one — a < ~0.53 AUC model has no real edge to trade on.
    _min_deploy_auc = float(os.getenv("ML_MIN_DEPLOY_AUC", "0.53"))
    if median_val_auc < _min_deploy_auc:
        deploy = False
        log.warning("New median val_auc %.4f < absolute floor %.2f — not deploying",
                    median_val_auc, _min_deploy_auc)
    if deploy and only_replace_if_better and CURRENT_POINTER.exists():
        try:
            prev_meta = json.loads(CURRENT_POINTER.read_text())
            prev_auc = prev_meta.get("val_auc") or 0
            if median_val_auc < prev_auc - 0.005:
                deploy = False
                log.warning(
                    "New median val_auc %.4f < previous %.4f — not deploying",
                    median_val_auc, prev_auc,
                )
        except (json.JSONDecodeError, OSError):
            pass

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_calibrated" if calibrated else ""
    # Add horizon / bucket tags to the file name so the per-event-type
    # and multi-horizon variants don't clobber each other.
    tag_parts: list[str] = []
    if _bucket and _bucket != "global":
        tag_parts.append(f"bucket-{_bucket}")
    if _horizon != 5:
        tag_parts.append(f"h{_horizon}d")
    tag = ("_" + "_".join(tag_parts)) if tag_parts else ""
    base_name = f"sklearn_hgb{tag}{suffix}_{version}"
    model_path = MODELS_DIR / f"{base_name}.joblib"
    meta_path = MODELS_DIR / f"{base_name}.meta.json"
    conformal_path = MODELS_DIR / f"{base_name}.conformal.json"
    joblib.dump(deployed_model, str(model_path))

    # Fit + save the conformal interval estimator on the calibration fold.
    # 90% coverage by default. Persisted as JSON so dashboard / sizing
    # modules can read it independent of joblib.
    conformal_alpha = 0.10
    try:
        cp = ConformalPredictor.fit_from_holdout(
            deployed_model, X_val_final, y_val_final, alpha=conformal_alpha,
        )
        cp.save(conformal_path)
        log.info("Conformal interval (alpha=%.2f) saved to %s (q_hat=%.4f)",
                 conformal_alpha, conformal_path.name, cp.q_hat)
        conformal_qhat = cp.q_hat
    except Exception as exc:  # noqa: BLE001
        log.warning("Conformal fit failed: %s — predictions will lack intervals", exc)
        conformal_qhat = None
    meta = {
        "version": version,
        "horizon": _horizon,
        "bucket": _bucket or "global",
        "label_col": _label_col,
        "model_type": ("sklearn.CalibratedClassifierCV(HistGradientBoosting, isotonic)"
                       if calibrated else "sklearn.HistGradientBoostingClassifier"),
        "feature_names": FEATURE_NAMES,
        "train_rows": len(last_train_idx),
        "val_rows": len(last_val_idx),
        "train_auc": last_train_auc,
        "val_auc": median_val_auc,            # median across folds
        "val_auc_last_fold": fold_val_aucs[-1],
        "val_auc_std": val_auc_std,
        "fold_aucs": fold_val_aucs,
        "n_splits": n_splits,
        "val_accuracy": val_acc,
        "val_positive_class_rate": val_pos_rate,
        "calibrated": calibrated,
        "calibration_method": calibration_method if calibrated else None,
        "conformal_qhat": conformal_qhat,
        "conformal_alpha": conformal_alpha,
        "conformal_path": str(conformal_path) if conformal_qhat is not None else None,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Update the GLOBAL pointer only for the default (5d, no bucket) run.
    # Per-event-type and per-horizon models register under their own
    # pointers so predict.py can find them.
    if _bucket is None and _horizon == 5:
        if deploy:
            CURRENT_POINTER.write_text(json.dumps({
                "version": version,
                "model_path": str(model_path),
                "meta_path": str(meta_path),
                "val_auc": median_val_auc,
                "calibrated": calibrated,
            }, indent=2))
    else:
        # Side-pointer for the variant
        side_pointer = MODELS_DIR / f"current_{(_bucket or 'global')}_h{_horizon}.json"
        side_pointer.write_text(json.dumps({
            "version": version,
            "model_path": str(model_path),
            "meta_path": str(meta_path),
            "val_auc": median_val_auc,
            "calibrated": calibrated,
            "horizon": _horizon,
            "bucket": _bucket or "global",
        }, indent=2))

    return TrainResult(
        success=True,
        version=version if deploy else None,
        train_rows=len(last_train_idx),
        val_rows=len(last_val_idx),
        train_auc=last_train_auc,
        val_auc=median_val_auc,
        val_accuracy=val_acc,
        val_positive_class_rate=val_pos_rate,
        model_path=str(model_path) if deploy else None,
        reason=None if deploy else "val_auc did not improve over previous model",
        fold_aucs=fold_val_aucs,
        val_auc_std=val_auc_std,
        calibrated=calibrated,
    )


def _post_calibration_table(cal_pred, y_val) -> None:
    """Diagnostic: confirm isotonic actually moved scores closer to truth."""
    bins = [(0.0, 0.40), (0.40, 0.50), (0.50, 0.55),
            (0.55, 0.60), (0.60, 0.65), (0.65, 1.00)]
    log.info("  Post-isotonic calibration:")
    for lo, hi in bins:
        mask = (cal_pred >= lo) & (cal_pred < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        actual = float(y_val[mask].mean())
        pred = float(cal_pred[mask].mean())
        tag = "OK " if abs(pred - actual) < 0.03 else "OFF"
        log.info(
            "    %s p[%.2f, %.2f): n=%5d  predicted=%.3f  actual=%.3f  diff=%+.3f",
            tag, lo, hi, n, pred, actual, actual - pred,
        )


def _load_training_rows(*, require_horizon: str = "return_5d_pct") -> list[dict]:
    """Pull labelled (resolved-outcome) rows for training.

    ``require_horizon`` is the column that must be non-null (so the row
    is "resolved" for the relevant horizon). Defaults to 5d for backward
    compatibility.

    Joins to ``llm_classifications`` when available — LLM-refined event_type
    and sentiment override the heuristic values for those rows. Rows without
    LLM classification fall back to the heuristic. Adds an extra feature
    ``has_llm_classification`` so the model knows when it's seeing a
    higher-quality label.
    """
    assert require_horizon in ("return_1d_pct", "return_5d_pct", "return_20d_pct"), \
        f"unsupported horizon: {require_horizon}"
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT
                COALESCE(lc.event_type, ss.event_type) AS event_type,
                COALESCE(lc.sentiment, ss.sentiment) AS sentiment,
                COALESCE(lc.sentiment_magnitude, ss.sentiment_magnitude) AS sentiment_magnitude,
                COALESCE(lc.factual, ss.factual) AS factual,
                lc.confidence AS llm_confidence,
                CASE WHEN lc.id IS NOT NULL THEN 1 ELSE 0 END AS has_llm_classification,
                ss.source_weight, ss.corroboration_count,
                ss.author_quality, ss.anti_pump_flag,
                ss.composite_score, ss.signal_class, ss.scored_at,
                ss.ticker AS ticker,
                rs.id AS signal_id,
                rs.source_tier, rs.published_at,
                st.confidence AS ticker_confidence,
                so.return_1d_pct, so.return_5d_pct, so.return_20d_pct,
                so.price_at_flag_ts
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            JOIN signal_tickers st ON st.signal_id = rs.id AND st.ticker = ss.ticker
            JOIN signal_outcomes so ON so.score_id = ss.id
            LEFT JOIN llm_classifications lc
                   ON lc.signal_id = ss.signal_id AND lc.ticker = ss.ticker
            WHERE so.{require_horizon} IS NOT NULL
              -- Noise filter: ~75% of resolved rows are routine regulatory
              -- backfill (event_type='other' or sec_edgar_backfill_* sources)
              -- with ~random forward returns. Training on them teaches the
              -- model the base rate → near-flat ~0.42 predictions. Exclude
              -- them so the model learns from genuine catalysts. Filter on
              -- the same COALESCE expression used for the event_type column.
              AND COALESCE(lc.event_type, ss.event_type) != 'other'
              AND rs.source NOT LIKE 'sec_edgar_backfill_%'
              -- Outlier filter: stock splits / reverse splits / ticker
              -- reuse can show 1,000%+ returns (e.g., INRE 3,002,400%).
              -- These poison the model's notion of "winners." Loosened from
              -- 50 → 200 so genuine +50-100% catalyst winners (the strategy's
              -- target trades) survive while split/data artifacts (>200%) drop.
              AND ABS(so.{require_horizon}) < 200.0
            """
        ).fetchall()
    return [dict(r) for r in rows]


# ---- Multi-horizon + per-event-type training -----------------------------
# (Tier 1 #5 + #6 — 2026-05-13)

# Buckets: groups of related event_types that share enough statistical
# structure to share a model. Bucket key → list of event_type strings.
EVENT_BUCKETS: dict[str, list[str]] = {
    "ma":       ["m_a_announcement", "m_a_rumor"],
    "earnings": ["earnings_beat", "earnings_miss", "earnings_announcement",
                 "guidance_raise", "guidance_cut"],
    "fda":      ["fda_approval", "fda_rejection", "clinical_trial_result"],
    "insider":  ["insider_buy", "insider_sell", "insider_transaction"],
    "activist": ["activist_position"],
    "macro":    ["macro"],
    # Everything else is the catch-all
}
MIN_BUCKET_ROWS = 300         # below this we don't train a bucket model
HORIZON_RETURN_COLS = {
    1:  "return_1d_pct",
    5:  "return_5d_pct",
    20: "return_20d_pct",
}


def _bucket_for_event(event_type: Optional[str]) -> str:
    """Map a raw event_type label → bucket key."""
    if not event_type:
        return "other"
    for bucket, members in EVENT_BUCKETS.items():
        if event_type in members:
            return bucket
    return "other"


def train_multi_horizon(
    *,
    horizons: tuple[int, ...] = (1, 5, 20),
    min_rows: int = MIN_TRAINING_ROWS,
    n_splits: int = DEFAULT_N_SPLITS,
    calibrate: bool = True,
) -> dict[int, TrainResult]:
    """Train one model per horizon. Wraps ``train_and_save`` per horizon
    with a custom label column. Results keyed by horizon (1/5/20)."""
    results: dict[int, TrainResult] = {}
    for h in horizons:
        col = HORIZON_RETURN_COLS[h]
        log.info("=== TRAINING horizon=%d (label=%s) ===", h, col)
        res = train_and_save(
            min_rows=min_rows,
            n_splits=n_splits,
            calibrate=calibrate,
            _horizon=h,
            _label_col=col,
        )
        results[h] = res
    return results


def train_per_event_type(
    *,
    horizons: tuple[int, ...] = (5,),
    min_rows_per_bucket: int = MIN_BUCKET_ROWS,
    n_splits: int = DEFAULT_N_SPLITS,
    calibrate: bool = True,
) -> dict[str, dict[int, TrainResult]]:
    """Train a separate model per (event-bucket, horizon) combination.

    Buckets with fewer than ``min_rows_per_bucket`` resolved rows skip
    training and fall back to the global model at predict time.

    Returns a nested dict: ``{bucket: {horizon: TrainResult}}``.
    """
    results: dict[str, dict[int, TrainResult]] = {}
    for h in horizons:
        col = HORIZON_RETURN_COLS[h]
        rows = _load_training_rows(require_horizon=col)
        if rows:
            _enrich_with_market_features(rows)
            compute_insider_clusters(rows)
            compute_corroboration_windows(rows)
            _enrich_with_external_features(rows)

        buckets: dict[str, list[dict]] = {}
        for r in rows:
            b = _bucket_for_event(r.get("event_type"))
            buckets.setdefault(b, []).append(r)
        log.info("Per-event-type bucket sizes (horizon=%d): %s",
                 h, {b: len(rs) for b, rs in buckets.items()})

        for bucket, bucket_rows in buckets.items():
            if len(bucket_rows) < min_rows_per_bucket:
                log.info("  skipping bucket=%s (n=%d < %d)",
                         bucket, len(bucket_rows), min_rows_per_bucket)
                continue
            log.info("=== TRAINING bucket=%s horizon=%d (n=%d) ===",
                     bucket, h, len(bucket_rows))
            res = train_and_save(
                min_rows=min(min_rows_per_bucket, 200),
                n_splits=min(n_splits, max(2, len(bucket_rows) // 200)),
                calibrate=calibrate,
                _horizon=h,
                _label_col=col,
                _bucket=bucket,
                _preloaded_rows=bucket_rows,
            )
            results.setdefault(bucket, {})[h] = res

    return results


def _enrich_with_market_features(rows: list[dict]) -> None:
    """Fetch market-context features for every row and attach to the dict.

    Walks all unique tickers in the training set, pre-fetches their full
    daily history over the spanning window, then computes per-row features.
    """
    from datetime import date, datetime, timezone

    # Identify unique tickers and date range
    tickers: set[str] = set()
    dates: list[date] = []
    for r in rows:
        t = (r.get("ticker") or "").upper()
        if t:
            # Strip T212-style suffix _US_EQ etc.
            if "_" in t:
                parts = t.split("_")
                if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
                    t = "_".join(parts[:-2]).replace("_", "-")
            tickers.add(t)
        # Use price_at_flag_ts (the anchor) or fall back to published_at
        ts = r.get("price_at_flag_ts") or r.get("published_at") or r.get("scored_at")
        if isinstance(ts, str):
            try:
                d = datetime.strptime(ts[:10], "%Y-%m-%d").date()
                dates.append(d)
            except ValueError:
                pass

    if not tickers or not dates:
        log.info("Market-feature enrichment skipped — no tickers/dates")
        return

    earliest = min(dates)
    latest = max(dates)
    # Pad earliest by 25 days to give the 20d window room
    from datetime import timedelta
    fetch_start = earliest - timedelta(days=35)
    fetch_end = latest + timedelta(days=2)

    log.info(
        "Enriching with market features: %d unique tickers from %s to %s",
        len(tickers), fetch_start, fetch_end,
    )
    cache = MarketFeatureCache()
    cache.warm(sorted(tickers), start=fetch_start, end=fetch_end)

    # Compute per-row
    for r in rows:
        t = (r.get("ticker") or "").upper()
        ts = r.get("price_at_flag_ts") or r.get("published_at") or r.get("scored_at")
        if not t or not isinstance(ts, str):
            continue
        try:
            d = datetime.strptime(ts[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        mf = cache.features_for(t, d)
        r["volume_ratio_5d_20d"] = mf.volume_ratio_5d_20d
        r["realized_vol_20d"]    = mf.realized_vol_20d
        r["spy_5d_return"]       = mf.spy_5d_return
        r["qqq_5d_return"]       = mf.qqq_5d_return
        r["vix_level"]           = mf.vix_level
        # Macro pack
        r["yield_curve_slope"]   = mf.yield_curve_slope
        r["dxy_5d_return"]       = mf.dxy_5d_return
        r["oil_5d_return"]       = mf.oil_5d_return
        r["gold_5d_return"]      = mf.gold_5d_return
    log.info("Market-feature enrichment complete")


def _enrich_with_external_features(rows: list[dict]) -> None:
    """Pull external-data features for every row from the staging tables
    populated by the various ingestors (FINRA short interest, FDA
    catalysts, 13F flow, Google Trends, Wikipedia pageviews, PEAD,
    fails-to-deliver, earnings whispers, insider transaction codes,
    news novelty, FinBERT sentiment).

    Best-effort. Any table that doesn't exist yet (fresh DB / ingestor not
    yet run) is silently skipped — the feature defaults to its no-info
    value via ``extract_features``.
    """
    from .external_features import attach_all_external_features
    try:
        attach_all_external_features(rows)
    except Exception as exc:  # noqa: BLE001
        log.warning("external feature enrichment failed: %s", exc)


def _enrich_with_ta_and_graph_features(rows: list[dict]) -> None:
    """Round-2 enrichments: TA indicators, calendar, graph peers.

    Uses a shared MarketFeatureCache already warmed for sector peers +
    BTC. Anything that fails defaults out gracefully.
    """
    from .market_features import CROSS_ASSETS, MarketFeatureCache
    from datetime import date, datetime, timedelta

    # Build a small extension cache that also covers sector peers + BTC.
    # We reuse the cache that's already warmed if the enrichment ran in
    # the same training pass; otherwise warm a fresh one.
    try:
        attach_calendar_features(rows)
    except Exception as exc:  # noqa: BLE001
        log.warning("calendar features failed: %s", exc)

    # Sector peer + BTC tickers we need cached
    tickers_to_warm: set[str] = {"BTC-USD"}
    from .graph_features import SECTOR_PEERS
    for r in rows:
        t = (r.get("ticker") or "").upper()
        if "_" in t:
            parts = t.split("_")
            if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
                t = "_".join(parts[:-2]).replace("_", "-")
        for peer in SECTOR_PEERS.get(t, []):
            tickers_to_warm.add(peer)

    # Build a tightened cache for TA + graph peers
    cache = MarketFeatureCache()
    try:
        dates: list[date] = []
        for r in rows:
            ts = r.get("price_at_flag_ts") or r.get("published_at") or r.get("scored_at")
            if isinstance(ts, str):
                try:
                    dates.append(datetime.strptime(ts[:10], "%Y-%m-%d").date())
                except ValueError:
                    pass
        if dates and tickers_to_warm:
            cache.warm(sorted(tickers_to_warm),
                       start=min(dates) - timedelta(days=40),
                       end=max(dates) + timedelta(days=2))
    except Exception as exc:  # noqa: BLE001
        log.warning("graph-peer cache warm failed: %s", exc)

    try:
        attach_ta_features(rows, cache)
    except Exception as exc:  # noqa: BLE001
        log.warning("TA features failed: %s", exc)

    try:
        peers = build_co_mention_graph()
        attach_graph_features(rows, cache, co_mention_peers=peers)
    except Exception as exc:  # noqa: BLE001
        log.warning("graph features failed: %s", exc)
