"""Stacked ensemble — HGB + LightGBM with a logistic meta-learner.

Earlier versions included a LogisticRegression base learner that scored
~0.49 AUC (worse than random) because it was fed the full 60-feature
matrix (mixed scales + sparse categoricals) and dragged the ensemble
below the single-best base.  Dropped on 2026-05-14.

Remaining design:
  - HGB (tree-based, captures non-linear interactions)
  - LightGBM (tree-based, different inductive bias) — optional, falls
    back gracefully when libomp.dylib is missing.
  - Logistic meta-learner on out-of-fold base predictions.

When only HGB is available (LightGBM missing) we still train the meta
on a single base; the meta becomes a calibration layer rather than a
true stack.  That's fine — it still produces a deployable artifact.

Output: ``data/models/ensemble_<version>.joblib`` containing a dict
{'hgb', 'lgb' (or None), 'meta'} plus the feature list.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import PROJECT_ROOT
from .features import FEATURE_NAMES

log = logging.getLogger(__name__)
MODELS_DIR = PROJECT_ROOT / "data" / "models"
CURRENT_ENSEMBLE_POINTER = MODELS_DIR / "current_ensemble.json"


@dataclass
class EnsembleResult:
    success: bool
    version: Optional[str]
    train_rows: int
    val_rows: int
    val_auc: Optional[float]
    base_aucs: dict
    fold_aucs: list[float]
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def train_ensemble(*, min_rows: int = 500, n_splits: int = 5) -> EnsembleResult:
    """Stacked ensemble. Reuses the standard pipeline's feature extraction."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib  # type: ignore
    import numpy as np  # type: ignore
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import roc_auc_score  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore
    # LightGBM's macOS wheel depends on libomp.dylib (provided by
    # Homebrew's libomp package). If the package is missing the import
    # raises OSError, not ImportError.
    try:
        import lightgbm as lgb  # type: ignore
        has_lgb = True
    except (ImportError, OSError) as exc:
        has_lgb = False
        log.warning(
            "lightgbm unavailable (%s: %s) — stacking with HGB only. "
            "To enable, run: brew install libomp",
            type(exc).__name__, exc,
        )

    # Reuse the training loader + enrichments.
    from .train import (
        _enrich_with_external_features, _enrich_with_market_features,
        _enrich_with_ta_and_graph_features, _load_training_rows,
    )
    from .features import (
        compute_corroboration_windows, compute_insider_clusters,
        extract_features_df,
    )

    rows = _load_training_rows()
    log.info("ensemble: loaded %d labelled rows", len(rows))
    if len(rows) < min_rows:
        return EnsembleResult(False, None, len(rows), 0, None, {}, [],
                              reason=f"too few rows: {len(rows)}")
    _enrich_with_market_features(rows)
    compute_insider_clusters(rows)
    compute_corroboration_windows(rows)
    _enrich_with_external_features(rows)
    _enrich_with_ta_and_graph_features(rows)

    rows.sort(key=lambda r: r.get("scored_at") or "")
    X = extract_features_df(rows)
    y = np.array(
        [1 if (r.get("return_5d_pct") or 0) > 0 else 0 for r in rows]
    )

    n_splits = min(n_splits, max(2, len(rows) // 500))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Out-of-fold predictions per base learner for the meta-learner.
    oof_hgb = np.zeros(len(rows))
    oof_lgb = np.zeros(len(rows)) if has_lgb else None
    fold_aucs: list[float] = []
    last_train_idx = None
    last_val_idx = None

    for fi, (tr, va) in enumerate(tscv.split(X), 1):
        hgb = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.03, max_iter=300,
            max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
            random_state=42,
        )
        hgb.fit(X.iloc[tr], y[tr])
        oof_hgb[va] = hgb.predict_proba(X.iloc[va])[:, 1]

        if has_lgb:
            lgb_clf = lgb.LGBMClassifier(
                objective="binary", learning_rate=0.05, n_estimators=200,
                num_leaves=31, min_child_samples=100, reg_lambda=1.0,
                random_state=43, verbose=-1,
            )
            lgb_clf.fit(X.iloc[tr], y[tr])
            oof_lgb[va] = lgb_clf.predict_proba(X.iloc[va])[:, 1]

        bases = [oof_hgb[va]] + ([oof_lgb[va]] if has_lgb else [])
        avg = np.mean(bases, axis=0)
        fold_aucs.append(float(roc_auc_score(y[va], avg)))
        log.info("  ensemble fold %d/%d  avg_val_auc=%.4f",
                 fi, n_splits, fold_aucs[-1])
        last_train_idx, last_val_idx = tr, va

    # Build meta features (out-of-fold predictions from each tree base).
    feats = [oof_hgb]
    if has_lgb:
        feats.append(oof_lgb)
    meta_X = np.column_stack(feats)
    # Use the last fold as the meta-learner's training data.
    valid_idx = last_val_idx if last_val_idx is not None else np.arange(len(rows))
    meta = LogisticRegression(max_iter=1000)
    meta.fit(meta_X[valid_idx], y[valid_idx])
    meta_pred = meta.predict_proba(meta_X[valid_idx])[:, 1]
    val_auc = float(roc_auc_score(y[valid_idx], meta_pred))

    # Refit deployment bases on the last training fold.
    hgb_deploy = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.03, max_iter=300,
        max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
        random_state=42,
    )
    hgb_deploy.fit(X.iloc[last_train_idx], y[last_train_idx])
    lgb_deploy = None
    if has_lgb:
        lgb_deploy = lgb.LGBMClassifier(
            objective="binary", learning_rate=0.05, n_estimators=200,
            num_leaves=31, min_child_samples=100, reg_lambda=1.0,
            random_state=43, verbose=-1,
        )
        lgb_deploy.fit(X.iloc[last_train_idx], y[last_train_idx])

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MODELS_DIR / f"ensemble_{version}.joblib"
    meta_path = MODELS_DIR / f"ensemble_{version}.meta.json"
    joblib.dump({
        "hgb": hgb_deploy,
        "lgb": lgb_deploy,
        "meta": meta,
        "feature_names": list(FEATURE_NAMES),
        "has_lgb": has_lgb,
    }, str(path))

    base_aucs: dict = {
        "hgb": float(roc_auc_score(y[last_val_idx], oof_hgb[last_val_idx])),
    }
    if has_lgb:
        base_aucs["lgb"] = float(
            roc_auc_score(y[last_val_idx], oof_lgb[last_val_idx])
        )

    meta_path.write_text(json.dumps({
        "version": version,
        "model_type": "stacked_ensemble",
        "feature_names": list(FEATURE_NAMES),
        "fold_aucs": fold_aucs,
        "val_auc": val_auc,
        "base_aucs": base_aucs,
        "has_lgb": has_lgb,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": "LogReg base dropped 2026-05-14 (was AUC ~0.49 — dragged ensemble).",
    }, indent=2))
    CURRENT_ENSEMBLE_POINTER.write_text(json.dumps({
        "version": version,
        "model_path": str(path),
        "meta_path": str(meta_path),
        "val_auc": val_auc,
        "base_aucs": base_aucs,
    }, indent=2))

    log.info("ensemble: val_auc=%.4f  base_aucs=%s", val_auc, base_aucs)
    return EnsembleResult(
        True, version, len(last_train_idx), len(last_val_idx),
        val_auc, base_aucs, fold_aucs,
    )
