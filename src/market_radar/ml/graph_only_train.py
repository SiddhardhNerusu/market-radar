"""Train a model that predicts price direction using **only** graph
features — sector peer momentum, co-mention peers, crypto-equity
sympathetic moves, and cross-asset macro context.

The point of this model is to answer: "from the network alone (peers
moving, sector strength, crypto sympathy), what's the probability this
ticker rises in 5 days?" It deliberately ignores news, sentiment,
event_type, and signal-intrinsic features so the graph contribution is
isolated.

Two uses:
  1. Dashboard: shows a "graph-only" probability next to the main
     probability. When the two agree, conviction is higher.
  2. Ensemble: a graph-only base learner is one of the three feeders
     into the stacked meta-learner (see ``ml/ensemble.py``).

Output: ``data/models/graph_only_<version>.joblib`` plus a side-pointer
at ``data/models/current_graph_only.json`` that ``predict.py`` reads.
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import PROJECT_ROOT
from ..storage import get_connection
from .graph_features import (
    GRAPH_ONLY_FEATURE_NAMES, attach_graph_features, build_co_mention_graph,
)
from .market_features import MarketFeatureCache

log = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "data" / "models"
CURRENT_GRAPH_POINTER = MODELS_DIR / "current_graph_only.json"


def _calibrate_prefit(base_estimator, method: str, X_val, y_val):
    """sklearn-version-portable isotonic/Platt calibration on a prefit base.

    sklearn < 1.6: ``CalibratedClassifierCV(estimator=base, cv='prefit')``
    sklearn >= 1.6: ``CalibratedClassifierCV(estimator=FrozenEstimator(base))``
    (cv='prefit' was removed in 1.6 in favor of FrozenEstimator.)
    """
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore
    try:
        from sklearn.frozen import FrozenEstimator  # type: ignore
        calib = CalibratedClassifierCV(estimator=FrozenEstimator(base_estimator),
                                       method=method)
    except ImportError:
        calib = CalibratedClassifierCV(estimator=base_estimator, method=method,
                                       cv="prefit")
    calib.fit(X_val, y_val)
    return calib


@dataclass
class GraphTrainResult:
    success: bool
    version: Optional[str]
    train_rows: int
    val_rows: int
    val_auc: Optional[float]
    fold_aucs: list[float]
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _load_rows() -> list[dict]:
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            """
            SELECT ss.ticker AS ticker, ss.scored_at,
                   rs.published_at,
                   so.price_at_flag_ts, so.return_5d_pct
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE so.return_5d_pct IS NOT NULL
            """
        ).fetchall()]


def _enrich(rows: list[dict]) -> None:
    """Warm market cache + attach all graph-relevant features."""
    from datetime import date, datetime, timedelta
    from .graph_features import CRYPTO_CORRELATED, SECTOR_PEERS

    # Tickers we need: all signal tickers + their sector peers + BTC
    needed: set[str] = {"SPY", "QQQ", "^VIX", "^TNX", "^IRX",
                        "UUP", "USO", "GLD", "BTC-USD"}
    dates: list[date] = []
    for r in rows:
        t = (r.get("ticker") or "").upper()
        if "_" in t:
            parts = t.split("_")
            if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
                t = "_".join(parts[:-2]).replace("_", "-")
        if t:
            needed.add(t)
            for peer in SECTOR_PEERS.get(t, [])[:8]:
                needed.add(peer)
        ts = r.get("price_at_flag_ts") or r.get("published_at") or r.get("scored_at")
        if isinstance(ts, str):
            try:
                dates.append(datetime.strptime(ts[:10], "%Y-%m-%d").date())
            except ValueError:
                pass

    if not dates:
        return
    cache = MarketFeatureCache()
    cache.warm(sorted(needed),
               start=min(dates) - timedelta(days=35),
               end=max(dates) + timedelta(days=2))

    # Compute MarketFeatures-derived fields needed by GRAPH_ONLY_FEATURE_NAMES
    for r in rows:
        t = (r.get("ticker") or "").upper()
        if "_" in t:
            parts = t.split("_")
            if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
                t = "_".join(parts[:-2]).replace("_", "-")
        ts = r.get("price_at_flag_ts") or r.get("published_at") or r.get("scored_at")
        if not isinstance(ts, str):
            continue
        try:
            d = datetime.strptime(ts[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        mf = cache.features_for(t, d)
        r["spy_5d_return"]       = mf.spy_5d_return or 0.0
        r["qqq_5d_return"]       = mf.qqq_5d_return or 0.0
        r["yield_curve_slope"]   = mf.yield_curve_slope or 0.0
        r["dxy_5d_return"]       = mf.dxy_5d_return or 0.0
        r["oil_5d_return"]       = mf.oil_5d_return or 0.0
        r["gold_5d_return"]      = mf.gold_5d_return or 0.0
        r["vix_level"]           = mf.vix_level or 18.0
        r["volume_ratio_5d_20d"] = mf.volume_ratio_5d_20d or 1.0
        r["realized_vol_20d"]    = mf.realized_vol_20d or 25.0
        r["unique_sources_24h"]  = 0  # graph-only model ignores this strictly
        r["institutional_buyers_minus_sellers_qoq"] = 0.0

    # Graph peer features
    peers = build_co_mention_graph()
    attach_graph_features(rows, cache, co_mention_peers=peers)


def _build_X(rows: list[dict]):
    import pandas as pd  # type: ignore
    matrix = []
    for r in rows:
        matrix.append([float(r.get(f) or 0.0) for f in GRAPH_ONLY_FEATURE_NAMES])
    return pd.DataFrame(matrix, columns=GRAPH_ONLY_FEATURE_NAMES)


def train_graph_only(*, min_rows: int = 500,
                     n_splits: int = 5) -> GraphTrainResult:
    """Train an HGB on GRAPH_ONLY_FEATURE_NAMES only. Walk-forward CV +
    isotonic calibration on the last fold.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib  # type: ignore
    import numpy as np  # type: ignore
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    from sklearn.metrics import roc_auc_score  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore

    rows = _load_rows()
    log.info("graph-only: loaded %d labelled rows", len(rows))
    if len(rows) < min_rows:
        return GraphTrainResult(False, None, len(rows), 0, None, [],
                                reason=f"too few rows: {len(rows)}")
    _enrich(rows)
    rows.sort(key=lambda r: r.get("scored_at") or "")
    X = _build_X(rows)
    y = np.array([1 if (r.get("return_5d_pct") or 0) > 0 else 0 for r in rows])

    n_splits = min(n_splits, max(2, len(rows) // 500))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_aucs: list[float] = []
    last_train_idx = None
    last_val_idx = None
    for fi, (tr, va) in enumerate(tscv.split(X), 1):
        clf = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.05, max_iter=200,
            max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
            random_state=42,
        )
        clf.fit(X.iloc[tr], y[tr])
        va_auc = float(roc_auc_score(y[va], clf.predict_proba(X.iloc[va])[:, 1]))
        fold_aucs.append(va_auc)
        log.info("  graph-only fold %d/%d  val_auc=%.4f", fi, n_splits, va_auc)
        last_train_idx, last_val_idx = tr, va

    median_auc = float(statistics.median(fold_aucs))
    log.info("graph-only: median val_auc=%.4f folds=%s",
             median_auc, ", ".join(f"{a:.4f}" for a in fold_aucs))

    # Refit + calibrate
    base = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=200,
        max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
        random_state=42,
    )
    base.fit(X.iloc[last_train_idx], y[last_train_idx])
    method = "isotonic" if len(last_val_idx) >= 1000 else "sigmoid"
    deployed = _calibrate_prefit(base, method, X.iloc[last_val_idx],
                                 y[last_val_idx])

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MODELS_DIR / f"graph_only_{version}.joblib"
    meta_path = MODELS_DIR / f"graph_only_{version}.meta.json"
    joblib.dump(deployed, str(path))
    meta_path.write_text(json.dumps({
        "version": version,
        "model_type": "graph_only(HGB+isotonic)",
        "feature_names": GRAPH_ONLY_FEATURE_NAMES,
        "fold_aucs": fold_aucs,
        "val_auc": median_auc,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2))
    CURRENT_GRAPH_POINTER.write_text(json.dumps({
        "version": version,
        "model_path": str(path),
        "meta_path": str(meta_path),
        "val_auc": median_auc,
    }, indent=2))

    return GraphTrainResult(True, version, len(last_train_idx),
                            len(last_val_idx), median_auc, fold_aucs)
