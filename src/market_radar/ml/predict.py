"""Load the current sklearn HistGradientBoosting model + predict on new signals.

Two entry points:
  * ``get_predictor()`` — returns a lazily-loaded singleton predictor.
  * ``ModelPredictor.predict_pending()`` — fills ``model_p_5d`` on signal_scores
    rows that don't have a prediction yet. Designed to be called periodically
    by the daemon.

If no current model exists yet, predict_pending() is a no-op and returns
zero work.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..storage import get_connection
from .external_features import attach_all_external_features
from .features import (FEATURE_NAMES, compute_corroboration_windows,
                       compute_insider_clusters, extract_features)
from .market_features import MarketFeatureCache
from .train import CURRENT_POINTER

log = logging.getLogger(__name__)


_PREDICTOR_LOCK = threading.Lock()
_PREDICTOR: Optional["ModelPredictor"] = None


@dataclass
class PredictionStats:
    candidates: int = 0
    predicted: int = 0
    skipped_no_model: bool = False


def get_predictor() -> Optional["ModelPredictor"]:
    """Return a cached predictor, or None if no model is deployed."""
    global _PREDICTOR
    with _PREDICTOR_LOCK:
        if _PREDICTOR is None or _PREDICTOR.is_stale():
            _PREDICTOR = ModelPredictor.try_load()
        return _PREDICTOR


class ModelPredictor:
    """Wraps a sklearn classifier + model metadata."""

    def __init__(self, model, version: str, model_path: Path,
                 feature_names: Optional[list[str]] = None):
        self.model = model
        self.version = version
        self.model_path = model_path
        # The exact feature list this specific model was trained on. Important
        # when ``features.FEATURE_NAMES`` has grown since the model was saved:
        # we subset X to only the columns the model knows, in the order it
        # expects. Falls back to the current FEATURE_NAMES list for models
        # whose meta we couldn't read.
        self.feature_names = feature_names or list(FEATURE_NAMES)
        self._loaded_mtime = model_path.stat().st_mtime if model_path.exists() else 0
        # Track the POINTER file's mtime too — when a retrain writes a new
        # model and updates current.json, the old file's mtime stays the
        # same so we'd never reload. Watching the pointer catches that.
        try:
            self._loaded_pointer_mtime = (CURRENT_POINTER.stat().st_mtime
                                           if CURRENT_POINTER.exists() else 0)
        except OSError:
            self._loaded_pointer_mtime = 0
        # Shared market-feature cache; warmed incrementally as new tickers
        # show up. Keeps yfinance calls amortised across predict cycles.
        self._market_cache = MarketFeatureCache()

    @classmethod
    def try_load(cls) -> Optional["ModelPredictor"]:
        if not CURRENT_POINTER.exists():
            return None
        try:
            pointer = json.loads(CURRENT_POINTER.read_text())
            model_path = Path(pointer["model_path"])
            version = pointer["version"]
            meta_path_str = pointer.get("meta_path")
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            log.warning("ML model pointer unreadable: %s", exc)
            return None
        if not model_path.exists():
            log.warning("ML model file missing: %s", model_path)
            return None
        try:
            import joblib  # type: ignore
            model = joblib.load(str(model_path))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load sklearn model %s: %s", model_path, exc)
            return None

        # Read the feature list this model was trained against, so we can
        # subset X correctly if the current FEATURE_NAMES has grown since.
        feature_names: Optional[list[str]] = None
        if meta_path_str:
            try:
                meta = json.loads(Path(meta_path_str).read_text())
                if isinstance(meta.get("feature_names"), list):
                    feature_names = [str(x) for x in meta["feature_names"]]
            except (OSError, json.JSONDecodeError) as exc:
                log.debug("Failed to read model meta %s: %s", meta_path_str, exc)

        log.info("Loaded ML model version=%s from %s (features=%d)",
                 version, model_path,
                 len(feature_names) if feature_names else len(FEATURE_NAMES))
        return cls(model, version, model_path, feature_names=feature_names)

    def is_stale(self) -> bool:
        """True if the pointer or model file has changed since load.

        Audit 2026-05-29: previous version only watched the model file's
        mtime, but a retrain writes a NEW file and updates the pointer —
        the old file's mtime never changes, so reload never triggered.
        Now also watches the pointer file's mtime.
        """
        if not CURRENT_POINTER.exists():
            return True
        try:
            ptr_mtime = CURRENT_POINTER.stat().st_mtime
            if ptr_mtime != self._loaded_pointer_mtime:
                return True
            return self.model_path.stat().st_mtime != self._loaded_mtime
        except OSError:
            return True

    def predict_one(self, row: dict[str, Any]) -> Optional[float]:
        try:
            import pandas as pd  # type: ignore
            features = extract_features(row)
            df = pd.DataFrame([features], columns=FEATURE_NAMES)
            # Subset/reorder to the model's training-time feature list. Any
            # features added since the model was trained get dropped here.
            df = df[[c for c in self.feature_names if c in df.columns]]
            proba = self.model.predict_proba(df)
            return float(proba[0, 1])
        except Exception as exc:  # noqa: BLE001
            log.warning("predict_one failed: %s", exc)
            return None

    def _enrich_market_features(self, rows: list[dict]) -> None:
        """Compute volume / vol / cross-asset features for each row."""
        from datetime import date, datetime, timedelta

        # Build the set of (normalized_ticker, signal_date) we need
        needed_tickers: set[str] = set()
        date_list: list[date] = []
        for r in rows:
            t = (r.get("ticker") or "").upper()
            if "_" in t:
                parts = t.split("_")
                if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
                    t = "_".join(parts[:-2]).replace("_", "-")
            if t:
                needed_tickers.add(t)
            ts = r.get("price_at_flag_ts") or r.get("published_at") or r.get("scored_at")
            if isinstance(ts, str):
                try:
                    date_list.append(datetime.strptime(ts[:10], "%Y-%m-%d").date())
                except ValueError:
                    pass

        if not needed_tickers or not date_list:
            return

        # Only fetch tickers we haven't cached yet
        missing = [t for t in needed_tickers if t not in self._market_cache.by_ticker]
        if missing:
            end = max(date_list) + timedelta(days=2)
            start = min(date_list) - timedelta(days=35)
            try:
                self._market_cache.warm(missing, start=start, end=end)
            except Exception as exc:  # noqa: BLE001
                log.warning("market feature warm failed: %s", exc)

        # Annotate each row
        for r in rows:
            t = (r.get("ticker") or "").upper()
            if "_" in t:
                parts = t.split("_")
                if len(parts) >= 3 and parts[-1] in {"EQ", "ETF", "STK"}:
                    t = "_".join(parts[:-2]).replace("_", "-")
            ts = r.get("price_at_flag_ts") or r.get("published_at") or r.get("scored_at")
            if not t or not isinstance(ts, str):
                continue
            try:
                d = datetime.strptime(ts[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            mf = self._market_cache.features_for(t, d)
            r["volume_ratio_5d_20d"] = mf.volume_ratio_5d_20d
            r["realized_vol_20d"]    = mf.realized_vol_20d
            r["spy_5d_return"]       = mf.spy_5d_return
            r["qqq_5d_return"]       = mf.qqq_5d_return
            r["vix_level"]           = mf.vix_level

    def predict_pending(self, *, batch_size: int = 5000) -> PredictionStats:
        """Fill ``model_p_5d`` on rows that don't have it yet."""
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            return PredictionStats(skipped_no_model=True)

        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT ss.id AS score_id, ss.ticker AS ticker,
                       COALESCE(lc.event_type, ss.event_type) AS event_type,
                       COALESCE(lc.sentiment, ss.sentiment) AS sentiment,
                       COALESCE(lc.sentiment_magnitude, ss.sentiment_magnitude) AS sentiment_magnitude,
                       COALESCE(lc.factual, ss.factual) AS factual,
                       lc.confidence AS llm_confidence,
                       CASE WHEN lc.id IS NOT NULL THEN 1 ELSE 0 END AS has_llm_classification,
                       ss.source_weight, ss.corroboration_count,
                       ss.author_quality, ss.anti_pump_flag, ss.composite_score,
                       ss.signal_class, ss.scored_at,
                       rs.id AS signal_id,
                       rs.source_tier, rs.published_at,
                       st.confidence AS ticker_confidence,
                       so.price_at_flag_ts
                FROM signal_scores ss
                JOIN raw_signals rs ON rs.id = ss.signal_id
                JOIN signal_tickers st ON st.signal_id = rs.id AND st.ticker = ss.ticker
                LEFT JOIN signal_outcomes so ON so.score_id = ss.id
                LEFT JOIN llm_classifications lc
                       ON lc.signal_id = ss.signal_id AND lc.ticker = ss.ticker
                WHERE ss.model_p_5d IS NULL
                ORDER BY ss.id DESC
                LIMIT ?
                """,
                (batch_size,),
            ).fetchall()

            if not rows:
                return PredictionStats(candidates=0, predicted=0)

            dicts = [dict(r) for r in rows]

            # Enrich each row with market-context features. We warm the cache
            # for any new tickers we haven't seen yet.
            self._enrich_market_features(dicts)
            # Insider-cluster feature (counts distinct Form-4 insiders in
            # 30d window for the same ticker). Cheap — one SQL pass.
            try:
                compute_insider_clusters(dicts)
            except Exception as exc:  # noqa: BLE001 — never block predictions
                log.debug("compute_insider_clusters failed: %s", exc)
            # Multi-source corroboration window (Tier 2 #19)
            try:
                compute_corroboration_windows(dicts)
            except Exception as exc:  # noqa: BLE001
                log.debug("compute_corroboration_windows failed: %s", exc)
            # External-data features (best-effort; defaults if staging
            # tables are empty)
            try:
                attach_all_external_features(dicts)
            except Exception as exc:  # noqa: BLE001
                log.debug("attach_all_external_features failed: %s", exc)

            X = pd.DataFrame(
                [extract_features(r) for r in dicts],
                columns=FEATURE_NAMES,
            )
            X = X[[c for c in self.feature_names if c in X.columns]]
            proba = self.model.predict_proba(X)[:, 1]

            for r, p in zip(dicts, proba):
                conn.execute(
                    "UPDATE signal_scores SET model_p_5d = ?, model_version = ? WHERE id = ?",
                    (float(p), self.version, r["score_id"]),
                )
                # Real-time notification: fire only on STRONG buy/sell.
                # Wrapped in try/except — a notify failure must NEVER
                # crash the predict loop or roll back the UPDATE above.
                try:
                    from market_radar.notifications.realtime import (
                        Notifier, AlertCandidate,
                    )
                    from market_radar.config import CONFIG as _CFG
                    p_val = float(p)
                    direction = None
                    action = None
                    if p_val >= _CFG.notify_buy_threshold:
                        direction, action = "buy", "STRONG_BUY"
                    elif p_val <= _CFG.notify_sell_threshold:
                        direction, action = "sell", "STRONG_SELL"
                    if direction is not None:
                        meta = conn.execute(
                            "SELECT rs.ingested_at, rs.title, rs.source "
                            "FROM signal_scores ss "
                            "JOIN raw_signals rs ON rs.id = ss.signal_id "
                            "WHERE ss.id = ?",
                            (r["score_id"],),
                        ).fetchone()
                        if meta:
                            Notifier().notify_if_eligible(AlertCandidate(
                                signal_id=int(r["score_id"]),
                                ticker=r["ticker"],
                                direction=direction, action=action, p=p_val,
                                title=meta["title"] or "",
                                source=meta["source"] or "?",
                                ingested_at=meta["ingested_at"] or "",
                            ))
                except Exception as exc:  # noqa: BLE001
                    log.warning("Notify failed for score_id=%s: %s",
                                r["score_id"], exc)
            return PredictionStats(candidates=len(rows), predicted=len(rows))
