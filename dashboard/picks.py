"""Pick-ranking endpoints for the action-oriented dashboard.

Read-only — never writes to the DB. Imports model-loading code from
``market_radar.ml.*`` but does not modify it.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market_radar.db"

# Make src/ importable so we can pull feature-extraction + projection code.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger("marketradar.dashboard.picks")


# Locked thresholds (per spec).
BUY_WATCH = 0.60
BUY_BUY = 0.67
BUY_STRONG = 0.75
SELL_WATCH = 0.40
SELL_SELL = 0.33
SELL_STRONG = 0.25

# Bound the candidate pool fed to the on-the-fly graph_only predictor.
_MAX_RE_PREDICT = 60


def action_label_for(p: Optional[float], direction: str) -> str:
    """Map a probability + direction to a UI action label."""
    if p is None:
        return "HOLD"
    direction = (direction or "buy").lower()
    if direction == "buy":
        if p >= BUY_STRONG:
            return "STRONG_BUY"
        if p >= BUY_BUY:
            return "BUY"
        if p >= BUY_WATCH:
            return "WATCH_BUY"
        return "HOLD"
    # sell
    if p <= SELL_STRONG:
        return "STRONG_SELL"
    if p <= SELL_SELL:
        return "SELL"
    if p <= SELL_WATCH:
        return "WATCH_SELL"
    return "HOLD"


def _connect_ro() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _p_col(horizon: int) -> str:
    return {1: "model_p_1d", 5: "model_p_5d", 20: "model_p_20d"}.get(
        int(horizon), "model_p_5d"
    )


_MODELS_CACHE: Optional[dict[str, Any]] = None


def _models() -> dict[str, Any]:
    """Load main + graph_only models once and cache."""
    global _MODELS_CACHE
    if _MODELS_CACHE is not None:
        return _MODELS_CACHE
    import joblib
    out: dict[str, Any] = {}
    for key, fname in (("main", "current.json"),
                       ("graph_only", "current_graph_only.json")):
        ptr_path = ROOT / "data" / "models" / fname
        if not ptr_path.exists():
            continue
        try:
            ptr = json.loads(ptr_path.read_text())
            meta_path = Path(ptr.get("meta_path", ""))
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            model_path = ptr.get("model_path")
            out[key] = {
                "model": joblib.load(model_path) if model_path else None,
                "meta": meta,
                "feature_names": meta.get("feature_names") or [],
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("picks: failed to load %s model: %s", key, exc)
    _MODELS_CACHE = out
    return out


def _enrich_rows(rows: list[dict]) -> bool:
    """Attach features needed for re-prediction. In-place.

    Returns True on success, False on hard failure.
    """
    if not rows:
        return True
    try:
        from market_radar.ml.features import (
            compute_insider_clusters,
            compute_corroboration_windows,
        )
        from market_radar.ml.external_features import attach_all_external_features
        from market_radar.ml.calendar_features import attach_calendar_features

        compute_insider_clusters(rows)
        compute_corroboration_windows(rows)
        attach_all_external_features(rows)
        attach_calendar_features(rows)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("picks: enrichment failed: %s", exc)
        return False


def _predict_batch(rows: list[dict], model_key: str) -> list[Optional[float]]:
    """Predict for a batch of pre-enriched rows. Returns None per row on failure."""
    if not rows:
        return []
    try:
        import pandas as pd
        from market_radar.ml.features import FEATURE_NAMES, extract_features
        models = _models()
        if model_key not in models or models[model_key].get("model") is None:
            return [None] * len(rows)
        m = models[model_key]
        feature_names = m["feature_names"]
        X: list[list[float]] = []
        keep_idx: list[int] = []
        for i, r in enumerate(rows):
            try:
                fv = extract_features(r)
                fmap = dict(zip(FEATURE_NAMES, fv))
                X.append([float(fmap.get(name, 0.0) or 0.0) for name in feature_names])
                keep_idx.append(i)
            except Exception as exc:  # noqa: BLE001
                log.debug("picks: feature extract failed row %d: %s", i, exc)
        if not X:
            return [None] * len(rows)
        Xdf = pd.DataFrame(X, columns=feature_names)
        ps = m["model"].predict_proba(Xdf)[:, 1]
        out: list[Optional[float]] = [None] * len(rows)
        for j, idx in enumerate(keep_idx):
            out[idx] = float(ps[j])
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("picks: batch predict for %s failed: %s", model_key, exc)
        return [None] * len(rows)


def build_picks(
    direction: str = "buy",
    limit: int = 20,
    horizon: int = 5,
    hours_window: int = 48,
    model_source: str = "main",
) -> list[dict]:
    """Return ranked picks for the dashboard.

    See module docstring for arg semantics.
    """
    direction = (direction or "buy").lower()
    if direction not in ("buy", "sell"):
        direction = "buy"
    model_source = (model_source or "main").lower()
    if model_source not in ("main", "graph_only"):
        model_source = "main"
    horizon = int(horizon) if int(horizon) in (1, 5, 20) else 5
    pcol = _p_col(horizon)
    hours = max(1, int(hours_window))
    limit = max(1, min(int(limit), 100))

    # Conformal width for the main model — used for the displayed interval.
    main_qhat = None
    try:
        main_qhat = float(_models().get("main", {}).get("meta", {}).get(
            "conformal_qhat", 0.0
        ))
    except Exception:
        main_qhat = None

    # Pull candidates.  For model=main we can prefilter on probability;
    # for model=graph_only we cannot, so we use composite_score as a coarse
    # initial ranking and re-predict the top _MAX_RE_PREDICT.
    sql_filter = ""
    if model_source == "main":
        if direction == "buy":
            sql_filter = f"AND ss.{pcol} >= {BUY_WATCH}"
        else:
            sql_filter = f"AND ss.{pcol} <= {SELL_WATCH}"

    con = _connect_ro()
    try:
        rows = con.execute(
            f"""
            SELECT ss.id AS score_id, ss.signal_id, ss.ticker, ss.event_type,
                   ss.sentiment, ss.sentiment_magnitude, ss.factual,
                   ss.source_weight, ss.corroboration_count, ss.author_quality,
                   ss.anti_pump_flag, ss.composite_score, ss.signal_class,
                   ss.scored_at,
                   ss.model_p_1d, ss.model_p_5d, ss.model_p_20d,
                   ss.model_version, ss.model_bucket,
                   rs.title, rs.url, rs.source, rs.source_tier,
                   rs.published_at, rs.author
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            WHERE rs.ingested_at >= datetime('now', '-{hours} hours')
              AND rs.source NOT LIKE 'sec_edgar_backfill_%'
              AND ss.{pcol} IS NOT NULL
              {sql_filter}
            ORDER BY ss.composite_score DESC, ss.id DESC
            LIMIT 200
            """,
        ).fetchall()
    finally:
        con.close()

    candidates = [dict(r) for r in rows]
    if not candidates:
        return []

    # Bound re-prediction work
    rerank_pool = candidates[:_MAX_RE_PREDICT]

    # Build features once (used for any re-prediction we do).
    p_graph_list: list[Optional[float]] = [None] * len(rerank_pool)
    if model_source == "graph_only" or True:
        # Always try to compute graph_only for agreement.  If enrichment
        # fails, we just leave agreement as None.
        if _enrich_rows(rerank_pool):
            p_graph_list = _predict_batch(rerank_pool, "graph_only")

    # Pick the primary p per candidate.
    enriched: list[tuple[float, dict, Optional[float], Optional[float]]] = []
    for i, c in enumerate(rerank_pool):
        p_main = c.get(pcol)
        p_graph = p_graph_list[i] if i < len(p_graph_list) else None
        if model_source == "graph_only":
            p_primary = p_graph
        else:
            p_primary = float(p_main) if p_main is not None else None
        if p_primary is None:
            continue
        # Direction filter (post re-prediction for graph_only)
        if direction == "buy" and p_primary < BUY_WATCH:
            continue
        if direction == "sell" and p_primary > SELL_WATCH:
            continue
        enriched.append((float(p_primary), c, p_main, p_graph))

    # Per-ticker dedup — most extreme p wins.
    by_ticker: dict[str, tuple[float, dict, Optional[float], Optional[float]]] = {}
    for p_primary, c, p_main, p_graph in enriched:
        t = (c.get("ticker") or "").upper()
        if not t:
            continue
        cur = by_ticker.get(t)
        if cur is None:
            by_ticker[t] = (p_primary, c, p_main, p_graph)
        else:
            cur_p = cur[0]
            if direction == "buy" and p_primary > cur_p:
                by_ticker[t] = (p_primary, c, p_main, p_graph)
            elif direction == "sell" and p_primary < cur_p:
                by_ticker[t] = (p_primary, c, p_main, p_graph)

    deduped = list(by_ticker.values())
    # Sort
    deduped.sort(key=lambda x: x[0], reverse=(direction == "buy"))
    deduped = deduped[:limit]

    out: list[dict] = []
    for p_primary, c, p_main, p_graph in deduped:
        agreement: Optional[float] = None
        if p_main is not None and p_graph is not None:
            try:
                agreement = 1.0 - abs(float(p_main) - float(p_graph))
            except Exception:
                agreement = None

        interval_lo: Optional[float] = None
        interval_hi: Optional[float] = None
        if main_qhat is not None and main_qhat > 0:
            interval_lo = max(0.0, p_primary - main_qhat)
            interval_hi = min(1.0, p_primary + main_qhat)

        out.append({
            "signal_id": int(c["score_id"]),
            "raw_signal_id": int(c["signal_id"]),
            "ticker": (c.get("ticker") or "").upper(),
            "action": action_label_for(p_primary, direction),
            "p": p_primary,
            "p_calibrated_interval_lo": interval_lo,
            "p_calibrated_interval_hi": interval_hi,
            "p_main_model": float(p_main) if p_main is not None else None,
            "p_graph_only_model": float(p_graph) if p_graph is not None else None,
            "event_type": c.get("event_type"),
            "source": c.get("source"),
            "source_tier": c.get("source_tier"),
            "scored_at": c.get("scored_at"),
            "published_at": c.get("published_at"),
            "title": c.get("title"),
            "url": c.get("url"),
            "composite_score": c.get("composite_score"),
            "signal_class": c.get("signal_class"),
            "multi_model_agreement": agreement,
        })
    return out


def build_pick_detail(signal_id: int) -> dict:
    """Return the full 'why' panel for one signal (signal_id = signal_scores.id)."""
    signal_id = int(signal_id)
    con = _connect_ro()
    try:
        row = con.execute(
            """
            SELECT ss.id AS score_id, ss.signal_id, ss.ticker, ss.event_type,
                   ss.sentiment, ss.sentiment_magnitude, ss.factual,
                   ss.source_weight, ss.corroboration_count, ss.author_quality,
                   ss.anti_pump_flag, ss.composite_score, ss.signal_class,
                   ss.scored_at,
                   ss.model_p_1d, ss.model_p_5d, ss.model_p_20d,
                   ss.model_version, ss.model_bucket,
                   rs.title, rs.url, rs.source, rs.source_tier,
                   rs.published_at, rs.author, rs.body
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            WHERE ss.id = ?
            """,
            (signal_id,),
        ).fetchone()
        if not row:
            return {"error": f"signal_id {signal_id} not found", "signal_id": signal_id}
        sig = dict(row)
        ticker = (sig.get("ticker") or "").upper()

        recent_news = [dict(r) for r in con.execute(
            """
            SELECT rs.id, rs.title, rs.url, rs.source, rs.source_tier,
                   rs.published_at,
                   ss.composite_score, ss.event_type, ss.sentiment,
                   ss.model_p_5d
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            WHERE ss.ticker = ?
              AND rs.published_at >= datetime('now', '-7 days')
              AND ss.id != ?
            ORDER BY rs.published_at DESC
            LIMIT 5
            """,
            (ticker, signal_id),
        ).fetchall()]

        insider_30d = [dict(r) for r in con.execute(
            """
            SELECT insider_name, officer_title, transaction_code, shares,
                   price, is_acquired, is_officer, is_director, is_10pct,
                   role_score, report_date
            FROM insider_transactions
            WHERE ticker = ? AND report_date >= date('now', '-30 days')
            ORDER BY report_date DESC
            LIMIT 20
            """,
            (ticker,),
        ).fetchall()]

        upcoming_catalysts = [dict(r) for r in con.execute(
            """
            SELECT decision_date, catalyst_type, description, source
            FROM catalysts
            WHERE ticker = ?
              AND decision_date >= date('now')
              AND decision_date <= date('now', '+60 days')
            ORDER BY decision_date ASC
            """,
            (ticker,),
        ).fetchall()]

        earnings_row = con.execute(
            """
            SELECT report_date, eps_actual, eps_estimate, eps_surprise_pct,
                   revenue_actual, revenue_estimate, revenue_surprise_pct
            FROM earnings_data
            WHERE ticker = ?
            ORDER BY report_date DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
        earnings_data = dict(earnings_row) if earnings_row else None
    finally:
        con.close()

    # Live re-predictions for this single signal.
    all_models: dict[str, Any] = {
        "stored_main_p_5d": sig.get("model_p_5d"),
        "stored_main_p_1d": sig.get("model_p_1d"),
        "stored_main_p_20d": sig.get("model_p_20d"),
        "stored_model_version": sig.get("model_version"),
        "stored_model_bucket": sig.get("model_bucket"),
    }
    rows_for_pred = [dict(sig)]
    if _enrich_rows(rows_for_pred):
        for key in ("main", "graph_only"):
            ps = _predict_batch(rows_for_pred, key)
            all_models[key] = {"p": ps[0] if ps else None}
    else:
        all_models["main"] = {"p": None}
        all_models["graph_only"] = {"p": None}

    shap_top10: list[dict] = []
    sp = ROOT / "data" / "models" / "shap_summary.json"
    if sp.exists():
        try:
            shap_data = json.loads(sp.read_text())
            shap_top10 = shap_data.get("feature_importance", [])[:10]
        except Exception:
            pass

    comparable_setups: dict[str, Any] = {}
    sc = sig.get("signal_class")
    if sc:
        try:
            from market_radar.storage import get_connection
            from market_radar.scoring.projections import project_for_class
            with get_connection() as gc:
                proj = project_for_class(gc, sc)
            if proj is not None:
                def _dist(d: Any) -> Optional[dict]:
                    if d is None:
                        return None
                    return {
                        "window_days": d.window_days,
                        "samples": d.samples,
                        "median_return_pct": d.median,
                        "p25_return_pct": d.p25,
                        "p75_return_pct": d.p75,
                        "p10_return_pct": d.p10,
                        "p90_return_pct": d.p90,
                        "hit_rate": d.hit_rate,
                    }
                comparable_setups = {
                    "signal_class": proj.signal_class,
                    "used_relaxed_match": proj.used_relaxed,
                    "one_day": _dist(proj.one_day),
                    "five_day": _dist(proj.five_day),
                    "twenty_day": _dist(proj.twenty_day),
                }
        except Exception as exc:  # noqa: BLE001
            comparable_setups = {"error": str(exc)}

    return {
        "signal_id": signal_id,
        "ticker": ticker,
        "source": {
            "title": sig.get("title"),
            "url": sig.get("url"),
            "source_name": sig.get("source"),
            "source_tier": sig.get("source_tier"),
            "published_at": sig.get("published_at"),
            "author": sig.get("author"),
        },
        "signal": {
            "event_type": sig.get("event_type"),
            "sentiment": sig.get("sentiment"),
            "sentiment_magnitude": sig.get("sentiment_magnitude"),
            "factual": sig.get("factual"),
            "composite_score": sig.get("composite_score"),
            "signal_class": sig.get("signal_class"),
            "scored_at": sig.get("scored_at"),
        },
        "recent_news": recent_news,
        "insider_30d": insider_30d,
        "upcoming_catalysts": upcoming_catalysts,
        "earnings_data": earnings_data,
        "all_models": all_models,
        "shap_top10": shap_top10,
        "comparable_setups": comparable_setups,
    }


def cfd_status() -> dict[str, Any]:
    """Report whether T212 CFD credentials are configured (env-only check)."""
    import os
    return {
        "cfd_configured": bool(os.getenv("T212_CFD_API_KEY")),
        "paper_only_message": "PAPER ONLY — no CFD account configured",
    }
