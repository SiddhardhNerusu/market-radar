"""SHAP feature attribution for the deployed model.

Computes mean(|SHAP|) per feature on the last fold's validation data
and writes a JSON summary that the dashboard can render as a bar chart.

Requires ``pip install shap`` (free). Skips gracefully if shap isn't
installed.

Output: ``data/models/shap_summary.json``
"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.config import PROJECT_ROOT  # noqa: E402

log = logging.getLogger("explain_model")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=500,
                   help="Number of validation rows to compute SHAP on")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        import joblib  # type: ignore
        import numpy as np  # type: ignore
        import shap  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError as exc:
        log.error("SHAP not installed (%s). Run: pip install shap", exc)
        return 1

    from market_radar.ml.train import (
        CURRENT_POINTER, _enrich_with_external_features,
        _enrich_with_market_features, _enrich_with_ta_and_graph_features,
        _load_training_rows,
    )
    from market_radar.ml.features import (
        FEATURE_NAMES, compute_corroboration_windows,
        compute_insider_clusters, extract_features_df,
    )

    if not CURRENT_POINTER.exists():
        log.error("No deployed model. Run train_ml.py first.")
        return 1
    pointer = json.loads(CURRENT_POINTER.read_text())
    model = joblib.load(pointer["model_path"])
    feat_names = list(FEATURE_NAMES)
    meta = json.loads(Path(pointer["meta_path"]).read_text())
    if isinstance(meta.get("feature_names"), list):
        feat_names = meta["feature_names"]

    rows = _load_training_rows()[-args.sample:]
    log.info("Computing SHAP on %d rows", len(rows))
    if not rows:
        return 1
    _enrich_with_market_features(rows)
    compute_insider_clusters(rows)
    compute_corroboration_windows(rows)
    _enrich_with_external_features(rows)
    _enrich_with_ta_and_graph_features(rows)
    X = extract_features_df(rows)
    X = X[[c for c in feat_names if c in X.columns]]

    # Calibrated wrappers don't have direct tree-shap support; fall back
    # to KernelExplainer on the underlying estimator if it's wrapped.
    inner = getattr(model, "calibrated_classifiers_", None)
    if inner:
        inner_model = inner[0].estimator
    else:
        inner_model = model

    try:
        explainer = shap.TreeExplainer(inner_model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class
    except Exception as exc:
        log.warning("TreeExplainer failed (%s). Falling back to KernelExplainer.", exc)
        sample = X.sample(min(50, len(X)), random_state=42)
        explainer = shap.KernelExplainer(inner_model.predict_proba, sample)
        # KernelExplainer is slow — sample for tractability
        X_eval = X.sample(min(100, len(X)), random_state=42)
        shap_vals = explainer.shap_values(X_eval)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        X = X_eval  # align column index with the sampled rows

    # Normalise to (n_samples, n_features). Newer SHAP returns 3-D
    # (n_samples, n_features, n_classes) from KernelExplainer+predict_proba.
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[..., 1]   # positive class

    mean_abs = np.abs(shap_vals).mean(axis=0)
    # Defensive: collapse any remaining axis so we get one scalar per feature
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=-1)
    pairs = sorted(zip(X.columns, mean_abs.tolist()), key=lambda kv: -kv[1])

    out = {
        "model_version": pointer.get("version"),
        "computed_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "sample_size": len(rows),
        "feature_importance": [
            {"feature": str(f), "mean_abs_shap": float(v)}
            for f, v in pairs
        ],
    }
    out_path = PROJECT_ROOT / "data" / "models" / "shap_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)
    log.info("Top 10 features by |SHAP|:")
    for f, v in pairs[:10]:
        log.info("  %-32s  %.4f", f, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
