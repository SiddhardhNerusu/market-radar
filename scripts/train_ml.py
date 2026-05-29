"""Train MARKET RADAR ML models on resolved outcomes.

Default: trains a single GLOBAL 5-day-horizon model with walk-forward CV
+ isotonic calibration + conformal interval. This is the model the
daemon's predict.py loads.

With ``--multi-horizon``, also trains separate 1-day and 20-day horizon
models so the dashboard can rank fast-movers vs slow-burn events.

With ``--per-event-type``, also trains one model per event bucket
(``ma``, ``earnings``, ``fda``, ``insider``, ``activist``, ``macro``).
Predict-time routing falls back to the global model when a bucket has
too little data.

Usage::

    python scripts/train_ml.py                                  # default
    python scripts/train_ml.py --multi-horizon                  # + 1d + 20d
    python scripts/train_ml.py --per-event-type                 # + per-bucket
    python scripts/train_ml.py --multi-horizon --per-event-type # everything
    python scripts/train_ml.py --always-deploy                  # override AUC gate

The new model is only deployed if its median walk-forward AUC is within
0.5 percentage points of (or better than) the previous deployed model
unless ``--always-deploy`` is passed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.ml.train import (  # noqa: E402
    train_and_save, train_multi_horizon, train_per_event_type,
)
from market_radar.ml.graph_only_train import train_graph_only  # noqa: E402
from market_radar.ml.ensemble import train_ensemble  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--min-rows", type=int, default=500,
                   help="Minimum labelled rows required to train (default 500)")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Walk-forward CV folds (default 5)")
    p.add_argument("--always-deploy", action="store_true",
                   help="Deploy even if median val AUC regresses vs previous")
    p.add_argument("--multi-horizon", action="store_true",
                   help="Also train 1d + 20d horizon models")
    p.add_argument("--per-event-type", action="store_true",
                   help="Also train per-event-type bucket models")
    p.add_argument("--horizons-only", action="store_true",
                   help="Skip the global default; only run multi-horizon")
    p.add_argument("--bucket-min-rows", type=int, default=300,
                   help="Min rows per event bucket to train a bucket model")
    p.add_argument("--graph-only", action="store_true",
                   help="Also train the graph-only model")
    p.add_argument("--ensemble", action="store_true",
                   help="Also train the stacked ensemble")
    p.add_argument("--all", action="store_true",
                   help="Shorthand for --multi-horizon --per-event-type --graph-only --ensemble")
    args = p.parse_args()
    if args.all:
        args.multi_horizon = True
        args.per_event_type = True
        args.graph_only = True
        args.ensemble = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("train_ml")

    overall_ok = True
    summary: dict = {}

    if not args.horizons_only:
        log.info("=== Training GLOBAL 5d model ===")
        result = train_and_save(
            min_rows=args.min_rows,
            n_splits=args.n_splits,
            only_replace_if_better=not args.always_deploy,
        )
        log.info("Global result: %s", result.to_dict())
        summary["global"] = result.to_dict()
        overall_ok = overall_ok and result.success

    if args.multi_horizon:
        log.info("\n=== Training MULTI-HORIZON models ===")
        horizons = (1, 5, 20) if args.horizons_only else (1, 20)
        mh = train_multi_horizon(horizons=horizons, n_splits=args.n_splits,
                                 min_rows=args.min_rows)
        summary["multi_horizon"] = {h: r.to_dict() for h, r in mh.items()}
        for h, r in mh.items():
            log.info("  h=%dd: success=%s val_auc=%s",
                     h, r.success, r.val_auc)
            overall_ok = overall_ok and r.success

    if args.per_event_type:
        log.info("\n=== Training PER-EVENT-TYPE models ===")
        pet = train_per_event_type(min_rows_per_bucket=args.bucket_min_rows,
                                   n_splits=args.n_splits)
        summary["per_event_type"] = {
            bucket: {h: r.to_dict() for h, r in by_h.items()}
            for bucket, by_h in pet.items()
        }
        for bucket, by_h in pet.items():
            for h, r in by_h.items():
                log.info("  bucket=%s h=%dd: success=%s val_auc=%s",
                         bucket, h, r.success, r.val_auc)
                overall_ok = overall_ok and r.success

    if args.graph_only:
        log.info("\n=== Training GRAPH-ONLY model ===")
        gr = train_graph_only(min_rows=args.min_rows, n_splits=args.n_splits)
        log.info("Graph-only: %s", gr.to_dict())
        summary["graph_only"] = gr.to_dict()
        overall_ok = overall_ok and gr.success

    if args.ensemble:
        log.info("\n=== Training STACKED ENSEMBLE ===")
        er = train_ensemble(min_rows=args.min_rows, n_splits=args.n_splits)
        log.info("Ensemble: %s", er.to_dict())
        summary["ensemble"] = er.to_dict()
        overall_ok = overall_ok and er.success

    print()
    print("=== Training summary ===")
    print(json.dumps(summary, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
