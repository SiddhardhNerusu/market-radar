"""Compute the deployed model's rolling-window AUC and update the
Page-Hinkley drift statistic. Call from cron weekly (or after the daily
outcome resolution job).

Usage::

    python scripts/check_model_drift.py
    python scripts/check_model_drift.py --window 1000 --lambda 0.05
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.ml.drift import (  # noqa: E402
    DEFAULT_DELTA, DEFAULT_LAMBDA, DEFAULT_WINDOW,
    compute_and_record_drift,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--delta",  type=float, default=DEFAULT_DELTA)
    p.add_argument("--lambda", dest="lambda_", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = compute_and_record_drift(
        window=args.window, delta=args.delta, lambda_=args.lambda_,
    )

    print("--- Drift check ---")
    print(f"observed_at:    {result.observed_at}")
    print(f"model_version:  {result.model_version}")
    print(f"window_n:       {result.window_n}")
    print(f"rolling_auc:    {result.rolling_auc}")
    print(f"ph_stat:        {result.ph_stat:.4f}")
    print(f"alerted:        {result.alerted}")
    print(f"reason:         {result.reason}")
    return 1 if result.alerted else 0


if __name__ == "__main__":
    sys.exit(main())
