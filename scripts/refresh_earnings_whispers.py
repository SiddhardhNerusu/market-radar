"""DEFERRED — EarningsWhispers.com serves only JS-rendered content.

History:
  EarningsWhispers publishes an unofficial buy-side whisper number that
  empirically out-performs the analyst consensus 70% of the time per
  their own 25-year data. The PEAD-feature on signals near earnings
  dates was meant to attach the most recent (whisper - consensus)
  divergence as a feature.

Why this script is now a no-op:
  As of 2026-05-14, https://www.earningswhispers.com/calendar serves
  a static HTML skeleton with no ticker/whisper/consensus data —
  the actual calendar is rendered client-side after JavaScript
  loads.  A 104 KB GET of the page yields the page chrome and a
  schema.org Article block, nothing usable for scraping.  Inline
  JS calls like ``togglecal('3')`` and ``adddownload('20260514','')``
  fetch data from internal endpoints we don't have credentials for.

Options for future revival:
  - Run a headless browser (Playwright / Selenium) and scrape the
    rendered DOM.  Cost: bundle + ~150MB Chromium + ~30s/run.
  - Subscribe to the EarningsWhispers paid API (~$50/mo).
  - Use a different free whisper source (Estimize, after a free-API
    application).

Per project rule for external parsers: "if the upstream page format
genuinely doesn't yield data, mark it as such in logs and move on —
DO NOT spend cycles trying to make a non-responsive endpoint work."

The earnings_whispers table is preserved (no rows deleted); the
feature attachment in ml/external_features.py degrades gracefully
when the table is empty.
"""
from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger("refresh_earnings_whispers")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.warning(
        "EarningsWhispers calendar serves only JS-rendered content "
        "(no usable static HTML as of 2026-05-14). Deferred — would "
        "need a headless-browser scrape. Exiting 0; "
        "earnings_whispers table untouched."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
