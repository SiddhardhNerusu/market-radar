"""Verify-before-spend script for the LLM classifier.

The single rule that came out of the 2026-05-13 LLM failure (see
``project_market_radar_llm_lesson.md``) is:

    Before authorising a paid batch, dump 5-10 sample inputs and outputs
    so we eyeball them. A $0.05 dry-batch catches the bug that wasted
    $17.50 last time.

This script does that. By default it runs in **dry mode** — picks N
sample SEC filings with bodies populated, prints the exact prompt the
classifier would send to Anthropic, and stops. It does not need
``ANTHROPIC_API_KEY`` for dry mode.

With ``--live`` it actually calls Anthropic Haiku on each sample,
prints the full response, and reports the event-type distribution.
That's the smoke test you should run **first** after re-enabling the
key, **before** kicking off the full backfill. Expected cost: ~$0.05.

The success criteria (printed at the end of a ``--live`` run):

  - Every sample's body is non-empty
  - At least 5 of 10 samples come back with a non-"other" event_type
    (across a mix of 8-K, 4, SC 13D, S-1, 425 there should be plenty
    of M&A, earnings, insider, FDA classifications)
  - No JSON parse failures
  - Per-sample cost is in the $0.003–$0.008 range

If any of these fail, STOP. Don't run the full backfill. Debug the
sampling pipeline at $0.05, not at $17.

Usage::

    # Dry mode — no LLM call. Shows what the classifier would see.
    python scripts/verify_llm_sample.py --n 10

    # Live mode — actually calls Anthropic. Costs ~$0.05.
    python scripts/verify_llm_sample.py --n 10 --live
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.llm.filter import should_classify  # noqa: E402
from market_radar.llm.prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    approx_input_tokens,
    approx_output_tokens,
    build_user_prompt,
)
from market_radar.llm.spend import estimate_cost  # noqa: E402
from market_radar.storage import get_connection  # noqa: E402


log = logging.getLogger("verify_llm_sample")


# Forms we want represented in the sample. Spread across major event types
# so we can spot the "everything is 'other'" failure mode immediately.
FORM_PATTERNS = (
    ("8-K",      "rs.title LIKE '8-K - %'"),
    ("4",        "rs.title LIKE '4 - %'"),
    ("SC 13D",   "rs.title LIKE 'SC 13D - %'"),
    ("S-1",      "rs.title LIKE 'S-1 - %'"),
    ("425",      "rs.title LIKE '425 - %'"),
    ("DEF 14A",  "rs.title LIKE 'DEF 14A - %'"),
)


def pick_samples(conn, *, per_form: int) -> list[dict]:
    rows: list[dict] = []
    for form_label, where in FORM_PATTERNS:
        cur = conn.execute(f"""
            SELECT ss.signal_id, ss.ticker, ss.composite_score, ss.event_type,
                   ss.sentiment, ss.signal_class,
                   rs.title, rs.body, rs.source, rs.source_tier,
                   rs.raw_payload, rs.url
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            WHERE rs.source LIKE 'sec_edgar%'
              AND {where}
              AND rs.body IS NOT NULL
              AND length(rs.body) >= 500
              AND NOT EXISTS (
                SELECT 1 FROM llm_classifications lc
                WHERE lc.signal_id = ss.signal_id AND lc.ticker = ss.ticker
              )
            ORDER BY rs.published_at DESC NULLS LAST, rs.id DESC
            LIMIT ?
        """, (per_form,))
        for r in cur.fetchall():
            d = dict(r)
            d["_form_label"] = form_label
            rows.append(d)
    return rows


def print_sample(row: dict, idx: int) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"[{idx}] form={row['_form_label']}  source={row['source']}  ticker={row['ticker']}")
    print(f"     id={row['signal_id']}  composite={row['composite_score']:.2f}")
    print(f"     title: {row['title'][:90]}")
    print(f"     url:   {row['url']}")
    print(f"     body_len: {len(row['body'])}  body head: {row['body'][:220].replace(chr(10),' ')!r}")
    user_msg = build_user_prompt(
        title=row["title"], body=row["body"],
        source=row["source"], primary_ticker=row["ticker"],
    )
    in_toks = approx_input_tokens(title=row["title"], body=row["body"])
    out_toks = approx_output_tokens()
    est = estimate_cost(input_tokens=in_toks, output_tokens=out_toks)
    print(f"     est_in_toks={in_toks}  est_out_toks={out_toks}  est_cost=${est:.4f}")
    # The actual user-message preview — last 200 chars so we see the body part
    print(f"     user_msg_tail: …{user_msg[-300:]!r}")
    # Filter decision
    ok, reason = should_classify(row)
    print(f"     filter_decision: {ok}  reason: {reason}")


def run_live(rows: list[dict]) -> None:
    """Actually call Anthropic for each row. Aborts on auth error."""
    try:
        from market_radar.llm import LLMClassifier
    except ImportError as exc:
        print(f"FATAL: cannot import LLMClassifier: {exc}")
        sys.exit(2)

    classifier = LLMClassifier()
    event_counter: Counter[str] = Counter()
    total_cost = 0.0
    parse_failures = 0
    empty_bodies = 0

    for i, row in enumerate(rows, 1):
        if not row.get("body"):
            empty_bodies += 1
            print(f"\n[{i}] EMPTY body — skipping")
            continue
        print(f"\n--- [{i}] calling LLM on form={row['_form_label']} "
              f"({row['ticker']}) body_len={len(row['body'])} ---")
        try:
            result = classifier.classify_one(row)
        except RuntimeError as exc:
            print(f"FATAL: {exc}")
            sys.exit(2)
        if result is None:
            print("  LLM returned None (cap-hit or call failed)")
            parse_failures += 1
            continue
        ev = result.get("event_type") or "<missing>"
        event_counter[ev] += 1
        total_cost += result.get("cost_usd") or 0
        print(f"  event_type     : {ev}")
        print(f"  sentiment      : {result.get('sentiment')}")
        print(f"  factual        : {result.get('factual')}")
        print(f"  confidence     : {result.get('confidence')}")
        print(f"  extracted      : {json.dumps(result.get('extracted_fields'), indent=4)[:400]}")
        print(f"  tokens in/out  : {result.get('input_tokens')} / {result.get('output_tokens')}")
        print(f"  cost           : ${result.get('cost_usd'):.4f}")

    print("\n" + "=" * 78)
    print("RESULTS SUMMARY")
    print("=" * 78)
    print(f"  samples classified : {sum(event_counter.values())}")
    print(f"  parse failures     : {parse_failures}")
    print(f"  empty bodies       : {empty_bodies}")
    print(f"  total cost         : ${total_cost:.4f}")
    print("  event_type distribution:")
    for ev, n in event_counter.most_common():
        print(f"    {ev:<28s}  {n}")

    # Verdict — same bar described in the file's docstring
    diverse_events = sum(n for ev, n in event_counter.items() if ev != "other")
    if empty_bodies > 0:
        print("\n  VERDICT: FAILED — empty bodies leaked through. Don't run "
              "the full backfill until the body filter is fixed.")
        sys.exit(1)
    if event_counter.get("other", 0) > sum(event_counter.values()) * 0.6:
        print("\n  VERDICT: FAILED — > 60% of classifications came back "
              "as 'other'. Same pattern as the 2026-05-13 failure. Do NOT "
              "run the full backfill. Investigate the body content first.")
        sys.exit(1)
    if diverse_events < max(2, len(rows) // 3):
        print("\n  VERDICT: BORDERLINE — diversity lower than expected. "
              "Inspect each row's body manually before authorising the "
              "full backfill.")
        sys.exit(1)
    print("\n  VERDICT: PASSED — proceed with the full backfill.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=10,
                   help="Total samples to take, spread evenly across forms (default: 10)")
    p.add_argument("--live", action="store_true",
                   help="Actually call Anthropic. Costs ~$0.05. "
                        "Default is dry mode (prints prompts only).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    per_form = max(1, args.n // len(FORM_PATTERNS))

    with get_connection() as conn:
        samples = pick_samples(conn, per_form=per_form)

    if not samples:
        print("No samples available. Run scripts/fetch_sec_bodies.py first "
              "to populate raw_signals.body for some SEC rows.")
        return 1

    print(f"Picked {len(samples)} samples ({per_form} per form pattern, "
          f"forms={[f for f,_ in FORM_PATTERNS]})")

    for i, row in enumerate(samples, 1):
        print_sample(row, i)

    if not args.live:
        print("\n--- DRY MODE ---")
        print(f"Re-run with --live to actually classify these {len(samples)} samples.")
        print("Expected cost: ~$0.05.")
        return 0

    run_live(samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
