"""Hourly counterfactual tracker for FILLED option spreads.

For every spread that actually filled (bot_option_spreads.filled_at IS NOT NULL)
and hasn't expired, re-price its two legs at the current market mid and log what
the spread WOULD be worth now if we'd held it — alongside what we actually
realized on exit. This answers "was cutting/de-risking the options the right
call, or would they have recovered?".

Appends one row per spread per run to data/option_counterfactuals.csv.
Runs hourly via com.marketradar.opttrack.plist.
Standalone:  .venv/bin/python scripts/track_option_counterfactuals.py
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from market_radar.config import CONFIG
from market_radar.storage import get_connection
from market_radar.execution.alpaca_client import AlpacaClient
from market_radar.execution.options.alpaca_options import AlpacaOptionsClient

CSV_PATH = os.path.join(CONFIG.project_root, "data", "option_counterfactuals.csv")


def main() -> int:
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()

    with get_connection() as conn:
        spreads = conn.execute(
            """SELECT id, underlying, direction, long_strike, short_strike,
                      expiration_date, contracts, entry_debit_usd, status,
                      closed_at, realized_pnl_usd, exit_reason
               FROM bot_option_spreads
               WHERE filled_at IS NOT NULL AND expiration_date >= ?
               ORDER BY id""",
            (today,),
        ).fetchall()
        legs_by: dict[int, dict[str, str]] = {}
        for s in spreads:
            legs = conn.execute(
                "SELECT role, contract_symbol FROM bot_option_legs WHERE spread_id=?",
                (s["id"],),
            ).fetchall()
            legs_by[s["id"]] = {l["role"]: l["contract_symbol"] for l in legs}

    if not spreads:
        print(f"[opttrack] {now:%Y-%m-%d %H:%M}Z — no filled, un-expired spreads to track")
        return 0

    opts = AlpacaOptionsClient(AlpacaClient())

    rows: list[dict] = []
    for s in spreads:
        legs = legs_by.get(s["id"], {})
        long_sym, short_sym = legs.get("long"), legs.get("short")
        if not long_sym or not short_sym:
            continue
        long_mid = short_mid = 0.0
        have_quote = False
        try:
            snaps = opts.get_snapshots([long_sym, short_sym])
            lq, sq = snaps.get(long_sym), snaps.get(short_sym)
            if lq is not None and sq is not None:
                long_mid = lq.effective_mid
                short_mid = sq.effective_mid
                have_quote = long_mid > 0 and short_mid > 0
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] snapshot failed for spread {s['id']} ({s['underlying']}): {exc}")

        contracts = int(s["contracts"])
        entry = float(s["entry_debit_usd"] or 0.0)
        our_pnl = round(float(s["realized_pnl_usd"] or 0.0), 2)
        if have_quote:
            cur_val = max(long_mid - short_mid, 0.0)            # per-spread $ value now
            would_be = round((cur_val - entry) * contracts * 100, 2)  # P&L if still held
            delta = round(would_be - our_pnl, 2)               # +ve = holding beats our exit
        else:
            cur_val = would_be = delta = None                  # no quote this tick

        rows.append({
            "checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "spread_id": s["id"],
            "underlying": s["underlying"],
            "spread": f"{s['long_strike']:.0f}/{s['short_strike']:.0f}{s['direction'][:1].upper()}",
            "exp": s["expiration_date"],
            "contracts": contracts,
            "entry_debit": round(entry, 2),
            "cur_value": cur_val,
            "would_be_pnl": would_be,
            "status": s["status"],
            "exit_reason": s["exit_reason"] or "",
            "our_exit_pnl": our_pnl,
            "hold_minus_actual": delta,
        })

    if rows:
        new_file = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new_file:
                w.writeheader()
            w.writerows(rows)

    print(f"[opttrack] {now:%Y-%m-%d %H:%M}Z — logged {len(rows)} spread(s) -> {CSV_PATH}")
    for r in rows:
        if r["would_be_pnl"] is None:
            print(f"  {r['underlying']:5} {r['spread']:9}  (no quote this tick)")
            continue
        verdict = ("holding BETTER" if (r["hold_minus_actual"] or 0) > 0
                   else "cutting BETTER/equal")
        print(f"  {r['underlying']:5} {r['spread']:9} would-be P&L ${r['would_be_pnl']:+8.0f}"
              f"  | we exited ${r['our_exit_pnl']:+8.0f}  -> {verdict} by ${abs(r['hold_minus_actual'] or 0):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
