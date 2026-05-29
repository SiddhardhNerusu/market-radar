"""Dump the KEY NAMES (not values) returned by each T212 endpoint, so we
can map them correctly in client.py without ever logging the actual values.

This script prints only:
  - JSON keys
  - value types (str/int/float/bool/null/list/dict)
  - list lengths
  - SHA-256 fingerprints for any string value (never the value itself)

Run after test_t212_connection.py succeeds, just once, to figure out the
real T212 response shape.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_radar.t212 import available_clients  # noqa: E402


def _shape(value: Any, depth: int = 0, max_depth: int = 4) -> Any:
    """Return a redacted representation: types + keys + lengths, never values."""
    if depth > max_depth:
        return "<truncated>"
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        # show length and a short fingerprint so we can distinguish ticker vs name
        h = hashlib.sha256(value.encode("utf-8")).hexdigest()[:6]
        return f"str(len={len(value)},sha={h})"
    if isinstance(value, list):
        if not value:
            return "[](len=0)"
        return [
            f"[](len={len(value)})",
            _shape(value[0], depth + 1, max_depth),
        ]
    if isinstance(value, dict):
        return {k: _shape(v, depth + 1, max_depth) for k, v in value.items()}
    return f"<{type(value).__name__}>"


def main() -> int:
    clients = available_clients()
    if not clients:
        print("No T212 credentials configured. Run test_t212_connection.py first.")
        return 1

    client = clients[0]
    print(f"Inspecting raw response shape via {client.account_type} account.\n")

    # account_summary
    try:
        resp = client._request("GET", "/equity/account/summary")
        print("/equity/account/summary keys:")
        print(_shape(resp))
        print()
    except Exception as exc:
        print(f"/equity/account/summary FAILED: {exc}")
        print()

    # account_cash
    try:
        resp = client._request("GET", "/equity/account/cash")
        print("/equity/account/cash keys:")
        print(_shape(resp))
        print()
    except Exception as exc:
        print(f"/equity/account/cash FAILED: {exc}")
        print()

    # positions — show first item's shape only
    try:
        resp = client._request("GET", "/equity/positions") or []
        print(f"/equity/positions: list of {len(resp)} item(s)")
        if resp:
            print("First position keys:")
            print(_shape(resp[0]))
        print()
    except Exception as exc:
        print(f"/equity/positions FAILED: {exc}")
        print()

    # open orders
    try:
        resp = client._request("GET", "/equity/orders") or []
        print(f"/equity/orders: list of {len(resp)} item(s)")
        if resp:
            print("First order keys:")
            print(_shape(resp[0]))
        print()
    except Exception as exc:
        print(f"/equity/orders FAILED: {exc}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
