"""SEC CIK → ticker lookup.

SEC publishes a JSON file mapping every CIK to one or more tickers. We
download it once and cache it on disk, refreshing weekly.

Endpoint: https://www.sec.gov/files/company_tickers.json
The file is small (~1 MB) and the SEC asks you set a descriptive User-Agent.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from ..config import PROJECT_ROOT

log = logging.getLogger(__name__)

CACHE_PATH = PROJECT_ROOT / "data" / "sec_company_tickers.json"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 1 week

# SEC requires a User-Agent that identifies the requester. Format per their
# guidance: "Name email@example.com". This is sent on all SEC requests.
DEFAULT_UA = "MARKET RADAR (research; redacted@example.com)"


class CikLookup:
    """Lazy, cached CIK→ticker map."""

    def __init__(
        self,
        cache_path: Path = CACHE_PATH,
        user_agent: str = DEFAULT_UA,
        ttl_seconds: int = CACHE_TTL_SECONDS,
    ):
        self.cache_path = cache_path
        self.user_agent = user_agent
        self.ttl_seconds = ttl_seconds
        self._map: dict[int, str] = {}
        self._name_map: dict[int, str] = {}

    # ------------------------------------------------------------------

    def get_ticker(self, cik: int) -> Optional[str]:
        if not self._map:
            self._load()
        return self._map.get(int(cik))

    def get_name(self, cik: int) -> Optional[str]:
        if not self._name_map:
            self._load()
        return self._name_map.get(int(cik))

    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._cache_valid():
            try:
                self._refresh()
            except Exception as exc:  # noqa: BLE001
                log.warning("CIK cache refresh failed; using stale cache if any: %s", exc)
                if not self.cache_path.exists():
                    self._map = {}
                    self._name_map = {}
                    return

        try:
            data = json.loads(self.cache_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("CIK cache read failed: %s", exc)
            self._map = {}
            self._name_map = {}
            return

        # SEC format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        m: dict[int, str] = {}
        names: dict[int, str] = {}
        if isinstance(data, dict):
            for row in data.values():
                if not isinstance(row, dict):
                    continue
                cik = row.get("cik_str")
                ticker = row.get("ticker")
                title = row.get("title")
                if isinstance(cik, int) and isinstance(ticker, str):
                    # First mapping wins (a CIK can have multiple share classes;
                    # we'd need a more nuanced match to pick the right one)
                    m.setdefault(cik, ticker)
                    if isinstance(title, str):
                        names.setdefault(cik, title)
        self._map = m
        self._name_map = names
        log.info("Loaded %d CIK→ticker mappings from %s", len(m), self.cache_path)

    def _cache_valid(self) -> bool:
        if not self.cache_path.exists():
            return False
        age = time.time() - self.cache_path.stat().st_mtime
        return age < self.ttl_seconds

    def _refresh(self) -> None:
        url = "https://www.sec.gov/files/company_tickers.json"
        log.info("Refreshing SEC CIK→ticker map from %s", url)
        resp = requests.get(
            url,
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(resp.text)


# Module-level singleton — cheap because everything is lazy.
CIK_LOOKUP = CikLookup()
