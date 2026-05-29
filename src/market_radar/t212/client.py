"""Trading 212 Public API client.

HTTP Basic auth with an (API_KEY, API_SECRET) pair. T212 issues a separate
pair per account (Invest vs Stocks ISA), so one ``T212Client`` instance is
bound to a single account type.

References:
  - https://docs.trading212.com/api/section/general-information/quickstart
  - https://docs.trading212.com/api/section/authentication
  - https://t212public-api-docs.redoc.ly/

Design notes:
  - All public methods raise ``T212Error`` (or a subclass) on failure with the
    response body attached so callers can log-and-continue or abort cleanly.
  - Rate limiting: T212 documents per-endpoint limits and returns
    ``x-ratelimit-*`` headers + a 429 with ``Retry-After`` when exceeded. We
    honour ``Retry-After`` and otherwise back off exponentially up to
    ``max_retries`` times.
  - We never log the API credentials. Only an opaque SHA-256 fingerprint.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import time
from dataclasses import asdict
from typing import Any, Literal, Optional

import requests
from requests.auth import HTTPBasicAuth

from ..config import CONFIG
from .types import (
    AccountCash,
    AccountCashBreakdown,
    AccountInfo,
    AccountInvestments,
    HistoricalOrder,
    Instrument,
    Order,
    Position,
    WalletImpact,
)

log = logging.getLogger(__name__)

AccountType = Literal["invest", "isa"]


class T212Error(Exception):
    """Raised when the Trading 212 API returns an error or the request fails."""

    def __init__(self, message: str, status: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class T212AuthError(T212Error):
    """Raised on 401/403 — credentials missing, wrong, or lacking required scope."""


class T212RateLimitError(T212Error):
    """Raised when we exhaust retries on a 429."""


def _fingerprint(value: str) -> str:
    """Stable, non-reversible identifier for log messages. Never logs the secret."""
    if not value:
        return "<empty>"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:8]}"


class T212Client:
    """Read-and-write client for one Trading 212 account."""

    DEFAULT_TIMEOUT = 15  # seconds

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        account_type: AccountType,
        *,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ):
        if not api_key or not api_secret:
            raise T212AuthError(
                f"Missing T212 credentials for account '{account_type}'. "
                "Set both T212_{ACCOUNT}_API_KEY and T212_{ACCOUNT}_API_SECRET in .env."
            )
        self.api_key = api_key
        self.api_secret = api_secret
        self.account_type = account_type
        self.base_url = (base_url or CONFIG.t212_base_url).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session or requests.Session()
        self._auth = HTTPBasicAuth(api_key, api_secret)
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "market-radar/0.1 (+local-only)",
            }
        )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def for_account(cls, account_type: AccountType) -> "T212Client":
        """Build a client using the credentials from .env."""
        if account_type == "invest":
            key = CONFIG.t212_invest_api_key
            secret = CONFIG.t212_invest_api_secret
        elif account_type == "isa":
            key = CONFIG.t212_isa_api_key
            secret = CONFIG.t212_isa_api_secret
        else:
            raise ValueError(f"Unknown account_type: {account_type!r}")
        return cls(api_key=key, api_secret=secret, account_type=account_type)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @property
    def key_fingerprint(self) -> str:
        return _fingerprint(self.api_key)

    @property
    def secret_fingerprint(self) -> str:
        return _fingerprint(self.api_secret)

    def basic_auth_preview(self) -> str:
        """Return a fingerprint of the Authorization header value (for debugging).

        Useful when troubleshooting 401s without exposing the credential itself.
        """
        joined = f"{self.api_key}:{self.api_secret}"
        encoded = base64.b64encode(joined.encode("utf-8")).decode("utf-8")
        return f"Basic {_fingerprint(encoded)} (len={len(encoded)})"

    # ------------------------------------------------------------------
    # Low-level request
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            attempt += 1
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    auth=self._auth,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                log.warning(
                    "T212 %s %s network error (attempt %d/%d) key=%s: %s",
                    method, path, attempt, self.max_retries + 1,
                    self.key_fingerprint, exc,
                )
                self._sleep_backoff(attempt)
                continue

            if resp.status_code in (401, 403):
                raise T212AuthError(
                    f"T212 auth failed ({resp.status_code}) for {self.account_type} "
                    f"key={self.key_fingerprint} secret={self.secret_fingerprint}. "
                    "Likely causes: wrong key, wrong secret, key/secret mismatch, "
                    "missing scope for this endpoint, or key generated against the "
                    "wrong environment (live vs demo). "
                    f"Body: {self._safe_body(resp)}",
                    status=resp.status_code,
                    body=self._safe_body(resp),
                )

            if resp.status_code == 429:
                wait = self._retry_after(resp, attempt)
                log.warning(
                    "T212 429 rate-limited on %s, sleeping %.1fs (attempt %d/%d)",
                    path, wait, attempt, self.max_retries + 1,
                )
                time.sleep(wait)
                continue

            if 500 <= resp.status_code < 600:
                log.warning(
                    "T212 %s %s returned %d (attempt %d/%d): %s",
                    method, path, resp.status_code, attempt, self.max_retries + 1,
                    self._safe_body(resp),
                )
                self._sleep_backoff(attempt)
                continue

            if 400 <= resp.status_code < 500:
                raise T212Error(
                    f"T212 {method} {path} failed with {resp.status_code}: "
                    f"{self._safe_body(resp)}",
                    status=resp.status_code,
                    body=self._safe_body(resp),
                )

            if resp.status_code == 204 or not resp.content:
                return None
            try:
                return resp.json()
            except ValueError as exc:
                raise T212Error(
                    f"T212 {method} {path} returned non-JSON ({resp.status_code}): "
                    f"{resp.text[:500]}",
                    status=resp.status_code,
                    body=resp.text,
                ) from exc

        if last_exc is not None:
            raise T212Error(f"T212 {method} {path} failed after retries") from last_exc
        raise T212RateLimitError(
            f"T212 {method} {path} rate-limited after {self.max_retries} retries"
        )

    @staticmethod
    def _safe_body(resp: requests.Response) -> Any:
        try:
            return resp.json()
        except ValueError:
            return resp.text[:500]

    @staticmethod
    def _retry_after(resp: requests.Response, attempt: int) -> float:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                pass
        return min(2 ** attempt, 30)

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        time.sleep(min(2 ** attempt, 30))

    # ------------------------------------------------------------------
    # Read endpoints (real path layout per T212 docs)
    # ------------------------------------------------------------------

    def ping(self) -> AccountInfo:
        """Cheap health check that proves the credentials work."""
        return self.account_summary()

    def account_summary(self) -> AccountInfo:
        """GET /api/v0/equity/account/summary — account id, currency, cash."""
        raw = self._request("GET", "/equity/account/summary") or {}
        cash_raw = raw.get("cash") or {}
        inv_raw = raw.get("investments") or {}
        return AccountInfo(
            id=raw.get("id"),
            currency=raw.get("currency"),
            total_value=raw.get("totalValue"),
            cash=AccountCashBreakdown(
                available_to_trade=cash_raw.get("availableToTrade"),
                reserved_for_orders=cash_raw.get("reservedForOrders"),
                in_pies=cash_raw.get("inPies"),
            ) if cash_raw else None,
            investments=AccountInvestments(
                current_value=inv_raw.get("currentValue"),
                total_cost=inv_raw.get("totalCost"),
                realized_profit_loss=inv_raw.get("realizedProfitLoss"),
                unrealized_profit_loss=inv_raw.get("unrealizedProfitLoss"),
            ) if inv_raw else None,
            raw=raw,
        )

    def account_cash(self) -> AccountCash:
        """GET /api/v0/equity/account/cash — full cash breakdown."""
        raw = self._request("GET", "/equity/account/cash") or {}
        return AccountCash(
            free=raw.get("free"),
            total=raw.get("total"),
            invested=raw.get("invested"),
            ppl=raw.get("ppl"),
            result=raw.get("result"),
            pie_cash=raw.get("pieCash"),
            blocked=raw.get("blocked"),
            raw=raw,
        )

    def positions(self) -> list[Position]:
        """GET /api/v0/equity/positions — all open positions.

        T212 returns nested objects: ``instrument.{ticker,name,isin,currency}``
        and ``walletImpact.{totalCost,currentValue,unrealizedProfitLoss,fxImpact}``.
        We flatten the common fields onto Position for convenience but keep
        the full structured pieces too.
        """
        raw = self._request("GET", "/equity/positions") or []
        out: list[Position] = []
        for p in raw:
            instrument_raw = p.get("instrument") or {}
            wallet_raw = p.get("walletImpact") or {}
            instrument = Instrument(
                ticker=instrument_raw.get("ticker", ""),
                name=instrument_raw.get("name"),
                isin=instrument_raw.get("isin"),
                currency=instrument_raw.get("currency"),
            ) if instrument_raw else None
            wallet_impact = WalletImpact(
                currency=wallet_raw.get("currency"),
                total_cost=wallet_raw.get("totalCost"),
                current_value=wallet_raw.get("currentValue"),
                unrealized_profit_loss=wallet_raw.get("unrealizedProfitLoss"),
                fx_impact=wallet_raw.get("fxImpact"),
            ) if wallet_raw else None
            out.append(
                Position(
                    ticker=(instrument.ticker if instrument else "") or "",
                    name=instrument.name if instrument else None,
                    isin=instrument.isin if instrument else None,
                    currency=instrument.currency if instrument else None,
                    quantity=float(p.get("quantity", 0.0)),
                    quantity_available_for_trading=p.get("quantityAvailableForTrading"),
                    quantity_in_pies=p.get("quantityInPies"),
                    average_price_paid=float(p.get("averagePricePaid", 0.0)),
                    current_price=p.get("currentPrice"),
                    created_at=p.get("createdAt"),
                    wallet_impact=wallet_impact,
                    instrument=instrument,
                    raw=p,
                )
            )
        return out

    def open_orders(self) -> list[Order]:
        """GET /api/v0/equity/orders — currently open/pending orders."""
        raw = self._request("GET", "/equity/orders") or []
        out: list[Order] = []
        for o in raw:
            out.append(
                Order(
                    id=o.get("id"),
                    ticker=o.get("ticker", ""),
                    quantity=float(o.get("quantity", 0.0)),
                    type=o.get("type"),
                    status=o.get("status"),
                    creation_time=o.get("creationTime"),
                    filled_quantity=o.get("filledQuantity"),
                    filled_value=o.get("filledValue"),
                    limit_price=o.get("limitPrice"),
                    stop_price=o.get("stopPrice"),
                    strategy=o.get("strategy"),
                    value=o.get("value"),
                    raw=o,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Paginated history endpoints — return raw page, callers paginate via
    # the nextPagePath the server includes in the response.
    # ------------------------------------------------------------------

    def historical_orders(
        self,
        *,
        cursor: Optional[int] = None,
        ticker: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": min(limit, 50)}
        if cursor is not None:
            params["cursor"] = cursor
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/equity/history/orders", params=params) or {
            "items": [],
            "nextPagePath": None,
        }

    def dividends(
        self,
        *,
        cursor: Optional[int] = None,
        ticker: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": min(limit, 50)}
        if cursor is not None:
            params["cursor"] = cursor
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/equity/history/dividends", params=params) or {
            "items": [],
            "nextPagePath": None,
        }

    def transactions(
        self,
        *,
        cursor: Optional[int] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": min(limit, 50)}
        if cursor is not None:
            params["cursor"] = cursor
        return self._request("GET", "/equity/history/transactions", params=params) or {
            "items": [],
            "nextPagePath": None,
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def iterate_history(
        self,
        endpoint: str,
        *,
        limit: int = 50,
        max_pages: int = 100,
    ):
        """Generic paginator that follows ``nextPagePath`` until exhausted.

        Yields one page (dict with ``items`` + ``nextPagePath``) at a time.
        ``endpoint`` must start with '/equity/history/...'.
        """
        params: dict[str, Any] = {"limit": min(limit, 50)}
        next_path = endpoint
        pages = 0
        while next_path and pages < max_pages:
            # If next_path is a full path with its own query string, strip the
            # base and re-use as-is — T212 returns absolute API paths.
            if next_path.startswith("/api/v0"):
                next_path = next_path[len("/api/v0"):]
            page = self._request("GET", next_path, params=params if pages == 0 else None) or {}
            yield page
            next_path = page.get("nextPagePath")
            pages += 1


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def available_clients() -> list[T212Client]:
    """Return a client for every account that has both key + secret set."""
    clients: list[T212Client] = []
    if CONFIG.has_t212_invest:
        clients.append(T212Client.for_account("invest"))
    if CONFIG.has_t212_isa:
        clients.append(T212Client.for_account("isa"))
    return clients


def position_to_dict(p: Position) -> dict[str, Any]:
    """Helper for the storage layer."""
    d = asdict(p)
    d.pop("raw", None)
    return d
