"""InsiderLaneTrader — the whole long-only insider lane in one readable loop.

Every loop (default 60 s):
  1. snapshot account + positions (fail-closed: no snapshot => no actions)
  2. day rollover: persist day-start equity; clear yesterday's halt
  3. daily-loss kill switch: equity drop >= daily_loss_pct of the BOOK =>
     cancel everything, flatten via the gateway (reduce-only by live sign),
     halt new entries for the day
  4. recover pending submits (idempotent via deterministic client_order_id)
  5. reconcile the lane's book against the broker (fills, bracket exits,
     unfilled parents past their TTL)
  6. time stops: positions held past max_hold_calendar_days are closed
  7. entries: fresh deduped Form 4 candidates -> quote -> ATR -> risk-sized
     bracket through the gateway; every qualifying candidate is written to
     edge_ledger whether or not it is traded.

What is deliberately NOT here: options, crypto, shorts, price action, an
LLM, an ML probability, Kelly, confluence multipliers, a trailing daily
profit-take, EOD flatten. Each either had measured negative edge or was a
no-op under the caps.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from ..execution.order_gateway import CLOSE, GateCaps, OrderGateway
from . import candidates as cands
from .config import LaneConfig
from .ledger import record_candidate
from .sessions import et_date, in_entry_window, parse_iso
from .sizing import EntryPlan, plan_entry

log = logging.getLogger("marketradar.lane")

_BENCH = "SPY"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


class InsiderLaneTrader:
    def __init__(
        self,
        cfg: LaneConfig,
        *,
        alpaca=None,
        conn_factory: Optional[Callable] = None,
        atr_fn: Optional[Callable[[str], Optional[float]]] = None,
        now_fn: Callable[[], datetime] = _now_utc,
        notify: Optional[Callable[[str], None]] = None,
    ) -> None:
        problems = cfg.validate()
        if problems:
            raise ValueError("LaneConfig incoherent: " + "; ".join(problems))
        self.cfg = cfg
        if alpaca is None:
            from ..execution.alpaca_client import AlpacaClient
            alpaca = AlpacaClient()
        self.alpaca = alpaca
        if conn_factory is None:
            from ..storage import get_connection
            conn_factory = get_connection
        self._conn_factory = conn_factory
        if atr_fn is None:
            from ..execution.sizer import get_atr
            atr_fn = get_atr
        self._atr = atr_fn
        self._now = now_fn
        self._notify_cb = notify
        book = cfg.book_equity_usd if cfg.book_equity_usd > 0 else 0.0
        gross = cfg.gross_cap_usd if cfg.gross_cap_usd > 0 else (book or 10_000.0)
        self.gateway = OrderGateway(
            alpaca,
            GateCaps(
                gross_cap_usd=gross,
                per_symbol_cap_usd=min(gross, (book or gross) * cfg.max_position_pct / 100.0),
                hard_notional_cap_usd=cfg.hard_order_notional_usd,
                allow_stock_shorts=False,
            ),
            alert=self._notify,
        )

    # ------------------------------------------------------------------ utils
    @contextmanager
    def _conn(self):
        with self._conn_factory() as c:
            yield c

    def _notify(self, msg: str) -> None:
        log.warning(msg)
        if self._notify_cb is None:
            try:
                from ..notifications.realtime import TradeAlert, notify_trade
                notify_trade(TradeAlert(kind="ALERT", symbol="LANE", extra=msg))
            except Exception:  # noqa: BLE001 — alerting never breaks execution
                pass
            return
        try:
            self._notify_cb(msg)
        except Exception:  # noqa: BLE001
            pass

    def _state_get(self, conn, key: str) -> Optional[str]:
        r = conn.execute("SELECT value FROM lane_state WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def _state_set(self, conn, key: str, value: str) -> None:
        conn.execute(
            """INSERT INTO lane_state(key, value, updated_at) VALUES (?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, _iso(self._now())))

    def _book(self, account) -> float:
        return self.cfg.book_equity_usd if self.cfg.book_equity_usd > 0 else float(account.equity)

    # ------------------------------------------------------------- main loop
    def run_forever(self) -> None:
        log.info("insider lane starting: %s dry_run=%s book=$%.0f",
                 self.cfg.candidate_name, self.cfg.dry_run, self.cfg.book_equity_usd)
        while True:
            t0 = time.time()
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                log.exception("lane loop error: %s", exc)
            time.sleep(max(1.0, self.cfg.interval_seconds - (time.time() - t0)))

    def run_once(self) -> dict:
        now = self._now()
        today = et_date(now)
        summary = {"placed": 0, "closed": 0, "halted": False, "skipped": 0}

        # 1. snapshot — fail closed
        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.get_positions()
        except Exception as exc:  # noqa: BLE001
            log.error("snapshot failed — no actions this loop: %s", exc)
            return summary
        held = {p.symbol.upper(): p for p in positions}
        book = self._book(account)

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO bot_account_snapshots
                   (snapshot_at, equity_usd, cash_usd, buying_power_usd, long_market_value, open_positions, open_orders)
                   VALUES (?,?,?,?,?,?,NULL)""",
                (_iso(now), float(account.equity), float(account.cash), float(account.buying_power),
                 float(getattr(account, "long_market_value", 0.0) or 0.0), len(positions)))

            # 2. day rollover
            day_key = f"day_start_equity:{today}"
            start_eq = self._state_get(conn, day_key)
            if start_eq is None:
                self._state_set(conn, day_key, str(float(account.equity)))
                start_eq = str(float(account.equity))
            halted_on = self._state_get(conn, "halted_on")
            halted = halted_on == today

            # 3. daily-loss kill switch
            drop = float(start_eq) - float(account.equity)
            cap = book * self.cfg.daily_loss_pct / 100.0
            if not halted and drop >= cap and cap > 0:
                self._notify(f"DAILY LOSS {drop:,.0f} >= cap {cap:,.0f} ({self.cfg.daily_loss_pct}% of book) — flattening + halting for {today}")
                self._flatten_all(conn, positions, reason="daily_loss")
                self._state_set(conn, "halted_on", today)
                halted = True
                summary["halted"] = True
                return summary
            summary["halted"] = halted

            # 4. recover pending submits
            self._recover_pending(conn)

            # 5. reconcile book vs broker
            summary["closed"] += self._reconcile(conn, held, now)

            # 6. time stops (regular hours only so the close is a market order that fills)
            if self._market_open():
                summary["closed"] += self._time_stops(conn, held, now)

            # 7. entries
            if halted:
                return summary
            if not (self._market_open() and in_entry_window(
                    now, first=self.cfg.no_entry_first_minutes, last=self.cfg.no_entry_last_minutes)):
                return summary
            placed, skipped = self._enter(conn, account, positions, held, now, today)
            summary["placed"] += placed
            summary["skipped"] += skipped
        return summary

    # ----------------------------------------------------------- sub-steps
    def _market_open(self) -> bool:
        try:
            return bool(self.alpaca.get_market_clock().get("is_open", False))
        except Exception as exc:  # noqa: BLE001
            log.warning("clock unavailable, treating market as closed: %s", exc)
            return False

    def _open_heat(self, conn) -> float:
        r = conn.execute(
            "SELECT COALESCE(SUM(risk_usd),0) FROM lane_positions WHERE status IN ('submitted','open','closing')"
        ).fetchone()
        return float(r[0] or 0.0)

    def _lane_open_count(self, conn) -> int:
        r = conn.execute(
            "SELECT COUNT(*) FROM lane_positions WHERE status IN ('submitted','open','closing')").fetchone()
        return int(r[0] or 0)

    def _flatten_all(self, conn, positions, *, reason: str) -> int:
        n = 0
        try:
            self.alpaca.cancel_all_orders()
        except Exception as exc:  # noqa: BLE001
            log.error("cancel_all_orders failed: %s", exc)
        for p in positions:
            if abs(float(p.qty)) < 1e-9:
                continue
            o = self.gateway.submit(intent=CLOSE, symbol=p.symbol, order_type="market",
                                    time_in_force="day", positions=positions)
            if o is not None:
                n += 1
                conn.execute(
                    """UPDATE lane_positions SET status='closing', exit_reason=?
                       WHERE ticker=? AND status IN ('submitted','open')""", (reason, p.symbol.upper()))
        return n

    def _close_position(self, conn, symbol: str, positions, *, reason: str) -> bool:
        try:
            self.alpaca.cancel_open_orders_for(symbol)   # release the bracket legs' hold
        except Exception as exc:  # noqa: BLE001
            log.error("cancel legs for %s failed: %s", symbol, exc)
            return False
        o = self.gateway.submit(intent=CLOSE, symbol=symbol, order_type="market",
                                time_in_force="day", positions=positions)
        if o is None:
            return False
        conn.execute("""UPDATE lane_positions SET status='closing', exit_reason=?
                        WHERE ticker=? AND status IN ('submitted','open')""", (reason, symbol.upper()))
        return True

    def _recover_pending(self, conn) -> None:
        rows = conn.execute(
            "SELECT * FROM lane_decisions WHERE outcome='pending_submit'").fetchall()
        for r in rows:
            r = dict(r)
            try:
                o = self.alpaca.get_order_by_client_order_id(r["client_order_id"])
            except Exception as exc:  # noqa: BLE001
                log.warning("pending lookup failed for %s: %s", r["client_order_id"], exc)
                continue
            if o is not None:
                conn.execute("UPDATE lane_decisions SET outcome='placed', parent_order_id=? WHERE id=?",
                             (o.id, r["id"]))
                exists = conn.execute("SELECT 1 FROM lane_positions WHERE client_order_id=?",
                                      (r["client_order_id"],)).fetchone()
                if not exists:
                    # A crash between submit and the position insert: adopt it with the
                    # broker's view (stop/target are in the legs; we record what we know).
                    conn.execute(
                        """INSERT INTO lane_positions (decision_id, ticker, qty, entry_limit, stop_loss,
                             take_profit, atr, risk_usd, parent_order_id, client_order_id, status, submitted_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?, 'submitted', ?)""",
                        (r["id"], r["ticker"], float(o.qty), float(o.limit_price or 0.0), 0.0, 0.0, None,
                         0.0, o.id, r["client_order_id"], o.submitted_at or _iso(self._now())))
                log.warning("recovered pending submit %s -> order %s", r["client_order_id"], o.id)
            else:
                age = self._now() - (parse_iso(r["decided_at"]) or self._now())
                if age > timedelta(minutes=10):
                    conn.execute("UPDATE lane_decisions SET outcome='failed', outcome_detail='pending_submit never reached broker' WHERE id=?",
                                 (r["id"],))

    def _reconcile(self, conn, held: dict, now: datetime) -> int:
        """Fills, exits and expiries. Side/qty are never inferred here — the
        broker's position snapshot is the truth; this only annotates our book."""
        closed = 0
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM lane_positions WHERE status IN ('submitted','open','closing')").fetchall()]
        for r in rows:
            sym = r["ticker"].upper()
            if r["status"] == "submitted":
                try:
                    o = self.alpaca.get_order(r["parent_order_id"])
                except Exception as exc:  # noqa: BLE001
                    log.warning("get_order(%s) failed: %s", r["parent_order_id"], exc)
                    continue
                st = (o.status or "").lower()
                if st == "filled" or (st in ("canceled", "expired", "done_for_day") and float(o.filled_qty or 0) > 0):
                    conn.execute(
                        """UPDATE lane_positions SET status='open', filled_at=?, filled_avg_price=?, qty=? WHERE id=?""",
                        (o.filled_at or _iso(now), o.filled_avg_price, float(o.filled_qty or r["qty"]), r["id"]))
                    self._notify(f"FILLED {sym} x{float(o.filled_qty or r['qty']):.0f} @ {o.filled_avg_price} "
                                 f"stop {r['stop_loss']} target {r['take_profit']}")
                elif st in ("canceled", "expired", "rejected", "done_for_day"):
                    conn.execute("UPDATE lane_positions SET status='canceled', closed_at=?, exit_reason='unfilled' WHERE id=?",
                                 (_iso(now), r["id"]))
                else:
                    sub = parse_iso(r["submitted_at"])
                    if sub and now - sub > timedelta(minutes=self.cfg.entry_ttl_minutes):
                        try:
                            self.alpaca.cancel_order(o.id)
                            log.info("canceled unfilled parent %s after TTL", sym)
                        except Exception as exc:  # noqa: BLE001
                            log.warning("cancel parent %s failed: %s", sym, exc)
                continue

            # open / closing: gone from the broker => closed
            if sym not in held:
                reason = r.get("exit_reason") or self._exit_reason_from_legs(r["parent_order_id"])
                conn.execute("UPDATE lane_positions SET status='closed', closed_at=?, exit_reason=? WHERE id=?",
                             (_iso(now), reason, r["id"]))
                closed += 1
                self._notify(f"CLOSED {sym} ({reason})")
        return closed

    def _exit_reason_from_legs(self, parent_order_id: str) -> str:
        try:
            o = self.alpaca.get_order(parent_order_id)
        except Exception:  # noqa: BLE001
            return "external"
        for leg in o.legs or []:
            if str(leg.get("status", "")).lower() == "filled":
                typ = str(leg.get("type") or leg.get("order_type") or "").lower()
                if typ.startswith("stop"):
                    return "stop_loss"
                if typ == "limit":
                    return "take_profit"
        return "external"

    def _time_stops(self, conn, held: dict, now: datetime) -> int:
        n = 0
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM lane_positions WHERE status='open'").fetchall()]
        positions = list(held.values())
        for r in rows:
            opened = parse_iso(r.get("filled_at") or r["submitted_at"])
            if opened is None:
                continue
            if now - opened < timedelta(days=self.cfg.max_hold_calendar_days):
                continue
            if r["ticker"].upper() not in held:
                continue
            if self.cfg.dry_run:
                log.info("[dry-run] time stop would close %s", r["ticker"])
                continue
            if self._close_position(conn, r["ticker"], positions, reason="time_stop"):
                n += 1
                self._notify(f"TIME STOP {r['ticker']} after {self.cfg.max_hold_calendar_days}d")
        return n

    def _enter(self, conn, account, positions, held: dict, now: datetime, today: str) -> tuple[int, int]:
        cfg = self.cfg
        placed = skipped = 0
        book = self._book(account)
        candidates = cands.recent_candidates(
            conn, now_utc=now, freshness_hours=cfg.freshness_hours,
            min_notional_usd=cfg.min_notional_usd, min_price=cfg.min_price,
            min_role_score=cfg.min_role_score, exclude_10b5_1=cfg.exclude_10b5_1)
        if not candidates:
            return 0, 0
        bench_px = self._ask(_BENCH)
        for c in candidates:
            if c.event_day > today:
                continue   # filed after today's close: it is tomorrow's bet
            if cands.already_decided(conn, c.ticker, c.event_day):
                continue
            detail = None
            if c.ticker in held:
                detail = "already held"
            elif cands.in_cooldown(conn, c.ticker, now_utc=now, cooldown_days=cfg.ticker_cooldown_days):
                detail = f"ticker cooldown {cfg.ticker_cooldown_days}d"
            elif self._lane_open_count(conn) >= cfg.max_positions:
                detail = f"max positions {cfg.max_positions}"
            quote = self._quote(c.ticker)
            ask = quote[1] if quote else None
            # Ledger first: the forward record must not depend on whether we traded.
            record_candidate(conn, candidate=cfg.candidate_name, cand=c,
                             price_at_flag=ask, bench_at_flag=bench_px, traded=False, now=now)
            if detail is not None:
                self._decide(conn, c, "skipped", detail, now)
                skipped += 1
                continue
            if quote is None:
                self._decide(conn, c, "skipped", "no quote", now)
                skipped += 1
                continue
            atr = self._atr(c.ticker)
            plan: EntryPlan = plan_entry(
                bid=quote[0], ask=quote[1], atr=float(atr or 0.0), book_equity=book,
                risk_pct=cfg.risk_per_trade_pct, stop_atr_mult=cfg.stop_atr_mult,
                tp_atr_mult=cfg.tp_atr_mult, max_position_pct=cfg.max_position_pct,
                limit_bps=cfg.entry_limit_bps, max_spread_bps=cfg.max_spread_bps,
                min_price=cfg.min_price, open_heat_usd=self._open_heat(conn),
                max_heat_pct=cfg.max_portfolio_heat_pct, hard_order_cap_usd=cfg.hard_order_notional_usd)
            if not plan.ok:
                self._decide(conn, c, "blocked", plan.reason, now)
                skipped += 1
                continue
            if cfg.dry_run:
                self._decide(conn, c, "skipped", f"dry-run: would buy {plan.qty} @ {plan.limit_price} "
                             f"stop {plan.stop_loss} tp {plan.take_profit} risk ${plan.risk_usd}", now)
                log.info("[dry-run] %s qty=%d limit=%.2f stop=%.2f tp=%.2f risk=$%.0f",
                         c.ticker, plan.qty, plan.limit_price, plan.stop_loss, plan.take_profit, plan.risk_usd)
                skipped += 1
                continue
            coid = f"mr-ins-{c.ticker}-{c.event_day.replace('-', '')}"[:48]
            dec_id = self._decide(conn, c, "pending_submit", None, now, client_order_id=coid)
            try:
                bo = self.gateway.submit_bracket(
                    symbol=c.ticker, qty=plan.qty, limit_price=plan.limit_price,
                    take_profit=plan.take_profit, stop_loss=plan.stop_loss,
                    time_in_force="gtc", client_order_id=coid, positions=positions)
            except Exception as exc:  # noqa: BLE001
                log.exception("submit_bracket %s raised: %s", c.ticker, exc)
                bo = None
                # leave pending_submit: _recover_pending resolves it from the broker
                continue
            if bo is None:
                conn.execute("UPDATE lane_decisions SET outcome='blocked', outcome_detail='gateway refused' WHERE id=?",
                             (dec_id,))
                skipped += 1
                continue
            parent = bo.parent
            conn.execute("UPDATE lane_decisions SET outcome='placed', parent_order_id=? WHERE id=?",
                         (parent.id, dec_id))
            conn.execute(
                """INSERT INTO lane_positions (decision_id, ticker, qty, entry_limit, stop_loss, take_profit,
                     atr, risk_usd, parent_order_id, client_order_id, status, submitted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?, 'submitted', ?)""",
                (dec_id, c.ticker, plan.qty, plan.limit_price, plan.stop_loss, plan.take_profit,
                 atr, plan.risk_usd, parent.id, coid, _iso(now)))
            conn.execute("UPDATE edge_ledger SET traded=1 WHERE candidate=? AND ticker=? AND event_day=?",
                         (cfg.candidate_name, c.ticker, c.event_day))
            placed += 1
            self._notify(f"PLACED {c.ticker} x{plan.qty} limit {plan.limit_price} stop {plan.stop_loss} "
                         f"target {plan.take_profit} risk ${plan.risk_usd:.0f} (insider ${c.notional_usd:,.0f})")
        return placed, skipped

    def _decide(self, conn, c, outcome: str, detail: Optional[str], now: datetime,
                *, client_order_id: Optional[str] = None) -> int:
        cur = conn.execute(
            """INSERT INTO lane_decisions (ticker, event_day, published_at, signal_ids, notional_usd,
                 role_score, outcome, outcome_detail, client_order_id, decided_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (c.ticker, c.event_day, c.published_at, ",".join(str(s) for s in c.signal_ids),
             c.notional_usd, c.role_score, outcome, detail, client_order_id, _iso(now)))
        return int(cur.lastrowid)

    def _quote(self, symbol: str):
        try:
            return self.alpaca.get_latest_quote(symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("quote %s failed: %s", symbol, exc)
            return None

    def _ask(self, symbol: str) -> Optional[float]:
        q = self._quote(symbol)
        return q[1] if q else None
