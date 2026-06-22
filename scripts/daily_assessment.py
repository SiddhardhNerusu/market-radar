"""Daily assessment digest — the honest 'how's it doing' (posts to Telegram).

Built around the deployed-state audit (2026-06-16) daily_watch_items so the daily
check reads TRUE numbers + system health, NOT green/red vibes. Scheduled by the
daemon (~21:35 UTC, after the US close). Reports:
  • TRUE P&L (the honest ledger, not the old clobber) + equity
  • EDGE PROGRESS (distinct clean days vs the 40-day promotion gate)
  • PIPELINE HEALTH (classify backlog/throughput, ML gate state, log size)
  • SAFETY (gateway blocks / circuit-breaker / reconcile-gap alerts today)
  • a blunt NOISE-vs-SIGNAL verdict so a lucky green day is never mistaken for edge.

Usage:  PYTHONPATH=src .venv/bin/python scripts/daily_assessment.py [--print]
"""
import os
import pathlib
import sys
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RANK_MIN_DAYS = 40  # promotion gate from the edge blueprint
LOG = pathlib.Path.home() / "Library/Logs/MarketRadar/daemon.out.log"
LIVELOG = pathlib.Path.home() / "Library/Logs/MarketRadar/livetrader.err.log"


def _q1(conn, sql, params=()):
    r = conn.execute(sql, params).fetchone()
    return r[0] if r else None


def build_report() -> str:
    from market_radar.storage import get_connection
    now = datetime.now(timezone.utc)
    wk = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    L = []
    with get_connection() as conn:
        # --- TRUE P&L: the EQUITY CURVE is authoritative (full-audit finding). The
        # per-trade ledger captured only ~13% of real P&L across the old 3-asset mix,
        # so all-time P&L = current equity - starting equity is the only honest number.
        today_eq = _q1(conn, "SELECT equity_usd FROM bot_account_snapshots ORDER BY id DESC LIMIT 1")
        start_eq = _q1(conn, "SELECT equity_usd FROM bot_account_snapshots ORDER BY id ASC LIMIT 1") or 0.0
        # Per-trade ledger (now fills-reconciled on the equity-only lane; still PARTIAL
        # over the historical mixed-asset period — labelled as such below).
        wk_realized = _q1(conn, "SELECT ROUND(SUM(realized_pnl_usd),2) FROM bot_daily_pnl WHERE trading_date>=?", (wk,)) or 0.0
        today_realized = _q1(conn, "SELECT ROUND(realized_pnl_usd,2) FROM bot_daily_pnl ORDER BY trading_date DESC LIMIT 1") or 0.0
        # --- EDGE PROGRESS (distinct clean days) ---
        # Promotion-gate metric = distinct LIVE SCORED days (scored_at). NOT
        # published_at: the SEC backfill spans ~515 historical published-dates but
        # was scored in a few batches — those are NOT independent live OOS
        # observations. Counting scored-days gives the honest count (~19/40).
        clean_days = _q1(conn,
            "SELECT COUNT(DISTINCT substr(ss.scored_at,1,10)) "
            "FROM signal_scores ss JOIN signal_outcomes so ON so.score_id=ss.id "
            "WHERE so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0 "
            "AND so.price_at_flag>=1") or 0
        # --- PIPELINE HEALTH ---
        backlog = _q1(conn,
            "SELECT COUNT(*) FROM signal_scores ss JOIN raw_signals rs ON rs.id=ss.signal_id "
            "LEFT JOIN llm_classifications lc ON lc.signal_id=ss.signal_id AND lc.ticker=ss.ticker "
            "WHERE lc.id IS NULL AND rs.source NOT LIKE 'sec_edgar_backfill_%' "
            "AND rs.source NOT LIKE 'price_action_%' "
            "AND rs.ingested_at >= strftime('%Y-%m-%dT%H:%M:%SZ', datetime('now','-2 days'))") or 0
        classified_today = _q1(conn,
            "SELECT COUNT(*) FROM llm_classifications WHERE classified_at >= "
            "strftime('%Y-%m-%dT%H:%M:%SZ', datetime('now','start of day'))") or 0
        scored_today = _q1(conn,
            "SELECT COUNT(*) FROM signal_scores WHERE scored_at >= "
            "strftime('%Y-%m-%dT%H:%M:%SZ', datetime('now','start of day'))") or 0
        mp_fill = _q1(conn,
            "SELECT ROUND(100.0*AVG(CASE WHEN model_p_5d IS NOT NULL THEN 1 ELSE 0 END),1) "
            "FROM signal_scores WHERE scored_at >= strftime('%Y-%m-%dT%H:%M:%SZ', datetime('now','start of day'))")

    # --- SAFETY events from the live log today ---
    def _count(path, *needles):
        try:
            t = path.read_text(errors="ignore")
        except OSError:
            return 0
        today = now.strftime("%Y-%m-%d")
        return sum(1 for ln in t.splitlines() if today in ln and any(n in ln for n in needles))
    gw_blocks = _count(LIVELOG, "gateway BLOCKED", "order gateway refused")
    breaker = _count(LIVELOG, "EQUITY CIRCUIT BREAKER")
    pnl_gap = _count(LIVELOG, "P&L GAP")
    log_mb = round(LOG.stat().st_size / 1e6, 1) if LOG.exists() else 0.0

    # --- VERDICT ---
    if clean_days < RANK_MIN_DAYS:
        verdict = (f"📉 STILL DATA-GATHERING — {clean_days}/{RANK_MIN_DAYS} clean days. "
                   f"Any green/red below is NOISE, not an edge. No conclusion yet.")
    else:
        verdict = ("📊 Enough days to test — run edge_screen_v2.py for the OOS verdict "
                   "(positive net-of-cost + OOS-persistent + p<0.05 before believing anything).")

    L.append(f"📋 <b>DAILY ASSESSMENT — {now:%Y-%m-%d} (PAPER)</b>")
    L.append(verdict)
    L.append("")
    _eq = float(today_eq or 0); _start = float(start_eq or 0)
    L.append(f"<b>P&L TRUTH (equity curve — authoritative):</b> equity ${_eq:,.0f} | "
             f"all-time ${_eq - _start:+,.0f} (from ${_start:,.0f} start)")
    L.append(f"<b>Per-trade ledger (partial — equity-lane fills only):</b> "
             f"today ${today_realized:+.2f} | 7d ${wk_realized:+.2f}")
    L.append(f"<b>Edge progress:</b> {clean_days}/{RANK_MIN_DAYS} distinct clean days")
    L.append(f"<b>Pipeline:</b> scored {scored_today} | classified {classified_today} "
             f"| backlog {backlog}" + (" ⚠STALLED" if (scored_today > 50 and classified_today == 0) else ""))
    L.append(f"<b>ML gate:</b> model_p fill {mp_fill if mp_fill is not None else 0}% "
             f"(~0% = correctly gated OFF; a jump to ~100% = a model passed OR gate bypassed — check)")
    L.append(f"<b>Safety today:</b> gateway-blocks {gw_blocks} | circuit-breaker {breaker} "
             f"| P&L-gap alerts {pnl_gap}")
    L.append(f"<b>Log size:</b> {log_mb} MB" + (" ⚠ROTATE" if log_mb > 200 else ""))
    return "\n".join(L)


def _notify(text: str) -> None:
    try:
        from market_radar.config import CONFIG
        import requests
        if not (CONFIG.telegram_bot_token and CONFIG.telegram_chat_id):
            return
        requests.post(
            f"https://api.telegram.org/bot{CONFIG.telegram_bot_token}/sendMessage",
            json={"chat_id": CONFIG.telegram_chat_id, "text": text,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10)
    except Exception:
        pass


def main() -> int:
    report = build_report()
    print(report)
    if "--print" not in sys.argv:
        _notify(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
