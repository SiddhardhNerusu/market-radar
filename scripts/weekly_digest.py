"""Sunday-evening weekly P/L digest.

Pulls the past 7 days of bot performance and posts a Telegram summary.
Scheduled via com.marketradar.digest.plist (Sunday 22:00 BST = 21:00 UTC).

Captures the metrics the user needs to evaluate £150/day weekly average:
  * Total realized P/L week
  * Win rate
  * Best + worst single trades
  * Best event_type for the week
  * Daily breakdown
  * Equity start → end
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _notify(text: str) -> None:
    """Telegram-only — silent no-op when not configured."""
    try:
        from market_radar.config import CONFIG
        import requests
        if not (CONFIG.telegram_bot_token and CONFIG.telegram_chat_id):
            return
        requests.post(
            f"https://api.telegram.org/bot{CONFIG.telegram_bot_token}/sendMessage",
            json={
                "chat_id": CONFIG.telegram_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
    except Exception:
        pass


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    from market_radar.storage import get_connection

    now = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    today_iso = now.strftime("%Y-%m-%d")

    with get_connection() as conn:
        # Daily P/L breakdown
        daily = conn.execute(
            """
            SELECT trading_date,
                   ROUND(realized_pnl_usd, 2) AS realized,
                   trades_count, wins, losses,
                   ROUND(largest_win, 2) AS biggest_win,
                   ROUND(largest_loss, 2) AS biggest_loss
            FROM bot_daily_pnl
            WHERE trading_date >= ?
            ORDER BY trading_date
            """,
            (week_start,),
        ).fetchall()

        # Best + worst stock/crypto trades
        top_orders = conn.execute(
            """
            SELECT ticker, ROUND(realized_pnl_usd, 2) AS pnl, exit_reason,
                   substr(filled_at, 1, 16) AS filled
            FROM bot_orders
            WHERE realized_pnl_usd IS NOT NULL
              AND filled_at >= ?
            ORDER BY realized_pnl_usd DESC
            LIMIT 3
            """,
            (week_start,),
        ).fetchall()
        worst_orders = conn.execute(
            """
            SELECT ticker, ROUND(realized_pnl_usd, 2) AS pnl, exit_reason
            FROM bot_orders
            WHERE realized_pnl_usd IS NOT NULL
              AND filled_at >= ?
            ORDER BY realized_pnl_usd ASC
            LIMIT 3
            """,
            (week_start,),
        ).fetchall()

        # Option spreads of the week
        top_spreads = conn.execute(
            """
            SELECT underlying, ROUND(realized_pnl_usd, 2) AS pnl,
                   ROUND(total_debit_usd, 2) AS debit, exit_reason
            FROM bot_option_spreads
            WHERE realized_pnl_usd IS NOT NULL
              AND closed_at >= ?
            ORDER BY realized_pnl_usd DESC
            LIMIT 3
            """,
            (week_start,),
        ).fetchall()

        # Best event_type by total P/L this week (via decisions → orders join)
        event_pnl = conn.execute(
            """
            SELECT ss.event_type, ROUND(SUM(bo.realized_pnl_usd), 2) AS total_pnl, COUNT(*) AS n
            FROM bot_orders bo
            JOIN bot_decisions bd ON bd.alpaca_order_id = bo.alpaca_order_id
            JOIN signal_scores ss ON ss.id = bd.score_id
            WHERE bo.realized_pnl_usd IS NOT NULL
              AND bo.filled_at >= ?
              AND ss.event_type IS NOT NULL
            GROUP BY ss.event_type
            ORDER BY total_pnl DESC
            LIMIT 5
            """,
            (week_start,),
        ).fetchall()

        # Equity curve endpoints
        equity_start = conn.execute(
            """
            SELECT ROUND(equity_usd, 2) FROM bot_account_snapshots
            WHERE snapshot_at >= ?
            ORDER BY snapshot_at ASC LIMIT 1
            """,
            (week_start,),
        ).fetchone()
        equity_end = conn.execute(
            """
            SELECT ROUND(equity_usd, 2) FROM bot_account_snapshots
            ORDER BY snapshot_at DESC LIMIT 1
            """,
        ).fetchone()

    # Aggregate
    total_realized = sum(float(d["realized"] or 0) for d in daily)
    total_trades = sum(int(d["trades_count"] or 0) for d in daily)
    total_wins = sum(int(d["wins"] or 0) for d in daily)
    total_losses = sum(int(d["losses"] or 0) for d in daily)
    win_rate = (total_wins / (total_wins + total_losses) * 100
                if (total_wins + total_losses) > 0 else 0)
    eq_s = float(equity_start[0]) if equity_start else 0
    eq_e = float(equity_end[0]) if equity_end else 0
    eq_delta = eq_e - eq_s
    daily_avg = total_realized / max(len(daily), 1)
    # GBP convert (rough, 1 USD = 0.80 GBP — adjust if needed)
    usd_to_gbp = 0.80
    daily_avg_gbp = daily_avg * usd_to_gbp
    total_gbp = total_realized * usd_to_gbp

    # Build the report
    lines = [
        f"📊 <b>WEEKLY DIGEST — {week_start} → {today_iso}</b>",
        "",
        f"<b>Total realized: ${total_realized:+.2f} (£{total_gbp:+.2f})</b>",
        f"<b>Daily avg: ${daily_avg:+.2f} (£{daily_avg_gbp:+.2f})</b>",
        f"Target: £150/day = £1,050/week",
        f"Hit rate vs target: {'✅ ABOVE' if daily_avg_gbp >= 150 else '❌ BELOW'}",
        "",
        f"📈 Equity: ${eq_s:,.0f} → ${eq_e:,.0f} (Δ ${eq_delta:+,.0f})",
        f"🎯 Win rate: {win_rate:.0f}% ({total_wins}W / {total_losses}L of {total_trades} closed)",
        "",
    ]

    if daily:
        lines.append("<b>Daily breakdown:</b>")
        for d in daily:
            d = dict(d)
            marker = "✅" if d["realized"] and d["realized"] > 0 else ("❌" if d["realized"] and d["realized"] < 0 else "—")
            lines.append(
                f"  {marker} {d['trading_date']}: ${d['realized']:+.2f} "
                f"({d['wins']}W/{d['losses']}L)"
            )
        lines.append("")

    if top_orders or top_spreads:
        lines.append("<b>Top winners:</b>")
        for s in top_spreads or []:
            s = dict(s)
            lines.append(f"  💰 {s['underlying']} options +${s['pnl']:.2f} ({s['exit_reason']})")
        for o in top_orders or []:
            o = dict(o)
            if (o["pnl"] or 0) > 0:
                lines.append(f"  💰 {o['ticker']} +${o['pnl']:.2f} ({o['exit_reason']})")
        lines.append("")

    if worst_orders:
        lines.append("<b>Worst losses:</b>")
        for o in worst_orders or []:
            o = dict(o)
            if (o["pnl"] or 0) < 0:
                lines.append(f"  📉 {o['ticker']} ${o['pnl']:.2f} ({o['exit_reason']})")
        lines.append("")

    if event_pnl:
        lines.append("<b>Best event types:</b>")
        for e in event_pnl:
            e = dict(e)
            sign = "+" if (e["total_pnl"] or 0) >= 0 else ""
            lines.append(f"  {e['event_type']}: {sign}${e['total_pnl']:.0f} ({e['n']} trades)")
        lines.append("")

    lines.append(
        f"<i>Auto-generated by scripts/weekly_digest.py at "
        f"{now.strftime('%Y-%m-%d %H:%M UTC')}</i>"
    )

    text = "\n".join(lines)
    logging.info("Digest:\n%s", text)
    _notify(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
