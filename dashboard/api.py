"""Read-only data API for the MARKET RADAR dashboard artifact.

Run from the dashboard via:
    python dashboard/api.py <command> [args...]

Always prints a single JSON document to stdout. Commands:

    overview
    portfolio
    signals [score_min=0] [source_tier=0] [ticker=""] [limit=100] [offset=0]
    ticker <TICKER>
    edge
    health
    metadata
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from market_radar.config import CONFIG  # noqa: E402
from market_radar.scoring.action_labels import label_signal  # noqa: E402
from market_radar.scoring.projections import project_for_class, projected_price_range  # noqa: E402
from market_radar.storage import get_connection  # noqa: E402

from dashboard.labels import display_event, display_source  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rows_to_list(rows) -> list[dict]:
    return [dict(r) for r in rows]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit(payload: dict[str, Any]) -> None:
    payload.setdefault("generated_at", _utc_now_iso())
    payload.setdefault("db_path", str(CONFIG.db_path))
    # Wrap output in sentinel markers so the dashboard can reliably extract
    # the JSON even if the calling tool prepends/appends text.
    sys.stdout.write("__MR_JSON_START__")
    json.dump(payload, sys.stdout, default=str, ensure_ascii=False)
    sys.stdout.write("__MR_JSON_END__\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_overview() -> dict[str, Any]:
    """Landing-tile data."""
    with get_connection() as conn:
        # T212 latest snapshot per account
        accounts = _rows_to_list(conn.execute(
            """
            SELECT t1.*
            FROM t212_account_snapshots t1
            JOIN (
                SELECT account_type, MAX(snapshot_at) AS latest
                FROM t212_account_snapshots
                GROUP BY account_type
            ) t2 ON t1.account_type = t2.account_type AND t1.snapshot_at = t2.latest
            ORDER BY t1.account_type
            """
        ).fetchall())

        # Combined totals
        total_value = sum((a.get("total_value") or 0) for a in accounts)
        total_invested = sum((a.get("invested") or 0) for a in accounts)
        total_pnl = sum((a.get("pnl") or 0) for a in accounts)
        total_cash = sum((a.get("cash") or 0) for a in accounts)

        # Signal counts in last 24h
        signal_counts = _rows_to_list(conn.execute(
            """
            SELECT source_tier, COUNT(*) AS n
            FROM raw_signals
            WHERE ingested_at >= datetime('now', '-1 day')
            GROUP BY source_tier
            ORDER BY source_tier
            """
        ).fetchall())

        # Top signals right now
        top_signals = _rows_to_list(conn.execute(
            """
            SELECT ss.id AS score_id, ss.ticker, ss.composite_score, ss.event_type,
                   ss.sentiment, ss.factual, ss.corroboration_count, ss.scored_at,
                   rs.title, rs.source, rs.source_tier, rs.published_at, rs.url
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            WHERE rs.ingested_at >= datetime('now', '-1 day')
            ORDER BY ss.composite_score DESC, ss.id DESC
            LIMIT 10
            """
        ).fetchall())

        # Per-source health
        health = _rows_to_list(conn.execute(
            """
            SELECT source, last_poll_at, last_success_at, last_error,
                   consecutive_errors
            FROM daemon_health
            ORDER BY last_poll_at DESC NULLS LAST
            """
        ).fetchall())

    return {
        "totals": {
            "total_value": total_value,
            "invested": total_invested,
            "cash": total_cash,
            "pnl": total_pnl,
            "currency": accounts[0]["raw_payload"] if accounts else None,
            "accounts": accounts,
        },
        "signal_counts_24h": signal_counts,
        "top_signals": top_signals,
        "daemon_health": health,
    }


# Map dashboard time-range tokens to SQLite ``datetime('now', '-N units')`` strings.
PORTFOLIO_RANGES: dict[str, Optional[str]] = {
    "1h":  "-1 hour",
    "6h":  "-6 hours",
    "1d":  "-1 day",
    "1w":  "-7 days",
    "1mo": "-30 days",
    "3mo": "-90 days",
    "6mo": "-180 days",
    "1y":  "-365 days",
    "max": None,            # no filter
}


def cmd_portfolio(*, range_token: str = "1d") -> dict[str, Any]:
    with get_connection() as conn:
        # Most recent account snapshots
        accounts = _rows_to_list(conn.execute(
            """
            SELECT t1.*
            FROM t212_account_snapshots t1
            JOIN (
                SELECT account_type, MAX(snapshot_at) AS latest
                FROM t212_account_snapshots
                GROUP BY account_type
            ) t2 ON t1.account_type = t2.account_type AND t1.snapshot_at = t2.latest
            ORDER BY t1.account_type
            """
        ).fetchall())

        # Most recent positions per account
        positions = _rows_to_list(conn.execute(
            """
            SELECT p.*
            FROM t212_positions p
            JOIN (
                SELECT account_type, MAX(snapshot_at) AS latest
                FROM t212_positions
                GROUP BY account_type
            ) m ON p.account_type = m.account_type AND p.snapshot_at = m.latest
            ORDER BY (p.quantity * COALESCE(p.current_price, 0)) DESC
            """
        ).fetchall())

        # Equity history — filterable by range_token
        offset = PORTFOLIO_RANGES.get(range_token, "-1 day")
        if offset is None:
            equity = _rows_to_list(conn.execute(
                """
                SELECT account_type, snapshot_at, total_value, cash, invested, pnl
                FROM t212_account_snapshots
                ORDER BY snapshot_at ASC
                """
            ).fetchall())
        else:
            equity = _rows_to_list(conn.execute(
                f"""
                SELECT account_type, snapshot_at, total_value, cash, invested, pnl
                FROM t212_account_snapshots
                WHERE snapshot_at >= datetime('now', '{offset}')
                ORDER BY snapshot_at ASC
                """
            ).fetchall())

    return {
        "accounts": accounts,
        "positions": positions,
        "equity_history": equity,
        "range_token": range_token,
        "available_ranges": list(PORTFOLIO_RANGES.keys()),
    }


def cmd_signals(
    *,
    score_min: float = 0.0,
    source_tier: int = 0,
    ticker: str = "",
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Paginated signals list with filters."""
    # Always exclude backfilled rows from the live Signals feed — they
    # exist in the DB for ML training + Edge stats only, not for browsing.
    clauses = ["rs.source NOT LIKE 'sec_edgar_backfill_%'"]
    params: list[Any] = []
    if score_min > 0:
        clauses.append("ss.composite_score >= ?")
        params.append(score_min)
    if source_tier in (1, 2, 3):
        clauses.append("rs.source_tier = ?")
        params.append(source_tier)
    if ticker:
        clauses.append("ss.ticker = ?")
        params.append(ticker.upper())

    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT ss.id AS score_id, ss.signal_id, ss.ticker, ss.composite_score,
               ss.event_type, ss.sentiment, ss.sentiment_magnitude, ss.factual,
               ss.corroboration_count, ss.signal_class, ss.scored_at,
               rs.title, rs.body, rs.source, rs.source_tier, rs.published_at,
               rs.url, rs.author,
               so.price_at_flag, so.return_1d_pct, so.return_5d_pct,
               so.return_20d_pct
        FROM signal_scores ss
        JOIN raw_signals rs ON rs.id = ss.signal_id
        LEFT JOIN signal_outcomes so ON so.score_id = ss.id
        WHERE {where_sql}
        ORDER BY ss.composite_score DESC, ss.id DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    with get_connection() as conn:
        items = _rows_to_list(conn.execute(sql, params).fetchall())
        total = conn.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            WHERE {where_sql}
            """,
            params[:-2],
        ).fetchone()["n"]

    return {"items": items, "total": total, "limit": limit, "offset": offset}


def _get_current_price_for(ticker: str) -> Optional[float]:
    """Cheap current-price fetcher for the projection display.

    Uses yfinance ``fast_info.last_price`` which is a single-call lookup.
    Returns None on failure — the projection just hides price ranges.
    """
    try:
        import yfinance as yf  # type: ignore
        sym = ticker.upper()
        if "_" in sym:  # T212 shape AAPL_US_EQ → AAPL
            sym = sym.split("_")[0]
        info = yf.Ticker(sym).fast_info
        price = getattr(info, "last_price", None)
        return float(price) if price else None
    except Exception:
        return None


def cmd_ticker(ticker: str) -> dict[str, Any]:
    ticker = ticker.upper()
    with get_connection() as conn:
        # Recent signals on this ticker
        signals = _rows_to_list(conn.execute(
            """
            SELECT ss.id AS score_id, ss.composite_score, ss.event_type,
                   ss.sentiment, ss.signal_class, ss.scored_at,
                   rs.title, rs.source, rs.source_tier, rs.published_at, rs.url,
                   so.price_at_flag, so.return_1d_pct, so.return_5d_pct,
                   so.return_20d_pct
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            LEFT JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE ss.ticker = ?
            ORDER BY ss.scored_at DESC
            LIMIT 200
            """,
            (ticker,),
        ).fetchall())

        # Position info (latest snapshot, any account)
        positions = _rows_to_list(conn.execute(
            """
            SELECT p.*
            FROM t212_positions p
            JOIN (
                SELECT account_type, MAX(snapshot_at) AS latest
                FROM t212_positions
                GROUP BY account_type
            ) m ON p.account_type = m.account_type AND p.snapshot_at = m.latest
            WHERE p.ticker LIKE ?
            """,
            (f"{ticker}\\_%",),  # match T212 ticker shape AAPL_US_EQ
        ).fetchall())

        # Aggregate stats
        agg = conn.execute(
            """
            SELECT COUNT(*) AS signals_total,
                   AVG(composite_score) AS avg_score,
                   MAX(composite_score) AS max_score,
                   SUM(CASE WHEN composite_score >= 7.5 THEN 1 ELSE 0 END) AS strong_count
            FROM signal_scores
            WHERE ticker = ?
            """,
            (ticker,),
        ).fetchone()

    # Projection from the most-recent signal's class
    projection_data: Optional[dict] = None
    if signals:
        # signal class lives on each row; get it from first (newest)
        latest_class = None
        # signals don't currently expose signal_class — pull from DB
        with get_connection() as conn2:
            cls_row = conn2.execute(
                "SELECT signal_class, composite_score FROM signal_scores "
                "WHERE id = ?", (signals[0]["score_id"],),
            ).fetchone()
            if cls_row:
                latest_class = cls_row["signal_class"]

        if latest_class:
            with get_connection() as conn3:
                proj = project_for_class(conn3, latest_class)
            if proj is not None:
                current_price = _get_current_price_for(ticker)
                projection_data = {
                    "signal_class": proj.signal_class,
                    "used_relaxed_match": proj.used_relaxed,
                    "current_price": current_price,
                    "one_day":    _proj_to_dict(proj.one_day,    current_price),
                    "five_day":   _proj_to_dict(proj.five_day,   current_price),
                    "twenty_day": _proj_to_dict(proj.twenty_day, current_price),
                }

    return {
        "ticker": ticker,
        "signals": signals,
        "positions": positions,
        "stats": dict(agg) if agg else {},
        "projection": projection_data,
    }


def _proj_to_dict(dist, current_price: Optional[float]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "window_days": dist.window_days,
        "samples": dist.samples,
        "median_return_pct": dist.median,
        "p25_return_pct": dist.p25,
        "p75_return_pct": dist.p75,
        "p10_return_pct": dist.p10,
        "p90_return_pct": dist.p90,
        "worst_return_pct": dist.worst,
        "best_return_pct": dist.best,
        "hit_rate": dist.hit_rate,
    }
    if current_price is not None and dist.samples > 0:
        prices = projected_price_range(current_price, dist)
        out["projection_prices"] = {
            k: v for k, v in prices.items()
            if k.endswith("_price")
        }
        out["current_price"] = current_price
    return out


def cmd_edge() -> dict[str, Any]:
    """Hit rate aggregations for the dashboard's Edge tab."""
    with get_connection() as conn:
        by_class = _rows_to_list(conn.execute(
            """
            SELECT
                ss.signal_class,
                COUNT(*) AS samples,
                ROUND(AVG(ss.composite_score), 2) AS avg_score,
                ROUND(AVG(so.return_1d_pct), 3) AS avg_1d,
                ROUND(AVG(so.return_5d_pct), 3) AS avg_5d,
                ROUND(AVG(so.return_20d_pct), 3) AS avg_20d,
                ROUND(AVG(CASE WHEN so.return_5d_pct > 0 THEN 1.0 ELSE 0.0 END), 3) AS hit_rate_5d
            FROM signal_scores ss
            LEFT JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE ss.signal_class IS NOT NULL
              AND COALESCE(so.data_corrupt, 0) = 0   -- P1: exclude corrupt-label rows
            GROUP BY ss.signal_class
            HAVING COUNT(*) >= 5
            ORDER BY samples DESC
            LIMIT 100
            """
        ).fetchall())

        by_event = _rows_to_list(conn.execute(
            """
            SELECT
                ss.event_type,
                COUNT(*) AS samples,
                ROUND(AVG(ss.composite_score), 2) AS avg_score,
                ROUND(AVG(so.return_5d_pct), 3) AS avg_5d_return,
                ROUND(AVG(CASE WHEN so.return_5d_pct > 0 THEN 1.0 ELSE 0.0 END), 3) AS hit_rate_5d
            FROM signal_scores ss
            LEFT JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE ss.event_type IS NOT NULL
              AND COALESCE(so.data_corrupt, 0) = 0   -- P1: exclude corrupt-label rows
            GROUP BY ss.event_type
            ORDER BY samples DESC
            """
        ).fetchall())

        by_source = _rows_to_list(conn.execute(
            """
            SELECT
                rs.source,
                rs.source_tier,
                COUNT(DISTINCT rs.id) AS signals,
                COUNT(DISTINCT ss.id) AS scored,
                ROUND(AVG(ss.composite_score), 2) AS avg_score,
                ROUND(AVG(CASE WHEN COALESCE(so.data_corrupt,0)=0
                               THEN so.return_5d_pct END), 3) AS avg_5d_return
            FROM raw_signals rs
            LEFT JOIN signal_scores ss ON ss.signal_id = rs.id
            LEFT JOIN signal_outcomes so ON so.score_id = ss.id
            GROUP BY rs.source
            ORDER BY signals DESC
            """
        ).fetchall())

        coverage = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM signal_scores) AS scored,
                (SELECT COUNT(*) FROM signal_outcomes) AS anchored,
                (SELECT COUNT(*) FROM signal_outcomes WHERE return_1d_pct IS NOT NULL) AS resolved_1d,
                (SELECT COUNT(*) FROM signal_outcomes WHERE return_5d_pct IS NOT NULL) AS resolved_5d,
                (SELECT COUNT(*) FROM signal_outcomes WHERE return_20d_pct IS NOT NULL) AS resolved_20d
            """
        ).fetchone()

    return {
        "by_signal_class": by_class,
        "by_event_type": by_event,
        "by_source": by_source,
        "coverage": dict(coverage) if coverage else {},
    }


def cmd_health() -> dict[str, Any]:
    with get_connection() as conn:
        rows = _rows_to_list(conn.execute(
            "SELECT * FROM daemon_health ORDER BY last_poll_at DESC NULLS LAST"
        ).fetchall())
    return {"sources": rows}


# ---------------------------------------------------------------------------
# Grouped feed — the main UI tab
# ---------------------------------------------------------------------------


def _measured_edge(
    conn,
    signal_class: Optional[str],
) -> tuple[Optional[float], int]:
    """Return (5d hit rate, sample size) for a signal class.

    Two-tier lookup:
      1. Exact signal_class match (most specific).
      2. If <50 samples, fall back to event_type + sentiment_dir (ignores
         tier, factual flag, corr band, megacap). This is critical because
         our backfill is all Tier 1 SEC; live signals are mostly Tier 3
         social and would otherwise never match historical edge data.
    """
    if not signal_class:
        return None, 0

    # 1. Exact match
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS n,
            AVG(CASE WHEN so.return_5d_pct > 0 THEN 1.0
                     WHEN so.return_5d_pct <= 0 THEN 0.0
                     ELSE NULL END) AS hit_rate
        FROM signal_scores ss
        JOIN signal_outcomes so ON so.score_id = ss.id
        WHERE ss.signal_class = ?
          AND so.return_5d_pct IS NOT NULL
          AND COALESCE(so.data_corrupt, 0) = 0
        """,
        (signal_class,),
    ).fetchone()
    if row and row["n"] and int(row["n"]) >= 50:
        return row["hit_rate"], int(row["n"])

    # 2. Relaxed match — event_type + sentiment_dir only. Signal class
    # format is "tier|event_type|sentiment_dir|factual|corr|cap"; we match
    # on positions 1 and 2 (event_type and sentiment_dir).
    parts = signal_class.split("|")
    if len(parts) < 3:
        return (row["hit_rate"] if row and row["n"] else None,
                int(row["n"]) if row and row["n"] else 0)
    event_type = parts[1]
    sentiment_dir = parts[2]
    relaxed = conn.execute(
        """
        SELECT
            COUNT(*) AS n,
            AVG(CASE WHEN so.return_5d_pct > 0 THEN 1.0
                     WHEN so.return_5d_pct <= 0 THEN 0.0
                     ELSE NULL END) AS hit_rate
        FROM signal_scores ss
        JOIN signal_outcomes so ON so.score_id = ss.id
        WHERE ss.event_type = ?
          AND (
              (? = 'bullish'  AND ss.sentiment >  0.2) OR
              (? = 'bearish'  AND ss.sentiment < -0.2) OR
              (? = 'neutral'  AND ss.sentiment BETWEEN -0.2 AND 0.2)
          )
          AND so.return_5d_pct IS NOT NULL
        """,
        (event_type, sentiment_dir, sentiment_dir, sentiment_dir),
    ).fetchone()
    if relaxed and relaxed["n"] and int(relaxed["n"]) > 0:
        return relaxed["hit_rate"], int(relaxed["n"])

    # No edge data available
    return None, 0


def _t212_holdings(conn) -> set[str]:
    """Latest T212 positions across all accounts, as a set of base tickers."""
    rows = conn.execute(
        """
        SELECT DISTINCT ticker
        FROM t212_positions p
        JOIN (
            SELECT account_type, MAX(snapshot_at) AS latest
            FROM t212_positions GROUP BY account_type
        ) m ON p.account_type = m.account_type AND p.snapshot_at = m.latest
        """
    ).fetchall()
    out: set[str] = set()
    for row in rows:
        t = (row["ticker"] or "").upper()
        # T212 ticker shape e.g. AAPL_US_EQ → AAPL
        base = t.split("_")[0] if "_" in t else t
        if base:
            out.add(base)
    return out


def cmd_grouped(*, limit_per_group: int = 10) -> dict[str, Any]:
    """Return signals grouped by action label (STRONG_BUY, BUY, etc).

    Each signal gets its action label computed inline with measured-edge
    data and T212 holdings awareness. Within each group, signals are
    sorted by composite_score descending.
    """
    with get_connection() as conn:
        holdings = _t212_holdings(conn)
        # Pull the top-N live signals (last 24h, scored only) — the backfill
        # rows have published_at far in the past, so they don't appear in
        # this feed but DO power the measured-edge stats.
        rows = conn.execute(
            """
            SELECT ss.id AS score_id, ss.signal_id, ss.ticker,
                   ss.composite_score, ss.event_type, ss.sentiment,
                   ss.sentiment_magnitude, ss.factual, ss.corroboration_count,
                   ss.signal_class, ss.scored_at, ss.model_p_5d, ss.model_version,
                   rs.title, rs.url, rs.source, rs.source_tier,
                   rs.published_at, rs.author,
                   so.price_at_flag, so.return_1d_pct, so.return_5d_pct,
                   so.return_20d_pct
            FROM signal_scores ss
            JOIN raw_signals rs ON rs.id = ss.signal_id
            LEFT JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE rs.ingested_at >= datetime('now', '-1 day')
              AND rs.source NOT LIKE 'sec_edgar_backfill_%'
            ORDER BY ss.composite_score DESC, ss.id DESC
            LIMIT 200
            """
        ).fetchall()

        # Cache measured edges by signal_class
        edge_cache: dict[str, tuple[Optional[float], int]] = {}

        enriched = []
        for row in rows:
            ticker = (row["ticker"] or "").upper()
            base_ticker = ticker.split("_")[0] if "_" in ticker else ticker
            holds = base_ticker in holdings

            sc = row["signal_class"]
            if sc not in edge_cache:
                edge_cache[sc] = _measured_edge(conn, sc)
            hit_rate, samples = edge_cache[sc]

            label = label_signal(
                event_type=row["event_type"],
                sentiment=row["sentiment"],
                composite_score=row["composite_score"],
                user_holds_ticker=holds,
                measured_hit_rate_5d=hit_rate,
                measured_samples=samples,
                model_p_5d=row["model_p_5d"],
            )

            d = dict(row)
            d["action"] = label.action
            d["action_display"] = label.display
            d["action_direction"] = label.direction
            d["action_edge_built"] = label.edge_built
            d["action_reason"] = label.reason
            d["action_rationale"] = label.rationale_short
            d["event_display"] = display_event(row["event_type"])
            d["source_display"] = display_source(row["source"])
            d["holding"] = holds
            d["measured_hit_rate_5d"] = hit_rate
            d["measured_samples"] = samples
            enriched.append(d)

        # Dedupe by (ticker, action) — keep the highest-scoring representative
        # per ticker within each action group, and attach a "+N more sources"
        # counter for the duplicates we hid.
        priority = [
            "STRONG_BUY", "STRONG_SHORT", "BUY", "SHORT",
            "TRIM", "WATCH", "AVOID", "SKIP",
        ]
        groups: dict[str, list[dict]] = {p: [] for p in priority}
        # Per-(ticker, action) representative; also track collected sources
        seen: dict[tuple[str, str], dict] = {}
        for sig in enriched:
            key = (sig["ticker"], sig["action"])
            if key in seen:
                rep = seen[key]
                rep["_other_count"] = rep.get("_other_count", 0) + 1
                # Collect other sources for tooltip-style listing later
                others = rep.setdefault("_other_sources", [])
                src_disp = sig.get("source_display") or sig.get("source")
                if src_disp and src_disp not in others:
                    others.append(src_disp)
            else:
                sig["_other_count"] = 0
                sig["_other_sources"] = []
                seen[key] = sig
                groups.setdefault(sig["action"], []).append(sig)

        # Cap each group
        for k in groups:
            groups[k] = groups[k][:limit_per_group]

    return {
        "groups": groups,
        "priority": priority,
        "counts": {k: len(v) for k, v in groups.items()},
        "t212_holdings": sorted(holdings),
    }


def cmd_metadata() -> dict[str, Any]:
    with get_connection() as conn:
        return {
            "db_path": str(CONFIG.db_path),
            "raw_signals_total": conn.execute("SELECT COUNT(*) AS n FROM raw_signals").fetchone()["n"],
            "scored_total": conn.execute("SELECT COUNT(*) AS n FROM signal_scores").fetchone()["n"],
            "outcomes_total": conn.execute("SELECT COUNT(*) AS n FROM signal_outcomes").fetchone()["n"],
            "outcomes_resolved": conn.execute("SELECT COUNT(*) AS n FROM signal_outcomes WHERE fully_resolved = 1").fetchone()["n"],
            "t212_snapshots": conn.execute("SELECT COUNT(*) AS n FROM t212_account_snapshots").fetchone()["n"],
            "configured": {
                "t212_invest": CONFIG.has_t212_invest,
                "t212_isa": CONFIG.has_t212_isa,
                "anthropic": CONFIG.has_anthropic,
                "reddit_praw": CONFIG.has_reddit,
            },
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_kwargs(args: list[str]) -> dict[str, Any]:
    """Parse `key=value` argv pairs."""
    out: dict[str, Any] = {}
    for a in args:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        # cast common types
        if v.isdigit():
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main(argv: list[str]) -> int:
    if not argv:
        _emit({"error": "usage: api.py <command> [args]"})
        return 1
    cmd = argv[0]
    rest = argv[1:]

    try:
        if cmd == "overview":
            _emit(cmd_overview())
        elif cmd == "grouped":
            _emit(cmd_grouped())
        elif cmd == "portfolio":
            _emit(cmd_portfolio())
        elif cmd == "signals":
            kwargs = _parse_kwargs(rest)
            _emit(cmd_signals(**kwargs))
        elif cmd == "ticker":
            if not rest:
                _emit({"error": "ticker command requires a symbol"})
                return 2
            _emit(cmd_ticker(rest[0]))
        elif cmd == "edge":
            _emit(cmd_edge())
        elif cmd == "health":
            _emit(cmd_health())
        elif cmd == "metadata":
            _emit(cmd_metadata())
        else:
            _emit({"error": f"unknown command: {cmd}"})
            return 3
    except Exception as exc:  # noqa: BLE001
        _emit({"error": str(exc), "command": cmd})
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
