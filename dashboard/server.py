"""Local Flask server for the MARKET RADAR dashboard.

Runs on the user's Mac, serves:
  - GET /              → dashboard/index.html
  - GET /api/<command> → JSON from api.py functions (in-process — no subprocess)

Binds to 127.0.0.1 (localhost only — never reachable from outside your machine).

    python dashboard/server.py            # foreground, defaults to port 8765
    PORT=9000 python dashboard/server.py  # custom port
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Re-use the api.py command functions directly — no subprocess hops.
from dashboard import api  # noqa: E402

DASHBOARD_DIR = ROOT / "dashboard"
DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"

log = logging.getLogger("marketradar.dashboard")


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="")

    @app.route("/")
    def index() -> Response:
        return send_from_directory(str(DASHBOARD_DIR), "index.html")

    @app.route("/health")
    def health() -> Response:
        return jsonify({"ok": True, "service": "market-radar-dashboard"})

    @app.route("/graph")
    def graph_page() -> Response:
        return send_from_directory(str(DASHBOARD_DIR), "graph.html")

    @app.route("/api/graph/<ticker>")
    def graph_data(ticker: str) -> Response:
        """Return graph-only model output + co-mention peers + SHAP top
        contributors for one ticker."""
        import json as _json
        from pathlib import Path as _P
        out: dict = {"ticker": ticker.upper()}

        # 1. Graph-only model probability (if trained)
        gp = _P("data/models/current_graph_only.json")
        if gp.exists():
            try:
                meta = _json.loads(gp.read_text())
                out["graph_only_model"] = {
                    "version": meta.get("version"),
                    "val_auc": meta.get("val_auc"),
                }
            except Exception:
                pass

        # 2. Co-mention peers from a fresh graph
        try:
            from market_radar.ml.graph_features import (
                build_co_mention_graph, SECTOR_PEERS, CRYPTO_CORRELATED,
            )
            peers = build_co_mention_graph(days=90, top_k=5)
            out["co_mention_peers"] = peers.get(ticker.upper(), [])
            out["sector_peers"] = SECTOR_PEERS.get(ticker.upper(), [])
            out["is_crypto_correlated"] = ticker.upper() in CRYPTO_CORRELATED
        except Exception as exc:
            out["co_mention_error"] = str(exc)

        # 3. SHAP top contributors (if computed)
        sp = _P("data/models/shap_summary.json")
        if sp.exists():
            try:
                shap = _json.loads(sp.read_text())
                out["shap_top10"] = shap.get("feature_importance", [])[:10]
            except Exception:
                pass

        return jsonify(out)

    @app.route("/api/<command>")
    def api_route(command: str) -> Response:
        # Allow simple GET-style filter args via query string for the
        # signals endpoint
        try:
            if command == "overview":
                data = api.cmd_overview()
            elif command == "grouped":
                data = api.cmd_grouped()
            elif command == "portfolio":
                rng = request.args.get("range", "1d")
                data = api.cmd_portfolio(range_token=rng)
            elif command == "signals":
                args = request.args
                kwargs = {}
                if "score_min" in args:
                    try:
                        kwargs["score_min"] = float(args.get("score_min"))
                    except ValueError:
                        pass
                if "source_tier" in args:
                    try:
                        kwargs["source_tier"] = int(args.get("source_tier"))
                    except ValueError:
                        pass
                if "ticker" in args:
                    kwargs["ticker"] = args.get("ticker", "").upper()
                if "limit" in args:
                    try:
                        kwargs["limit"] = int(args.get("limit"))
                    except ValueError:
                        pass
                if "offset" in args:
                    try:
                        kwargs["offset"] = int(args.get("offset"))
                    except ValueError:
                        pass
                data = api.cmd_signals(**kwargs)
            elif command == "edge":
                data = api.cmd_edge()
            elif command == "health":
                data = api.cmd_health()
            elif command == "metadata":
                data = api.cmd_metadata()
            else:
                return jsonify({"error": f"unknown command: {command}"}), 404

            # Add metadata fields api._emit normally adds
            from datetime import datetime, timezone
            data.setdefault(
                "generated_at",
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            return jsonify(data)
        except Exception as exc:  # noqa: BLE001
            log.exception("api error in %s", command)
            return jsonify({"error": str(exc), "command": command}), 500

    @app.route("/api/ticker/<ticker>")
    def api_ticker(ticker: str) -> Response:
        try:
            return jsonify(api.cmd_ticker(ticker))
        except Exception as exc:  # noqa: BLE001
            log.exception("api ticker error")
            return jsonify({"error": str(exc), "ticker": ticker}), 500

    @app.route("/api/picks")
    def picks_route() -> Response:
        from dashboard.picks import build_picks, cfd_status
        direction = request.args.get("direction", "buy")
        try:
            limit = int(request.args.get("limit", 20))
        except ValueError:
            limit = 20
        try:
            horizon = int(request.args.get("horizon", 5))
        except ValueError:
            horizon = 5
        try:
            hours_window = int(request.args.get("hours_window", 48))
        except ValueError:
            hours_window = 48
        model_source = request.args.get("model", "main")
        try:
            picks = build_picks(
                direction=direction,
                limit=limit,
                horizon=horizon,
                hours_window=hours_window,
                model_source=model_source,
            )
            return jsonify({
                "direction": direction,
                "horizon": horizon,
                "model": model_source,
                "hours_window": hours_window,
                "picks": picks,
                "cfd": cfd_status(),
            })
        except Exception as exc:  # noqa: BLE001
            log.exception("picks error")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/picks/why/<int:signal_id>")
    def pick_detail_route(signal_id: int) -> Response:
        from dashboard.picks import build_pick_detail
        try:
            return jsonify(build_pick_detail(signal_id))
        except Exception as exc:  # noqa: BLE001
            log.exception("pick detail error")
            return jsonify({"error": str(exc), "signal_id": signal_id}), 500

    @app.route("/api/feed")
    def feed_route() -> Response:
        import sqlite3 as _sq
        try:
            limit = min(int(request.args.get("limit", 50)), 200)
        except ValueError:
            limit = 50
        try:
            con = _sq.connect(
                f"file:{ROOT}/data/market_radar.db?mode=ro", uri=True,
            )
            con.row_factory = _sq.Row
            rows = con.execute(
                """
                SELECT ss.id AS score_id, ss.ticker, ss.event_type,
                       ss.composite_score, ss.model_p_5d AS p, ss.scored_at,
                       rs.source, rs.title, rs.ingested_at, rs.published_at,
                       rs.url
                FROM signal_scores ss
                JOIN raw_signals rs ON rs.id = ss.signal_id
                WHERE rs.source NOT LIKE 'sec_edgar_backfill_%'
                  AND ss.model_p_5d IS NOT NULL
                ORDER BY ss.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            con.close()
            return jsonify({
                "limit": limit,
                "count": len(rows),
                "items": [dict(r) for r in rows],
            })
        except Exception as exc:  # noqa: BLE001
            log.exception("feed error")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/latency_stats")
    def latency_stats_route() -> Response:
        import sqlite3 as _sq
        try:
            con = _sq.connect(
                f"file:{ROOT}/data/market_radar.db?mode=ro", uri=True,
            )
            con.row_factory = _sq.Row
            r = con.execute(
                """
                SELECT
                  COUNT(*)              AS n,
                  AVG(latency_seconds)  AS mean_latency_s,
                  MIN(latency_seconds)  AS min_latency_s,
                  MAX(latency_seconds)  AS max_latency_s,
                  CASE WHEN COUNT(*) = 0 THEN NULL
                       ELSE SUM(CASE WHEN latency_seconds <= 60 THEN 1
                                     ELSE 0 END) * 1.0 / COUNT(*)
                  END                   AS pct_under_60s
                FROM notifications_sent
                WHERE sent_at >= datetime('now','-7 days')
                  AND latency_seconds IS NOT NULL
                """
            ).fetchone()
            con.close()
            d = dict(r) if r else {}
            return jsonify({
                "window": "7d",
                "n": d.get("n") or 0,
                "mean_latency_s": d.get("mean_latency_s"),
                "min_latency_s": d.get("min_latency_s"),
                "max_latency_s": d.get("max_latency_s"),
                "pct_under_60s": d.get("pct_under_60s"),
            })
        except Exception as exc:  # noqa: BLE001
            log.exception("latency_stats error")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/risk/status")
    def risk_status() -> Response:
        try:
            from market_radar.risk import RiskManager
            from market_radar.config import CONFIG
            rm = RiskManager()
            return jsonify({
                "emergency_stop": CONFIG.risk_emergency_stop,
                "daily_loss_cap_usd": CONFIG.risk_daily_loss_cap_usd,
                "today_pnl_usd": rm._today_realized_pnl_usd(),
                "max_gross_exposure_usd": CONFIG.risk_max_gross_exposure_usd,
                "current_gross_exposure_usd": rm._current_gross_exposure_usd(),
                "max_daily_trades": CONFIG.risk_max_daily_trades,
                "today_trade_count": rm._today_trade_count(),
                "drift_block_hours": CONFIG.risk_drift_block_hours,
                "hours_since_drift_alert": rm._hours_since_last_drift_alert(),
                "min_calibrated_p": CONFIG.risk_min_calibrated_p,
                "max_position_pct": CONFIG.risk_max_position_pct,
                "max_sector_pct": CONFIG.risk_max_sector_pct,
            })
        except Exception as exc:  # noqa: BLE001
            log.exception("risk status error")
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------
    # /bot/status — live execution layer state for the auto-trader
    # Added 2026-05-27 as part of the live trading layer.
    # ------------------------------------------------------------------
    @app.route("/bot/status")
    def bot_status() -> Response:
        try:
            from market_radar.storage import get_connection
            with get_connection() as conn:
                today_pnl = conn.execute(
                    "SELECT COALESCE(realized_pnl_usd,0), COALESCE(trades_count,0), "
                    "       COALESCE(wins,0), COALESCE(losses,0) "
                    "FROM bot_daily_pnl WHERE trading_date = date('now')"
                ).fetchone() or (0.0, 0, 0, 0)

                open_orders = conn.execute(
                    "SELECT COUNT(*) FROM bot_orders "
                    "WHERE status NOT IN ('filled','canceled','expired','rejected')"
                ).fetchone()[0]

                # Last 20 decisions (combined stock + options)
                recent_stock = conn.execute(
                    "SELECT 'stock' kind, ticker, direction, model_p, "
                    "       outcome, decided_at "
                    "FROM bot_decisions ORDER BY id DESC LIMIT 20"
                ).fetchall()
                try:
                    recent_opt = conn.execute(
                        "SELECT 'option' kind, underlying ticker, direction, "
                        "       model_p, outcome, decided_at "
                        "FROM bot_option_decisions ORDER BY id DESC LIMIT 20"
                    ).fetchall()
                except Exception:  # noqa: BLE001
                    recent_opt = []
                recent = sorted(
                    [dict(r) for r in list(recent_stock) + list(recent_opt)],
                    key=lambda d: d.get("decided_at", ""), reverse=True,
                )[:20]

                # 7-day rolling stats
                week = conn.execute(
                    "SELECT SUM(realized_pnl_usd), SUM(trades_count), "
                    "       SUM(wins), SUM(losses) FROM bot_daily_pnl "
                    "WHERE trading_date >= date('now','-7 days')"
                ).fetchone() or (0.0, 0, 0, 0)

                latest_eq = conn.execute(
                    "SELECT equity_usd, cash_usd, open_positions, snapshot_at "
                    "FROM bot_account_snapshots ORDER BY id DESC LIMIT 1"
                ).fetchone()

            # Macro regime (optional — fail-soft)
            regime_dict = None
            try:
                from market_radar.signals.macro_regime import get_regime
                r = get_regime()
                regime_dict = {
                    "bias": r.bias,
                    "size_multiplier": r.size_multiplier,
                    "allow_longs": r.allow_longs,
                    "allow_shorts": r.allow_shorts,
                    "vix": r.vix,
                    "spy_50d_trend": r.spy_50d_trend,
                    "reason": r.reason,
                }
            except Exception:  # noqa: BLE001
                pass

            week_pnl, week_trades, week_wins, week_losses = week
            win_rate_7d = (week_wins / max(week_trades, 1) * 100) if week_trades else 0
            return jsonify({
                "today": {
                    "pnl_usd": today_pnl[0],
                    "trades": today_pnl[1],
                    "wins": today_pnl[2],
                    "losses": today_pnl[3],
                },
                "week": {
                    "pnl_usd": week_pnl or 0,
                    "trades": week_trades or 0,
                    "win_rate_pct": win_rate_7d,
                },
                "open_orders": open_orders,
                "account": (
                    dict(latest_eq) if latest_eq else None
                ),
                "regime": regime_dict,
                "recent_decisions": recent,
            })
        except Exception as exc:  # noqa: BLE001
            log.exception("bot status error")
            return jsonify({"error": str(exc)}), 500

    return app


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    host = os.environ.get("HOST", DEFAULT_HOST)
    app = create_app()
    log.info("MARKET RADAR dashboard → http://%s:%d/", host, port)
    log.info("(Ctrl-C to stop)")
    # use_reloader=False so it works under launchd
    app.run(host=host, port=port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
