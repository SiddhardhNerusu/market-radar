"""MARKET RADAR daemon — long-running orchestrator.

Wires every ingestor, scorer, outcome tracker, T212 snapshotter, and
notifier into one APScheduler-driven process. Designed to run under
launchd: ``KeepAlive=true`` means crashes are restarted automatically.

EXECUTION GATE — when an execution adapter (e.g. T212 CFD) is added,
EVERY order MUST be evaluated by ``market_radar.risk.RiskManager`` first.
The daemon today is paper-only: nothing in this file places trades.
The risk manager exists at src/market_radar/risk/manager.py and exposes
RiskManager().evaluate(TradeProposal(...)) -> RiskDecision; see its
docstring for the seven hard rules and the fail-closed contract.
DO NOT bypass the gate when wiring execution.

Schedule (defaults):
  - SEC EDGAR poll          every 5 min
  - RSS news poll           every 3 min
  - Reddit / StockTwits     every 3 min
  - T212 snapshot           every 5 min (read-only)
  - Score pending           every 30 sec
  - Snapshot outcomes       every 1 min
  - Update due outcomes     every 1 hour
  - Dispatch notifications  every 30 sec (after scoring)

All polls run defensively — one failed source never crashes the daemon.
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .config import CONFIG
from .ingestors import (
    RedditPublicIngestor,
    RssNewsIngestor,
    SecEdgarIngestor,
    StockTwitsTrendingIngestor,
)
from .notifications import dispatch_pending
from .outcomes import snapshot_pending_outcomes, update_due_outcomes
from .scoring import score_pending
from .storage import init_db
from .t212.snapshotter import snapshot_all as snapshot_t212

log = logging.getLogger("marketradar.daemon")


# ---------------------------------------------------------------------------
# Configurable intervals (seconds)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "sec_edgar_seconds":   60,          # stub-first ingest (no HTTP in poll loop)
    "sec_hydrate_seconds": 20,          # out-of-band SEC body hydration (latency fix)
    "rss_news_seconds":    3 * 60,
    "alpaca_news_seconds": 30,          # catalyst firehose — tightened 60->30 for latency
    "halts_seconds":       60,          # Nasdaq trading-halt feed — real-time bang detector
    "movers_seconds":      120,         # market-wide top-gainers/most-active — bang scanner
    "reddit_seconds":      3 * 60,
    "stocktwits_seconds":  2 * 60,
    "earnings_calendar_seconds": 12 * 60 * 60,  # twice a day — Finnhub free tier
    "t212_snap_seconds":   5 * 60,
    "scoring_seconds":     60,          # was 30 — a backlog batch ran >30s and overran the tick
                                        # (~8.9k 'max instances reached' warns/audit); 60s fits the window
    "outcome_snap_seconds": 60,
    "outcome_update_seconds": 60 * 60,
    "notify_seconds":      30,
    "ml_predict_seconds":  60,
    "llm_classify_seconds": 45,         # tightened 90->45 — LLM classify is on the catalyst critical path
    "price_action_seconds": 60,
    "health_check_seconds": 30 * 60,    # stalled-source detector (blueprint #8)
    "finra_short_interest_seconds": 12 * 60 * 60,  # FINRA bi-monthly short interest (#5)
    "clinicaltrials_seconds": 8 * 60 * 60,         # ClinicalTrials.gov v2 readouts (#5)
    "near_dup_seconds": 6 * 60 * 60,               # SimHash near-dup re-cluster (#6)
    "reconcile_pnl_seconds": 30 * 60,              # rebuild bot_daily_pnl realized from Alpaca fills (source of truth)
}


# Cached price-action scanner — instantiated on first call so daemon
# still loads cleanly when Alpaca creds are missing.
_PA_SCANNER = None


# Cache ingestor instances so their requests.Session (and TCP connection pool)
# persists across polls. Re-creating an ingestor every interval tick opened a
# fresh Session each time — the likely cause of the prior file-descriptor
# exhaustion. Lazily built so missing creds never break daemon load.
_INGESTORS: dict = {}


def _cached(key: str, factory):
    inst = _INGESTORS.get(key)
    if inst is None:
        inst = factory()
        _INGESTORS[key] = inst
    return inst


# ---------------------------------------------------------------------------
# Job wrappers
# ---------------------------------------------------------------------------

def _safe(name: str, fn):
    """Wrap a job so an exception in one job never kills the scheduler."""
    def runner():
        t0 = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            log.exception("[job:%s] crashed: %s", name, exc)
        else:
            log.debug("[job:%s] ok in %.1fs", name, time.time() - t0)
    runner.__name__ = f"run_{name}"
    return runner


def _job_sec_edgar() -> None:
    # Stub-first (blueprint #2 latency fix): fetch_bodies=False keeps ALL HTTP
    # out of the poll loop so the SQLite writer is never held across the network.
    # Bodies are filled out-of-band by _job_sec_hydrate below.
    _cached("sec_edgar", lambda: SecEdgarIngestor(fetch_bodies=False)).poll()

def _job_sec_hydrate() -> None:
    from .ingestors.sec_hydrate import hydrate_sec_bodies
    hydrate_sec_bodies()

def _job_rss_news() -> None:
    _cached("rss_news", RssNewsIngestor).poll()

def _job_alpaca_news() -> None:
    """Pull the market-wide Alpaca/Benzinga news firehose (every ticker).

    This is the blind-spot fix for micro-cap catalyst PRs (e.g. VERU +159% on a
    Novo Nordisk deal that we never saw). No-ops cleanly when creds are missing.
    """
    if not (CONFIG.alpaca_api_key and CONFIG.alpaca_api_secret):
        return
    from .ingestors.alpaca_news import AlpacaNewsIngestor
    _cached("alpaca_news", AlpacaNewsIngestor).poll()

def _job_reddit() -> None:
    # Disabled 2026-05-29: Reddit public JSON returns 6,930 HTTP 403s/day
    # without authentication. Ingestor was silently logging warnings and
    # contributing zero usable signals. Re-enable when PRAW OAuth is wired.
    if not CONFIG.has_reddit:
        return
    RedditPublicIngestor().poll()

def _job_stocktwits() -> None:
    StockTwitsTrendingIngestor().poll()

def _job_halts() -> None:
    """Pull the Nasdaq Trader trading-halt feed — the real-time 'this stock is
    banging right now' detector (LULD volatility halts on micro-caps, plus news +
    regulatory halts). Public feed, no credentials needed."""
    from .ingestors.halts import NasdaqHaltsIngestor
    _cached("halts", NasdaqHaltsIngestor).poll()

def _job_movers() -> None:
    """Market-wide top-gainers + most-active scanner — catches the micro-cap bangs
    the ~85-name price-action scanner is blind to (INHD +1897% etc.). No-ops
    cleanly when Alpaca creds are missing."""
    if not (CONFIG.alpaca_api_key and CONFIG.alpaca_api_secret):
        return
    from .ingestors.market_movers import MarketMoversIngestor
    _cached("movers", MarketMoversIngestor).poll()

def _job_earnings_calendar() -> None:
    """Pull next 30 days of US earnings from Finnhub. Runs twice daily."""
    from .ingestors.earnings_calendar import EarningsCalendarIngestor
    _cached("earnings_calendar", EarningsCalendarIngestor).poll()

def _job_t212_snap() -> None:
    # DEAD PATH (P3 hardening 2026-06-15): T212 is a READ-ONLY snapshot with NO
    # execution adapter wired (see module docstring) — it polls LIVE-MONEY
    # credentials on a timer for a path that never trades. Disabled unless the
    # operator explicitly opts in via ENABLE_T212_SNAPSHOT, so live-money creds
    # aren't exercised for nothing.
    import os as _os
    if _os.getenv("ENABLE_T212_SNAPSHOT", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    if not CONFIG.has_t212:
        return
    snapshot_t212()

def _job_score() -> None:
    score_pending()

def _job_outcome_snap() -> None:
    snapshot_pending_outcomes()

def _job_outcome_update() -> None:
    update_due_outcomes()

def _job_notify() -> None:
    dispatch_pending()

def _job_ml_predict() -> None:
    from .ml import get_predictor
    predictor = get_predictor()
    if predictor is None:
        return
    predictor.predict_pending()

def _job_llm_classify() -> None:
    if not CONFIG.has_anthropic:
        return
    try:
        from .llm import classify_pending
        classify_pending(batch_size=30)
    except RuntimeError as exc:
        log.warning("LLM classify skipped: %s", exc)
    except Exception as exc:  # noqa: BLE001
        log.exception("LLM classify job error: %s", exc)

def _job_price_action() -> None:
    """Scan the liquid equity universe and emit price-action signals.

    No-ops cleanly when Alpaca creds are missing.
    """
    if not (CONFIG.alpaca_api_key and CONFIG.alpaca_api_secret):
        return
    global _PA_SCANNER
    try:
        from .signals import PriceActionScanner
        if _PA_SCANNER is None:
            _PA_SCANNER = PriceActionScanner()
        _PA_SCANNER.run()
    except Exception as exc:  # noqa: BLE001
        log.exception("price_action job error: %s", exc)


def _job_ml_retrain() -> None:
    from .ml import train_and_save
    try:
        result = train_and_save(only_replace_if_better=True)
        log.info(
            "[ml_retrain] success=%s version=%s val_auc=%s reason=%s",
            result.success, result.version, result.val_auc, result.reason,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("[ml_retrain] failed: %s", exc)


# Throttle health alerts to once/source/day.
_HEALTH_ALERTED_ON: dict = {}


def _job_health_check() -> None:
    """Alert on sources that fetch OK but have silently STOPPED producing
    (blueprint #8 — a broken parser/source). Throttled once/source/day."""
    from datetime import date as _date
    from .storage import get_connection
    from .storage.db import stalled_sources
    today = _date.today().isoformat()
    with get_connection() as conn:
        stalled = stalled_sources(conn)
    for src in stalled:
        if _HEALTH_ALERTED_ON.get(src) == today:
            continue
        _HEALTH_ALERTED_ON[src] = today
        log.error("[health] source '%s' fetched OK but produced 0 signals for many "
                  "consecutive polls — likely broken", src)
        try:
            from .notifications.realtime import TradeAlert, notify_trade
            notify_trade(TradeAlert(
                kind="ALERT", symbol="DAEMON",
                extra=f"source '{src}' stalled — fetching OK but producing 0 signals "
                      f"(likely a broken parser/source)"))
        except Exception:  # noqa: BLE001 — never let alerting crash the job
            pass


def _job_finra_short_interest() -> None:
    """FINRA consolidated short interest -> short_interest table (blueprint #5).
    Public, no-auth; idempotent UPSERT keyed on (ticker, report_date)."""
    from .ingestors.finra_short_interest import FinraShortInterestIngestor
    _cached("finra_short_interest", FinraShortInterestIngestor).poll()


def _job_clinicaltrials() -> None:
    """ClinicalTrials.gov v2 Phase 2/3 readouts -> catalysts (blueprint #5).
    Public API, best-effort sponsor->ticker. Never raises."""
    from .ingestors.clinicaltrials import ClinicalTrialsIngestor
    _cached("clinicaltrials", ClinicalTrialsIngestor).poll()


def _job_near_dup() -> None:
    """Re-cluster trailing-14d raw_signals into near-dup groups (SimHash+LSH,
    blueprint #6). Runs the standalone backfill in a subprocess so the heavy
    clustering stays out of the scheduler thread."""
    import subprocess
    subprocess.run(
        [sys.executable, str(CONFIG.project_root / "scripts" / "backfill_near_dup.py"),
         "--days", "14"],
        check=False, timeout=900,
    )


def _job_daily_assessment() -> None:
    """Post the honest daily-assessment digest to Telegram (deployed-state audit
    watch-items: true P&L, edge-day progress vs the 40-day gate, pipeline health,
    safety events). Subprocess so the queries never block the scheduler thread."""
    import subprocess
    subprocess.run(
        [sys.executable, str(CONFIG.project_root / "scripts" / "daily_assessment.py")],
        check=False, timeout=180,
    )


# ---------------------------------------------------------------------------
# Daemon entry point
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Console + rotating file logs. Keeps daemon.log under 10 MB with 5 rolls."""
    from logging.handlers import RotatingFileHandler

    log_dir = CONFIG.project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "daemon.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=5,                # keep 5 rotations → max ~60 MB on disk
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    # Quiet werkzeug's per-request access log to avoid log spam
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    # apscheduler logs every job execution + "max instances reached" at INFO —
    # ~25k lines/day that filled the UNROTATED launchd stdout to 228MB (audit R2).
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    # yfinance emits its own ERROR lines for crypto/index symbols it can't price
    # ("$BTCUSD: possibly delisted", "$ES_F: no timezone found") — ~3,400/run of
    # benign spam that buries real errors. Our price fetcher already catches those
    # failures at debug level, so silence yfinance's logger.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler],
        force=True,
    )


def _job_reconcile_pnl() -> None:
    """Rebuild bot_daily_pnl realized P&L from Alpaca equity fills — the SOURCE OF
    TRUTH (full-audit ch6). The per-trade ledger had drifted to ~13% of real P&L
    because exits were mis-booked; this idempotent fills FIFO replaces realized /
    trades / wins / losses for the equity-lane era. The equity CURVE remains the
    authoritative headline (daily_assessment); this keeps the per-trade view honest."""
    from .execution.alpaca_client import AlpacaClient
    from .execution.pnl_reconcile import reconcile_daily_pnl_from_fills
    from .storage import get_connection
    since = os.getenv("PNL_RECONCILE_SINCE", "2026-06-15")
    summ = reconcile_daily_pnl_from_fills(AlpacaClient(), get_connection, since_date=since)
    log.info("[reconcile_pnl] rebuilt %d day(s) since %s | equity-lane realized=%s | uncovered=%d",
             summ["dates_written"], since, summ["total_realized"], summ["uncovered"])


def build_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(
        timezone="UTC",
        job_defaults={
            "max_instances": 1,
            "coalesce": True,
            "misfire_grace_time": 60,
        },
    )

    jobs = [
        ("sec_edgar",      _job_sec_edgar,      DEFAULTS["sec_edgar_seconds"]),
        ("sec_hydrate",    _job_sec_hydrate,    DEFAULTS["sec_hydrate_seconds"]),
        ("rss_news",       _job_rss_news,       DEFAULTS["rss_news_seconds"]),
        ("alpaca_news",    _job_alpaca_news,    DEFAULTS["alpaca_news_seconds"]),
        ("halts",          _job_halts,          DEFAULTS["halts_seconds"]),
        ("movers",         _job_movers,         DEFAULTS["movers_seconds"]),
        ("reddit",         _job_reddit,         DEFAULTS["reddit_seconds"]),
        # DISABLED 2026-06-08: StockTwits is Tier 3 — 0 trades ever, measured negative
        # edge, and EXCLUDED from corroboration (only Tier 1/2 count). It was ~44% of
        # scoring volume, backing up the queue for zero trading benefit. Verified safe
        # to cut. Re-enable by uncommenting this single line.
        # ("stocktwits",     _job_stocktwits,     DEFAULTS["stocktwits_seconds"]),
        ("earnings_cal",   _job_earnings_calendar, DEFAULTS["earnings_calendar_seconds"]),
        ("t212_snapshot",  _job_t212_snap,      DEFAULTS["t212_snap_seconds"]),
        ("score",          _job_score,          DEFAULTS["scoring_seconds"]),
        ("outcome_snap",   _job_outcome_snap,   DEFAULTS["outcome_snap_seconds"]),
        ("outcome_update", _job_outcome_update, DEFAULTS["outcome_update_seconds"]),
        ("notify",         _job_notify,         DEFAULTS["notify_seconds"]),
        ("ml_predict",     _job_ml_predict,     DEFAULTS["ml_predict_seconds"]),
        ("llm_classify",   _job_llm_classify,   DEFAULTS["llm_classify_seconds"]),
        ("price_action",   _job_price_action,   DEFAULTS["price_action_seconds"]),
        ("health_check",   _job_health_check,   DEFAULTS["health_check_seconds"]),
        ("finra_short_int", _job_finra_short_interest, DEFAULTS["finra_short_interest_seconds"]),
        ("clinicaltrials", _job_clinicaltrials, DEFAULTS["clinicaltrials_seconds"]),
        ("near_dup",       _job_near_dup,       DEFAULTS["near_dup_seconds"]),
        ("reconcile_pnl",  _job_reconcile_pnl,  DEFAULTS["reconcile_pnl_seconds"]),
    ]
    for i, (name, fn, interval) in enumerate(jobs):
        sched.add_job(
            _safe(name, fn),
            trigger=IntervalTrigger(seconds=interval),
            id=name,
            replace_existing=True,
            # Stagger startup ~3s/job so a restart doesn't fire all ~15 jobs
            # (50 RSS feeds + SEC + movers + scoring + ML) at once — that thundering
            # herd spiked connections/FDs. tz-aware (datetime.utcnow() is deprecated).
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=i * 3),
        )
        log.info("scheduled %-20s every %ds", name, interval)

    # Weekly model retrain — Monday 04:00 UTC. The job is a no-op if there
    # aren't enough resolved outcomes yet (returns reason="Not enough rows").
    sched.add_job(
        _safe("ml_retrain", _job_ml_retrain),
        trigger=CronTrigger(day_of_week="mon", hour=4, minute=0, timezone="UTC"),
        id="ml_retrain",
        replace_existing=True,
    )
    log.info("scheduled %-20s weekly Mon 04:00 UTC", "ml_retrain")

    # Daily honest assessment digest — 21:35 UTC (after the US close).
    sched.add_job(
        _safe("daily_assessment", _job_daily_assessment),
        trigger=CronTrigger(hour=21, minute=35, timezone="UTC"),
        id="daily_assessment",
        replace_existing=True,
    )
    log.info("scheduled %-20s daily 21:35 UTC", "daily_assessment")
    return sched


def start_dashboard_server() -> None:
    """Start the Flask dashboard server in a daemon thread.

    Imported lazily so the daemon still runs even if Flask is missing
    (e.g. before requirements.txt has been pip-installed).
    """
    import os
    import threading
    try:
        sys.path.insert(0, str(CONFIG.project_root))
        from dashboard.server import create_app  # type: ignore
    except ImportError as exc:
        log.warning("Dashboard server not started — Flask import failed: %s", exc)
        log.warning("Install with: pip install flask")
        return

    port = int(os.environ.get("DASHBOARD_PORT", 8765))
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")

    def _run():
        try:
            app = create_app()
            # use_reloader=False because we're in a thread; threaded=True so
            # multiple browser tabs can hit it without serialising.
            app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("Dashboard server crashed: %s", exc)

    thread = threading.Thread(target=_run, name="dashboard-server", daemon=True)
    thread.start()
    log.info("Dashboard: http://%s:%d/", host, port)


def main() -> int:
    configure_logging()
    log.info("=" * 70)
    log.info("MARKET RADAR daemon starting")
    log.info("DB:       %s", CONFIG.db_path)
    log.info("T212:     %s", "configured" if CONFIG.has_t212 else "MISSING")
    log.info("  invest: %s", "ok" if CONFIG.has_t212_invest else "missing")
    log.info("  isa:    %s", "ok" if CONFIG.has_t212_isa else "missing")
    log.info("LLM:      %s", "configured" if CONFIG.has_anthropic else "missing (heuristics only)")
    log.info("Reddit:   %s", "PRAW configured" if CONFIG.has_reddit else "public-JSON only")
    log.info("Notif threshold: %.1f", CONFIG.notification_threshold)
    log.info("=" * 70)

    init_db()
    start_dashboard_server()
    sched = build_scheduler()
    sched.start()

    stop_flag = {"stop": False}

    def handle_sig(signum, frame):  # noqa: ARG001
        log.info("Received signal %s — shutting down", signum)
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    while not stop_flag["stop"]:
        time.sleep(1)

    log.info("Stopping scheduler …")
    sched.shutdown(wait=True)
    log.info("MARKET RADAR daemon stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
