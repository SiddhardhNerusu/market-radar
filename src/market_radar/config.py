"""Central config loader. Reads .env from the project root."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # Trading 212 (HTTP Basic auth — key + secret per account)
    t212_base_url: str
    t212_invest_api_key: str
    t212_invest_api_secret: str
    t212_isa_api_key: str
    t212_isa_api_secret: str

    # Anthropic (LLM classification)
    anthropic_api_key: str
    anthropic_model: str

    # Free data sources
    finnhub_api_key: str
    newsapi_key: str

    # Alpaca (already from AUTO TRADER)
    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_base_url: str

    # Reddit
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str

    # Scoring + behavior
    notification_threshold: float
    hide_penny_default: bool

    # Paths
    db_path: Path
    project_root: Path

    # Risk infrastructure (see src/market_radar/risk/manager.py)
    risk_daily_loss_cap_usd: float = 200.0
    risk_max_gross_exposure_usd: float = 2000.0
    risk_max_position_pct: float = 5.0
    risk_max_sector_pct: float = 25.0
    risk_max_daily_trades: int = 10
    risk_max_concurrent_positions: int = 6   # anti pile-on: max distinct open names
    risk_drift_block_hours: int = 24
    risk_min_calibrated_p: float = 0.62
    risk_emergency_stop: bool = False
    # Options reserve: USD of gross exposure reserved for option spreads only.
    # Non-option proposals (stocks, crypto) cap at (gross - reserve). Ensures
    # the highest-EV path (debit spreads) always has budget.
    risk_options_reserve_usd: float = 2500.0
    # Crypto-only ceiling: crypto positions can't exceed this absolute $ value
    # WHILE THE US MARKET IS OPEN. Forces crypto selectivity so it doesn't
    # monopolize the gross budget before US-session stocks/options compete.
    risk_max_crypto_exposure_usd: float = 2000.0
    # Weekend / market-closed crypto ceiling: when stocks + options can't
    # trade, the stock+options budget sits idle, so crypto is allowed to use
    # more capital. Reverts to the weekday ceiling (above) the moment the US
    # market opens, and the Monday pre-open trim brings exposure back down.
    risk_max_crypto_exposure_weekend_usd: float = 5000.0
    # Monthly drawdown halt: if equity drops this much below 30-day rolling
    # peak, halt all trading and require manual review. Catches structural
    # issues (broken strategy, bad config) before they compound.
    risk_monthly_drawdown_usd: float = 2000.0

    # Real-time notifications (see src/market_radar/notifications/notifier.py)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    notify_macos_banner: bool = True
    notify_buy_threshold: float = 0.75
    notify_sell_threshold: float = 0.25
    notify_per_ticker_cooldown_min: int = 30
    notify_max_daily: int = 20

    @property
    def has_t212(self) -> bool:
        return bool(
            (self.t212_invest_api_key and self.t212_invest_api_secret)
            or (self.t212_isa_api_key and self.t212_isa_api_secret)
        )

    @property
    def has_t212_invest(self) -> bool:
        return bool(self.t212_invest_api_key and self.t212_invest_api_secret)

    @property
    def has_t212_isa(self) -> bool:
        return bool(self.t212_isa_api_key and self.t212_isa_api_secret)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_reddit(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret)


def load_config() -> Config:
    default_db = PROJECT_ROOT / "data" / "market_radar.db"
    db_path_env = os.getenv("DB_PATH", "").strip()
    db_path = Path(db_path_env) if db_path_env else default_db

    return Config(
        t212_base_url=os.getenv("T212_BASE_URL", "https://live.trading212.com/api/v0").strip(),
        t212_invest_api_key=os.getenv("T212_INVEST_API_KEY", "").strip(),
        t212_invest_api_secret=os.getenv("T212_INVEST_API_SECRET", "").strip(),
        t212_isa_api_key=os.getenv("T212_ISA_API_KEY", "").strip(),
        t212_isa_api_secret=os.getenv("T212_ISA_API_SECRET", "").strip(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip(),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY", "").strip(),
        newsapi_key=os.getenv("NEWSAPI_KEY", "").strip(),
        alpaca_api_key=os.getenv("ALPACA_API_KEY", "").strip(),
        alpaca_api_secret=os.getenv("ALPACA_API_SECRET", "").strip(),
        alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip(),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID", "").strip(),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "market-radar/0.1").strip(),
        notification_threshold=_float(os.getenv("NOTIFICATION_THRESHOLD"), 7.5),
        hide_penny_default=_bool(os.getenv("HIDE_PENNY_DEFAULT"), True),
        db_path=db_path,
        project_root=PROJECT_ROOT,
        risk_daily_loss_cap_usd=_float(os.getenv("RISK_DAILY_LOSS_CAP_USD"), 200.0),
        risk_max_gross_exposure_usd=_float(os.getenv("RISK_MAX_GROSS_EXPOSURE_USD"), 2000.0),
        risk_max_position_pct=_float(os.getenv("RISK_MAX_POSITION_PCT"), 5.0),
        risk_max_sector_pct=_float(os.getenv("RISK_MAX_SECTOR_PCT"), 25.0),
        risk_max_daily_trades=_int(os.getenv("RISK_MAX_DAILY_TRADES"), 10),
        risk_max_concurrent_positions=_int(os.getenv("RISK_MAX_CONCURRENT_POSITIONS"), 6),
        risk_drift_block_hours=_int(os.getenv("RISK_DRIFT_BLOCK_HOURS"), 24),
        risk_min_calibrated_p=_float(os.getenv("RISK_MIN_CALIBRATED_P"), 0.62),
        risk_emergency_stop=_bool(os.getenv("RISK_EMERGENCY_STOP"), False),
        risk_options_reserve_usd=_float(os.getenv("RISK_OPTIONS_RESERVE_USD"), 2500.0),
        risk_max_crypto_exposure_usd=_float(os.getenv("RISK_MAX_CRYPTO_EXPOSURE_USD"), 2000.0),
        risk_max_crypto_exposure_weekend_usd=_float(os.getenv("RISK_MAX_CRYPTO_EXPOSURE_WEEKEND_USD"), 5000.0),
        risk_monthly_drawdown_usd=_float(os.getenv("RISK_MONTHLY_DRAWDOWN_USD"), 2000.0),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        notify_macos_banner=_bool(os.getenv("NOTIFY_MACOS_BANNER"), True),
        notify_buy_threshold=_float(os.getenv("NOTIFY_BUY_THRESHOLD"), 0.75),
        notify_sell_threshold=_float(os.getenv("NOTIFY_SELL_THRESHOLD"), 0.25),
        notify_per_ticker_cooldown_min=_int(os.getenv("NOTIFY_PER_TICKER_COOLDOWN_MIN"), 30),
        notify_max_daily=_int(os.getenv("NOTIFY_MAX_DAILY"), 20),
    )


# Module-level singleton
CONFIG = load_config()

# Startup sanity check on critical risk caps. Today's lesson: someone set
# RISK_DAILY_LOSS_CAP_USD=99999 and the cap silently never fired, letting
# losses run to -$2k. Refuse to start with absurdly loose caps.
import logging as _logging
_log = _logging.getLogger("marketradar.config")
_SANITY_DAILY_LOSS_MAX = 2000.0  # absolute ceiling — anything above is a bug
if CONFIG.risk_daily_loss_cap_usd > _SANITY_DAILY_LOSS_MAX:
    _log.error(
        "REFUSING TO START: RISK_DAILY_LOSS_CAP_USD=%.0f exceeds sanity ceiling $%.0f. "
        "Set a reasonable cap (e.g. $500) in .env. Bot exiting.",
        CONFIG.risk_daily_loss_cap_usd, _SANITY_DAILY_LOSS_MAX,
    )
    import sys as _sys
    _sys.exit(2)
_log.info(
    "Risk caps loaded: daily_loss=$%.0f gross=$%.0f opt_reserve=$%.0f pos=%.1f%% sector=%.1f%% "
    "trades/day=%d drift_block=%dh min_p=%.2f emergency_stop=%s",
    CONFIG.risk_daily_loss_cap_usd,
    CONFIG.risk_max_gross_exposure_usd,
    CONFIG.risk_options_reserve_usd,
    CONFIG.risk_max_position_pct,
    CONFIG.risk_max_sector_pct,
    CONFIG.risk_max_daily_trades,
    CONFIG.risk_drift_block_hours,
    CONFIG.risk_min_calibrated_p,
    CONFIG.risk_emergency_stop,
)
_log.info(
    "Notify config: telegram=%s macos=%s buy>=%.2f sell<=%.2f cooldown=%dm daily_cap=%d",
    bool(CONFIG.telegram_bot_token and CONFIG.telegram_chat_id),
    CONFIG.notify_macos_banner,
    CONFIG.notify_buy_threshold,
    CONFIG.notify_sell_threshold,
    CONFIG.notify_per_ticker_cooldown_min,
    CONFIG.notify_max_daily,
)
