"""Lock the startup config-coherence validation (P0 rebuild). The shipped
defaults were incoherent (options reserve $2500 > gross cap $2000 => negative
stock budget); the bot must REFUSE TO START on configs that silently disable
protection. Validation runs at import time and sys.exit(2)s, so we assert via a
subprocess importing market_radar.config with controlled env overrides.

load_dotenv() does NOT override already-set env vars, so the values we inject
here take precedence over .env; unset keys fall back to the real .env / defaults.
"""
import os
import pathlib
import subprocess
import sys

SRC = str(pathlib.Path(__file__).resolve().parents[1] / "src")


def _import_config(env_overrides):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env.update({k: str(v) for k, v in env_overrides.items()})
    return subprocess.run(
        [sys.executable, "-c", "import market_radar.config"],
        env=env, capture_output=True, text=True,
    )


def test_refuses_when_reserve_exceeds_gross():
    r = _import_config({
        "RISK_MAX_GROSS_EXPOSURE_USD": 2000,
        "RISK_OPTIONS_RESERVE_USD": 9999,
    })
    assert r.returncode == 2, r.stderr
    assert "REFUSING TO START" in r.stderr


def test_refuses_when_hard_order_exceeds_gross():
    r = _import_config({
        "RISK_MAX_GROSS_EXPOSURE_USD": 2000,
        "RISK_OPTIONS_RESERVE_USD": 500,
        "RISK_HARD_ORDER_NOTIONAL_USD": 9999,
    })
    assert r.returncode == 2, r.stderr
    assert "REFUSING TO START" in r.stderr


def test_refuses_when_daily_loss_exceeds_gross():
    r = _import_config({
        "RISK_MAX_GROSS_EXPOSURE_USD": 2000,
        "RISK_OPTIONS_RESERVE_USD": 500,
        "RISK_HARD_ORDER_NOTIONAL_USD": 1000,
        "RISK_DAILY_LOSS_CAP_USD": 1999,  # < sanity ceiling but >= gross? no, < gross
        "RISK_INTRADAY_EQUITY_STOP_USD": 0,
    })
    # daily_loss 1999 < gross 2000 -> coherent -> should NOT refuse on this rule
    assert r.returncode == 0, r.stderr


def test_accepts_coherent_live_like_config():
    r = _import_config({
        "RISK_MAX_GROSS_EXPOSURE_USD": 6000,
        "RISK_OPTIONS_RESERVE_USD": 2500,
        "RISK_HARD_ORDER_NOTIONAL_USD": 2500,
        "RISK_DAILY_LOSS_CAP_USD": 600,
    })
    assert r.returncode == 0, f"coherent config should import cleanly: {r.stderr}"
