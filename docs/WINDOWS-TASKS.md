# Windows Task Scheduler — equivalents to Mac launchd jobs

## 1. Daily DB backup (3am)
Action: Start a program
Program: C:\MARKET_RADAR\.venv\Scripts\python.exe
Arguments: C:\MARKET_RADAR\scripts\backup_db.py
Start in: C:\MARKET_RADAR
Trigger: Daily at 03:00

## 2. Weekly auto-retrain (Sunday 4am)
Action: Start a program
Program: C:\MARKET_RADAR\.venv\Scripts\python.exe
Arguments: C:\MARKET_RADAR\scripts\train_ml.py --all
Start in: C:\MARKET_RADAR
Trigger: Weekly on Sunday at 04:00

## 3. Drift check (every 6h)
Action: Start a program
Program: C:\MARKET_RADAR\.venv\Scripts\python.exe
Arguments: C:\MARKET_RADAR\scripts\check_model_drift.py --window 500
Start in: C:\MARKET_RADAR
Trigger: Daily at 00:00, repeat every 6 hours, for 1 day

## 4. Daemon as a Windows service (via NSSM)
Download NSSM: https://nssm.cc/download
nssm install MarketRadarDaemon
  Path: C:\MARKET_RADAR\.venv\Scripts\python.exe
  Startup directory: C:\MARKET_RADAR
  Arguments: scripts\run_daemon.py
  Stdout: C:\MARKET_RADAR\logs\daemon.out.log
  Stderr: C:\MARKET_RADAR\logs\daemon.err.log
nssm set MarketRadarDaemon Start SERVICE_AUTO_START
net start MarketRadarDaemon
