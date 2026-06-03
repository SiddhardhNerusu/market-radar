# MARKET RADAR — Model retrain history

Auto-appended by `scripts/scheduled_retrain.py`. Each entry records what changed and whether the new model deployed.

---

## 2026-05-29 18:34 UTC

- **Exit code**: 0 (success)
- **Old model**: `20260529T175456Z` (val_auc=0.6148)
- **New model**: `20260529T183900Z` (val_auc=0.6144)
- **Deployed**: YES
- **Horizon**: 5 day
- **Train rows**: 116712 / **Val rows**: 23342
- **Log file**: `retrain-20260529-1834.log`

---

## 2026-05-30 03:00 UTC

- **Exit code**: 0 (success)
- **Old model**: `20260529T183900Z` (val_auc=0.6144)
- **New model**: `20260529T183900Z` (val_auc=0.6144)
- **Deployed**: no — AUC gate rejected new model
- **Horizon**: 5 day
- **Train rows**: 116712 / **Val rows**: 23342
- **Log file**: `retrain-20260530-0300.log`

---

## 2026-05-31 03:00 UTC

- **Exit code**: 0 (success)
- **Old model**: `20260529T183900Z` (val_auc=0.6144)
- **New model**: `20260529T183900Z` (val_auc=0.6144)
- **Deployed**: no — AUC gate rejected new model
- **Horizon**: 5 day
- **Train rows**: 116712 / **Val rows**: 23342
- **Log file**: `retrain-20260531-0300.log`

---

## 2026-06-01 03:00 UTC

- **Exit code**: 0 (success)
- **Old model**: `20260529T183900Z` (val_auc=0.6144)
- **New model**: `20260529T183900Z` (val_auc=0.6144)
- **Deployed**: no — AUC gate rejected new model
- **Horizon**: 5 day
- **Train rows**: 116712 / **Val rows**: 23342
- **Log file**: `retrain-20260601-0300.log`

---

## 2026-06-02 03:00 UTC

- **Exit code**: 0 (success)
- **Old model**: `20260529T183900Z` (val_auc=0.6144)
- **New model**: `20260529T183900Z` (val_auc=0.6144)
- **Deployed**: no — AUC gate rejected new model
- **Horizon**: 5 day
- **Train rows**: 116712 / **Val rows**: 23342
- **Log file**: `retrain-20260602-0300.log`

---

## 2026-06-03 03:00 UTC

- **Exit code**: 1 (FAILED)
- **Old model**: `20260529T183900Z` (val_auc=0.6144)
- **New model**: `20260603T030033Z` (val_auc=0.6578)
- **Deployed**: YES
- **Horizon**: 5 day
- **Train rows**: 2839 / **Val rows**: 567
- **Log file**: `retrain-20260603-0300.log`
- **Failure reason**: exit code 1 — see log

---

