"""Validate whether the signal scoring is actually predictive of forward returns.

The core question: do higher-scored signals actually go up more? A good score
shows (a) MONOTONIC mean forward return across score deciles, and (b) a positive
rank Information Coefficient (IC). This is the measurement foundation the audit
flagged as missing — and the prerequisite for safely rebalancing the composite
weights (you can't reset trade thresholds without knowing where the edge lives).

Read-only. Run from the project root: .venv/bin/python scripts/validate_scoring.py
"""
import sqlite3
import pandas as pd

DB = "data/market_radar.db"
HORIZON = "return_5d_pct"  # primary horizon the gate cares about


def _deciles(df, score_col):
    d = df.dropna(subset=[score_col, HORIZON]).copy()
    if len(d) < 100:
        return None, None, len(d)
    try:
        d["bucket"] = pd.qcut(d[score_col].rank(method="first"), 10, labels=False)
    except ValueError:
        return None, None, len(d)
    tbl = d.groupby("bucket")[HORIZON].agg(["mean", "count"])
    ic = d[score_col].corr(d[HORIZON], method="spearman")
    return tbl, ic, len(d)


def _report(df, label):
    print(f"\n===== {label}  (n={len(df)}) =====")
    for score_col in ("composite_score", "model_p_5d"):
        tbl, ic, n = _deciles(df, score_col)
        if tbl is None:
            print(f"  {score_col}: too few rows ({n})")
            continue
        lo = tbl["mean"].iloc[0]
        hi = tbl["mean"].iloc[-1]
        steps = tbl["mean"].values
        up = sum(1 for i in range(1, len(steps)) if steps[i] > steps[i - 1])
        mono = up / (len(steps) - 1)
        print(f"  {score_col:16} rank IC={ic:+.3f}  "
              f"bottom-decile={lo:+.2f}%  top-decile={hi:+.2f}%  "
              f"top-minus-bottom={hi - lo:+.2f}%  monotonic_steps={mono:.0%}")


conn = sqlite3.connect(DB)
df = pd.read_sql_query(
    f"""
    SELECT ss.composite_score, ss.model_p_5d, ss.event_type, so.{HORIZON}
    FROM signal_scores ss
    JOIN signal_outcomes so ON so.score_id = ss.id
    WHERE so.{HORIZON} IS NOT NULL AND ABS(so.{HORIZON}) < 200
    """,
    conn,
)
conn.close()

print("Higher score should mean higher forward return: rising deciles + positive IC.")
print("(IC ~0.03-0.05 stable = a real usable signal; ~0 = no edge; negative = inverted.)")
_report(df, "ALL signals")
_report(df[df["event_type"] != "other"], "TRADEABLE (event_type != 'other')")
alpha = df[df["event_type"].isin([
    "m_a_announcement", "fda_approval", "fda_rejection", "earnings_beat",
    "earnings_miss", "guidance_raise", "activist_position", "insider_buy",
])]
_report(alpha, "HIGH-ALPHA event types only")
