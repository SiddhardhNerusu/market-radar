"""The fragile -> real promotion gate. Pure functions over edge_ledger rows.

The June research promised this gate in prose (EDGE_HUNT §2.4); here it is
in code. A cohort is promoted only if, on the forward record alone:

  1. >= ``min_days`` distinct event days, spanning >= 2 ISO weeks;
  2. net-of-cost MEDIAN abnormal return > 0;
  3. 10%-trimmed mean > 0;
  4. mean after dropping the top k = max(1, round(10% of n)) winners > 0
     (a fixed "drop 5" is 17% of a 30-bet sample and 1% of a 500-bet one);
  5. mean > 0 in BOTH the earliest-70%-of-days and latest-30% halves;
  6. a one-sided cluster bootstrap (resampling EVENT DAYS, so same-day bets
     are not treated as independent) rejects mean <= 0 at ``alpha``, after a
     Bonferroni divisor for the number of cohorts you are testing.

Anything that needs the mean but not the median stays fragile. Sizing
against a cohort that has not passed is the mistake this exists to prevent.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional


@dataclass
class GateResult:
    passed: bool
    n: int
    n_days: int
    stats: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)   # name -> (bool, detail)

    def summary(self) -> str:
        lines = [f"promotion gate: {'PASS' if self.passed else 'FAIL'}  (n={self.n}, days={self.n_days})"]
        for k, (ok, detail) in self.checks.items():
            lines.append(f"  [{'x' if ok else ' '}] {k}: {detail}")
        return "\n".join(lines)


def _net_values(rows: Iterable[dict], horizon: str) -> list[tuple[str, float]]:
    col = f"abn_{horizon}_pct"
    out = []
    for r in rows:
        v = r.get(col)
        if v is None:
            continue
        cost = float(r.get("cost_frac") or 0.0) * 100.0
        out.append((str(r["event_day"]), float(v) - cost))
    return out


def _trimmed_mean(xs: list[float], frac: float = 0.10) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(math.floor(len(s) * frac))
    core = s[k:len(s) - k] if len(s) - 2 * k > 0 else s
    return sum(core) / len(core)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def drop_top_k_mean(xs: list[float]) -> tuple[float, int]:
    k = max(1, int(round(0.10 * len(xs))))
    s = sorted(xs)
    rest = s[:len(s) - k]
    return (sum(rest) / len(rest) if rest else float("nan")), k


def early_late(pairs: list[tuple[str, float]], frac: float = 0.70) -> tuple[list[float], list[float]]:
    days = sorted({d for d, _ in pairs})
    cut = max(1, int(math.ceil(len(days) * frac)))
    early_days = set(days[:cut])
    return ([v for d, v in pairs if d in early_days], [v for d, v in pairs if d not in early_days])


def cluster_bootstrap_p(pairs: list[tuple[str, float]], *, n_iter: int = 5000, seed: int = 0) -> float:
    """One-sided p-value for H0: mean <= 0, resampling event DAYS with
    replacement. p = share of resampled means <= 0."""
    by_day: dict[str, list[float]] = {}
    for d, v in pairs:
        by_day.setdefault(d, []).append(v)
    days = list(by_day)
    if len(days) < 2:
        return 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_iter):
        sample = [v for d in rng.choices(days, k=len(days)) for v in by_day[d]]
        if sum(sample) / len(sample) <= 0:
            hits += 1
    return (hits + 1) / (n_iter + 1)


def _iso_weeks(days: Iterable[str]) -> int:
    weeks = set()
    for d in days:
        try:
            y, w, _ = date.fromisoformat(d).isocalendar()
            weeks.add((y, w))
        except ValueError:
            continue
    return len(weeks)


def evaluate(rows: Iterable[dict], *, horizon: str = "5d", min_days: int = 15,
             exclude_days: Iterable[str] = (), alpha: float = 0.05,
             n_families: int = 1, n_iter: int = 5000, seed: int = 0) -> GateResult:
    """Apply the gate to edge_ledger rows (dicts). ``exclude_days`` lets you
    drop a known backfill batch day so it cannot dominate the sample."""
    ex = set(exclude_days)
    pairs = [(d, v) for d, v in _net_values(rows, horizon) if d not in ex]
    xs = [v for _, v in pairs]
    n = len(xs)
    days = sorted({d for d, _ in pairs})
    res = GateResult(passed=False, n=n, n_days=len(days))
    if n == 0:
        res.checks["sample"] = (False, "no resolved rows")
        return res

    med = _median(xs)
    trim = _trimmed_mean(xs)
    dtk, k = drop_top_k_mean(xs)
    early, late = early_late(pairs)
    e_mean = sum(early) / len(early) if early else float("nan")
    l_mean = sum(late) / len(late) if late else float("nan")
    p = cluster_bootstrap_p(pairs, n_iter=n_iter, seed=seed)
    alpha_adj = alpha / max(1, n_families)
    weeks = _iso_weeks(days)
    mean = sum(xs) / n

    res.stats = {"mean": mean, "median": med, "trim10": trim, f"drop_top_{k}": dtk,
                 "early_mean": e_mean, "late_mean": l_mean, "p_cluster_bootstrap": p,
                 "alpha_adj": alpha_adj, "iso_weeks": weeks, "win_rate": sum(1 for x in xs if x > 0) / n}
    res.checks = {
        "sample_size": (len(days) >= min_days and weeks >= 2,
                        f"{len(days)} distinct days (need {min_days}), {weeks} ISO weeks (need 2)"),
        "median_gt_0": (med > 0, f"median {med:+.2f}%"),
        "trim10_gt_0": (trim > 0, f"10%-trimmed mean {trim:+.2f}%"),
        f"drop_top_{k}_gt_0": (dtk > 0, f"mean after dropping top {k}: {dtk:+.2f}%"),
        "early_and_late_gt_0": (e_mean > 0 and l_mean > 0,
                                f"early {e_mean:+.2f}% (n={len(early)}), late {l_mean:+.2f}% (n={len(late)})"),
        "cluster_bootstrap": (p < alpha_adj, f"p={p:.3f} vs alpha/{n_families}={alpha_adj:.3f}"),
    }
    res.passed = all(ok for ok, _ in res.checks.values())
    return res
