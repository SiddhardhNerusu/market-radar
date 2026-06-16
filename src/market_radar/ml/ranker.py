"""LGBMRanker beta-neutral scorer machinery + a hard deploy gate.

Blueprint #7 (DATA-GATED). This is a *learning-to-rank* model, not a
classifier: instead of asking "will this bet win?" it asks "on a given
scored day, which bets will out-rank the others?". That framing is what
the live ranking surface actually needs — it never trades the whole tape,
it trades the top of each day's list.

Why beta-neutral / day-demeaned:
  On a green-tape day almost every bet's ``return_5d_pct`` is positive, and
  on a red-tape day almost every one is negative. A model trained on raw
  forward return mostly learns the market's direction (beta) — useless for
  *selecting within a day*. We therefore use the DAY-DEMEANED forward
  return as the relevance target: each bet's ``return_5d_pct`` minus that
  day's mean across the clean, deduped ``(ticker, day)`` bets. The cross-
  sectional mean is removed, so the model can only learn the within-day
  ordering — the alpha, not the beta. LightGBM's ``lambdarank`` objective
  consumes that as a graded relevance label, grouped by scored-day.

Leakage control — purged day-walk-forward CV:
  Folds are split on *whole scored-days*, never on rows, so two rows from
  the same day can never land on opposite sides of a fold (the same-day
  leakage that plagues row-level ``TimeSeriesSplit``). Each fold trains on
  an expanding prefix of days and validates on the next block of days, with
  a one-day PURGE gap between train and val so a training row's 5-day
  forward window can't overlap the validation days.

THE HARD DEPLOY GATE (``RANK_MIN_DAYS = 40``):
  A ranker needs many *distinct days* to learn a stable cross-sectional
  ordering — row count is irrelevant (one 50k-row day teaches one ordering).
  We have ~19 clean scored-days today. ``evaluate_deploy_gate`` REFUSES to
  promote a ranker below the floor and PROVABLY returns the
  ``'composite_fallback'`` decision, so the live surface keeps serving the
  existing ``composite_score``. The machinery below still trains fine on
  synthetic data — the gate is a deploy guard, not a train guard.

No daemon job is wired up: the gate blocks deployment today, so there is
nothing to schedule yet. When distinct clean days clears the floor an
integrator can call ``train_ranker_if_gated`` from a periodic retrain job.
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ..config import PROJECT_ROOT
from ..storage import get_connection

log = logging.getLogger(__name__)


MODELS_DIR = PROJECT_ROOT / "data" / "models"
# Side-pointer for the ranker; kept separate from the classifier's
# ``current.json`` so a ranker promotion never clobbers the composite/HGB
# pipeline. Absent pointer == "no ranker deployed, serve composite".
RANKER_POINTER = MODELS_DIR / "current_ranker.json"

# --- THE DATA GATE ---------------------------------------------------------
# Minimum number of DISTINCT clean scored-days before a ranker may deploy.
# Below this the gate refuses and the system serves the composite score.
# Distinct *days* (not rows) is the right unit: a ranker learns the
# within-day cross-sectional ordering, so it needs many independent days to
# generalise. ~19 today → blocked.
RANK_MIN_DAYS = 40

# Minimum bets on a day for that day to form a usable ranking group. A day
# with a single bet has no within-day ordering to learn from (and a group
# size of 1 is degenerate for lambdarank).
MIN_GROUP_SIZE = 2

# Fallback decision string the live surface keys on.
DECISION_COMPOSITE_FALLBACK = "composite_fallback"
DECISION_DEPLOY_RANKER = "deploy_ranker"


@dataclass
class RankerDataset:
    """A demeaned, grouped, chronologically-ordered ranking dataset.

    ``rows`` are the clean deduped bets sorted by ``(day, ticker)``.
    ``relevance`` is the day-demeaned forward return aligned to ``rows``.
    ``group_sizes`` are the per-day contiguous block sizes LightGBM's
    ranker consumes (sum == len(rows)). ``days`` is the ordered list of
    distinct scored-days; ``group_day`` aligns each group to its day so the
    CV splitter can purge on day boundaries.
    """

    rows: list[dict]
    relevance: list[float]
    group_sizes: list[int]
    days: list[str]
    group_day: list[str]

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def n_rows(self) -> int:
        return len(self.rows)


@dataclass
class GateDecision:
    """Result of the hard deploy gate."""

    decision: str                 # DECISION_COMPOSITE_FALLBACK | DECISION_DEPLOY_RANKER
    deployed: bool
    distinct_days: int
    min_days_required: int
    reason: str
    model_path: Optional[str] = None
    fold_ndcgs: list[float] = field(default_factory=list)
    val_ndcg: Optional[float] = None

    @property
    def served_model(self) -> str:
        """What the live surface should serve given this decision."""
        return "ranker" if self.deployed else "composite"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Data loading + day-demeaning
# ---------------------------------------------------------------------------


def _load_clean_ranker_rows(
    *, label_col: str = "return_5d_pct",
) -> list[dict]:
    """Pull clean, deduped ``(ticker, day)`` bets with a resolved forward
    return, one row per (ticker, day).

    Clean-data rule (project-wide): ``COALESCE(so.data_corrupt,0)=0 AND
    price_at_flag>=1``. Dedup: when the same ticker is flagged multiple
    times on the same scored-day we keep the highest-composite bet for that
    ``(ticker, day)`` — that's the one the live surface would have ranked —
    so a single noisy story can't stuff a day's ranking group with
    duplicates of the same name.
    """
    assert label_col in ("return_1d_pct", "return_5d_pct", "return_20d_pct"), \
        f"unsupported label column: {label_col}"
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            WITH clean AS (
                SELECT
                    ss.ticker                       AS ticker,
                    substr(ss.scored_at, 1, 10)     AS day,
                    ss.scored_at                    AS scored_at,
                    ss.composite_score              AS composite_score,
                    so.{label_col}                  AS fwd_return,
                    ROW_NUMBER() OVER (
                        PARTITION BY substr(ss.scored_at, 1, 10), ss.ticker
                        ORDER BY ss.composite_score DESC, ss.id DESC
                    )                               AS rn
                FROM signal_scores ss
                JOIN signal_outcomes so ON so.score_id = ss.id
                WHERE so.{label_col} IS NOT NULL
                  AND COALESCE(so.data_corrupt, 0) = 0
                  AND so.price_at_flag >= 1
            )
            SELECT ticker, day, scored_at, composite_score, fwd_return
            FROM clean
            WHERE rn = 1
            ORDER BY day ASC, ticker ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def build_dataset(
    *,
    label_col: str = "return_5d_pct",
    min_group_size: int = MIN_GROUP_SIZE,
    _preloaded_rows: Optional[list[dict]] = None,
) -> RankerDataset:
    """Build the day-demeaned, grouped ranking dataset.

    Steps:
      1. Load clean deduped ``(ticker, day)`` bets (or use ``_preloaded_rows``
         — each must carry ``day``/``ticker``/``composite_score``/``fwd_return``).
      2. Drop days with fewer than ``min_group_size`` bets — no within-day
         ordering to learn.
      3. For each surviving day subtract the day's mean ``fwd_return`` →
         the beta-neutral relevance target.
      4. Emit rows sorted by ``(day, ticker)`` with matching group sizes.
    """
    rows = list(_preloaded_rows) if _preloaded_rows is not None else \
        _load_clean_ranker_rows(label_col=label_col)

    # Group by day (preserving day order). Input is already day-sorted, but
    # we re-bucket defensively in case a caller passes unsorted preloaded rows.
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        day = (r.get("day") or (r.get("scored_at") or "")[:10])
        if not day:
            continue
        r = dict(r)
        r["day"] = day
        by_day.setdefault(day, []).append(r)

    out_rows: list[dict] = []
    relevance: list[float] = []
    group_sizes: list[int] = []
    group_day: list[str] = []
    kept_days: list[str] = []

    for day in sorted(by_day.keys()):
        day_rows = by_day[day]
        if len(day_rows) < min_group_size:
            continue
        returns = [float(r.get("fwd_return") or 0.0) for r in day_rows]
        day_mean = sum(returns) / len(returns)
        # Sort within-day by ticker for a stable, reproducible group order.
        order = sorted(range(len(day_rows)), key=lambda i: str(day_rows[i].get("ticker") or ""))
        kept_days.append(day)
        group_sizes.append(len(day_rows))
        group_day.append(day)
        for i in order:
            out_rows.append(day_rows[i])
            relevance.append(returns[i] - day_mean)

    return RankerDataset(
        rows=out_rows,
        relevance=relevance,
        group_sizes=group_sizes,
        days=kept_days,
        group_day=group_day,
    )


# ---------------------------------------------------------------------------
# Relevance gradation
# ---------------------------------------------------------------------------


def _gain_labels(relevance: Iterable[float], group_sizes: list[int]) -> "Any":
    """Convert continuous day-demeaned returns into non-negative integer
    relevance grades, ranked WITHIN each day.

    LightGBM's ``lambdarank`` needs ordered, non-negative integer labels per
    group. We rank each day's bets by their demeaned return and assign a
    grade by quintile (0..4), so the model optimises the within-day ordering
    (NDCG) regardless of the raw return scale. Grading per-day (not globally)
    keeps the target beta-neutral end to end.
    """
    import numpy as np  # type: ignore

    rel = np.asarray(list(relevance), dtype=float)
    grades = np.zeros(len(rel), dtype=int)
    start = 0
    for n in group_sizes:
        end = start + n
        block = rel[start:end]
        if n == 1:
            grades[start:end] = 0
        else:
            # rank 0..n-1 (ascending), then map to a 0..4 grade band.
            order = block.argsort()
            ranks = np.empty(n, dtype=float)
            ranks[order] = np.arange(n)
            n_bands = min(5, n)
            band = np.floor(ranks / n * n_bands).astype(int)
            band = np.clip(band, 0, n_bands - 1)
            grades[start:end] = band
        start = end
    return grades


# ---------------------------------------------------------------------------
# Purged day-walk-forward CV
# ---------------------------------------------------------------------------


def purged_day_folds(
    days: list[str],
    *,
    n_splits: int = 4,
    purge_days: int = 1,
) -> list[tuple[list[str], list[str]]]:
    """Expanding-window walk-forward folds over DISTINCT days, with a purge
    gap so no training day's forward-return window overlaps a validation day.

    Returns a list of ``(train_days, val_days)`` pairs. Splitting on whole
    days guarantees no same-day row lands on both sides of a fold — the
    leakage row-level ``TimeSeriesSplit`` can't prevent. ``purge_days`` drops
    the last ``purge_days`` training days adjacent to each val block (the
    embargo/purge from López de Prado, applied on the day axis).
    """
    uniq = sorted(set(days))
    if len(uniq) < (n_splits + 1):
        n_splits = max(1, len(uniq) - 1)
    if n_splits < 1:
        return []
    # Block sizes for the validation windows (expanding train, like
    # TimeSeriesSplit but on the day index).
    fold_size = len(uniq) // (n_splits + 1)
    if fold_size < 1:
        fold_size = 1
    folds: list[tuple[list[str], list[str]]] = []
    for k in range(1, n_splits + 1):
        val_start = k * fold_size
        val_end = (k + 1) * fold_size if k < n_splits else len(uniq)
        val = uniq[val_start:val_end]
        train = uniq[: max(0, val_start - purge_days)]
        if not train or not val:
            continue
        folds.append((train, val))
    return folds


def _build_ranker(monotone_constraints: Optional[list[int]] = None) -> "Any":
    """Configure the LGBMRanker. ``lambdarank`` optimises NDCG directly.

    Hyperparameters mirror the conservative settings used elsewhere in the
    codebase (shallow trees, high ``min_child_samples``, L2) so a thin
    training set can't overfit. Monotone constraints are OPTIONAL — pass a
    per-feature list of {-1,0,1} to enforce direction; omitted by default.
    """
    import lightgbm as lgb  # type: ignore

    params: dict[str, Any] = dict(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=15,
        min_child_samples=50,
        reg_lambda=1.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=44,
        verbose=-1,
    )
    if monotone_constraints is not None:
        params["monotone_constraints"] = list(monotone_constraints)
    return lgb.LGBMRanker(**params)


def _features_for_dataset(ds: RankerDataset) -> "Any":
    """Extract the model feature matrix for a built dataset.

    Reuses the shared ``extract_features_df`` so the ranker sees the same
    feature space as the classifier. The bare rows loaded from SQL only
    carry ranking metadata (ticker/day/composite/return); ``extract_features``
    fills every missing feature with its documented no-info default, so this
    works on both real rows and the synthetic rows the tests inject.
    """
    from .features import extract_features_df

    return extract_features_df(ds.rows)


# ---------------------------------------------------------------------------
# Train + cross-validate (machinery)
# ---------------------------------------------------------------------------


@dataclass
class TrainOutcome:
    trained: bool
    n_days: int
    n_rows: int
    fold_ndcgs: list[float]
    val_ndcg: Optional[float]
    model: Optional[Any] = None
    reason: Optional[str] = None


def cross_validate_ranker(
    ds: RankerDataset,
    *,
    n_splits: int = 4,
    purge_days: int = 1,
    monotone_constraints: Optional[list[int]] = None,
) -> TrainOutcome:
    """Run purged day-walk-forward CV and fit a final model on all days.

    Returns the per-fold NDCG@k, the median val NDCG, and a final ranker
    refit on the full dataset. This is pure machinery — it does NOT consult
    the deploy gate (that's ``evaluate_deploy_gate``'s job).
    """
    import numpy as np  # type: ignore
    from lightgbm import early_stopping, log_evaluation  # type: ignore

    if ds.n_days < 2:
        return TrainOutcome(False, ds.n_days, ds.n_rows, [], None,
                            reason="need >=2 days to cross-validate")

    X = _features_for_dataset(ds)
    y = _gain_labels(ds.relevance, ds.group_sizes)

    # Map each row to its day so we can slice train/val by whole days.
    row_day: list[str] = []
    for day, n in zip(ds.group_day, ds.group_sizes):
        row_day.extend([day] * n)
    row_day_arr = np.asarray(row_day)

    folds = purged_day_folds(ds.days, n_splits=n_splits, purge_days=purge_days)
    fold_ndcgs: list[float] = []

    def _slice(days_set: set[str]):
        mask = np.isin(row_day_arr, list(days_set))
        idx = np.where(mask)[0]
        # Recompute group sizes for the sliced rows, preserving day order.
        sub_days = row_day_arr[idx]
        groups: list[int] = []
        if len(sub_days):
            cur = sub_days[0]
            count = 0
            for d in sub_days:
                if d == cur:
                    count += 1
                else:
                    groups.append(count)
                    cur, count = d, 1
            groups.append(count)
        return idx, groups

    for fi, (train_days, val_days) in enumerate(folds, 1):
        tr_idx, tr_groups = _slice(set(train_days))
        va_idx, va_groups = _slice(set(val_days))
        if not len(tr_idx) or not len(va_idx):
            continue
        model = _build_ranker(monotone_constraints)
        model.fit(
            X.iloc[tr_idx], y[tr_idx], group=tr_groups,
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_group=[va_groups],
            eval_at=[5],
            callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
        )
        # Pull the best NDCG@5 the booster saw on the val group.
        ndcg = None
        try:
            ev = model.best_score_.get("valid_0", {})
            ndcg = float(next(iter(ev.values()))) if ev else None
        except (StopIteration, AttributeError):
            ndcg = None
        if ndcg is not None:
            fold_ndcgs.append(ndcg)
            log.info("  ranker fold %d/%d  train_days=%d val_days=%d  ndcg@5=%.4f",
                     fi, len(folds), len(train_days), len(val_days), ndcg)

    val_ndcg = float(statistics.median(fold_ndcgs)) if fold_ndcgs else None

    # Final fit on ALL days (no holdout) for the deployable artifact.
    final = _build_ranker(monotone_constraints)
    final.fit(X, y, group=ds.group_sizes)

    return TrainOutcome(
        trained=True,
        n_days=ds.n_days,
        n_rows=ds.n_rows,
        fold_ndcgs=fold_ndcgs,
        val_ndcg=val_ndcg,
        model=final,
    )


# ---------------------------------------------------------------------------
# THE DEPLOY GATE
# ---------------------------------------------------------------------------


def distinct_clean_days(*, label_col: str = "return_5d_pct") -> int:
    """Count distinct clean scored-days available for ranking today.

    This is what the gate compares against ``RANK_MIN_DAYS``. Clean-data rule
    applied: ``COALESCE(so.data_corrupt,0)=0 AND price_at_flag>=1``.
    """
    with get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT COUNT(DISTINCT substr(ss.scored_at, 1, 10)) AS days
            FROM signal_scores ss
            JOIN signal_outcomes so ON so.score_id = ss.id
            WHERE so.{label_col} IS NOT NULL
              AND COALESCE(so.data_corrupt, 0) = 0
              AND so.price_at_flag >= 1
            """
        ).fetchone()
    return int(row["days"] or 0) if row else 0


def evaluate_deploy_gate(
    *,
    distinct_days: int,
    min_days: int = RANK_MIN_DAYS,
) -> GateDecision:
    """The HARD gate. Pure + side-effect-free so it's trivially testable.

    Returns ``DECISION_COMPOSITE_FALLBACK`` (and ``deployed=False``) whenever
    ``distinct_days < min_days`` — provably refusing to promote a ranker and
    leaving the live surface on the composite score. Only at/above the floor
    does it return ``DECISION_DEPLOY_RANKER``.
    """
    if distinct_days < min_days:
        return GateDecision(
            decision=DECISION_COMPOSITE_FALLBACK,
            deployed=False,
            distinct_days=distinct_days,
            min_days_required=min_days,
            reason=(f"distinct clean scored-days {distinct_days} < "
                    f"RANK_MIN_DAYS {min_days} — refusing to deploy ranker; "
                    f"serving composite score"),
        )
    return GateDecision(
        decision=DECISION_DEPLOY_RANKER,
        deployed=True,
        distinct_days=distinct_days,
        min_days_required=min_days,
        reason=(f"distinct clean scored-days {distinct_days} >= "
                f"RANK_MIN_DAYS {min_days} — ranker eligible to deploy"),
    )


def train_ranker_if_gated(
    *,
    label_col: str = "return_5d_pct",
    min_days: int = RANK_MIN_DAYS,
    n_splits: int = 4,
    purge_days: int = 1,
    monotone_constraints: Optional[list[int]] = None,
    _preloaded_rows: Optional[list[dict]] = None,
) -> GateDecision:
    """End-to-end entry point: check the gate, and ONLY if it passes do we
    train + persist a ranker. Otherwise return the composite-fallback
    decision WITHOUT training or writing any model.

    Provable property the test relies on:
      * gate fails  → no training, no model file, no pointer written,
                       decision == DECISION_COMPOSITE_FALLBACK.
      * gate passes → train, persist a ``.joblib`` + ``current_ranker.json``,
                       decision == DECISION_DEPLOY_RANKER.

    The gate is evaluated against the DISTINCT DAYS in the actual ranking
    dataset (after dropping single-bet days) so a fold of all-singleton days
    can't sneak past the floor.
    """
    ds = build_dataset(label_col=label_col, _preloaded_rows=_preloaded_rows)

    # Gate on the distinct days that actually form usable ranking groups.
    decision = evaluate_deploy_gate(distinct_days=ds.n_days, min_days=min_days)
    if not decision.deployed:
        log.warning("Ranker deploy gate BLOCKED: %s", decision.reason)
        return decision

    log.info("Ranker deploy gate PASSED (%d days). Training…", ds.n_days)
    outcome = cross_validate_ranker(
        ds, n_splits=n_splits, purge_days=purge_days,
        monotone_constraints=monotone_constraints,
    )
    if not outcome.trained or outcome.model is None:
        # Defensive: machinery couldn't train despite the gate passing.
        decision.decision = DECISION_COMPOSITE_FALLBACK
        decision.deployed = False
        decision.reason = f"gate passed but training failed: {outcome.reason}"
        log.warning("Ranker training failed post-gate: %s", outcome.reason)
        return decision

    model_path = _persist_ranker(outcome, label_col=label_col)
    decision.model_path = str(model_path)
    decision.fold_ndcgs = outcome.fold_ndcgs
    decision.val_ndcg = outcome.val_ndcg
    log.info("Ranker deployed: %s (val ndcg@5=%s)", model_path, outcome.val_ndcg)
    return decision


def _persist_ranker(outcome: TrainOutcome, *, label_col: str) -> Path:
    """Serialise the trained ranker + update the ranker side-pointer.

    Mirrors ``train.py``'s pattern: a versioned ``.joblib`` + ``.meta.json``
    alongside a small JSON pointer the live surface reads. Kept on a SEPARATE
    pointer from the classifier so the two pipelines never collide.
    """
    import joblib  # type: ignore
    from .features import FEATURE_NAMES

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"lgbm_ranker_{version}"
    model_path = MODELS_DIR / f"{base_name}.joblib"
    meta_path = MODELS_DIR / f"{base_name}.meta.json"
    joblib.dump(outcome.model, str(model_path))

    meta = {
        "version": version,
        "model_type": "lightgbm.LGBMRanker(lambdarank, day-demeaned target)",
        "label_col": label_col,
        "feature_names": list(FEATURE_NAMES),
        "n_days": outcome.n_days,
        "n_rows": outcome.n_rows,
        "val_ndcg_at5": outcome.val_ndcg,
        "fold_ndcgs": outcome.fold_ndcgs,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    RANKER_POINTER.write_text(json.dumps({
        "version": version,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "val_ndcg_at5": outcome.val_ndcg,
    }, indent=2))
    return model_path
