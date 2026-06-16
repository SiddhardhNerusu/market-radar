"""Lock the LGBMRanker hard deploy gate (blueprint #7, DATA-GATED).

Two provable properties:
  (a) With < RANK_MIN_DAYS distinct clean days the gate returns the
      'composite_fallback' decision and does NOT deploy a ranker (no model
      file, no pointer written, no training run).
  (b) With >= RANK_MIN_DAYS synthetic days the machinery trains an
      LGBMRanker on the day-demeaned target and WOULD deploy it.

The synthetic-day tests feed ``_preloaded_rows`` so they never touch the
real DB, and redirect the model output dir to a tmp path so a passing-gate
run can't pollute the real ``data/models/`` pointer.
"""
import importlib

import pytest

ranker = importlib.import_module("market_radar.ml.ranker")

# Skip the train half cleanly if libomp/lightgbm isn't importable on this box
# (same guard the ensemble module uses). The gate-block half needs no lightgbm.
try:
    import lightgbm  # type: ignore  # noqa: F401
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False


def _synthetic_rows(n_days: int, bets_per_day: int = 6, seed: int = 0):
    """Build ``n_days`` of clean (ticker, day) bets with a learnable
    within-day ordering: a per-bet ``composite_score`` that correlates with
    the day-demeaned forward return, so the ranker has real signal."""
    import random
    rng = random.Random(seed)
    rows = []
    base = 20100101
    for d in range(n_days):
        # Synthetic ISO day; spacing doesn't matter, only distinctness.
        yyyy = 2020 + d // 250
        rem = d % 250
        mm = 1 + rem // 28
        dd = 1 + rem % 28
        day = f"{yyyy:04d}-{mm:02d}-{dd:02d}"
        day_beta = rng.uniform(-3, 3)  # whole-day market move (the beta)
        for b in range(bets_per_day):
            alpha = rng.uniform(-2, 2)            # within-day edge
            comp = 0.5 + 0.1 * alpha + rng.uniform(-0.02, 0.02)
            rows.append({
                "ticker": f"T{b:02d}",
                "day": day,
                "scored_at": f"{day}T12:00:00Z",
                "composite_score": comp,
                # forward return = beta (same for the day) + alpha (per bet)
                "fwd_return": day_beta + alpha,
                # minimal feature inputs; extract_features fills the rest
                "event_type": "m_a_announcement",
                "source_tier": 1,
                "sentiment": 0.5,
            })
    return rows


# ---------------------------------------------------------------------------
# (a) Gate BLOCKS below the floor
# ---------------------------------------------------------------------------


def test_gate_pure_blocks_below_floor():
    """The pure gate refuses below RANK_MIN_DAYS and serves composite."""
    d = ranker.evaluate_deploy_gate(distinct_days=19, min_days=ranker.RANK_MIN_DAYS)
    assert d.decision == ranker.DECISION_COMPOSITE_FALLBACK
    assert d.deployed is False
    assert d.served_model == "composite"
    assert d.model_path is None
    assert d.min_days_required == ranker.RANK_MIN_DAYS


def test_gate_pure_passes_at_floor():
    d = ranker.evaluate_deploy_gate(distinct_days=ranker.RANK_MIN_DAYS,
                                    min_days=ranker.RANK_MIN_DAYS)
    assert d.decision == ranker.DECISION_DEPLOY_RANKER
    assert d.deployed is True
    assert d.served_model == "ranker"


def test_train_if_gated_blocks_and_does_not_deploy(tmp_path, monkeypatch):
    """End-to-end: < floor days → composite_fallback, NO model written, NO
    training attempted (we point MODELS_DIR/pointer at an empty tmp dir and
    assert nothing lands there)."""
    monkeypatch.setattr(ranker, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ranker, "RANKER_POINTER", tmp_path / "current_ranker.json")
    # Tripwire: if training were attempted it'd call cross_validate_ranker.
    def _boom(*a, **k):  # noqa: ANN001
        raise AssertionError("cross_validate_ranker must NOT run when gate blocks")
    monkeypatch.setattr(ranker, "cross_validate_ranker", _boom)

    rows = _synthetic_rows(n_days=19)   # below the 40-day floor
    decision = ranker.train_ranker_if_gated(_preloaded_rows=rows)

    assert decision.decision == ranker.DECISION_COMPOSITE_FALLBACK
    assert decision.deployed is False
    assert decision.served_model == "composite"
    assert decision.distinct_days == 19
    # PROVABLY no deploy: no joblib, no pointer.
    assert decision.model_path is None
    assert not (tmp_path / "current_ranker.json").exists()
    assert list(tmp_path.glob("*.joblib")) == []


def test_real_db_is_below_floor():
    """Sanity: today's real DB is below the floor, so the gate blocks now.
    (Confirms ACCEPTANCE: data-gate blocks today, serves composite.)"""
    try:
        days = ranker.distinct_clean_days()
    except Exception:  # noqa: BLE001 — no DB in some CI sandboxes
        pytest.skip("DB unavailable")
    d = ranker.evaluate_deploy_gate(distinct_days=days)
    assert days < ranker.RANK_MIN_DAYS, (
        f"expected < {ranker.RANK_MIN_DAYS} clean days today, got {days}")
    assert d.decision == ranker.DECISION_COMPOSITE_FALLBACK
    assert d.served_model == "composite"


# ---------------------------------------------------------------------------
# (b) With >= floor synthetic days the machinery trains + would deploy
# ---------------------------------------------------------------------------


def test_dataset_target_is_day_demeaned():
    """Each day's relevance target must sum to ~0 (beta removed)."""
    ds = ranker.build_dataset(_preloaded_rows=_synthetic_rows(n_days=45))
    start = 0
    for n in ds.group_sizes:
        block = ds.relevance[start:start + n]
        assert abs(sum(block)) < 1e-6, "day-demeaned target must sum to zero per group"
        start += n
    assert start == ds.n_rows


def test_purged_folds_have_no_day_overlap():
    days = [f"2020-01-{i:02d}" for i in range(1, 28)]
    folds = ranker.purged_day_folds(days, n_splits=4, purge_days=1)
    assert folds, "expected at least one fold"
    for train_days, val_days in folds:
        assert set(train_days).isdisjoint(set(val_days)), "no day on both sides"
        # purge: the last train day must be strictly before the first val day,
        # with at least one purged day of gap.
        assert max(train_days) < min(val_days)


@pytest.mark.skipif(not HAS_LGBM, reason="lightgbm/libomp not importable")
def test_train_if_gated_deploys_with_enough_days(tmp_path, monkeypatch):
    """End-to-end: >= floor synthetic days → trains an LGBMRanker and WOULD
    deploy (writes a joblib + pointer into the tmp models dir)."""
    monkeypatch.setattr(ranker, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ranker, "RANKER_POINTER", tmp_path / "current_ranker.json")

    rows = _synthetic_rows(n_days=45, bets_per_day=8)
    decision = ranker.train_ranker_if_gated(_preloaded_rows=rows)

    assert decision.decision == ranker.DECISION_DEPLOY_RANKER
    assert decision.deployed is True
    assert decision.served_model == "ranker"
    assert decision.distinct_days >= ranker.RANK_MIN_DAYS
    # PROVABLY deployed: a model file + a readable pointer exist.
    assert decision.model_path is not None
    pointer = tmp_path / "current_ranker.json"
    assert pointer.exists()
    import json
    meta = json.loads(pointer.read_text())
    assert meta["model_path"].endswith(".joblib")
    assert (tmp_path / "current_ranker.json").exists()
    assert list(tmp_path.glob("lgbm_ranker_*.joblib")), "ranker artifact written"


@pytest.mark.skipif(not HAS_LGBM, reason="lightgbm/libomp not importable")
def test_cross_validate_produces_fold_scores():
    """The CV machinery yields per-fold NDCG@5 on synthetic data."""
    ds = ranker.build_dataset(_preloaded_rows=_synthetic_rows(n_days=45, bets_per_day=8))
    outcome = ranker.cross_validate_ranker(ds, n_splits=4)
    assert outcome.trained is True
    assert outcome.model is not None
    assert len(outcome.fold_ndcgs) >= 1
    assert outcome.val_ndcg is not None
