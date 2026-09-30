"""End-to-end continual-learning behaviour of ADAPT-X."""

import inspect
import os
import re

import numpy as np
import pandas as pd
import pytest

from experiments.baselines import NaiveIncrementalModel, StaticModel
from experiments.distribution_shifts import FAMILIES, create_benchmark_stream
from src import continual_learner
from src.adaptation_controller import AdaptationController
from src.config import ADAPTXConfig
from src.continual_learner import ContinualLearner
from src.metrics import evaluate_predictions
from src.sequential_evaluation import SequentialEvaluator, split_stream
from src.shift_detection import ShiftReport
from tests.conftest import make_classification, make_regression


def _skill(m, X, y, task):
    if task == "regression":
        return evaluate_predictions(task, y, m.predict(X))["skill"]
    return evaluate_predictions(task, y, None, m.predict_proba(X)[:, 1])["skill"]


def test_initial_learning(small_cfg):
    for maker, task, floor in ((make_regression, "regression", 0.8), (make_classification, "binary_classification", 0.5)):
        X, y, spec = maker(1500, 0)
        Xt, yt, _ = maker(500, 1)
        m = ContinualLearner(small_cfg).initialize(X, y, spec)
        assert _skill(m, Xt, yt, task) > floor


def test_sequential_update_is_incremental_not_retraining(small_cfg):
    X, y, spec = make_regression(1000, 0)
    m = ContinualLearner(small_cfg).initialize(X, y, spec)
    base = m.state.stages[0].estimator
    probe = make_regression(200, 9)[0]
    base_out = base.predict(m.encoder.transform(probe)).copy()
    for t in range(1, 4):
        rep = m.update(*make_regression(1000, t, shift=1.5 * t)[:2])
        assert isinstance(rep, ShiftReport)
    # the batch-0 learner is still the same object with identical output: it was never refitted
    assert m.state.stages[0].estimator is base
    assert np.allclose(base.predict(m.encoder.transform(probe)), base_out)
    assert len(m.state.stages) > 1 and all(s.kind == "correction" for s in m.state.stages[1:])
    assert m.n_batches_seen == 4 and len(m.history) == 3


def test_catastrophic_forgetting_is_prevented(small_cfg):
    """Same concept, disjoint input regions: region A must not be forgotten after learning B, C."""
    cfg = small_cfg.replace(replay_capacity=300)
    XA, yA, spec = make_regression(1200, 0, shift=-3)
    XA_test, yA_test, _ = make_regression(400, 50, shift=-3)
    m, naive = ContinualLearner(cfg).initialize(XA, yA, spec), NaiveIncrementalModel(cfg).initialize(XA, yA, spec)
    before = _skill(m, XA_test, yA_test, "regression")
    for t, s in enumerate((0.0, 3.0), start=1):
        Xt, yt, _ = make_regression(1200, t, shift=s)
        m.update(Xt, yt)
        naive.update(Xt, yt)
    after, naive_after = _skill(m, XA_test, yA_test, "regression"), _skill(naive, XA_test, yA_test, "regression")
    assert after > before - 0.1, (before, after)
    assert after > naive_after + 0.3, (after, naive_after)


def test_adaptation_to_concept_shift(small_cfg):
    X, y, spec = make_regression(1200, 0)
    m, static = ContinualLearner(small_cfg).initialize(X, y, spec), StaticModel(small_cfg).initialize(X, y, spec)
    X1, y1, _ = make_regression(1200, 1, concept=True)
    Xt, yt, _ = make_regression(400, 2, concept=True)
    m.update(X1, y1)
    assert _skill(m, Xt, yt, "regression") > 0.7 > _skill(static, Xt, yt, "regression")


def test_adaptive_update_strength():
    # replay_weight_min < 1 so the stale-replay mechanism is visible (the tuned default is 1.0)
    ctl = AdaptationController(ADAPTXConfig(replay_weight_min=0.5))
    none = ctl.plan(ShiftReport(1, 500), 1)
    cov = ctl.plan(ShiftReport(1, 500, overall_drift_score=0.8, covariate_drift_score=0.8, covariate_drift=True,
                               drift_type="covariate", severity="high"), 1)
    con = ctl.plan(ShiftReport(1, 500, overall_drift_score=0.8, concept_drift_score=0.8, concept_drift=True,
                               drift_type="concept", severity="high"), 1)
    assert none.mode == "MAINTAIN" and cov.mode == "COVARIATE_ADAPT" and con.mode == "CONCEPT_ADAPT"
    assert none.stability > con.stability and cov.stability == none.stability      # only concept drift unlocks plasticity
    assert con.replay_weight < cov.replay_weight == none.replay_weight
    assert cov.stage_trees > none.stage_trees and con.stage_trees > none.stage_trees
    assert con.memory_new_share > none.memory_new_share
    # concept evidence outside the reference support does not unlock global plasticity
    far = ctl.plan(ShiftReport(1, 500, overall_drift_score=0.8, concept_drift_score=0.8, concept_drift=True,
                               drift_type="concept", severity="high", in_support_fraction=0.1), 1)
    assert far.stability > con.stability and far.plasticity_demand < con.plasticity_demand


def test_learner_plasticity_follows_detected_drift(small_cfg):
    """Integrated check: a concept batch moves predictions on old inputs more than a stationary batch."""
    X, y, spec = make_regression(1200, 0)
    probe = make_regression(300, 77)[0]
    moves = {}
    for name, (Xt, yt, _) in {"stationary": make_regression(1200, 1), "concept": make_regression(1200, 1, concept=True)}.items():
        m = ContinualLearner(small_cfg).initialize(X, y, spec)
        p0 = m.predict(probe)
        rep = m.update(Xt, yt)
        moves[name] = (float(np.mean(np.abs(m.predict(probe) - p0))), rep.drift_type)
    assert moves["stationary"][1] == "none" and moves["concept"][1] == "concept"
    assert moves["concept"][0] > 5 * moves["stationary"][0]


def test_no_future_batch_access():
    batches, spec = create_benchmark_stream(FAMILIES["linear_regression"], 300, 0)
    splits = split_stream(batches, 0)
    log = []

    class Spy:
        def initialize(self, X, y, task):
            log.append(("init", len(log), X)); self.task = task
        def update(self, X, y):
            log.append(("update", len(log), X))
        def predict(self, X):
            return np.zeros(len(X))
    SequentialEvaluator().run(Spy(), splits, spec, [b[0] for b in batches], "spy", "lin", 0)
    assert [e[0] for e in log] == ["init"] + ["update"] * 5
    for t, (_, _, X) in enumerate(log):
        pd.testing.assert_frame_equal(X, splits[t][0])                       # only batch t's update part
        seen = set(map(tuple, X.round(12).to_numpy()))
        for j, sp in enumerate(splits):
            holdout = set(map(tuple, sp[2].round(12).to_numpy()))
            assert not seen & holdout                                         # never an evaluation row
            if j > t:
                assert not seen & set(map(tuple, sp[0].round(12).to_numpy()))  # never a future row


def test_reproducibility():
    batches, spec = create_benchmark_stream(FAMILIES["mixed_classification"], 500, 3)
    outs = []
    for _ in range(2):
        m = ContinualLearner(ADAPTXConfig(random_seed=11)).initialize(batches[0][1], batches[0][2], spec)
        reps = [m.update(X, y).to_dict() for _, X, y in batches[1:4]]
        outs.append((m.predict_proba(batches[4][1])[:, 1], [r["drift_type"] for r in reps], [r["overall_drift_score"] for r in reps]))
    assert np.array_equal(outs[0][0], outs[1][0]) and outs[0][1:] == outs[1][1:]


def test_cross_dataset_generalization():
    """The same code, with no per-dataset settings, runs and adapts on all 8 families."""
    ev, wins = SequentialEvaluator(), 0
    for fam, gen in FAMILIES.items():
        batches, spec = create_benchmark_stream(gen, 500, 0)
        splits, names = split_stream(batches, 0), [b[0] for b in batches]
        a = ev.run(ContinualLearner(ADAPTXConfig(random_seed=0)), splits, spec, names, "a", fam, 0).summary()
        s = ev.run(StaticModel(ADAPTXConfig(random_seed=0)), splits, spec, names, "s", fam, 0).summary()
        assert np.isfinite(a["post_update_skill"])
        wins += a["post_update_skill"] > s["post_update_skill"]
    assert wins >= 7


def test_core_is_domain_agnostic():
    """No feature names, dataset families or domain words anywhere in the algorithm package."""
    src_dir = os.path.dirname(inspect.getfile(continual_learner))
    code = " ".join(open(os.path.join(src_dir, f)).read() for f in os.listdir(src_dir) if f.endswith(".py"))
    for fam in FAMILIES:
        assert fam not in code
    for word in ("temperature", "study_hours", "income", "attendance", "churn", "price", "cat_id", "x_0", "feat_0"):
        assert not re.search(rf"\b{word}\b", code, re.IGNORECASE), word


def test_column_names_do_not_matter(small_cfg):
    X, y, spec = make_regression(800, 0)
    X1, y1, _ = make_regression(800, 1, shift=1.0)
    ren = {c: f"zz{i}_{c[::-1]}" for i, c in enumerate(X.columns)}
    spec2 = type(spec)("regression", numerical_features=[ren[c] for c in spec.numerical_features])
    a = ContinualLearner(small_cfg).initialize(X, y, spec)
    b = ContinualLearner(small_cfg).initialize(X.rename(columns=ren), y, spec2)
    a.update(X1, y1)
    b.update(X1.rename(columns=ren), y1)
    assert np.allclose(a.predict(X1), b.predict(X1.rename(columns=ren)))


@pytest.mark.parametrize("flags", [dict(use_drift_detection=False), dict(replay_capacity=0), dict(adaptive_scaling=False),
                                   dict(persistent_state=False), dict(replay_strategy="reservoir")])
def test_ablation_variants_run(small_cfg, flags):
    X, y, spec = make_classification(800, 0)
    m = ContinualLearner(small_cfg.replace(**flags)).initialize(X, y, spec)
    for t in range(1, 3):
        m.update(*make_classification(800, t, pos_rate=0.2 if t == 2 else None)[:2])
    assert m.predict_proba(X).shape == (800, 2)
