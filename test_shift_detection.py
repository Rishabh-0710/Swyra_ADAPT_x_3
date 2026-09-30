"""Distribution-shift detector: sensitivity, specificity, drift typing, interpretability."""

import numpy as np

from experiments.distribution_shifts import FAMILIES, create_stationary_stream
from src.config import ADAPTXConfig
from src.continual_learner import ContinualLearner
from tests.conftest import make_classification, make_regression


def _fit(X, y, spec, **kw):
    return ContinualLearner(ADAPTXConfig(random_seed=0, base_max_iter=150, **kw)).initialize(X, y, spec)


def test_stationary_false_alarm_rate():
    alarms, total = 0, 0
    for fam in ("linear_regression", "interaction_classification", "mixed_classification"):
        for seed in (0, 1):
            batches, spec = create_stationary_stream(FAMILIES[fam], n_batches=5, batch_size=800, seed=seed)
            m = _fit(batches[0][1], batches[0][2], spec)
            for _, X, y in batches[1:]:
                rep = m.update(X, y)
                alarms += rep.severity in ("medium", "high")
                total += 1
    assert alarms / total <= 0.15, f"false alarm rate {alarms}/{total}"


def test_distribution_shift_detection_covariate():
    X, y, spec = make_regression(1000, 0)
    m = _fit(X, y, spec)
    X1, y1, _ = make_regression(1000, 1, shift=2.0)          # same P(y|x), x0 moved
    rep = m.update(X1, y1)
    assert rep.covariate_drift and "c0" in rep.affected_features
    assert rep.affected_features[0] == "c0"
    assert rep.drift_type in ("covariate", "compound")
    fs = {f.feature: f for f in rep.feature_shifts}
    assert fs["c0"].psi > 0.25 and fs["c0"].q_value < 0.01
    assert not fs["c3"].drifted


def test_concept_shift_detection():
    X, y, spec = make_regression(1000, 0)
    m = _fit(X, y, spec)
    X1, y1, _ = make_regression(1000, 1, concept=True)       # same P(x), new P(y|x)
    rep = m.update(X1, y1)
    assert rep.concept_drift and not rep.covariate_drift
    assert rep.drift_type == "concept"
    assert rep.loss_ratio_in_support > 2 and rep.concept_p_value < 0.01
    assert rep.severity == "high"


def test_correlation_shift_detected_when_marginals_do_not_move():
    X, y, spec = make_regression(1500, 0)
    m = _fit(X, y, spec)
    X1, y1, _ = make_regression(1500, 1)
    # c2, c3 become correlated (rho = 0.8) while each stays exactly N(0, 1)
    X1["c3"] = 0.8 * X1["c2"] + 0.6 * X1["c3"]
    rep = m.update(X1, y1)
    assert rep.correlation_drift and rep.correlation_max_delta > 0.5
    assert not any(f.drifted for f in rep.feature_shifts)          # no marginal test fires


def test_prior_shift_detection_classification():
    X, y, spec = make_classification(1500, 0)
    m = _fit(X, y, spec)
    X1, y1, _ = make_classification(1500, 1, pos_rate=0.1)
    rep = m.update(X1, y1)
    assert rep.prior_shift and rep.drift_type in ("prior", "compound")


def test_report_is_interpretable():
    X, y, spec = make_regression(800, 0)
    m = _fit(X, y, spec)
    rep = m.update(*make_regression(800, 1, shift=2.0)[:2])
    d = rep.to_dict()
    for key in ("overall_drift_score", "severity", "covariate_drift_score", "concept_drift_score",
                "affected_features", "confidence", "drift_type"):
        assert key in d
    assert 0 <= rep.overall_drift_score <= 1 and 0 <= rep.confidence <= 1
    assert "c0" in rep.summary()
