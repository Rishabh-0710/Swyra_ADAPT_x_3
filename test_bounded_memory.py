"""Bounded memory: replay, drift reference, model state; no raw historical data retained."""

import numpy as np
import pandas as pd

from experiments.baselines import FullRetrainOracle
from src.config import ADAPTXConfig
from src.continual_learner import ContinualLearner
from src.memory_accounting import large_arrays
from src.replay_memory import BoundedReplayMemory
from tests.conftest import make_regression


def test_bounded_replay_memory_both_strategies():
    for strategy in ("stratified_diversity", "reservoir"):
        mem = BoundedReplayMemory("regression", 150, np.zeros(4, bool), np.ones(4), strategy=strategy, random_seed=0)
        rng = np.random.default_rng(0)
        for t in range(30):
            mem.update(rng.normal(size=(400, 4)), rng.normal(size=400), t, new_share=0.3)
            assert mem.size <= 150
        assert mem.nbytes() == 150 * 4 * 8 + 150 * 8 + 150 * 4
        assert len(np.unique(mem.batch_index)) > 3        # keeps history, not only the last batch


def test_replay_keeps_minority_class():
    mem = BoundedReplayMemory("binary_classification", 100, np.zeros(3, bool), np.ones(3), random_seed=0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    y = (rng.uniform(size=2000) < 0.03).astype(float)
    mem.update(X, y, 0, new_share=1.0)
    _, yr, w = mem.get()
    assert yr.sum() >= 25                                  # floor keeps the rare class
    assert abs(np.sum(w * yr) / np.sum(w) - y.mean()) < 0.02   # prior-correction weights


def test_bounded_reference_memory_independent_of_initial_size(small_cfg):
    sizes = {}
    for n0 in (600, 20_000):
        X, y, spec = make_regression(n0, 0)
        m = ContinualLearner(small_cfg).initialize(X, y, spec)
        sizes[n0] = (m.detector.nbytes(), len(m.detector.ref_X), m.memory.size)
    assert sizes[600][1] <= small_cfg.reference_capacity and sizes[20_000][1] == small_cfg.reference_capacity
    assert sizes[20_000][0] <= sizes[600][0] * 1.01 + 1e4
    assert sizes[20_000][2] == small_cfg.replay_capacity


def test_memory_does_not_scale_with_stream_length(small_cfg):
    cfg = small_cfg.replace(max_stages=5)
    X, y, spec = make_regression(800, 0)
    m = ContinualLearner(cfg).initialize(X, y, spec)
    footprints = []
    for t in range(1, 26):
        Xt, yt, _ = make_regression(800, t, shift=0.8 * np.sin(t), concept=(t % 7 == 0))
        m.update(Xt, yt)
        fp = m.memory_footprint()
        footprints.append(fp)
        assert fp["replay_rows"] <= cfg.replay_capacity
        assert fp["reference_rows"] <= cfg.reference_capacity
        assert fp["n_stages"] <= cfg.max_stages
    assert len(m.history) <= cfg.history_length
    early = max(f["total_bytes"] for f in footprints[4:10])
    late = max(f["total_bytes"] for f in footprints[15:])
    assert late < 1.5 * early, (early, late)
    # contrast: the full-retraining oracle grows linearly with the stream
    o = FullRetrainOracle(cfg).initialize(X, y, spec)
    b1 = o.memory_footprint()["total_bytes"]
    for t in range(1, 6):
        Xt, yt, _ = make_regression(800, t)
        o.update(Xt, yt)
    assert o.memory_footprint()["total_bytes"] > 3 * b1 * 0.5


def test_no_historical_raw_data_retention(small_cfg):
    X, y, spec = make_regression(1000, 0)
    m = ContinualLearner(small_cfg).initialize(X, y, spec)
    inputs = [X]
    for t in range(1, 8):
        Xt, yt, _ = make_regression(1000, t, shift=0.5)
        m.update(Xt, yt)
        inputs.append(Xt)
    limit = max(small_cfg.replay_capacity, small_cfg.reference_capacity)
    offenders = large_arrays(m, min_rows=limit)
    assert offenders == [], f"arrays larger than the declared budgets: {offenders}"
    frames = [p for p, _ in large_arrays(m, min_rows=0) if "DataFrame" in p]
    assert not frames
    # nothing inside the learner is a view onto a caller's batch
    arrays = []

    def collect(o, seen=set()):
        if id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            arrays.append(o)
        elif isinstance(o, dict):
            [collect(v) for v in o.values()]
        elif isinstance(o, (list, tuple)):
            [collect(v) for v in o]
        elif hasattr(o, "__dict__") and not isinstance(o, type):
            collect(vars(o))
    collect(m)
    for Xin in inputs:
        raw = Xin.to_numpy()
        assert not any(np.shares_memory(a, raw) for a in arrays if a.dtype == raw.dtype)
