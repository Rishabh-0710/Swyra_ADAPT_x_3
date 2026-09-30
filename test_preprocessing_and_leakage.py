"""Feature-schema stability, bounded encoder state and target-encoding leakage."""

import numpy as np
import pandas as pd

from src.config import ADAPTXConfig
from src.continual_learner import ContinualLearner
from src.memory_accounting import deep_nbytes
from src.preprocessing import StreamingFeatureEncoder
from src.task import TaskSpec


def _mixed(n, seed, id_offset=0, with_nan=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                       "k": rng.choice(["p", "q", "r"], n),
                       "uid": [f"u{id_offset + i}" for i in range(n)]})
    if with_nan:
        df.loc[: n // 10, "a"] = np.nan
        df.loc[: n // 20, "k"] = None
    y = pd.Series(rng.normal(size=n))
    return df, y


SPEC = lambda: TaskSpec("regression", numerical_features=["a", "b"], categorical_features=["k", "uid"])


def test_feature_schema_stability():
    enc = StreamingFeatureEncoder(SPEC(), max_categories=8, n_hash_buckets=4).fit_schema(_mixed(200, 0)[0])
    width = len(enc.feature_names_out_)
    X_nan, _ = _mixed(200, 1, id_offset=10_000, with_nan=True)     # NaNs + unseen categories
    X_missing_col = _mixed(50, 2)[0].drop(columns=["b"])          # a whole column absent
    for X in (_mixed(300, 3)[0], X_nan, X_missing_col):
        Z = enc.transform(X)
        assert Z.shape[1] == width
    # missing indicators exist (and are all zero) even for batches without NaN
    Z0 = enc.transform(_mixed(100, 4)[0])
    isna_cols = [i for i, n in enumerate(enc.feature_names_out_) if n.endswith("__isna")]
    assert len(isna_cols) == 2 and np.all(Z0[:, isna_cols] == 0)
    assert np.all(enc.transform(X_nan)[: 200 // 10, isna_cols[0]] == 1)


def test_learner_survives_schema_perturbations(small_cfg):
    X0, y0 = _mixed(400, 0)
    m = ContinualLearner(small_cfg).initialize(X0, y0, SPEC())
    X1, y1 = _mixed(400, 1, id_offset=5000, with_nan=True)
    m.update(X1, y1)                       # the original implementation crashed here
    assert m.predict(X1).shape == (400,)


def test_bounded_categorical_state_under_high_cardinality_stream():
    enc = StreamingFeatureEncoder(SPEC(), max_categories=16, n_hash_buckets=8, use_target_encoding=True)
    X0, y0 = _mixed(500, 0)
    enc.fit_schema(X0)
    enc.observe(X0, y0)
    sizes = []
    for t in range(1, 15):                  # 14 x 500 = 7000 never-seen category values
        X, y = _mixed(500, t, id_offset=t * 1000)
        enc.transform(X)
        enc.observe(X, y)
        sizes.append(deep_nbytes(enc))
    assert len(enc.vocab["uid"]) <= 16
    assert max(sizes) == min(sizes), "encoder state must not grow with new categories"
    codes = enc.transform(_mixed(500, 99, id_offset=10**6)[0])[:, 3]
    assert codes.max() < enc.n_codes_["uid"]


def test_no_target_encoding_leakage_features_independent_of_own_labels():
    """Features used to learn batch t must not depend on y_t (permuting y_t changes nothing)."""
    cfg = ADAPTXConfig(use_target_encoding=True, replay_capacity=100, reference_capacity=128,
                       base_max_iter=50, stage_trees_max=30, random_seed=0)
    X0, y0 = _mixed(400, 0)
    X1, y1 = _mixed(400, 1)
    X1["uid"] = X0["uid"].values            # categories seen before -> target encoding is active
    captured = []
    for y_variant in (y1, pd.Series(np.random.default_rng(5).permutation(y1.values))):
        m = ContinualLearner(cfg).initialize(X0, y0, SPEC())
        orig = m.encoder.transform
        m.encoder.transform = lambda X, _o=orig: captured.append(_o(X)) or captured[-1]
        m.update(X1, y_variant)
    assert np.array_equal(captured[0], captured[1]), "training features of batch t depend on y_t (leakage)"


def test_target_encoding_not_correlated_with_own_label_on_unique_ids():
    """Unique-ID categorical + pure-noise target: the original pipeline gave corr(TE, y) = 1.0."""
    enc = StreamingFeatureEncoder(SPEC(), use_target_encoding=True, random_seed=0)
    X0, y0 = _mixed(1000, 0)
    enc.fit_schema(X0)
    Z0 = enc.initial_transform(X0, y0.values)          # batch 0: out-of-fold
    te = enc.feature_names_out_.index("uid__target_enc")
    assert abs(np.corrcoef(Z0[:, te], y0)[0, 1]) < 0.15
    enc.observe(X0, y0.values)
    X1, y1 = _mixed(1000, 1)
    Z1 = enc.transform(X1)                             # later batch: pre-update state
    assert abs(np.corrcoef(Z1[:, te], y1)[0, 1]) < 0.15


def test_out_of_fold_batch0_row_encoding_ignores_own_label():
    enc = StreamingFeatureEncoder(SPEC(), use_target_encoding=True, random_seed=0)
    X0, y0 = _mixed(300, 0)
    X0["k"] = "p"
    enc.fit_schema(X0)
    te = enc.feature_names_out_.index("k__target_enc")
    a = enc.initial_transform(X0, y0.values)[:, te]
    y_mod = y0.values.copy()
    y_mod[7] += 1000.0
    b = enc.initial_transform(X0, y_mod)[:, te]
    assert a[7] == b[7]
