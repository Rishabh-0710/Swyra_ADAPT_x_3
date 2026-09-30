"""
Benchmark streams: generic shift operators + 8 independent data-generating families.

The operators act on column *positions* (first numeric columns), never on
feature meaning. The learner never sees the family name. ``DEV_FAMILIES`` are
the only families (with dev seeds only) used to tune the adaptation controller;
the other six families are never used for tuning.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.task import TaskSpec


# =====================================================================
# 1. Pure Mathematical Distribution Shift Operators (Domain-Agnostic)
# =====================================================================

def apply_mean_shift(
    df: pd.DataFrame,
    cols: List[str],
    shift_magnitude: float = 2.0,
) -> pd.DataFrame:
    """Mathematical shift: X_{:, j} <- X_{:, j} + delta."""
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
            std = float(df_out[col].std()) if df_out[col].std() > 0 else 1.0
            df_out[col] = df_out[col] + shift_magnitude * std
    return df_out


def apply_scale_shift(
    df: pd.DataFrame,
    cols: List[str],
    scale_factor: float = 2.5,
) -> pd.DataFrame:
    """Mathematical shift: X_{:, j} <- gamma * (X_{:, j} - mu) + mu."""
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
            mean = float(df_out[col].mean())
            df_out[col] = mean + scale_factor * (df_out[col] - mean)
    return df_out


def apply_rotation_shift(
    df: pd.DataFrame,
    col_pair: Tuple[str, str],
    theta_rad: float = np.pi / 4,
) -> pd.DataFrame:
    """Mathematical feature correlation shift via 2D rotation matrix."""
    df_out = df.copy()
    c1, c2 = col_pair
    if c1 in df_out.columns and c2 in df_out.columns:
        x1 = df_out[c1].to_numpy(dtype=float)
        x2 = df_out[c2].to_numpy(dtype=float)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        df_out[c1] = cos_t * x1 - sin_t * x2
        df_out[c2] = sin_t * x1 + cos_t * x2
    return df_out


def apply_concept_shift_regression(
    df: pd.DataFrame,
    y: pd.Series,
    interaction_cols: Tuple[str, str],
    shift_gain: float = 3.0,
) -> pd.Series:
    """Mathematical concept shift: P(Y|X) changes via new non-linear interaction."""
    c1, c2 = interaction_cols
    x1 = df[c1].to_numpy(dtype=float)
    x2 = df[c2].to_numpy(dtype=float)
    delta_y = shift_gain * np.sin(x1 * x2)
    return pd.Series(y.to_numpy(dtype=float) + delta_y, name=y.name)


def apply_concept_shift_classification(
    df: pd.DataFrame,
    y: pd.Series,
    invert_cols: List[str],
    flip_rate: float = 0.35,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """Mathematical concept shift: Decision boundary flips in a specific subspace."""
    if rng is None:
        rng = np.random.default_rng(42)
    y_arr = y.to_numpy().copy()
    mask = np.ones(len(df), dtype=bool)
    for col in invert_cols:
        if col in df.columns:
            mask &= (df[col] > df[col].median())

    subspace_idx = np.where(mask)[0]
    if len(subspace_idx) > 0:
        flip_picks = rng.choice(
            subspace_idx,
            size=int(len(subspace_idx) * flip_rate),
            replace=False,
        )
        y_arr[flip_picks] = 1 - y_arr[flip_picks]
    return pd.Series(y_arr, name=y.name)


def apply_noise_shift(
    y: pd.Series,
    noise_sigma: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """Additive measurement noise increase: y <- y + N(0, sigma^2)."""
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_sigma, size=len(y))
    return pd.Series(y.to_numpy() + noise, name=y.name)


def apply_prior_shift_classification(
    df: pd.DataFrame,
    y: pd.Series,
    target_pos_ratio: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Mathematical prior shift: changes class prevalence P(Y=1)."""
    if rng is None:
        rng = np.random.default_rng(42)
    y_arr = y.to_numpy()
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    n_total = len(y_arr)
    n_pos_desired = int(n_total * target_pos_ratio)
    n_neg_desired = n_total - n_pos_desired

    chosen_pos = rng.choice(pos_idx, size=n_pos_desired, replace=True)
    chosen_neg = rng.choice(neg_idx, size=n_neg_desired, replace=True)
    all_picks = rng.permutation(np.concatenate([chosen_pos, chosen_neg]))

    return df.iloc[all_picks].reset_index(drop=True), y.iloc[all_picks].reset_index(drop=True)


# =====================================================================
# 2. Independent data-generating families
#    Each family = input sampler + label mechanism P(y|x). The label
#    mechanism is reused after covariate shifts so that "covariate shift"
#    really keeps P(y|x) fixed (the original harness kept the old labels
#    after moving x, which silently turned covariate shift into concept shift).
# =====================================================================

def _spec(task, num, cat=()):
    return TaskSpec(task_type=task, target_name="target", numerical_features=list(num), categorical_features=list(cat))


def _bernoulli(rng, logits):
    return (rng.uniform(size=len(logits)) < 1 / (1 + np.exp(-np.asarray(logits)))).astype(int)


def _num(df, cols):
    return df[cols].to_numpy(dtype=float)


class Family:
    def __init__(self, name, task, cols, sample_x, label, cat=()):
        self.name, self.task, self.cols, self.cat = name, task, list(cols), list(cat)
        self.sample_x, self.label = sample_x, label
        self.__name__ = name

    def __call__(self, n_samples=6000, seed=0):
        rng = np.random.default_rng(seed)
        df = self.sample_x(rng, n_samples)
        y = self.label(df, rng, seed)
        return df, pd.Series(y, name="target"), _spec(self.task, self.cols, self.cat)

    def relabel(self, df, seed, t):
        return self.label(df, np.random.default_rng([seed, t, 99]), seed)


def _friedman_x(rng, n):
    return pd.DataFrame(rng.uniform(0, 1, (n, 10)), columns=[f"x_{i}" for i in range(10)])


def _friedman_y(df, rng, seed):
    X = df.to_numpy(dtype=float)
    return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + rng.normal(0, 0.5, len(X))


def _corr_gauss(d, rho, prefix):
    L = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(d), np.arange(d))))
    return lambda rng, n: pd.DataFrame(rng.normal(size=(n, d)) @ L.T, columns=[f"{prefix}_{i}" for i in range(d)])


def _linear_y(df, rng, seed):
    beta = np.random.default_rng(seed + 1).uniform(-3, 3, size=df.shape[1])
    return df.to_numpy(dtype=float) @ beta + rng.normal(0, 1.0, len(df))


def _gauss_x(d, prefix, lo=None, hi=None):
    if lo is None:
        return lambda rng, n: pd.DataFrame(rng.normal(size=(n, d)), columns=[f"{prefix}_{i}" for i in range(d)])
    return lambda rng, n: pd.DataFrame(rng.uniform(lo, hi, (n, d)), columns=[f"{prefix}_{i}" for i in range(d)])


def _nonlin_clf_y(df, rng, seed):
    X = df.to_numpy(dtype=float)
    return _bernoulli(rng, 1.5 * X[:, 0] - 2 * X[:, 1] + 3 * X[:, 2] * X[:, 3] - 1.2 * X[:, 4] ** 2 + rng.normal(0, 0.4, len(X)))


def _interaction_y(df, rng, seed):
    X = df.to_numpy(dtype=float)
    return _bernoulli(rng, 2.0 * X[:, 0] * X[:, 1] - 1.5 * X[:, 2] * X[:, 3] + 2.0 * np.sin(2 * X[:, 4]) * X[:, 5] + 0.5 * X[:, 6])


_MIXED_NUM = [f"num_{i}" for i in range(8)]


def _mixed_x(rng, n):
    df = pd.DataFrame(rng.normal(size=(n, 8)), columns=_MIXED_NUM)
    df["cat_tier"] = rng.choice(["A", "B", "C", "D"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    df["cat_zone"] = rng.choice(["N", "S", "E", "W"], size=n)
    df["cat_id"] = [f"id{i}" for i in rng.integers(0, 300, size=n)]
    for c in ("num_4", "num_5"):
        df.loc[rng.uniform(size=n) < 0.05, c] = np.nan
    return df


def _mixed_y(df, rng, seed):
    X = df[_MIXED_NUM].to_numpy(dtype=float)
    id_effect = np.random.default_rng(seed + 1).normal(0, 0.6, size=300)
    ids = df["cat_id"].str[2:].astype(int).to_numpy()
    tier = df["cat_tier"].map({"A": 1.5, "B": 0.5, "C": -0.8, "D": -1.8}).to_numpy()
    z = 1.8 * X[:, 0] - 1.4 * X[:, 1] + 0.8 * X[:, 2] * X[:, 3] + tier + id_effect[ids] + rng.normal(0, 0.5, len(df))
    return _bernoulli(rng, z)


def _highdim_y(df, rng, seed):
    beta = np.zeros(df.shape[1])
    beta[[2, 7, 15, 25, 38]] = [4.0, -3.5, 2.0, -2.5, 5.0]
    return df.to_numpy(dtype=float) @ beta + rng.normal(0, 1.0, len(df))


def _hetero_y(df, rng, seed):
    X = df.to_numpy(dtype=float)
    s = 3 * np.tanh(X[:, 0]) + 2 * X[:, 1] - 1.5 * X[:, 2] ** 2
    return s + rng.normal(size=len(X)) * (0.3 + 0.5 * np.linalg.norm(X[:, :3], axis=1))


def _imbalanced_y(df, rng, seed):
    X = df.to_numpy(dtype=float)
    return _bernoulli(rng, -3.3 + 1.5 * X[:, 0] + 1.2 * X[:, 1] * X[:, 2] - 0.8 * X[:, 3] ** 2 + 1.0 * X[:, 4])


family_nonlinear_regression = Family("nonlinear_regression", "regression", [f"x_{i}" for i in range(10)], _friedman_x, _friedman_y)
family_linear_regression = Family("linear_regression", "regression", [f"v_{i}" for i in range(8)], _corr_gauss(8, 0.3, "v"), _linear_y)
family_nonlinear_classification = Family("nonlinear_classification", "binary_classification", [f"feat_{i}" for i in range(12)], _gauss_x(12, "feat"), _nonlin_clf_y)
family_interaction_classification = Family("interaction_classification", "binary_classification", [f"u_{i}" for i in range(10)], _gauss_x(10, "u", -2, 2), _interaction_y)
family_mixed_classification = Family("mixed_classification", "binary_classification", _MIXED_NUM, _mixed_x, _mixed_y, cat=["cat_tier", "cat_zone", "cat_id"])
family_highdim_regression = Family("highdim_regression", "regression", [f"dim_{i}" for i in range(40)], _corr_gauss(40, 0.5, "dim"), _highdim_y)
family_heteroscedastic_regression = Family("heteroscedastic_regression", "regression", [f"var_{i}" for i in range(10)], _gauss_x(10, "var", -2, 2), _hetero_y)
family_imbalanced_classification = Family("imbalanced_classification", "binary_classification", [f"s_{i}" for i in range(10)], _gauss_x(10, "s"), _imbalanced_y)


FAMILIES: Dict[str, Callable] = {
    "nonlinear_regression": family_nonlinear_regression,
    "linear_regression": family_linear_regression,
    "nonlinear_classification": family_nonlinear_classification,
    "interaction_classification": family_interaction_classification,
    "mixed_classification": family_mixed_classification,
    "highdim_regression": family_highdim_regression,
    "heteroscedastic_regression": family_heteroscedastic_regression,
    "imbalanced_classification": family_imbalanced_classification,
}
DEV_FAMILIES = ["nonlinear_regression", "nonlinear_classification"]

ENV_NAMES = [
    "B0_baseline", "B1_covariate", "B2_prior_label", "B3_concept",
    "B4_noise_rotation", "B5_hidden_compound",
]


# =====================================================================
# 3. Stream construction: Batch 0 -> 1 -> 2 -> 3 -> 4 -> hidden compound shift
# =====================================================================

def create_benchmark_stream(generator_fn: Callable, batch_size: int = 1000, seed: int = 42):
    """Return ([(env_name, X_t, y_t) for t = 0..5], task_spec)."""
    rng = np.random.default_rng(seed + 12345)
    df_all, y_all, spec = generator_fn(n_samples=batch_size * 6, seed=seed)
    num = spec.numerical_features
    clf = spec.task_type == "binary_classification"
    batches = []
    for t in range(6):
        df = df_all.iloc[t * batch_size:(t + 1) * batch_size].reset_index(drop=True)
        y = y_all.iloc[t * batch_size:(t + 1) * batch_size].reset_index(drop=True)
        if t == 1:  # covariate: mean + scale shift of two inputs, P(y|x) unchanged
            df = apply_scale_shift(apply_mean_shift(df, num[:2], 2.2), num[:2], 1.8)
            y = _relabel(generator_fn, df, y, spec, seed, t)
        elif t == 2:  # prior / label shift
            if clf:
                df, y = apply_prior_shift_classification(df, y, target_pos_ratio=0.15 if y.mean() > 0.2 else 0.30, rng=rng)
            else:
                y = pd.Series(y.to_numpy() + 0.75 * y.std(), name=y.name)
        elif t == 3:  # concept: P(y|x) changes
            if clf:
                y = apply_concept_shift_classification(df, y, num[:2], flip_rate=0.4, rng=rng)
            else:
                y = apply_concept_shift_regression(df, y, (num[0], num[1]), shift_gain=0.8 * y.std())
        elif t == 4:  # correlation (rotation) + noise
            df = apply_rotation_shift(df, (num[0], num[1]), np.pi / 3)
            y = _relabel(generator_fn, df, y, spec, seed, t)
            if clf:
                arr = y.to_numpy().copy()
                flip = rng.choice(len(arr), size=int(0.10 * len(arr)), replace=False)
                arr[flip] = 1 - arr[flip]
                y = pd.Series(arr, name=y.name)
            else:
                y = apply_noise_shift(y, 0.4 * y.std(), rng)
        elif t == 5:  # hidden compound: scale + mean shift + new concept + noise
            df = apply_mean_shift(apply_scale_shift(df, num[:3], 2.0), [num[2]], 1.8)
            y = _relabel(generator_fn, df, y, spec, seed, t)
            if clf:
                y = apply_concept_shift_classification(df, y, [num[1]], flip_rate=0.35, rng=rng)
                arr = y.to_numpy().copy()
                flip = rng.choice(len(arr), size=int(0.08 * len(arr)), replace=False)
                arr[flip] = 1 - arr[flip]
                y = pd.Series(arr, name=y.name)
            else:
                y = apply_concept_shift_regression(df, y, (num[1], num[2]), shift_gain=0.7 * y.std())
                y = apply_noise_shift(y, 0.25 * y.std(), rng)
        batches.append((ENV_NAMES[t], df, y))
    return batches, spec


def _relabel(generator_fn, df, y, spec, seed, t):
    """Re-draw labels from the family's unchanged P(y|x) after the inputs moved."""
    if hasattr(generator_fn, "relabel"):
        return pd.Series(generator_fn.relabel(df, seed, t), name=y.name)
    return y


def create_stationary_stream(generator_fn: Callable, n_batches: int = 6, batch_size: int = 1000, seed: int = 0):
    """No shift at all: used to measure the detector's false-alarm rate."""
    df_all, y_all, spec = generator_fn(n_samples=batch_size * n_batches, seed=seed)
    return [(f"stationary_{t}", df_all.iloc[t * batch_size:(t + 1) * batch_size].reset_index(drop=True),
             y_all.iloc[t * batch_size:(t + 1) * batch_size].reset_index(drop=True)) for t in range(n_batches)], spec
