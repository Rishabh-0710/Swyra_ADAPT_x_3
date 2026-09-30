import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

from src.config import ADAPTXConfig
from src.task import TaskSpec


def make_regression(n, seed, shift=0.0, concept=False, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[:, 0] += shift
    y = 2 * X[:, 0] + np.sin(2 * X[:, 1]) + 0.5 * X[:, 2] * X[:, 3]
    if concept:
        y = -2 * X[:, 0] + np.sin(2 * X[:, 1]) + 1.5 * X[:, 2]
    y = y + rng.normal(0, 0.3, n)
    cols = [f"c{i}" for i in range(d)]
    return pd.DataFrame(X, columns=cols), pd.Series(y), TaskSpec("regression", numerical_features=cols)


def make_classification(n, seed, pos_rate=None, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    z = 1.5 * X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    if pos_rate is not None:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        k = int(pos_rate * n)
        idx = rng.permutation(np.concatenate([rng.choice(pos, k), rng.choice(neg, n - k)]))
        df, y = df.iloc[idx].reset_index(drop=True), y[idx]
    return df, pd.Series(y), TaskSpec("binary_classification", numerical_features=list(df.columns))


@pytest.fixture
def small_cfg():
    return ADAPTXConfig(replay_capacity=200, reference_capacity=256, base_max_iter=100, stage_trees_max=80, random_seed=3)
