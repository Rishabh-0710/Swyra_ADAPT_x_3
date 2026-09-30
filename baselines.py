"""
Baselines. Every baseline uses the same frozen StreamingFeatureEncoder and
the same histogram-GBDT hyper-parameters as ADAPT-X's base stage, so any
difference comes from the continual-learning strategy, not the learner.

1. StaticModel            fit on batch 0, never updated
2. NaiveIncrementalModel  a fresh model on each new batch only (no memory)
3. StandardReplayModel    a fresh model on new batch + FIFO replay (K rows)
4. FullRetrainOracle      a fresh model on ALL history (unbounded memory;
                          violates the problem constraints; reference only)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.config import ADAPTXConfig
from src.memory_accounting import deep_nbytes
from src.preprocessing import StreamingFeatureEncoder
from src.task import TaskSpec, infer_task_spec


class _GBDTBaseline:
    def __init__(self, config: Optional[ADAPTXConfig] = None) -> None:
        self.cfg = config or ADAPTXConfig()
        self.task: Optional[TaskSpec] = None
        self.encoder: Optional[StreamingFeatureEncoder] = None
        self.model = None
        self.n_updates = 0

    def _new_model(self):
        c = self.cfg
        kw = dict(max_iter=c.base_max_iter, learning_rate=c.base_learning_rate, max_leaf_nodes=c.max_leaf_nodes,
                  min_samples_leaf=c.min_samples_leaf, l2_regularization=c.l2_regularization, early_stopping=True,
                  validation_fraction=0.15, n_iter_no_change=10, categorical_features=self.encoder.categorical_mask_,
                  random_state=c.random_seed + self.n_updates)
        return HistGradientBoostingClassifier(**kw) if self.task.task_type == "binary_classification" else HistGradientBoostingRegressor(**kw)

    def initialize(self, X, y, task_spec: Optional[TaskSpec] = None):
        self.task = task_spec or infer_task_spec(X, y)
        self.encoder = StreamingFeatureEncoder(self.task, self.cfg.max_categories, self.cfg.n_hash_buckets).fit_schema(X)
        Z = self.encoder.transform(X)
        yy = np.asarray(y, dtype=float)
        self._on_init(Z, yy)
        self.model = self._new_model().fit(*self._training_set(Z, yy))
        return self

    def update(self, X, y):
        self.n_updates += 1
        Z = self.encoder.transform(X)
        yy = np.asarray(y, dtype=float)
        if self._retrains():
            self.model = self._new_model().fit(*self._training_set(Z, yy))
        self._after_update(Z, yy)
        return None

    # hooks
    def _on_init(self, Z, y): ...
    def _after_update(self, Z, y): ...
    def _retrains(self) -> bool: return True
    def _training_set(self, Z, y): return Z, y

    def predict(self, X):
        Z = self.encoder.transform(X)
        return self.model.predict(Z)

    def predict_proba(self, X):
        return self.model.predict_proba(self.encoder.transform(X))

    def memory_footprint(self):
        total = deep_nbytes(self.model) + deep_nbytes(self.encoder) + self._stored_bytes()
        return {"total_bytes": int(total)}

    def _stored_bytes(self) -> int:
        return 0


class StaticModel(_GBDTBaseline):
    def _retrains(self):
        return False


class NaiveIncrementalModel(_GBDTBaseline):
    pass


class StandardReplayModel(_GBDTBaseline):
    """FIFO replay of the most recent K rows; no drift detection, no stability control."""

    def _on_init(self, Z, y):
        K = self.cfg.replay_capacity
        self.rX, self.ry = Z[-K:].copy(), y[-K:].copy()

    def _training_set(self, Z, y):
        if self.n_updates == 0:
            return Z, y
        return np.vstack([Z, self.rX]), np.concatenate([y, self.ry])

    def _after_update(self, Z, y):
        K = self.cfg.replay_capacity
        self.rX = np.vstack([self.rX, Z])[-K:].copy()
        self.ry = np.concatenate([self.ry, y])[-K:].copy()

    def _stored_bytes(self):
        return self.rX.nbytes + self.ry.nbytes


class FullRetrainOracle(_GBDTBaseline):
    """Keeps every row ever seen and retrains from scratch: O(total rows) memory and time."""

    def _on_init(self, Z, y):
        self.allX: List[np.ndarray] = [Z]
        self.ally: List[np.ndarray] = [y]

    def _training_set(self, Z, y):
        if self.n_updates == 0:
            return Z, y
        self.allX.append(Z)
        self.ally.append(y)
        return np.vstack(self.allX), np.concatenate(self.ally)

    def _stored_bytes(self):
        return sum(a.nbytes for a in self.allX) + sum(a.nbytes for a in self.ally)
