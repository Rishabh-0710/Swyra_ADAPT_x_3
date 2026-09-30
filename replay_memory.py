"""
Bounded replay memory: the only place where labelled rows of past batches
may survive, and never more than ``capacity`` of them.

Rows are stored in the encoder's frozen feature space, which never changes
meaning over time, so a stored row stays valid for later updates.

Selection (``stratified_diversity``):
  * target stratification: classes, or target quantiles for regression, with
    proportional quotas and a floor so rare strata are not lost;
  * inside each stratum, half the quota is chosen by greedy farthest-point
    sampling (covers the feature space, including rare regions) and half at
    random (keeps the density representative);
  * the controller sets ``new_share``, the part of the capacity given to the
    newest batch. Balanced mode keeps an approximately uniform history, and
    concept-drift mode gives more room to the new regime.

``reservoir`` is the random-replay ablation: uniform random selection with
the same capacity and the same new/old split.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class BoundedReplayMemory:
    def __init__(
        self,
        task_type: str,
        capacity: int,
        categorical_mask: np.ndarray,
        distance_scale: np.ndarray,
        strategy: str = "stratified_diversity",
        n_strata: int = 10,
        random_seed: int = 0,
    ) -> None:
        if strategy not in ("stratified_diversity", "reservoir"):
            raise ValueError(strategy)
        self.task_type = task_type
        self.capacity = max(0, int(capacity))
        self.categorical_mask = np.asarray(categorical_mask, dtype=bool)
        n_num = len(distance_scale)
        self._num_idx = np.arange(n_num)
        self._scale = np.asarray(distance_scale, dtype=float)
        self._cat_idx = np.where(self.categorical_mask)[0]
        self.strategy = strategy
        self.n_strata = 2 if task_type == "binary_classification" else int(n_strata)
        self.rng = np.random.default_rng(random_seed)

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.batch_index: Optional[np.ndarray] = None
        self.n_observed: int = 0             # scalar counters only
        self.observed_pos_rate: Optional[float] = None

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return 0 if self.X is None else int(len(self.X))

    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.X, self.y, self.batch_index) if a is not None)

    def _strata(self, y: np.ndarray) -> np.ndarray:
        if self.task_type == "binary_classification":
            return (y > 0.5).astype(int)
        if len(y) < self.n_strata:
            return np.zeros(len(y), dtype=int)
        edges = np.unique(np.quantile(y, np.linspace(0, 1, self.n_strata + 1)[1:-1]))
        return np.searchsorted(edges, y, side="right")

    def _embed(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num = X[:, self._num_idx] / self._scale if len(self._num_idx) else np.zeros((len(X), 0))
        num = np.nan_to_num(num, nan=0.0)
        num = np.clip(num, -50, 50)
        cat = X[:, self._cat_idx] if len(self._cat_idx) else np.zeros((len(X), 0))
        return num, cat

    def _farthest_points(self, X: np.ndarray, k: int) -> np.ndarray:
        num, cat = self._embed(X)
        first = int(self.rng.integers(len(X)))
        chosen = [first]
        d = np.sum((num - num[first]) ** 2, axis=1) + np.sum(cat != cat[first], axis=1)
        for _ in range(1, k):
            nxt = int(np.argmax(d))
            chosen.append(nxt)
            d = np.minimum(d, np.sum((num - num[nxt]) ** 2, axis=1) + np.sum(cat != cat[nxt], axis=1))
        return np.asarray(chosen, dtype=int)

    def _select(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        n = len(y)
        if k >= n:
            return np.arange(n)
        if k <= 0:
            return np.zeros(0, dtype=int)
        if self.strategy == "reservoir":
            return np.sort(self.rng.choice(n, size=k, replace=False))
        strata = self._strata(y)
        labels, counts = np.unique(strata, return_counts=True)
        floor = np.minimum(counts, int(np.ceil(k * 0.5 / max(len(labels), 1))))
        quota = np.maximum(floor, np.floor(k * counts / n).astype(int))
        while quota.sum() > k:
            j = int(np.argmax(quota - floor)) if (quota > floor).any() else int(np.argmax(quota))
            quota[j] -= 1
        while quota.sum() < k:
            room = counts - quota
            j = int(np.argmax(room))
            quota[j] += 1
        picked = []
        for lab, q in zip(labels, quota):
            idx = np.where(strata == lab)[0]
            if q >= len(idx):
                picked.append(idx)
                continue
            n_fps = q // 2
            fps = idx[self._farthest_points(X[idx], n_fps)] if n_fps > 0 else np.zeros(0, dtype=int)
            rest = np.setdiff1d(idx, fps)
            rnd = self.rng.choice(rest, size=q - n_fps, replace=False)
            picked.append(np.concatenate([fps, rnd]))
        return np.sort(np.concatenate(picked))

    # ------------------------------------------------------------------
    def update(self, X_new: np.ndarray, y_new: np.ndarray, batch_index: int, new_share: Optional[float] = None) -> None:
        X_new = np.asarray(X_new, dtype=float)
        y_new = np.asarray(y_new, dtype=float)
        self.n_observed += len(y_new)
        if self.task_type == "binary_classification":
            rate = float(np.mean(y_new)) if len(y_new) else 0.5
            self.observed_pos_rate = rate if self.observed_pos_rate is None else 0.7 * self.observed_pos_rate + 0.3 * rate
        if self.capacity == 0:
            return
        n_old = self.size
        if new_share is None:
            new_share = 1.0 if n_old == 0 else 0.5
        k_new = int(round(new_share * self.capacity))
        k_new = min(len(y_new), max(k_new, self.capacity - n_old))
        k_old = min(n_old, self.capacity - k_new)

        sel_new = self._select(X_new, y_new, k_new)
        parts_X, parts_y, parts_b = [X_new[sel_new]], [y_new[sel_new]], [np.full(len(sel_new), batch_index, dtype=np.int32)]
        if n_old:
            sel_old = self._select(self.X, self.y, k_old)
            parts_X.insert(0, self.X[sel_old])
            parts_y.insert(0, self.y[sel_old])
            parts_b.insert(0, self.batch_index[sel_old])
        # np.concatenate always allocates new arrays: no view of the caller's batch survives
        self.X = np.ascontiguousarray(np.concatenate(parts_X))
        self.y = np.ascontiguousarray(np.concatenate(parts_y))
        self.batch_index = np.concatenate(parts_b)
        assert self.size <= self.capacity

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, y, prior-correction weights). Empty arrays if memory is empty."""
        if self.size == 0:
            return np.zeros((0, len(self.categorical_mask))), np.zeros(0), np.zeros(0)
        w = np.ones(self.size)
        if self.task_type == "binary_classification" and self.observed_pos_rate is not None:
            mem_rate = float(np.clip(np.mean(self.y), 1e-3, 1 - 1e-3))
            obs = float(np.clip(self.observed_pos_rate, 1e-3, 1 - 1e-3))
            w = np.where(self.y > 0.5, obs / mem_rate, (1 - obs) / (1 - mem_rate))
        return self.X.copy(), self.y.copy(), w
