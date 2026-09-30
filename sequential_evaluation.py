"""
Sequential evaluator: the single authoritative evaluation protocol.

For each stream (dataset family x seed) every batch t is split ONCE into an
update part (given to the learner) and a holdout part (kept by the evaluator).
The split depends only on (seed, t), so every model and every ablation sees
exactly the same rows. (In the original harness one RNG was shared across
models, so each model was scored on different rows.)

Protocol per batch t >= 1 (test-then-train):
    P[t]    = skill on holdout_t BEFORE the learner sees batch t (prequential)
    update(update_t)                               (only this batch is passed)
    R[t, j] = skill on holdout_j for every j <= t   (accuracy matrix)

Summary (standard continual-learning metrics):
    prequential_skill   mean_t P[t]                 robustness to arriving shift
    post_update_skill   mean_t R[t, t]              adaptation
    final_avg_skill     mean_j R[T, j]              knowledge over all regimes
    forgetting          mean_j max_{t<T} R[t, j] - R[T, j]
    hidden_prequential  P[T] on the final compound-shift batch
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.memory_accounting import deep_nbytes
from src.metrics import evaluate_predictions
from src.task import TaskSpec

Split = Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]


def split_stream(batches: List[Tuple[str, pd.DataFrame, pd.Series]], seed: int, holdout_fraction: float = 0.2) -> List[Split]:
    out = []
    for t, (_, X, y) in enumerate(batches):
        rng = np.random.default_rng([seed, t, 7919])
        perm = rng.permutation(len(X))
        n_ho = int(round(holdout_fraction * len(X)))
        ho, up = perm[:n_ho], perm[n_ho:]
        out.append((X.iloc[up].reset_index(drop=True), y.iloc[up].reset_index(drop=True),
                    X.iloc[ho].reset_index(drop=True), y.iloc[ho].reset_index(drop=True)))
    return out


@dataclass
class StreamResult:
    model: str
    dataset: str
    seed: int
    task_type: str
    env_names: List[str]
    R: np.ndarray
    prequential: np.ndarray
    latency: np.ndarray
    state_bytes: np.ndarray
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        T = len(self.env_names)
        last = T - 1
        R, P = self.R, self.prequential
        forgetting = [max(R[t, j] for t in range(j, last)) - R[last, j] for j in range(last)]
        return {
            "dataset": self.dataset, "seed": self.seed, "model": self.model, "task_type": self.task_type,
            "prequential_skill": float(np.mean(P[1:])),
            "worst_prequential_skill": float(np.min(P[1:])),
            "post_update_skill": float(np.mean([R[t, t] for t in range(1, T)])),
            "final_avg_skill": float(np.mean(R[last, :])),
            "forgetting": float(np.mean(forgetting)),
            "backward_transfer": float(np.mean([R[last, j] - R[j, j] for j in range(last)])),
            "hidden_prequential_skill": float(P[last]),
            "hidden_post_update_skill": float(R[last, last]),
            "mean_update_latency_sec": float(np.mean(self.latency[1:])),
            "init_latency_sec": float(self.latency[0]),
            "final_state_kb": float(self.state_bytes[-1] / 1024),
            "peak_state_kb": float(np.max(self.state_bytes) / 1024),
        }


def _state_bytes(model: Any) -> int:
    if hasattr(model, "memory_footprint"):
        return int(model.memory_footprint()["total_bytes"])
    return deep_nbytes(model)


class SequentialEvaluator:
    def __init__(self, holdout_fraction: float = 0.2) -> None:
        self.holdout_fraction = holdout_fraction

    def _skill(self, model: Any, task: TaskSpec, X, y) -> float:
        if task.task_type == "binary_classification":
            prob = model.predict_proba(X)[:, 1]
            return evaluate_predictions(task.task_type, y, None, prob)["skill"]
        return evaluate_predictions(task.task_type, y, model.predict(X))["skill"]

    def run(self, model: Any, splits: List[Split], task: TaskSpec, env_names: List[str],
            model_name: str, dataset: str, seed: int,
            on_update: Optional[Callable[[int, Any], None]] = None) -> StreamResult:
        T = len(splits)
        R = np.full((T, T), np.nan)
        P = np.full(T, np.nan)
        lat = np.zeros(T)
        mem = np.zeros(T)
        reports = []
        for t in range(T):
            X_up, y_up, X_ho, y_ho = splits[t]
            if t == 0:
                t0 = time.perf_counter()
                model.initialize(X_up, y_up, task)
                lat[t] = time.perf_counter() - t0
            else:
                P[t] = self._skill(model, task, X_ho, y_ho)          # test ...
                t0 = time.perf_counter()
                rep = model.update(X_up, y_up)                        # ... then train
                lat[t] = time.perf_counter() - t0
                if rep is not None and hasattr(rep, "to_dict"):
                    reports.append(rep.to_dict())
            for j in range(t + 1):
                R[t, j] = self._skill(model, task, splits[j][2], splits[j][3])
            mem[t] = _state_bytes(model)
            if on_update is not None:
                on_update(t, model)
        return StreamResult(model_name, dataset, seed, task.task_type, env_names, R, P, lat, mem,
                            extra={"reports": reports})
