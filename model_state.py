"""
Persistent adaptive model state.

The model is an additive function in margin space (identity for regression,
logit for binary classification):

    F(x) = b + sum_k beta_k * h_k(x)

* ``h_0`` is the **base stage** learned from batch 0 (a histogram GBDT).
* ``h_1..h_m`` are **correction stages**. Each is a small GBDT fitted on a
  later batch to the pseudo-residuals of the *then-current* F (functional
  gradient boosting continued across batches). Once fitted, a stage's trees
  are frozen and never refitted.
* ``(b, beta)`` are the **combiner** parameters. They are re-estimated on
  every batch by a closed-form or Newton solve with an explicit
  stability-plasticity objective (see :func:`fit_combiner`).

This state persists across batches. An update adds at most one stage and
re-weights the existing ones; it never retrains from scratch. The number of
stages is capped (``max_stages``), so the model size is bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from src.memory_accounting import deep_nbytes


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


@dataclass
class Stage:
    estimator: Any
    kind: str          # "base" | "correction"
    born_at_batch: int
    n_trees: int

    def output(self, Z: np.ndarray, task_type: str) -> np.ndarray:
        if self.kind == "base" and task_type == "binary_classification":
            return self.estimator.decision_function(Z)
        return self.estimator.predict(Z)


def fit_combiner(
    task_type: str,
    H: np.ndarray, y: np.ndarray, w: np.ndarray,
    H_anchor: Optional[np.ndarray], F_anchor: Optional[np.ndarray],
    stability: float,
    theta_prior: np.ndarray,
    ridge: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """
    Solve for theta = (b, beta) minimising

        sum_i w_i * loss(y_i, b + H_i beta)                      (fit new evidence)
      + lam * sum_a (b + H_a beta - F_old(x_a))^2                 (don't forget: distillation)
      + eps * sum_k s_k (theta_k - theta_prior_k)^2                (numerical ridge)

    ``lam = stability * sum(w) / n_anchor`` (x 0.25 for log-loss, to match its
    curvature), so ``stability`` is a dimensionless trade-off: at 1 the old
    function on the anchor rows carries as much weight as the labelled data.
    Returns (theta, objective value).
    """
    n, k = H.shape
    A = np.hstack([np.ones((n, 1)), H])
    sw = float(np.sum(w))
    scale = np.concatenate([[1.0], np.mean(H ** 2, axis=0) + 1e-12])
    eps = ridge * sw
    if H_anchor is not None and len(H_anchor) and stability > 0:
        Aa = np.hstack([np.ones((len(H_anchor), 1)), H_anchor])
        lam = stability * sw / len(H_anchor) * (0.25 if task_type != "regression" else 1.0)
    else:
        Aa, lam = np.zeros((0, k + 1)), 0.0
        F_anchor = np.zeros(0)

    def objective(theta: np.ndarray) -> float:
        m = A @ theta
        if task_type == "regression":
            data = float(np.sum(w * (y - m) ** 2))
        else:
            data = float(np.sum(w * (np.logaddexp(0, m) - y * m)))
        dist = lam * float(np.sum((Aa @ theta - F_anchor) ** 2)) if lam else 0.0
        return data + dist + eps * float(np.sum(scale * (theta - theta_prior) ** 2))

    P = np.diag(eps * scale)
    if task_type == "regression":
        M = A.T @ (A * w[:, None]) + lam * Aa.T @ Aa + P
        rhs = A.T @ (w * y) + lam * Aa.T @ F_anchor + P @ theta_prior
        theta = np.linalg.solve(M, rhs)
        return theta, objective(theta)

    theta = theta_prior.astype(float).copy()
    f = objective(theta)
    for _ in range(30):
        p = sigmoid(A @ theta)
        g = A.T @ (w * (p - y)) + 2 * lam * Aa.T @ (Aa @ theta - F_anchor) + 2 * P @ (theta - theta_prior)
        Hm = A.T @ (A * (w * p * (1 - p))[:, None]) + 2 * lam * Aa.T @ Aa + 2 * P
        step = np.linalg.solve(Hm + 1e-9 * np.eye(k + 1), g)
        t = 1.0
        while t > 1e-4:
            cand = theta - t * step
            fc = objective(cand)
            if fc <= f:
                break
            t *= 0.5
        if t <= 1e-4:
            break
        converged = f - fc < 1e-9 * max(1.0, abs(f))
        theta, f = cand, fc
        if converged:
            break
    return theta, f


class AdaptiveModelState:
    def __init__(self, task_type: str, max_stages: int) -> None:
        self.task_type = task_type
        self.max_stages = int(max_stages)
        self.stages: List[Stage] = []
        self.beta = np.zeros(0)
        self.intercept = 0.0

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([[self.intercept], self.beta])

    def set_theta(self, theta: np.ndarray) -> None:
        self.intercept = float(theta[0])
        self.beta = np.asarray(theta[1:], dtype=float).copy()

    def stage_outputs(self, Z: np.ndarray) -> np.ndarray:
        if not self.stages:
            return np.zeros((len(Z), 0))
        return np.column_stack([s.output(Z, self.task_type) for s in self.stages])

    def margin(self, Z: np.ndarray, H: Optional[np.ndarray] = None) -> np.ndarray:
        H = self.stage_outputs(Z) if H is None else H
        return self.intercept + H @ self.beta

    def predict_value(self, Z: np.ndarray, H: Optional[np.ndarray] = None) -> np.ndarray:
        m = self.margin(Z, H)
        return m if self.task_type == "regression" else sigmoid(m)

    def add_stage(self, stage: Stage, prior_weight: float = 1.0) -> None:
        self.stages.append(stage)
        self.beta = np.concatenate([self.beta, [prior_weight]])

    def remove_stage(self, idx: int) -> None:
        del self.stages[idx]
        self.beta = np.delete(self.beta, idx)

    def total_trees(self) -> int:
        return int(sum(s.n_trees for s in self.stages))

    def nbytes(self) -> int:
        return deep_nbytes(self.stages) + self.beta.nbytes
