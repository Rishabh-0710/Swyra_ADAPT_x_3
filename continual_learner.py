"""
ADAPT-X: Adaptive continual learning for one tabular supervised task under
dynamic distribution shift.

    LEARN  ->  DETECT  ->  ADAPT  ->  RETAIN  ->  LEARN AGAIN

``initialize(X0, y0)``  LEARN
    freeze the feature schema, fit the base stage, calibrate the bounded
    drift reference on held-out rows, seed the bounded replay memory.

``update(X_t, y_t)``    one continual-learning step, strictly ordered:
    1. encode X_t with the encoder state from BEFORE this batch
    2. pre-update prediction with the persistent model state
    3. DETECT   covariate / correlation / concept / label shift
    4. ADAPT    the controller turns the report into an UpdatePlan
    5. grow one correction stage on pseudo-residuals of the current model
       (new rows + half of replay, replay down-weighted by the plan)
    6. RETAIN   re-fit combiner weights: new-evidence loss + functional
       distillation toward the pre-update model on replayed inputs
    7. enforce the stage budget (evict the least useful learned function)
    8. update the bounded replay memory, then refresh the drift reference
    9. only now let the encoder absorb (X_t, y_t) for the NEXT batch

Nothing in the learner holds historical raw data: past labelled rows exist
only in the replay memory (<= replay_capacity rows) and past inputs only in
the drift reference (<= reference_capacity rows).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.adaptation_controller import AdaptationController, UpdatePlan
from src.config import ADAPTXConfig
from src.model_state import AdaptiveModelState, Stage, fit_combiner, sigmoid
from src.preprocessing import StreamingFeatureEncoder
from src.replay_memory import BoundedReplayMemory
from src.shift_detection import DistributionShiftDetector, ShiftReport, per_row_loss
from src.task import TaskSpec, infer_task_spec

ArrayLike = Union[pd.DataFrame, np.ndarray]


class ContinualLearner:
    """ADAPT-X continual learner (domain-agnostic; regression or binary classification)."""

    def __init__(self, config: Optional[ADAPTXConfig] = None, task_spec: Optional[TaskSpec] = None) -> None:
        self.cfg = config or ADAPTXConfig()
        self.task_spec = task_spec
        self.encoder: Optional[StreamingFeatureEncoder] = None
        self.state: Optional[AdaptiveModelState] = None
        self.memory: Optional[BoundedReplayMemory] = None
        self.detector: Optional[DistributionShiftDetector] = None
        self.controller = AdaptationController(self.cfg)
        self.history: deque = deque(maxlen=self.cfg.history_length)
        self.n_batches_seen = 0
        self.n_rows_seen = 0
        self.rng = np.random.default_rng(self.cfg.random_seed)

    # ============================================================ helpers
    @property
    def task_type(self) -> str:
        return self.task_spec.task_type

    def _target(self, y) -> np.ndarray:
        y = np.asarray(y)
        if self.task_type == "binary_classification" and self.task_spec.pos_label is not None and y.dtype.kind not in "biuf":
            return (y == self.task_spec.pos_label).astype(float)
        return y.astype(float)

    def _split(self, y: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
        """Stratified random split into (fit rows, held-out check rows)."""
        n = len(y)
        idx = self.rng.permutation(n)
        if self.task_type == "binary_classification":
            check = np.concatenate([idx[y[idx] == c][: int(round(frac * np.sum(y == c)))] for c in (0.0, 1.0)])
        else:
            check = idx[: int(round(frac * n))]
        mask = np.zeros(n, dtype=bool)
        mask[check] = True
        return np.where(~mask)[0], np.where(mask)[0]

    def _gbdt(self, kind: str, n_trees: int, lr: float, leaves: int, seed_offset: int):
        common = dict(
            max_iter=int(max(5, n_trees)), learning_rate=lr, max_leaf_nodes=leaves,
            min_samples_leaf=self.cfg.min_samples_leaf, l2_regularization=self.cfg.l2_regularization,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=10,
            categorical_features=self.encoder.categorical_mask_, random_state=self.cfg.random_seed + seed_offset,
        )
        if kind == "classifier":
            return HistGradientBoostingClassifier(**common)
        return HistGradientBoostingRegressor(**common)

    def _pseudo_residuals(self, y: np.ndarray, F: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Negative gradient (regression) or Newton step (log-loss) of the current model."""
        if self.task_type == "regression":
            return y - F, w
        p = sigmoid(F)
        h = np.maximum(p * (1 - p), 1e-3)
        return np.clip((y - p) / h, -4.0, 4.0), w * h

    # ============================================================ LEARN
    def initialize(self, X0: ArrayLike, y0, task_spec: Optional[TaskSpec] = None) -> "ContinualLearner":
        cfg = self.cfg
        if task_spec is not None:
            self.task_spec = task_spec
        if self.task_spec is None:
            self.task_spec = infer_task_spec(X0, y0)
        y = self._target(y0)

        self.encoder = StreamingFeatureEncoder(
            self.task_spec, cfg.max_categories, cfg.n_hash_buckets, cfg.use_target_encoding, random_seed=cfg.random_seed
        ).fit_schema(X0)
        Z = self.encoder.initial_transform(X0, y)
        fit_idx, chk_idx = self._split(y, cfg.internal_holdout)

        base = self._gbdt("classifier" if self.task_type == "binary_classification" else "regressor",
                          cfg.base_max_iter, cfg.base_learning_rate, cfg.max_leaf_nodes, 0)
        base.fit(Z[fit_idx], y[fit_idx])
        self.state = AdaptiveModelState(self.task_type, cfg.max_stages)
        self.state.add_stage(Stage(base, "base", 0, int(base.n_iter_)))

        pred_chk = self.state.predict_value(Z[chk_idx])
        n_num = len(self.encoder.numerical_features)
        self.detector = DistributionShiftDetector(
            self.task_type, self.encoder.feature_names_out_, n_num, len(self.encoder.categorical_features),
            reference_capacity=cfg.reference_capacity, alpha=cfg.alpha, alpha_model=cfg.alpha_model, psi_min_effect=cfg.psi_min_effect,
            corr_min_effect=cfg.corr_min_effect, loss_min_effect=cfg.loss_min_effect,
            min_support_rows=cfg.min_support_rows, random_seed=cfg.random_seed,
        )
        self.detector.fit_reference(Z, per_row_loss(self.task_type, y[chk_idx], pred_chk), y[chk_idx] - pred_chk)

        self.memory = BoundedReplayMemory(
            self.task_type, cfg.replay_capacity, self.encoder.categorical_mask_, self.encoder.distance_scale_,
            strategy=cfg.replay_strategy, random_seed=cfg.random_seed,
        )
        self.memory.update(Z, y, batch_index=0, new_share=1.0)
        self.encoder.observe(X0, y)           # statistics become visible only from batch 1 on
        self.n_batches_seen, self.n_rows_seen = 1, len(y)
        return self

    # ============================================================ update
    def update(self, X: ArrayLike, y_raw) -> ShiftReport:
        if self.state is None:
            raise RuntimeError("call initialize() first")
        cfg, t0 = self.cfg, time.perf_counter()
        batch = self.n_batches_seen
        y = self._target(y_raw)

        # 1-2. encode with pre-batch state; pre-update prediction
        Z = self.encoder.transform(X)
        H_new = self.state.stage_outputs(Z)
        pred_pre = self.state.predict_value(Z, H_new)

        # 3-4. DETECT -> ADAPT decision
        report = self.detector.detect(Z, y, pred_pre, batch_index=batch)
        plan = self.controller.plan(report, self.n_batches_seen)

        fit_idx, chk_idx = self._split(y, cfg.internal_holdout)
        Zr, yr, wr = self.memory.get()
        perm = self.rng.permutation(len(yr))
        rA, rB = perm[: len(perm) // 2], perm[len(perm) // 2:]

        if cfg.persistent_state:
            self._continual_step(Z, y, H_new, fit_idx, chk_idx, Zr, yr, wr, rA, rB, plan, batch)
        else:
            self._refit_from_scratch(Z, y, fit_idx, Zr, yr, wr, batch)

        # 8. RETAIN: bounded replay memory + drift reference move forward
        self.memory.update(Z, y, batch_index=batch, new_share=plan.memory_new_share)
        pred_chk = self.state.predict_value(Z[chk_idx])
        self.detector.refresh(Z, plan.reference_refresh,
                              per_row_loss(self.task_type, y[chk_idx], pred_chk), y[chk_idx] - pred_chk)
        # 9. encoder statistics absorb this batch only after the update is complete
        self.encoder.observe(X, y)

        self.n_batches_seen += 1
        self.n_rows_seen += len(y)
        self.history.append({
            "batch": batch, "report": report.to_dict(), "plan": plan.to_dict(),
            "n_stages": len(self.state.stages), "total_trees": self.state.total_trees(),
            "weights": np.round(self.state.beta, 4).tolist(), "intercept": round(self.state.intercept, 4),
            "latency_sec": time.perf_counter() - t0,
        })
        return report

    def _continual_step(self, Z, y, H_new, fit_idx, chk_idx, Zr, yr, wr, rA, rB, plan: UpdatePlan, batch: int) -> None:
        cfg, st = self.cfg, self.state
        theta_old = st.theta
        Hr = st.stage_outputs(Zr) if len(yr) else np.zeros((0, len(st.stages)))
        F_new = st.intercept + H_new @ st.beta
        F_rep = st.intercept + Hr @ st.beta if len(yr) else np.zeros(0)

        # 5. grow one correction stage on pseudo-residuals of the current model
        if plan.grow_stage and len(fit_idx) >= 50:
            Zs = np.vstack([Z[fit_idx], Zr[rA]]) if len(rA) else Z[fit_idx]
            ys = np.concatenate([y[fit_idx], yr[rA]])
            ws = np.concatenate([np.ones(len(fit_idx)), plan.replay_weight * wr[rA]])
            Fs = np.concatenate([F_new[fit_idx], F_rep[rA]])
            target, sw = self._pseudo_residuals(ys, Fs, ws)
            stage_est = self._gbdt("regressor", plan.stage_trees, plan.stage_learning_rate,
                                   cfg.stage_max_leaf_nodes, 1000 + batch)
            stage_est.fit(Zs, target, sample_weight=sw)
            stage = Stage(stage_est, "correction", batch, int(stage_est.n_iter_))
            st.add_stage(stage, prior_weight=1.0)
            H_new = np.column_stack([H_new, stage.output(Z, self.task_type)])
            if len(yr):
                Hr = np.column_stack([Hr, stage.output(Zr, self.task_type)])

        # 6. combiner: fit held-out new rows + replay half B, distill old function on replay half B
        rows = chk_idx if cfg.combiner_rows == "heldout" else np.arange(len(y))
        Hd = np.vstack([H_new[rows], Hr[rB]]) if len(rB) else H_new[rows]
        yd = np.concatenate([y[rows], yr[rB]])
        wd = np.concatenate([np.ones(len(rows)), plan.replay_weight * wr[rB]])
        if len(rB):
            H_anchor, F_anchor = Hr[rB], F_rep[rB]
        else:  # no replay available: anchor on new inputs (pure LwF-style stability)
            H_anchor, F_anchor = H_new[chk_idx], F_new[chk_idx]
        prior = np.concatenate([theta_old, np.ones(len(st.stages) + 1 - len(theta_old))])

        def solve(cols):
            idx = np.array([0] + [c + 1 for c in cols])
            return fit_combiner(self.task_type, Hd[:, cols], yd, wd, H_anchor[:, cols], F_anchor,
                                plan.stability, prior[idx])

        cols = list(range(len(st.stages)))
        theta, obj = solve(cols)
        # 7. bounded model: drop dead stages and evict the least useful one when over budget
        f_scale = float(np.sqrt(np.mean((Hd @ prior[1:len(st.stages) + 1]) ** 2))) + 1e-12
        while len(cols) > 1:
            weak = [i for i, c in enumerate(cols)
                    if abs(theta[i + 1]) * np.sqrt(np.mean(Hd[:, c] ** 2)) < 1e-3 * f_scale]
            if not weak and len(cols) <= st.max_stages:
                break
            if weak:
                drop = weak[0]
            else:
                costs = [solve(cols[:i] + cols[i + 1:])[1] - obj for i in range(len(cols))]
                drop = int(np.argmin(costs))
            cols = cols[:drop] + cols[drop + 1:]
            theta, obj = solve(cols)
        for i in sorted(set(range(len(st.stages))) - set(cols), reverse=True):
            st.remove_stage(i)
        st.set_theta(theta)

    def _refit_from_scratch(self, Z, y, fit_idx, Zr, yr, wr, batch: int) -> None:
        """Ablation D: no persistent state; a fresh model on (new batch + replay)."""
        cfg = self.cfg
        Zs = np.vstack([Z[fit_idx], Zr]) if len(yr) else Z[fit_idx]
        ys = np.concatenate([y[fit_idx], yr])
        ws = np.concatenate([np.ones(len(fit_idx)), wr])
        est = self._gbdt("classifier" if self.task_type == "binary_classification" else "regressor",
                         cfg.base_max_iter, cfg.base_learning_rate, cfg.max_leaf_nodes, 2000 + batch)
        est.fit(Zs, ys, sample_weight=ws)
        self.state = AdaptiveModelState(self.task_type, cfg.max_stages)
        self.state.add_stage(Stage(est, "base", batch, int(est.n_iter_)))

    # ============================================================ inference
    def predict(self, X: ArrayLike) -> np.ndarray:
        v = self.state.predict_value(self.encoder.transform(X))
        return v if self.task_type == "regression" else (v >= 0.5).astype(int)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.task_type != "binary_classification":
            raise ValueError("predict_proba is only defined for binary classification")
        p = self.state.predict_value(self.encoder.transform(X))
        return np.column_stack([1 - p, p])

    # ============================================================ audit
    def memory_footprint(self) -> Dict[str, int]:
        from src.memory_accounting import deep_nbytes

        parts = {
            "model_state_bytes": self.state.nbytes(),
            "replay_memory_bytes": self.memory.nbytes(),
            "drift_reference_bytes": self.detector.nbytes(),
            "encoder_state_bytes": deep_nbytes(self.encoder),
            "history_bytes": deep_nbytes(self.history),
        }
        parts["total_bytes"] = int(sum(parts.values()))
        parts["replay_rows"] = self.memory.size
        parts["reference_rows"] = 0 if self.detector.ref_X is None else len(self.detector.ref_X)
        parts["n_stages"] = len(self.state.stages)
        return parts
