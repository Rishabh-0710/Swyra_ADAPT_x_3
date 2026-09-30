"""
Adaptation controller: turns a ShiftReport into an UpdatePlan.

The controller decides how much the model is allowed to change on each batch.
It works from two drift quantities:

* ``c`` (capacity demand) = overall drift score. New regions of input space or
  a new mapping both need new model capacity, so this sets the tree budget of
  the new correction stage and how fast the detector's reference moves.
* ``q`` (plasticity demand) = concept/label drift score x in-support fraction.
  Only a changed P(y|x) *where the model already had knowledge* justifies
  changing predictions on old inputs, so this lowers the stability
  (distillation) strength, down-weights stale replay labels, and gives the
  newest batch more room in replay memory.

Resulting behaviour:
  no drift            -> MAINTAIN: small refinement stage, strong stability
  covariate drift     -> COVARIATE_ADAPT: more trees for new regions, stability kept
  concept drift       -> CONCEPT_ADAPT: plasticity up, stale replay down-weighted
  label/prior drift   -> PRIOR_CORRECT: intercept/bias re-fit dominates
  compound drift      -> COMPOUND_ADAPT: capacity and plasticity both high

The four free constants (``stability_max``, ``replay_weight_min``,
``stage_trees_max`` and ``stage_learning_rate``) were chosen by
``experiments/tune_controller.py`` on development families and seeds only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.config import ADAPTXConfig
from src.shift_detection import ShiftReport


@dataclass
class UpdatePlan:
    mode: str
    capacity_demand: float
    plasticity_demand: float
    stability: float           # weight of functional distillation toward the pre-update model
    replay_weight: float       # sample weight of replayed rows relative to new rows
    stage_trees: int           # tree budget of the new correction stage (early stopping may use fewer)
    stage_learning_rate: float
    grow_stage: bool
    memory_new_share: float    # share of replay capacity given to the newest batch
    reference_refresh: float   # share of the drift reference replaced by the newest batch
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


class AdaptationController:
    def __init__(self, config: ADAPTXConfig) -> None:
        self.cfg = config

    def _balanced_share(self, n_batches_seen: int) -> float:
        # 1/(t+1) keeps the replay memory ~uniform over all batches seen so far
        return float(np.clip(1.0 / (n_batches_seen + 1), self.cfg.min_new_share, self.cfg.max_new_share))

    def _make(self, mode: str, c: float, q: float, n_batches_seen: int, grow: bool, why: str) -> UpdatePlan:
        cfg = self.cfg
        stability = cfg.stability_min + (cfg.stability_max - cfg.stability_min) * (1 - q) ** 2
        replay_w = cfg.replay_weight_min + (1 - cfg.replay_weight_min) * (1 - q)
        trees = int(round(cfg.stage_trees_min + (cfg.stage_trees_max - cfg.stage_trees_min) * c))
        share = float(np.clip(max(self._balanced_share(n_batches_seen), q), cfg.min_new_share, cfg.max_new_share))
        return UpdatePlan(
            mode=mode, capacity_demand=float(c), plasticity_demand=float(q), stability=float(stability),
            replay_weight=float(replay_w), stage_trees=trees, stage_learning_rate=cfg.stage_learning_rate,
            grow_stage=grow, memory_new_share=share, reference_refresh=float(max(0.25, c)), rationale=why,
        )

    def plan(self, report: ShiftReport, n_batches_seen: int) -> UpdatePlan:
        cfg = self.cfg
        if not cfg.use_drift_detection:
            return self._make("FIXED", 0.5, 0.5, n_batches_seen, True, "ablation: drift detection disabled, fixed plan")
        if not cfg.adaptive_scaling:
            if report.drift_type == "none":
                return self._make("SKIP", 0.0, 0.0, n_batches_seen, False, "ablation: binary trigger, no drift -> no model change")
            return self._make("TRIGGERED", 0.5, 0.5, n_batches_seen, True, "ablation: binary trigger, drift -> fixed plan")

        c = report.overall_drift_score
        t = report.drift_type
        # Plasticity is only justified where the old model had knowledge. The
        # combiner's parameters (intercept, stage weights) act on ALL inputs, so
        # concept evidence is scaled by the share of the batch that lies inside
        # the reference support. A shift into unseen regions is absorbed by the
        # new (local) correction stage instead of by rewriting the old function.
        # A label/prior shift is a global bias by definition, so it is not scaled.
        q = report.concept_drift_score
        if t != "prior":
            q *= report.in_support_fraction
        if t == "none":
            # Nothing changed: only a small stage (stage_trees_min, early-stopped)
            # plus the combiner re-fit, under maximal stability. Skipping growth
            # entirely was tested and rejected: on stationary streams the model
            # then stops improving from new data (results/archive_v3_*).
            return self._make("MAINTAIN", 0.0, 0.0, n_batches_seen, True,
                              "no significant shift: small refinement stage, strong stability")
        if t == "covariate":
            return self._make("COVARIATE_ADAPT", c, 0.0, n_batches_seen, True,
                              "inputs moved but mapping holds in-support: add capacity, keep old function")
        if t == "prior":
            return self._make("PRIOR_CORRECT", c, q, n_batches_seen, True,
                              "systematic residual bias: re-fit intercept/bias with moderate plasticity")
        if t == "compound":
            return self._make("COMPOUND_ADAPT", c, q, n_batches_seen, True,
                              "inputs and mapping both changed: high capacity and plasticity")
        return self._make("CONCEPT_ADAPT", c, q, n_batches_seen, True,
                          f"{t} drift of P(y|x) in-support: raise plasticity, give the new regime more replay room")
