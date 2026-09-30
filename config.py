"""
Deterministic configuration for ADAPT-X.

All tunable behaviour lives in one dataclass, which can be loaded from
``config.yaml``. The project needs no environment variables or secrets.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ADAPTXConfig:
    # ---------------- bounded memory budgets ----------------
    replay_capacity: int = 500            # K: max labelled rows kept in replay memory
    reference_capacity: int = 512         # R: max rows kept by the shift detector's reference sample
    max_stages: int = 8                   # max persistent learned functions in the model state
    max_categories: int = 64              # frozen vocabulary size per categorical feature
    n_hash_buckets: int = 16              # buckets for categories never seen in batch 0
    history_length: int = 50              # bounded diagnostic history (reports/plans kept)

    # ---------------- base learner (one model component) ----------------
    base_max_iter: int = 300
    base_learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    stage_max_leaf_nodes: int = 15
    internal_holdout: float = 0.2         # share of each batch held out from tree fitting
    combiner_rows: str = "all_new"        # rows of the new batch used by the combiner: heldout | all_new (tuned)

    # ---------------- shift detection ----------------
    alpha: float = 0.01                   # FDR level across per-feature shift tests
    alpha_model: float = 0.05             # level of the single concept / label-bias test per batch
    psi_min_effect: float = 0.10          # practical-significance gate for a marginal shift
    corr_min_effect: float = 0.20         # practical-significance gate for a correlation shift
    loss_min_effect: float = 0.15         # >= 15% worse loss needed to call concept drift
    min_support_rows: int = 30            # rows needed for the in-support concept test

    # ---------------- adaptation controller (tuned on dev families) ----------------
    stability_max: float = 0.25           # distillation strength when nothing changed (tuned)
    stability_min: float = 0.05
    replay_weight_min: float = 1.0        # replay weight under maximal concept drift (tuned; 1.0 = never down-weight)
    stage_trees_min: int = 30
    stage_trees_max: int = 300            # (tuned)
    stage_learning_rate: float = 0.1
    min_new_share: float = 0.2            # min share of replay memory given to newest batch
    max_new_share: float = 0.8

    # ---------------- ablation switches ----------------
    use_drift_detection: bool = True      # A: False -> fixed plan, no detection
    adaptive_scaling: bool = True         # C: False -> binary trigger, fixed plan
    persistent_state: bool = True         # D: False -> discard state, refit on batch + replay
    replay_strategy: str = "stratified_diversity"  # E: "reservoir" -> uniform random replay
    use_target_encoding: bool = False     # optional leak-free streaming target encoding

    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def replace(self, **kwargs: Any) -> "ADAPTXConfig":
        d = self.to_dict()
        d.update(kwargs)
        return ADAPTXConfig(**d)


def load_config(path: Optional[str] = None) -> ADAPTXConfig:
    """Load ``adaptx`` section of a YAML file into an :class:`ADAPTXConfig`."""
    if path is None:
        return ADAPTXConfig()
    import yaml

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    section = raw.get("adaptx", raw)
    flat: Dict[str, Any] = {}
    for value in section.values() if all(isinstance(v, dict) for v in section.values()) else [section]:
        flat.update(value)
    known = {f.name for f in fields(ADAPTXConfig)}
    unknown = set(flat) - known
    if unknown:
        raise ValueError(f"Unknown ADAPT-X config keys: {sorted(unknown)}")
    return ADAPTXConfig(**flat)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
