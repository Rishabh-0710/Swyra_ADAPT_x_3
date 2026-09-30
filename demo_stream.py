"""
One-minute demo: run ADAPT-X on one stream and print what it detected and
how it adapted at every batch.

    python -m experiments.demo_stream --family mixed_classification --seed 0
"""

from __future__ import annotations

import argparse
import warnings

from experiments.distribution_shifts import FAMILIES, create_benchmark_stream
from src.config import load_config
from src.continual_learner import ContinualLearner

warnings.filterwarnings("ignore")


def main(family: str, seed: int, config: str) -> None:
    batches, spec = create_benchmark_stream(FAMILIES[family], 1000, seed)
    learner = ContinualLearner(load_config(config).replace(random_seed=seed))
    learner.initialize(batches[0][1], batches[0][2], spec)                 # LEARN
    print(f"{batches[0][0]}: learned {spec.task_type} task, {len(learner.state.stages)} stage(s)")
    for env, X, y in batches[1:]:
        report = learner.update(X, y)                                      # DETECT -> ADAPT -> RETAIN
        h = learner.history[-1]
        p = h["plan"]
        print(f"\n{env}\n  DETECT  {report.summary()}")
        print(f"  ADAPT   {p['mode']}: stability={p['stability']:.2f} replay_weight={p['replay_weight']:.2f} "
              f"tree_budget={p['stage_trees']} new_stage={p['grow_stage']}  ({p['rationale']})")
        print(f"  RETAIN  stages={h['n_stages']} weights={h['weights']} replay_rows={learner.memory.size} "
              f"latency={h['latency_sec']:.2f}s")
    fp = learner.memory_footprint()
    print(f"\nbounded state: {fp['total_bytes'] / 1024:.0f} KB  (replay {fp['replay_rows']} rows, "
          f"reference {fp['reference_rows']} rows, {fp['n_stages']} stages)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mixed_classification", choices=list(FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    main(a.family, a.seed, a.config)
