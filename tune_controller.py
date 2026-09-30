"""
Controller tuning on DEVELOPMENT data only.

* families: DEV_FAMILIES (nonlinear regression + nonlinear classification)
* seeds:    1001, 1002 (never used by the benchmark, which uses 0..4)
* batches:  B0..B4 only; the hidden compound batch B5 is never used here

Objective = mean of prequential, post-update and final-average skill.
The winning values are copied into config.yaml; results/controller_tuning.csv
records every candidate so the choice is auditable.

    python -m experiments.tune_controller
"""

from __future__ import annotations

import itertools
import os
import time

import numpy as np
import pandas as pd

from experiments.distribution_shifts import DEV_FAMILIES, FAMILIES, create_benchmark_stream
from src.config import ADAPTXConfig
from src.continual_learner import ContinualLearner
from src.sequential_evaluation import SequentialEvaluator, split_stream

DEV_SEEDS = [1001, 1002]
GRID = {
    "stability_max": [0.05, 0.25, 1.0, 4.0],
    "replay_weight_min": [0.1, 0.5, 1.0],
    "stage_trees_max": [100, 300],
    "combiner_rows": ["heldout", "all_new"],
}


def main(out_dir: str = "results") -> pd.DataFrame:
    ev = SequentialEvaluator()
    streams = []
    for fam in DEV_FAMILIES:
        for seed in DEV_SEEDS:
            batches, spec = create_benchmark_stream(FAMILIES[fam], 1000, seed)
            batches = batches[:5]
            streams.append((fam, seed, spec, [b[0] for b in batches], split_stream(batches, seed)))
    rows = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        t0 = time.time()
        scores = []
        for fam, seed, spec, names, splits in streams:
            cfg = ADAPTXConfig(random_seed=seed, **params)
            s = ev.run(ContinualLearner(cfg), splits, spec, names, "adaptx", fam, seed).summary()
            scores.append({"dataset": fam, "seed": seed, **params,
                           "objective": np.mean([s["prequential_skill"], s["post_update_skill"], s["final_avg_skill"]]),
                           **{k: s[k] for k in ("prequential_skill", "post_update_skill", "final_avg_skill", "forgetting")}})
        rows.extend(scores)
        print(params, "objective=%.4f" % np.mean([r["objective"] for r in scores]), "(%.1fs)" % (time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "controller_tuning.csv"), index=False)
    agg = df.groupby(keys)[["objective", "prequential_skill", "post_update_skill", "final_avg_skill"]].mean()
    agg = agg.sort_values("objective", ascending=False)
    print(agg.head(10).round(4))
    return agg


if __name__ == "__main__":
    main()
