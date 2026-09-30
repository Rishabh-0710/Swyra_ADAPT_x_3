"""
THE authoritative benchmark: baselines, ADAPT-X and all ablations in one run.

Every model sees the same data. For each (family, seed) the stream and its
update/holdout splits are generated once and reused for every model and every
ablation, with the same metrics and the same evaluator.

    python -m experiments.run_benchmark                    # full run (config.yaml)
    python -m experiments.run_benchmark --quick            # 2 seeds, smaller batches

Outputs (results/):
    stream_results.csv      one row per (family, seed, model)
    benchmark_summary.csv   baselines vs ADAPT-X
    ablation_summary.csv    ablations A-E vs full ADAPT-X (F)
    per_family_summary.csv  per-family means
    paired_tests.csv        ADAPT-X vs each alternative: wins and Wilcoxon p-values
    drift_reports.csv       every ShiftReport produced by ADAPT-X
    RESULTS.md              the tables above, as markdown (the README quotes it)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from experiments.baselines import FullRetrainOracle, NaiveIncrementalModel, StandardReplayModel, StaticModel
from experiments.distribution_shifts import FAMILIES, create_benchmark_stream
from src.config import ADAPTXConfig, load_config
from src.continual_learner import ContinualLearner
from src.sequential_evaluation import SequentialEvaluator, split_stream

ADAPTX = "ADAPT-X (proposed)"
BASELINES = ["Static", "Naive incremental", "Standard replay (FIFO)", "Full retrain oracle*"]
ABLATIONS = {
    "A: no drift detection": dict(use_drift_detection=False),
    "B: no replay (K=0)": dict(replay_capacity=0),
    "C: no adaptive scaling": dict(adaptive_scaling=False),
    "D: no stable persistent state": dict(persistent_state=False),
    "E: random replay": dict(replay_strategy="reservoir"),
}
METRICS = ["post_update_skill", "prequential_skill", "final_avg_skill", "forgetting", "worst_prequential_skill",
           "hidden_prequential_skill", "hidden_post_update_skill", "mean_update_latency_sec", "peak_state_kb"]


def model_factories(cfg: ADAPTXConfig) -> Dict[str, Callable[[], object]]:
    f: Dict[str, Callable[[], object]] = {
        "Static": lambda: StaticModel(cfg),
        "Naive incremental": lambda: NaiveIncrementalModel(cfg),
        "Standard replay (FIFO)": lambda: StandardReplayModel(cfg),
        "Full retrain oracle*": lambda: FullRetrainOracle(cfg),
        ADAPTX: lambda: ContinualLearner(cfg),
    }
    for name, flags in ABLATIONS.items():
        f[name] = (lambda fl: (lambda: ContinualLearner(cfg.replace(**fl))))(flags)
    return f


def _md(df: pd.DataFrame, floatfmt: int = 3) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        vals = [f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row.values]
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(lines)


def main(config_path: str = "config.yaml", quick: bool = False, out_dir: str = "results") -> None:
    raw = yaml.safe_load(open(config_path)) if os.path.exists(config_path) else {}
    bench = raw.get("benchmark", {})
    seeds: List[int] = bench.get("seeds", [0, 1, 2, 3, 4])
    batch_size: int = bench.get("batch_size", 1000)
    families: List[str] = bench.get("families", list(FAMILIES))
    if quick:
        seeds, batch_size = seeds[:2], 600
    base_cfg = load_config(config_path) if os.path.exists(config_path) else ADAPTXConfig()
    os.makedirs(out_dir, exist_ok=True)
    ev = SequentialEvaluator(bench.get("holdout_fraction", 0.2))

    rows, drift_rows = [], []
    t_start = time.time()
    for fam in families:
        for seed in seeds:
            batches, spec = create_benchmark_stream(FAMILIES[fam], batch_size, seed)
            splits, names = split_stream(batches, seed, ev.holdout_fraction), [b[0] for b in batches]
            cfg = base_cfg.replace(random_seed=seed)
            for name, make in model_factories(cfg).items():
                res = ev.run(make(), splits, spec, names, name, fam, seed)
                s = res.summary()
                s["R_matrix"] = json.dumps(np.round(res.R, 4).tolist())
                s["prequential_by_batch"] = json.dumps(np.round(res.prequential, 4).tolist())
                rows.append(s)
                if name == ADAPTX:
                    for rep in res.extra["reports"]:
                        drift_rows.append({"dataset": fam, "seed": seed, "env": names[rep["batch_index"]],
                                           **{k: v for k, v in rep.items() if not isinstance(v, list)},
                                           "affected_features": ";".join(rep["affected_features"])})
            print(f"[{time.time() - t_start:7.1f}s] {fam} seed={seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "stream_results.csv"), index=False)
    pd.DataFrame(drift_rows).to_csv(os.path.join(out_dir, "drift_reports.csv"), index=False)

    agg = df.groupby("model")[METRICS].mean()
    bench_tbl = agg.loc[BASELINES + [ADAPTX]]
    abl_tbl = agg.loc[list(ABLATIONS) + [ADAPTX]].rename(index={ADAPTX: "F: full ADAPT-X"})
    bench_tbl.to_csv(os.path.join(out_dir, "benchmark_summary.csv"))
    abl_tbl.to_csv(os.path.join(out_dir, "ablation_summary.csv"))
    fam_tbl = df.pivot_table(index=["dataset", "model"], values=["post_update_skill", "prequential_skill", "final_avg_skill", "forgetting"])
    fam_tbl.to_csv(os.path.join(out_dir, "per_family_summary.csv"))

    # paired comparison over identical streams
    paired = []
    key = ["dataset", "seed"]
    a = df[df.model == ADAPTX].set_index(key)
    for other in BASELINES + list(ABLATIONS):
        b = df[df.model == other].set_index(key).loc[a.index]
        for m in ["post_update_skill", "prequential_skill", "final_avg_skill", "forgetting"]:
            diff = (a[m] - b[m]).to_numpy()
            better = diff < 0 if m == "forgetting" else diff > 0
            try:
                p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                p = 1.0
            paired.append({"vs": other, "metric": m, "mean_diff_adaptx_minus_other": float(diff.mean()),
                           "adaptx_better_streams": int(better.sum()), "n_streams": len(diff), "wilcoxon_p": p})
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(os.path.join(out_dir, "paired_tests.csv"), index=False)

    # post-update skill per family (baselines + ADAPT-X)
    fam_post = df[df.model.isin(BASELINES + [ADAPTX])].pivot_table(index="dataset", columns="model", values="post_update_skill")[BASELINES + [ADAPTX]]
    fam_final = df[df.model.isin(BASELINES + [ADAPTX])].pivot_table(index="dataset", columns="model", values="final_avg_skill")[BASELINES + [ADAPTX]]

    with open(os.path.join(out_dir, "RESULTS.md"), "w") as fh:
        fh.write(f"# Benchmark results\n\nGenerated by `python -m experiments.run_benchmark` "
                 f"({len(families)} families x {len(seeds)} seeds, batch size {batch_size}, "
                 f"{len(df)} stream runs, {time.time() - t_start:.0f}s).\n\n")
        fh.write("Skill = R^2 (regression) or 2*AUC-1 (classification). * = oracle keeps all history (violates the bounded-memory constraint).\n\n")
        fh.write("## Baselines vs ADAPT-X (mean over all streams)\n\n" + _md(bench_tbl.round(4)) + "\n\n")
        fh.write("## Ablations (same streams, same splits)\n\n" + _md(abl_tbl.round(4)) + "\n\n")
        fh.write("## Post-update skill per family\n\n" + _md(fam_post.round(3)) + "\n\n")
        fh.write("## Final average skill (retention over all batches) per family\n\n" + _md(fam_final.round(3)) + "\n\n")
        fh.write("## Paired comparison: ADAPT-X vs alternative\n\n" + _md(paired_df.set_index("vs").round(4), 4) + "\n")
    print(open(os.path.join(out_dir, "RESULTS.md")).read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    main(args.config, args.quick, args.out)
