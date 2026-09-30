"""
Long-stream runtime / memory experiment (SDG 9 evidence).

A 30-batch stream with recurring covariate shifts and periodic concept shifts.
Records, per batch, the update latency and the full state size of ADAPT-X,
standard replay and the full-retraining oracle.

    python -m experiments.scaling
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from experiments.baselines import FullRetrainOracle, StandardReplayModel
from experiments.distribution_shifts import FAMILIES, apply_mean_shift
from src.config import load_config
from src.continual_learner import ContinualLearner


def long_stream(n_batches: int = 30, batch_size: int = 1000, seed: int = 0):
    fam = FAMILIES["heteroscedastic_regression"]
    df, y, spec = fam(n_samples=n_batches * batch_size, seed=seed)
    out = []
    for t in range(n_batches):
        X = df.iloc[t * batch_size:(t + 1) * batch_size].reset_index(drop=True)
        X = apply_mean_shift(X, spec.numerical_features[:2], 0.8 * np.sin(t / 3))
        yt = pd.Series(fam.relabel(X, seed, t))
        if (t // 10) % 2 == 1:                     # concept regime switches every 10 batches
            yt = yt + 1.5 * np.sign(X[spec.numerical_features[3]].to_numpy())
        out.append((X, yt))
    return out, spec


def main(out_dir: str = "results", n_batches: int = 30) -> pd.DataFrame:
    cfg = load_config("config.yaml") if os.path.exists("config.yaml") else None
    stream, spec = long_stream(n_batches)
    rows = []
    for name, model in [("ADAPT-X", ContinualLearner(cfg)), ("Standard replay (FIFO)", StandardReplayModel(cfg)),
                        ("Full retrain oracle*", FullRetrainOracle(cfg))]:
        for t, (X, y) in enumerate(stream):
            t0 = time.perf_counter()
            model.initialize(X, y, spec) if t == 0 else model.update(X, y)
            lat = time.perf_counter() - t0
            fp = model.memory_footprint()
            rows.append({"model": name, "batch": t, "update_latency_sec": lat, "state_kb": fp["total_bytes"] / 1024,
                         "replay_rows": fp.get("replay_rows"), "reference_rows": fp.get("reference_rows"),
                         "n_stages": fp.get("n_stages")})
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "scaling_long_stream.csv"), index=False)
    view = df[df.batch.isin([1, 5, 10, 20, n_batches - 1])].pivot_table(index="batch", columns="model",
                                                                          values=["update_latency_sec", "state_kb"])
    print(view.round(3))
    return df


def stationary_efficiency(out_dir: str = "results", n_batches: int = 15) -> pd.DataFrame:
    """Stream with NO shift: full ADAPT-X (MAINTAIN = small stage, max stability)
    vs ablation A (no detection: fixed mid-size plan every batch)."""
    from experiments.distribution_shifts import create_stationary_stream
    from src.metrics import evaluate_predictions

    base = load_config("config.yaml") if os.path.exists("config.yaml") else None
    rows = []
    for fam in ("linear_regression", "nonlinear_classification", "mixed_classification"):
        for seed in (0, 1):
            batches, spec = create_stationary_stream(FAMILIES[fam], n_batches + 1, 1000, seed)
            X_te, y_te = batches[-1][1], batches[-1][2]
            for name, cfg in (("ADAPT-X", base.replace(random_seed=seed)),
                              ("A: no drift detection", base.replace(random_seed=seed, use_drift_detection=False))):
                m = ContinualLearner(cfg)
                lat = []
                for t, (_, X, y) in enumerate(batches[:-1]):
                    t0 = time.perf_counter()
                    m.initialize(X, y, spec) if t == 0 else m.update(X, y)
                    if t:
                        lat.append(time.perf_counter() - t0)
                if spec.task_type == "regression":
                    skill = evaluate_predictions(spec.task_type, y_te, m.predict(X_te))["skill"]
                else:
                    skill = evaluate_predictions(spec.task_type, y_te, None, m.predict_proba(X_te)[:, 1])["skill"]
                modes = [h["plan"]["mode"] for h in m.history]
                rows.append({"dataset": fam, "seed": seed, "model": name, "test_skill": skill,
                             "mean_update_latency_sec": float(np.mean(lat)), "total_trees": m.state.total_trees(),
                             "state_kb": m.memory_footprint()["total_bytes"] / 1024,
                             "maintain_updates": sum(mo == "MAINTAIN" for mo in modes)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "stationary_efficiency.csv"), index=False)
    print(df.groupby("model")[["test_skill", "mean_update_latency_sec", "total_trees", "state_kb", "maintain_updates"]].mean().round(4))
    return df


if __name__ == "__main__":
    main()
    stationary_efficiency()
