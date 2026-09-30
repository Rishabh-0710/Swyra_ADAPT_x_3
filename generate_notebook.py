"""
Generates notebooks/analysis.ipynb (a walkthrough of one ADAPT-X stream plus
the benchmark tables). Plain JSON, no nbformat dependency:

    python notebooks/generate_notebook.py
"""

import json
import os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip().splitlines(keepends=True)})


md("""
# ADAPT-X walkthrough
LEARN -> DETECT -> ADAPT -> RETAIN on one stream, then the authoritative benchmark tables.
""")
code("""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from src import ContinualLearner, load_config
from experiments.distribution_shifts import FAMILIES, create_benchmark_stream
cfg = load_config("../config.yaml")
batches, spec = create_benchmark_stream(FAMILIES["mixed_classification"], batch_size=1000, seed=0)
""")
md("## 1. Learn batch 0, then process each batch sequentially")
code("""
learner = ContinualLearner(cfg).initialize(batches[0][1], batches[0][2], spec)
for env, X, y in batches[1:]:
    report = learner.update(X, y)          # only the current batch is ever passed
    plan = learner.history[-1]["plan"]
    print(f"{env:22s} {report.summary()}")
    print(f"{'':22s} -> {plan['mode']}: stability={plan['stability']:.2f} replay_w={plan['replay_weight']:.2f} "
          f"trees<={plan['stage_trees']} | stages={learner.history[-1]['n_stages']}")
""")
md("## 2. Bounded memory audit")
code("""
pd.Series(learner.memory_footprint())
""")
md("## 3. Benchmark and ablation results (from `python -m experiments.run_benchmark`)")
code("""
print(open("../results/RESULTS.md").read())
""")
code("""
import matplotlib.pyplot as plt
df = pd.read_csv("../results/scaling_long_stream.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for m, g in df.groupby("model"):
    ax[0].plot(g.batch, g.state_kb, label=m); ax[1].plot(g.batch, g.update_latency_sec, label=m)
ax[0].set_title("state size (KB)"); ax[1].set_title("update latency (s)"); ax[0].legend(); plt.show()
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis.ipynb")
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path)
