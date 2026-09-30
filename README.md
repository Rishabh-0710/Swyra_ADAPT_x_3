# ADAPT-X: Continual Learning under Dynamic Distribution Shift

ADAPT-X learns **one fixed supervised tabular task** (regression or binary
classification) from an initial dataset. It then adapts to a stream of labelled
batches whose distribution keeps changing, under a hard memory budget and
without ever retraining on the full history.

```
LEARN  ->  DETECT  ->  ADAPT  ->  RETAIN  ->  LEARN AGAIN
```

The algorithm is domain-agnostic. It only uses numeric and categorical
distributions, rank correlations, residuals, per-row losses and model
behaviour. Column names and meanings are never used; a test renames every
column and checks that predictions are unchanged.

---

## 1. Architecture

```
                 INITIAL DATA (batch 0)
                          |
                          v
     Initial learning: freeze feature schema, fit base stage h0,
     calibrate drift reference on held-out rows, seed replay memory
                          |
                          v
      PERSISTENT MODEL STATE   F(x) = b + sum_k beta_k * h_k(x)
                          |
            +-------------+-------------+
            |      Incoming batch t     |   (only batch t is ever passed in)
            +-------------+-------------+
                          |
          encode with the encoder state from BEFORE batch t
                          |
                          v
               PRE-UPDATE PREDICTION  F_{t-1}(x_t)
                          |
                          v
         DISTRIBUTION SHIFT DETECTOR  (bounded reference, R rows)
       +------------------+----------------------+
       |                                         |
  Feature drift                             Residual drift
  KS / chi2 + BH-FDR, PSI,               pre-update loss vs held-out
  Wasserstein, Spearman-corr             reference loss (in-support rows),
  Fisher-z test                          label-bias test, Page-Hinkley
       +------------------+----------------------+
                          |  ShiftReport (type, severity, score,
                          v   affected features, confidence)
               ADAPTATION CONTROLLER -> UpdatePlan
       +------------------+----------------------+
       |                                         |
  Stable memory                             New evidence
  replay rows (<= K) +                      one new correction stage h_new
  distillation toward F_{t-1}               fitted to pseudo-residuals of F_{t-1}
       +------------------+----------------------+
                          |
                          v
      CONTROLLED UPDATE: re-fit (b, beta) with
      loss(new evidence) + stability * ||F - F_{t-1}||^2 on replay inputs;
      evict the least useful stage if more than max_stages
                          |
                          v
               UPDATED PERSISTENT MODEL STATE
                          |
                          v
      BOUNDED REPLAY MEMORY update (<= K rows) -> refresh drift
      reference (<= R rows) -> encoder absorbs (X_t, y_t) -> next batch
```

| Component | File | What persists / what changes |
|---|---|---|
| `ContinualLearner` | `src/continual_learner.py` | Runs the ordered LEARN→DETECT→ADAPT→RETAIN step. |
| `AdaptiveModelState` | `src/model_state.py` | **Stages persist and are frozen once fitted.** Each update adds at most one small correction stage (continued functional gradient boosting across batches). Only the combiner weights `(b, beta)` are re-estimated each batch. The number of stages is capped at `max_stages`: dead stages are dropped, and the least useful one is evicted when over budget. |
| `DistributionShiftDetector` | `src/shift_detection.py` | A bounded reference sample (≤ R rows), ≤ R held-out losses and residuals, PSI bins, and one CUSUM float. The reference moves toward the newest batch at the rate the controller chooses. |
| `AdaptationController` | `src/adaptation_controller.py` | Stateless. Maps a report to an `UpdatePlan`: stability, replay weight, tree budget, memory share and reference refresh rate. |
| `BoundedReplayMemory` | `src/replay_memory.py` | ≤ K labelled rows in the frozen feature space, selected by target stratification plus farthest-point and random sampling. Prior-correction weights handle imbalanced classes. |
| `StreamingFeatureEncoder` | `src/preprocessing.py` | Frozen schema. Bounded categorical vocabulary plus CRC32 hash buckets. Optional leak-free target encoding. |
| `SequentialEvaluator` | `src/sequential_evaluation.py` | The single test-then-train protocol used by every experiment. |

### Why this counts as incremental learning

On an update, the batch-0 learner and every earlier correction stage stay in
memory **unchanged**. `test_sequential_update_is_incremental_not_retraining`
checks that the base stage is the same object, with identical output, after
three updates. New knowledge enters in two ways:

- **One correction stage**, a small GBDT fitted to the *pseudo-residuals of the
  current model* (the Newton step for log-loss) on the new batch plus half of
  the replay memory.
- **A re-fit of the combiner** `(b, beta)`, which solves

  `sum_i w_i loss(y_i, F(x_i))  +  stability * sum_{a in replay} (F(x_a) - F_old(x_a))^2`

  This is a closed-form solve for regression and a few Newton steps for
  classification.

The second term is *functional distillation* toward the pre-update model on
remembered inputs. It is the explicit anti-forgetting mechanism, and the
controller sets its weight. Retraining from scratch on the new batch plus
replay exists only as **ablation D**, where it is measurably worse
(section 5).

GBDT is one component here: each stage is a scikit-learn
`HistGradientBoosting*` model. The continual-learning logic (stages, combiner,
distillation, replay, detection, control) sits around it.

### How the adaptation controller works (`src/adaptation_controller.py`)

The controller uses two drift quantities from the report:

- **`c` = overall drift score.** New regions of input space or a new mapping
  both need capacity, so `c` sets the tree budget of the new stage
  (`30 → 300`, early-stopped) and the reference refresh rate.
- **`q` = concept / label score × in-support fraction.** Only a change in
  P(y|x) *where the model already had knowledge* justifies changing
  predictions on old inputs. `q` lowers the distillation strength
  (`0.25 → 0.05`) and gives the newest batch more room in replay.
  Concept evidence from rows outside the reference support goes to the new
  (local) stage, not into rewriting the global function. Label or prior
  shifts are global by definition, so `q` is not scaled for them.

| Detected | Mode | Behaviour |
|---|---|---|
| none | MAINTAIN | Small refinement stage, maximal stability |
| covariate | COVARIATE_ADAPT | More trees for new regions; old function protected |
| prior / label | PRIOR_CORRECT | Intercept/bias re-fit dominates |
| concept (in-support) | CONCEPT_ADAPT | Stability down, more replay room for the new regime |
| compound | COMPOUND_ADAPT | Capacity and plasticity both high |

The controller's constants were **tuned only on development data**:
`python -m experiments.tune_controller`, using 2 families, dev seeds 1001 and
1002, and batches B0–B4 only. The benchmark uses seeds 0–4. Every candidate is
logged in `results/controller_tuning.csv`. The tuning chose
`replay_weight_min = 1.0`, meaning stale replay is **not** down-weighted, which
is the opposite of our prior. We kept that value rather than overriding it.

### Shift detector: statistics and thresholds (`src/shift_detection.py`)

- **Marginal shift.** KS (numeric) or chi-square (categorical codes and
  missingness) per feature, with Benjamini-Hochberg FDR at α = 1%. A feature
  is flagged only if PSI ≥ 0.10 as well (the standard "moderate shift" level),
  so huge batches can't flag trivial differences.
- **Joint shift.** Fisher-z test on every pair of Spearman correlations,
  Bonferroni-corrected, with |Δρ| ≥ 0.2. This catches rotations that leave
  every marginal unchanged (tested).
- **Concept shift.** The **pre-update** per-row loss on rows inside the
  reference support is compared with held-out reference losses using a
  one-sided Mann-Whitney U test at 5%, plus a loss ratio ≥ 1.15. Restricting to
  in-support rows separates "the mapping changed" from "we are extrapolating".
- **Label/prior shift.** A residual-bias Welch t-test (regression) or a
  calibration-in-the-large z-test (classification).
- **Gradual drift.** A Page-Hinkley statistic on the log loss ratio across
  batches.
- **Report contents.** `overall_drift_score`, `severity`, `drift_type`
  (covariate / concept / prior / compound / gradual), `covariate_drift_score`,
  `concept_drift_score`, `affected_features` (ranked by PSI, with p/q-values,
  PSI and Wasserstein), `confidence`, and a one-line `summary()`.
- **False-alarm rate on stationary streams:** at most 15% medium or high
  severity, checked in the tests.

---

## 2. Leakage fixes (exact)

| Problem in attempt 1 | Fix | Test |
|---|---|---|
| `update()` called `partial_fit(X_t, y_t)` and then `transform(X_t)` again, so target-encoded features of batch *t* contained y_t. With a unique-ID categorical and pure-noise y, corr(feature, own label) was **1.00**. | Two-phase encoder. `transform` reads only state from earlier batches, and `observe(X_t, y_t)` runs **after** the model update and replay/reference refresh. Batch 0 uses 5-fold out-of-fold encoding. By default, categoricals use native GBDT categorical splits on frozen codes, and target encoding is optional (`use_target_encoding`). | `test_no_target_encoding_leakage_features_independent_of_own_labels` permutes y_t and requires identical training features. Re-inserting the old order makes this test fail (checked). Also `test_target_encoding_not_correlated_with_own_label_on_unique_ids` and `test_out_of_fold_batch0_row_encoding_ignores_own_label`. |
| Replay stored features computed after the leak, in a scaling that drifted over time. | Replay stores rows in the frozen, unscaled schema (trees need no scaling), so stored rows keep their meaning. | `test_no_historical_raw_data_retention` |
| The evaluator shared one RNG across models, so each model was scored on different rows. | `split_stream` derives each split from `(seed, t)` only, so every model and ablation gets the same rows. | `test_no_future_batch_access` (the learner only ever receives batch t's update part: never holdout rows, never future rows) |

## 3. Memory fixes (exact)

| Unbounded in attempt 1 | Now |
|---|---|
| `DriftDetector.reference_X = X.copy()` (all of batch 0) | ≤ `reference_capacity` (512) sampled rows plus ≤ 512 held-out losses, refreshed by weighted resampling |
| `cat_counts` / `cat_target_sums` grew with every new category | Frozen top-64 vocabulary + 16 CRC32 buckets per feature. All per-category arrays have a fixed size. |
| Growing `drift_history`, `adaptation_history`, `update_times` lists | `deque(maxlen=50)` |
| Model memory invisible (LightGBM C++ trees) | `memory_footprint()` counts tree node arrays, replay, reference, encoder and history. `max_stages` caps model size. |
| Missing indicators added only when NaNs appeared (the schema changed, then `update()` crashed) | Every numeric feature always has its indicator column |

Tests: `test_bounded_reference_memory_independent_of_initial_size` (600 vs
20 000 initial rows) and `test_memory_does_not_scale_with_stream_length`
(25 batches; the full-retraining oracle is the growing contrast).
`test_bounded_categorical_state_under_high_cardinality_stream` streams 7 000
new categories with byte-identical state.
`test_no_historical_raw_data_retention` walks the whole object graph and finds
no array larger than the declared budgets and no view onto any caller's batch.

---

## 4. Why the attempt-1 numbers disagreed (0.6909 vs 0.7796)

Neither number was trustworthy:

1. **Different datasets.** The benchmark used 5 families (3 regression); the
   ablation used 3 families (2 classification).
2. **The composite score is not comparable across task types.** Regression was
   scored as 1/(1+RMSE) (about 0.3), classification by AUC (about 0.85).
   Restricted to the ablation's datasets and seeds, the benchmark's own
   proposed-model score is **0.7763**, which accounts for almost the whole gap.
3. **Different seeds.** The benchmark used {42, 101, 777}; the ablation used
   {42, 101}.
4. **Different splits per model**, from the shared evaluator RNG.
5. **A stale README table** that didn't match the CSVs.

There is now one pipeline, `experiments/run_benchmark.py`. It runs the
baselines, ADAPT-X and all ablations on the same 40 streams (8 families × 5
seeds), the same splits and the same scale-free metric:
**skill = R² for regression, 2·AUC−1 for classification**. It writes every
table below to `results/RESULTS.md`.

---

## 5. Results (`python -m experiments.run_benchmark`)

Each stream has six batches:

- B0 baseline
- B1 covariate shift (labels re-drawn from the same P(y|x))
- B2 prior/label shift
- B3 concept shift
- B4 rotation + noise
- B5 hidden compound shift, never used for tuning

Batches have 1 000 rows, 20% held out by the evaluator. Metrics:

- **post-update**: skill on batch t after learning it (adaptation)
- **prequential**: skill on batch t *before* learning it (robustness)
- **final avg**: skill of the final model on the holdouts of all batches (retention)
- **forgetting**: mean over batches of (best earlier skill − final skill)

### Baselines vs ADAPT-X (mean of 40 streams)

| model | post-update ↑ | prequential ↑ | final avg (retention) ↑ | forgetting ↓ | hidden B5 before learning ↑ | update latency s ↓ | peak state KB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Static | 0.424 | 0.424 | 0.486 | 0.000 | -0.035 | 0.002 | 338.280 |
| Naive incremental | 0.719 | 0.347 | -0.726 | 1.753 | -0.088 | 0.244 | 487.675 |
| Standard replay (FIFO) | 0.729 | 0.387 | 0.436 | 0.375 | -0.002 | 0.311 | 640.540 |
| Full retrain oracle* | 0.733 | 0.436 | 0.728 | 0.036 | 0.051 | 0.431 | 1498.240 |
| ADAPT-X (proposed) | 0.729 | 0.417 | 0.582 | 0.200 | 0.047 | 0.180 | 987.235 |

*The oracle keeps every row and retrains from scratch each batch, so it breaks
the bounded-memory rule. It is a reference ceiling, not a competitor.*

What the table shows:

- **Adaptation (post-update).** ADAPT-X ties the best bounded baselines: 0.729 vs 0.729 for replay (p = 0.53) and 0.719 for naive.
- **Robustness (prequential).** ADAPT-X is better than every bounded baseline: +0.030 vs replay (31/40 streams, p < 1e-4) and +0.070 vs naive.
- **Retention.** Final-average skill is 0.582, vs 0.436 for replay (38/40 streams, p < 1e-6) and −0.726 for naive (which catastrophically forgets). Forgetting is 0.20, vs 0.375 for replay.
- **Latency.** ADAPT-X has the lowest update latency of all updating models: 0.18 s, vs 0.31 s for replay and 0.43 s for the oracle.
- **The oracle is still better.** The unbounded oracle beats ADAPT-X on retention (0.728) and slightly on prequential skill. This is the honest price of bounded memory.
- **ADAPT-X is not the best everywhere.** On classification post-update skill (e.g. interaction, imbalanced) FIFO replay or the oracle is higher. On heteroscedastic regression, retention (0.12) is far below the oracle's (0.67): the hidden B5 batch moves inputs and y far outside anything seen before, and every bounded model loses old-regime accuracy there.
- **Memory.** ADAPT-X's state (≈1 MB peak) is larger than FIFO replay's, because it holds up to 8 tree stages. It is bounded, and it stops growing (see the scaling table).

### Ablations (same 40 streams, same splits)

| model | post-update ↑ | prequential ↑ | final avg (retention) ↑ | forgetting ↓ | hidden B5 before learning ↑ | update latency s ↓ | peak state KB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A: no drift detection | 0.727 | 0.414 | 0.584 | 0.195 | 0.041 | 0.175 | 972.909 |
| B: no replay (K=0) | 0.725 | 0.399 | -0.027 | 0.922 | 0.022 | 0.161 | 937.517 |
| C: no adaptive scaling | 0.728 | 0.414 | 0.585 | 0.195 | 0.039 | 0.173 | 967.168 |
| D: no stable persistent state | 0.720 | 0.400 | 0.539 | 0.239 | -0.015 | 0.352 | 726.609 |
| E: random replay | 0.734 | 0.417 | 0.587 | 0.197 | 0.038 | 0.169 | 992.079 |
| F: full ADAPT-X | 0.729 | 0.417 | 0.582 | 0.200 | 0.047 | 0.180 | 987.235 |

What each component contributes (paired over the same 40 streams; `results/paired_tests.csv`):

- **B, replay memory: essential.** Without it, final-average skill collapses from 0.582 to −0.027 and forgetting goes from 0.20 to 0.92 (40/40 streams, p < 1e-7).
- **D, persistent stable state (vs refitting on batch + replay): clearly helps.**
  - Retention +0.043 (35/40 streams, p < 1e-4)
  - Prequential skill +0.017 (p < 1e-4)
  - Post-update skill +0.009 (p = 0.036)
  - Latency roughly halved: 0.18 s vs 0.35 s
- **A and C, drift detection and adaptive scaling: no significant accuracy effect on average.**
  - Retention differs by −0.003 (p = 0.06).
  - They help on all four classification families and on nonlinear regression (e.g. nonlinear classification 0.658 vs 0.616, mixed 0.576 vs 0.544) and hurt on heteroscedastic regression (0.123 vs 0.273).
  - Their measurable benefit is **compute**. On stationary streams, MAINTAIN mode matches the fixed plan's skill (0.791 vs 0.786) with 24% fewer trees, 15% less state and about 10% lower latency (runtime section). They also make every update explainable. Across streams, ADAPT-X beats A on retention in 24 of 40.
- **E, stratified-diversity replay vs random replay: no measurable difference** (p = 0.87). We keep it as the default only because it guarantees minority-class coverage (`test_replay_keeps_minority_class`). The benchmark does not show an accuracy gain from it, and we do not claim one.

### Per family

**Post-update skill**

| dataset | Static | Naive incremental | Standard replay (FIFO) | Full retrain oracle* | ADAPT-X (proposed) |
|---|---|---|---|---|---|
| heteroscedastic_regression | 0.258 | 0.773 | 0.742 | 0.713 | 0.743 |
| highdim_regression | 0.543 | 0.816 | 0.804 | 0.790 | 0.825 |
| imbalanced_classification | 0.408 | 0.633 | 0.637 | 0.655 | 0.618 |
| interaction_classification | 0.552 | 0.590 | 0.663 | 0.703 | 0.636 |
| linear_regression | 0.486 | 0.824 | 0.812 | 0.788 | 0.832 |
| mixed_classification | 0.572 | 0.623 | 0.624 | 0.666 | 0.617 |
| nonlinear_classification | 0.575 | 0.624 | 0.673 | 0.700 | 0.669 |
| nonlinear_regression | -0.002 | 0.871 | 0.874 | 0.851 | 0.893 |

**Final average skill (retention)**

| dataset | Static | Naive incremental | Standard replay (FIFO) | Full retrain oracle* | ADAPT-X (proposed) |
|---|---|---|---|---|---|
| heteroscedastic_regression | 0.360 | -7.184 | 0.068 | 0.668 | 0.123 |
| highdim_regression | 0.606 | 0.558 | 0.707 | 0.793 | 0.735 |
| imbalanced_classification | 0.459 | 0.253 | 0.412 | 0.664 | 0.603 |
| interaction_classification | 0.570 | 0.198 | 0.445 | 0.692 | 0.572 |
| linear_regression | 0.552 | 0.530 | 0.663 | 0.786 | 0.726 |
| mixed_classification | 0.595 | 0.391 | 0.518 | 0.678 | 0.576 |
| nonlinear_classification | 0.594 | 0.301 | 0.559 | 0.718 | 0.658 |
| nonlinear_regression | 0.155 | -0.857 | 0.116 | 0.824 | 0.659 |

### Drift detection on the benchmark streams (200 ADAPT-X updates)

| injected shift | none | covariate | prior | concept | compound | gradual | mean support |
|---|---|---|---|---|---|---|---|
| B1_covariate | 0 | 30 | 0 | 0 | 10 | 0 | 0.28 |
| B2_prior_label | 0 | 0 | 0 | 0 | 40 | 0 | 0.82 |
| B3_concept | 0 | 4 | 22 | 1 | 12 | 1 | 0.99 |
| B4_noise_rotation | 17 | 0 | 0 | 7 | 15 | 1 | 0.91 |
| B5_hidden_compound | 0 | 7 | 0 | 0 | 33 | 0 | 0.14 |

B3 reverts B2's label/prior shift while adding a concept change, so "prior" (22/40) is a correct reading of the dominant change. B1 is sometimes labelled compound because tree extrapolation near the support boundary raises in-support loss. B4 (rotation + 10% noise) is missed in 17/40 streams, because its loss increase is often below the 15% effect gate.

### Runtime and memory on a 30-batch stream (`python -m experiments.scaling`)

| model | state KB @1 | @10 | @20 | @29 | mean update s | max update s |
|---|---|---|---|---|---|---|
| ADAPT-X | 417 | 769 | 842 | 797 | 0.129 | 0.165 |
| Full retrain oracle* | 569 | 2141 | 3785 | 5265 | 0.452 | 0.723 |
| Standard replay (FIFO) | 306 | 418 | 512 | 306 | 0.213 | 0.4 |

**Stationary stream (15 batches, no shift; 3 families × 2 seeds, `stationary_efficiency.csv`)**: this is where detection pays off in compute.

| model | test skill | update s | trees in state | state KB |
|---|---|---|---|---|
| A: no drift detection | 0.786 | 0.156 | 284 | 930 |
| ADAPT-X | 0.791 | 0.141 | 215 | 787 |

---

## 6. SDG 9: Industry, Innovation and Infrastructure

The link is about compute and memory, not marketing. A deployed model
monitored on shifting data is usually kept healthy by periodic **full
retraining on all accumulated data**. The cost of that grows with the history:
the oracle column above uses O(total rows) memory, and its update time rises
with every batch.

ADAPT-X replaces this with the following:

1. **Continual learning instead of retraining.** An update fits one small
   early-stopped GBDT on about 1 000 new + 250 replay rows and solves a
   (k+1)-dimensional regularised regression for the combiner. Its cost depends
   on the batch size and the budgets, not on stream length.
2. **Bounded memory.** State = K replay rows + R reference rows + ≤ max_stages
   tree ensembles + fixed-size encoder arrays. It plateaus instead of growing,
   so the model can run on fixed hardware (edge gateways, industrial
   controllers) for an unlimited stream.
3. **Resilience.** The detector gives an auditable reason for every model
   change (which feature moved, whether the mapping changed, and with what
   confidence), and distillation limits how much a single bad batch can
   overwrite.

This targets SDG 9.4 (resource-efficient, cleaner technology in
infrastructure) and 9.5 (innovation capacity), by making adaptive ML systems
cheaper to keep correct over long deployments. It does not claim any energy
figure that was not measured; the runtime and memory tables above are the
measured evidence.

---

## 7. Remaining limitations (honest)

- **Retention gap to the oracle.** ADAPT-X retains less than the unbounded full-retraining oracle (0.58 vs 0.73 final-average skill). Bounded memory costs retention, especially after extreme extrapolating shifts such as heteroscedastic B5.
- **Detection and adaptive scaling.** They do not improve average accuracy over a well-chosen fixed plan (ablations A and C). Their measured value is compute savings on stationary data plus interpretability. The in-support plasticity rule was added after inspecting attempt-2's first full benchmark (`results/archive_v1_global_plasticity/`, retention 0.504 → 0.582). That is a design iteration informed by held-out families, and we disclose it. The "skip all growth when no drift" variant was also tried and rejected (`results/archive_v3_maintain_without_growth/`).
- **Replay selection.** Stratified-diversity replay showed no accuracy benefit over random replay.
- **Controller tuning.** Constants were tuned on two dev families with dev seeds. The resulting optimum sits near the low end of the stability range, so the controller has limited leverage.
- **Correlation drift.** Correlation shift is detected, but in-support filtering for the concept test uses marginal ranges only, so joint-novelty rows count as in-support.
- **Detection accuracy on the benchmark.** B1 (pure covariate) is sometimes reported as compound (10/40), because tree extrapolation near the support boundary raises in-support loss. B4 (rotation + noise) is reported as none in about 40% of streams.
- **Model state size.** The state is up to 8 stages and larger than a single refit model. It is bounded but not minimal.
- **Environment.** LightGBM could not be installed in the build environment, so the base learner is scikit-learn `HistGradientBoosting` (native categorical and missing-value support). All results use it.
- **pytest.** pytest was not installable in the build sandbox. The 33 tests were executed with a minimal pytest-compatible runner. Please run `python -m pytest -q` locally.

---

## 8. Run it

No environment variables or secrets are needed, so there is intentionally no
`.env.example`. All configuration is in `config.yaml` plus CLI flags.

```bash
pip install -r requirements.txt
python -m pytest -q                                   # test suite
python -m experiments.demo_stream --family mixed_classification   # one stream, narrated
python -m experiments.run_benchmark                   # authoritative benchmark + ablations (~11 min, 2 cores)
python -m experiments.scaling                         # long-stream runtime/memory + stationary stream
python -m experiments.tune_controller                 # dev-only controller tuning (~5 min)
```

```python
from src import ContinualLearner, load_config
learner = ContinualLearner(load_config("config.yaml")).initialize(X0, y0)   # task type inferred
for X_t, y_t in stream:
    report = learner.update(X_t, y_t)      # ShiftReport; learner.history[-1]["plan"] = UpdatePlan
    print(report.summary())
preds = learner.predict(X_new)             # predict_proba for classification
learner.memory_footprint()                 # audited bytes per component
```

## 9. Repository map

```
src/            algorithm (domain-agnostic; no dataset or feature names anywhere)
experiments/    8 data families + shift operators, baselines, benchmark, tuning, scaling, demo
tests/          33 behavioural tests (leakage, bounded memory, forgetting, detection, reproducibility ...)
results/        RESULTS.md + CSVs from the runs above; archive_v*/ = earlier design iterations
notebooks/      walkthrough notebook (generate_notebook.py rebuilds it)
```
