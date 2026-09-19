# Independent single-source recovery: P0 revision 3

## Scope and audit

This revision evaluates independent source scenarios under a correctly specified
synthetic PDE. It does not test noise, sensor-count variation, meteorological
misspecification, multiple sources, Almaty observations, or relative superiority
over a classical estimator. "Held-out" means held out from design/tuning: each
scenario's sparse sensor series is intentionally used to fit its own fresh PINN.
The dense field and source parameters are not training targets.

The inspected legacy inverse script did not explicitly feed source coordinates,
Q, or dense concentration values into its optimization. Nevertheless, the mixed
archive and broad configuration interface were weak structural guarantees. The
new fit API has strict, copied sparse-data and optimizer-settings records; it
cannot receive scenario paths, truth fields, metadata or evaluation grids.
Unexpected settings are rejected. A loader opens only the two input files, and
post-training evaluation requires a persisted terminal attempt record before
opening any truth. Tests actively deny truth-file access while fitting and
compare repeated state dictionaries. This verifies the implemented dataflow, not
a security sandbox or the historical provenance of earlier hyperparameter choices.

No existing physics, differentiation, source parameterization, MLP, loss, or
optimizer module was modified. Source coordinates remain sigmoid parameters;
Q remains positive via softplus. The model's unconstrained concentration output
is not clipped or replaced with a positivity-enforcing architecture.

## Preregistration and reproducibility

The full design is in [the frozen protocol](single_source_benchmark_protocol.md)
and `configs/single_source_benchmark.yaml`. The executable freeze was made at
2026-09-19T15:35:59.697814+00:00, before any benchmark reference or held-out fit.

- Protocol SHA-256: `a960d1418561b989b57f5b9046d45f769ce27ccdc3ef2485d001fd44ad0abbac`.
- Sensor-file SHA-256: `c0533d4c030bc4ce020eaaebf6531a19453d36b5e322e6c63eb9d27baef2ebd2`.
- Reference-solver commit: `342b16125c6c7baf7232e1d3eb7d0d53a580459d`
  ("Add high-accuracy synthetic reference pipeline").
- Actual uncommitted revision-3 executable bytes are archived in
  `results/single_source_benchmark/benchmark_code.zip` and individually hashed in
  `protocol_lock.json`. Git HEAD alone is not claimed to contain revision 3.
- Environment/package versions, scenario plan, exact geometry and later reference
  manifest are retained alongside the run. Code/config mutation causes resume
  verification to fail. Failed terminal attempts are not retried.

The plan draws 10 development and 30 held-out sources independently, with xs,ys
uniform on [0.25,0.75] and Q uniform on [0.8,1.4]. Sigma=0.08; u=0.35,v=-0.15,
D=0.005; unit square; zero IC and zero concentration on all edges. Sensor geometry
is the same 20 coordinates for every case, with 81 readings per sensor at
t=0:0.01:0.8 and exactly zero added noise. All references use 641², dt=0.00003125;
each also has 161²/321² refinement evidence. Selected dense times are
0.1,0.2,0.4,0.8, not an entire saved high-resolution time history.

The unchanged fit uses 64×64×64 tanh hidden layers, Adam lr=0.001, 6,000 updates,
data/PDE/IC/BC weights 10/1/1/1, 512 data and 512 interior samples, 128 IC points
and 64 per edge per update. All source guesses are (0.5,0.5,0.5). Primary seed 42
runs on all 40 scenarios; seeds 314 and 2718 additionally run on preselected
heldout_001,007,013,019,025. This is a 50-fit reduced design, not 30×3 held-out fits.
No outcome-dependent tuning or selective restarts are allowed.

## Interpretation conventions

Continuous localization error is primary. The secondary descriptive recovery
criterion is location error <=0.08 (one sigma) **and** relative peak-strength
error <=0.20. This cutoff is not a validated universal success standard, and
parameter recovery does not certify a physically admissible concentration field.
Computational completion, evaluation completion and scientific recovery are
separate fields. Missing metrics/failures remain in raw results and recovery
denominators. Primary held-out statistics do not pool optimizer restarts.

Dense concentration metrics equally weight all 641² nodes of the four selected
snapshots. They are discrete snapshot-ensemble errors, not a time integral over
the nonuniform snapshot times. Independent PDE/IC/BC diagnostics use fresh seeded
points; finite sampled residuals do not prove continuous-domain compliance.
Negative predictions are measured exactly, with no clipping/tolerance masking.

The identifiability diagnostic is for predetermined heldout_001, at fixed true Q
and known physics, using a separate 201² solver. Its observation-MSE landscape is
not a posterior or calibrated confidence region and cannot establish joint
location/strength uniqueness across all scenarios.

## Recovery results

All **50/50** scheduled fits completed computationally and were evaluated:
10 development, 30 primary held-out, and 10 additional-seed fits. Every successful
fit used exactly 6,000 updates; each source estimate matches its final history
record. There were no training crashes, NaNs, missing evaluations or excluded
cases. Computational success did **not** imply scientific recovery.

Primary held-out recovery was **14/30 = 46.67%**, with a **95% Wilson interval
30.23–63.86%**. Sixteen cases failed the predeclared combined criterion. Location
alone was within one source width in **22/30** cases; strength was within 20% in
**14/30**. All 30 strengths were underestimated. This latter signed observation
is descriptive; the preregistered strength endpoint remains absolute relative
error. The confidence interval concerns the sampled scenario distribution,
not a calibrated confidence region for any particular source.

All statistics below use **30 primary-seed held-out cases**, with zero missing
values. Standard deviation is the sample SD (`ddof=1`); quantiles use NumPy's
linear interpolation. Relative errors are shown in percent. L denotes the
synthetic domain-length unit, and C denotes synthetic concentration.

| Metric | Mean | Median | SD | Q25 | Q75 | Q90 | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Localization (L) | 0.043417 | 0.028428 | 0.044180 | 0.004061 | 0.082442 | 0.113895 | 0.120041 |
| Relative peak-strength error (%) | 30.967 | 25.888 | 29.677 | 2.833 | 61.929 | 68.836 | 88.182 |
| Concentration relative L2 (%) | 34.690 | 29.118 | 30.701 | 5.515 | 67.198 | 74.111 | 85.726 |
| Concentration RMSE (C) | 0.018475 | 0.018414 | 0.016279 | 0.002491 | 0.033103 | 0.041861 | 0.048693 |
| Concentration MAE (C) | 0.006287 | 0.006285 | 0.004556 | 0.001803 | 0.010075 | 0.012829 | 0.014538 |

Development results remain separate: **4/10** recovered, mean localization
0.048464 L, median 0.036460 L, SD 0.052577 L, Q90 0.114196 L, worst 0.158515 L;
mean strength error 34.239% and mean concentration L2 37.717%. No settings were
changed in response. Across all 50 fits, 22 met the criterion, but that pooled
number is **not** an independent-scenario recovery-rate estimate because some
scenarios have three seeds.

## Physical diagnostics: observed violations, not a validity certificate

All **30/30** primary held-out fields contain negative predictions. The mean
negative fraction is **36.08%**, with a range **5.39–65.60%**. The most negative
prediction is **−0.026666 C**, in `heldout_017`. No negative values were clamped,
masked or excluded from concentration errors. Fraction alone does not measure
negative mass or its magnitude, especially where true C is close to zero.

These diagnostics use independent, previously unused sampled points. RMSE/MAE
columns are means of per-scenario values, not pooled RMSE across all points.

| Diagnostic | Mean RMSE | Mean MAE | Worst scenario RMSE | Largest sampled absolute violation |
| --- | ---: | ---: | ---: | ---: |
| PDE residual (C/time) | 0.007807 | 0.004635 | 0.011416 | 0.101527 |
| Zero boundary condition (C) | 0.003320 | 0.001913 | 0.011568 | 0.124454 |
| Zero initial condition (C) | 0.003039 | 0.002140 | 0.005193 | 0.024881 |

Thus the predictions are **not strictly physically admissible**: positivity and
zero IC/BC are violated, sometimes substantially at particular sampled points.
Small average residuals are not a recovery certificate. For example, the worst
localized primary case (`heldout_007`) has PDE RMSE only 0.003972 C/time while
its concentration L2 error is 82.86%. The best-localized case has PDE RMSE
0.007792 C/time. Residual magnitude alone would mis-rank these source estimates.

## Optimizer sensitivity

This comparison varies MLP initialization and training-sample randomness while
keeping the physical source guess fixed. It does **not** test different initial
source-location guesses. Only the five preregistered IDs were run three times.

| Scenario | Localization range over 3 seeds (L) | Within-case sample SD (L) | Recoveries |
| --- | ---: | ---: | ---: |
| heldout_001 | 0.077024–0.089771 | 0.007220 | 0/3 |
| heldout_007 | 0.120041–0.141252 | 0.011459 | 0/3 |
| heldout_013 | 0.037513–0.040450 | 0.001473 | 0/3 |
| heldout_019 | 0.006749–0.007519 | 0.000413 | 3/3 |
| heldout_025 | 0.005200–0.006189 | 0.000499 | 3/3 |

No subset case changed its recovery status across seeds. The largest within-case
location range was 0.021211 L; within-case SDs were below the primary between-case
SD of 0.044180 L. In this limited subset, failures persist across the tested
optimizer randomness. This is descriptive evidence, not a formal variance
decomposition or proof that other initializations cannot help. No best seed was
substituted into the primary analysis.

## Worst failures and mechanically selected examples

- **Worst localization: heldout_007.** True (xs,ys,Q) =
  (0.734204,0.495653,0.871811); estimate = (0.620174,0.458147,0.180060).
  Location error 0.120041 L, strength error 79.35%, concentration L2 82.86%.
  This case's sensor reference estimate missed the 1% target (1.31%); that
  limitation is not hidden and the case is not removed.
- **Worst strength and field reconstruction: heldout_017.** True source
  (0.743138,0.725770,1.249677); estimate (0.700473,0.831568,0.147689).
  Strength error **88.18%**, concentration L2 **85.73%**, RMSE 0.048693 C;
  minimum predicted C **−0.026666**. Its reference met the assessed target.
- **Largest sampled boundary violation: heldout_021**, |C|=**0.124454 C**
  on a boundary prescribed to be exactly zero; boundary RMSE 0.011568 C.
- **Largest negative fraction: heldout_011**, **65.60%** of evaluated entries.

Examples were selected by localization rank, never by appearance:
`best_case` = **heldout_029** (0.000444 L), `median_case` = **heldout_016**
(lower-middle rank, 0.024294 L), `worst_case` = **heldout_007** (0.120041 L).
The lower-middle example is not the arithmetic median of two middle errors.
Every example displays truth, unclipped prediction and absolute error with
sensors and true/estimated source markers. Every fit also retains its own
reconstruction, losses, source history, checkpoint and metrics.

## Scientific interpretation

**Supported classification: unreliable recovery for this frozen configuration.**
Only 14/30 unseen sources met the declared location-and-strength criterion,
mean strength error was 31.0%, and concentration L2 had median 29.1% and Q90
74.1%. The continuous endpoints show substantial failures independently of any
debate over the secondary 20% strength cutoff. Accurate cases exist, but their
existence does not establish reliable independent-source recovery across the
tested sampling distribution. This is not a theorem that all PINNs must fail.

**Do not advance to broad noise/sensor/misspecification claims on this evidence.**
First understand the baseline failures and distinguish observation geometry,
joint identifiability, finite training budget and optimization effects using
development data. Any changed model/protocol requires a fresh untouched held-out
set or a clearly separated new validation design. This revision neither changes
the model to rescue these outcomes nor runs the next experimental stages.

The evidence is conditional on one geometry, one wind/diffusion setting, one
known-width/time-independent source family, clean zero Dirichlet boundaries,
zero IC and zero added noise. Numerical references are approximate and two
sensor trajectories miss the assessed accuracy target. Five seed-repeated
scenarios do not provide a complete optimizer uncertainty analysis. No calibrated
source posterior, atmospheric validation, emitter identification or comparison
claim against classical methods follows from this benchmark.

## Identifiability landscape

The predetermined scenario is **heldout_001**: true (xs,ys,Q) =
(0.559291,0.318557,1.357689). Its primary PINN estimate is
(0.593744,0.387446,0.539181). At each location in the 21×21 candidate grid we
held Q at **1.357689**, not at its PINN estimate, and ran the separate 201²
numerical solver with the same sensor coordinates and times.

| Conditional observation-MSE diagnostic | Value (C²) |
| --- | ---: |
| Exact true location (separately evaluated) | 5.02172e−8 |
| Best grid location, (0.575,0.325) | 8.12958e−6 |
| Best grid location more than one source width away | 4.65548e−5 |
| Best grid location more than two source widths away | 4.65548e−5 |
| PINN-estimated location, but **true Q fixed** | 1.06145e−3 |

The lowest sampled basin is near the true source. Its grid-minimum location error
is 0.016979 L; the 0.025 grid spacing limits localization of that minimum. There
is also a secondary lower-mismatch region near (0.700,0.400). Its mismatch is
about 5.7 times the grid minimum and 927 times the separately evaluated true-
location mismatch. Thus the noise-free, **known-Q conditional diagnostic favors
the true location**, rather than establishing equally compatible distant sources.

The PINN-location value above is **not** the trained network's sensor MSE and
does not use its estimated Q; it evaluates location alone under oracle strength
control. This distinction prevents an unfair comparison with the fitted network.
The true-location mismatch is not zero because the 201² and 641² numerical
operators differ. A finite grid and one fixed-Q scenario cannot prove global or
joint (xs,ys,Q) uniqueness, exclude location/strength trade-offs, or diagnose all
failures as purely optimization failures. No error threshold was fitted to turn
this landscape into a likelihood, confidence region or posterior probability map.

## Artifact verification and outputs

The complete command exited successfully. A post-run audit verified all 40
reference/input/source hashes, the frozen code/configuration, all 50 checkpoints,
and **300,000** sequential update records. Recomputing concentration metrics and
negativity statistics from every saved prediction gave maximum difference
**0.0** from the per-run metrics. Primary aggregate means, medians, sample SDs,
quantiles and extrema were recomputed from raw CSV. All **108 PNG** artifacts
passed image-file integrity checks. Selected figures were visually inspected;
no examples were swapped after viewing them.

The run is in `results/single_source_benchmark/` (Git-ignored):

- `raw_results.csv`: all 50 rows, including every recovery failure and all true /
  estimated source parameters; `summary.csv` and `summary.json`: primary-scenario
  aggregates, confidence interval, separate development and optimizer statistics.
- `scenario_manifest.csv`, `manifest_lock.json`, `scenario_plan.json`,
  `protocol.yaml`, `protocol.md`, `protocol_lock.json`, `sensor_geometry.json`:
  immutable scientific design and source/input/reference hashes.
- `environment.json`, `provenance.json`, `benchmark_code.zip`,
  `source_snapshot.zip`: exact environment, Git provenance and executed code.
- `true_vs_predicted_sources`, `localization_error_distribution`,
  `source_strength_recovery`, `physical_diagnostics`,
  `source_location_loss_landscape`: 300-dpi PNG and PDF figures.
- `best_case/`, `median_case/`, `worst_case/`: mechanical selection metadata and
  final-time truth/prediction/error figures; `identifiability/`: candidate CSV,
  numerical landscape checkpoint, config and metrics.
- `datasets/<scenario>/`: sparse input files separated from source truth,
  selected dense fields, three-grid refinement evidence and generation metadata.
- `runs/<scenario>/seed_<seed>/`: exact fit recipe/input hashes, per-update JSONL
  and text logs, model/source/optimizer checkpoint, terminal attempt status,
  source estimate, dense prediction checkpoint, metrics, per-snapshot metrics,
  loss/source-history plots and reconstruction figures.

Reproduce into a **new** output directory with:

```bash
python scripts/run_single_source_benchmark.py --output-dir results/single_source_benchmark_replication
```

Existing data are never silently overwritten. `--resume` verifies the freeze and
does not retry terminal failures. Deterministic scientific arrays/metrics are
expected in the saved environment; timestamps, runtimes and archive-byte details
are not guaranteed identical across machines. Running this command again on the
same scenario plan is a replication, **not** a new untouched test set.

## Repository changes and Git state

The Karpathy coding guidelines kept changes confined to benchmark adapters,
tests and reporting; existing model, source parameterization, losses and solver
equations were reused without modification. Modified: `README.md`. Created:

- `configs/single_source_benchmark.yaml`
- `configs/single_source_sensors.json`
- `docs/single_source_benchmark_protocol.md`
- `docs/single_source_benchmark.md`
- `scripts/run_single_source_benchmark.py`
- `src/inverpinn/data/single_source_benchmark.py`
- `src/inverpinn/training/single_source.py`
- `src/inverpinn/evaluation/single_source_benchmark.py`
- `src/inverpinn/visualization/single_source_benchmark.py`
- `tests/test_single_source_benchmark.py`

Current branch: `scientific-revision`. Latest committed milestone:
`342b16125c6c7baf7232e1d3eb7d0d53a580459d`, "Add high-accuracy synthetic reference
pipeline". That commit was made before revision 3 and left a clean working tree.
The revision-3 changes listed above remain **uncommitted**. Generated binary
references/checkpoints/results are ignored
and were not added to Git. No history was rewritten. No noise/sensor-count sweep,
real-data pipeline change, multi-source inference or classical-method comparison
was performed.

## Reference qualification (completed before fitting)

All **10 development and 30 held-out** references were generated. Their manifest
was frozen at **2026-09-19T16:33:00.092696+00:00**, before the first development fit.
`scenario_manifest.csv` SHA-256 is
`63bc3b2a958bf0577fa986ccde331f456a5eb847203aa24cb7980aa9be6c74cc`.

**38/40** scenarios meet the assessed <1% target across all four fields and the
pooled sensor trajectory. The misses are `development_004` (sensor estimate
**1.5301%**) and `heldout_007` (**1.3137%**). Both remain in all scheduled results;
the latter also retains its preselected additional optimization seeds. Their
dense final-field estimates are 0.9732% and 0.9548%, respectively. Across all
160 assessed dense snapshots the maximum estimate is **0.9950%**. Thus every
dense field meets the assessed target, but not every sparse observation trajectory
does. Mean sensor-trajectory estimate across all 40 cases is **0.6622%**.

These are three-grid, conditional Richardson estimates, not certified bounds.
"Zero noise" means zero **added measurement noise**, not exact continuum-PDE
observations. No references were adaptively replaced, excluded, or relabeled
qualified after their errors were measured. Numerical bias is a limitation of
the recovery evidence, especially for the two flagged trajectories.

## Automated validation

- Targeted benchmark suite: **23 passed**, 27.89 s, no warnings or failures.
- Full project suite with the final tests: **290 passed**, 170.82 s, **120
  warnings**, no failures. This run shared CPU resources with reference generation;
  its wall time is not a test-performance comparison.
- Warnings: one NumPy/netCDF4 binary-compatibility runtime warning and 119
  xarray/netCDF4 NumPy-2.5 shape-assignment deprecation warnings. These already
  existed before revision 3; they were not suppressed or "fixed" in this scope.
- The suite covers all twelve requested benchmark safeguards, including actual
  small-grid reference generation, bitwise-zero noise, strict training input
  records, denied truth access during training, reproducible state dictionaries,
  correct analytical metrics/physical diagnostics, retained failed attempts,
  separate completion/recovery flags, report generation, controlled landscape
  generation and non-retrying resume behavior. No previous test was weakened.

The checks are software/mathematical validation, not proof of successful source
recovery. The scientific evidence comes from the scheduled independent cases.
