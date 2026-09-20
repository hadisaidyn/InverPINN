# P0 revision 5: fresh blind benchmark report

Completed on 2026-09-20. **Preregistered classification: unreliable controlled
recovery.** B_revised improved several paired errors but recovered only 18/30
sources, underestimated Q in every primary case, and failed the preregistered
mean-PDE-RMSE non-regression gate. No model was changed after these results.

The prospective definitions are in [the protocol](fresh_blind_benchmark_protocol.md).
Exact values, all observations, reference fields, predictions, checkpoints,
per-update logs and provenance are retained in the local-only archive
`results/fresh_blind_benchmark/`, particularly `summary.json`, `summary.csv`
and `all_runs.csv`. These ignored files are not included in a clean checkout;
see the [archive-access boundary](reproducibility.md#reference-and-full-archive-boundary).
All numerical units below are synthetic, not metres, hours or real PM2.5 units.
Q denotes Gaussian **peak source intensity**, in concentration/time, not
area-integrated emissions. Relative errors are fractions unless marked %.

## A. Independence from model development

Yes, for the new source scenarios. The initial working tree was clean on
`scientific-revision` at `9dc90c247b0a596b789cb0ee5f1c06f938db40c4`.
Protocol and seed were frozen at **2026-09-20T03:16:05.917917+00:00**, before
fresh generation. Master seed **2009202605** generated exactly fresh_001 through
fresh_030. Their source hashes and child seeds are disjoint from the 40 prior
development/held-out sources reconstructed from frozen generation rules. No
previous outcomes or datasets were opened to select, exclude or interpret cases.
No fresh draw was rejected, replaced, tuned on or used to select a model.

The input-only fitting interfaces and process-level accidental-access guard
exclude source truth and prior results. Evaluation opens truth only after a
terminal fitting record. Tests exercise the guard in a live subprocess. This
is not a claim of OS-level isolation against a malicious process. The primary
optimization seed was 42 for every case/model. All 60 primary and 20 additional
fits finished; there were no retries or replacements.

Independence does **not** mean a new sensor layout or transport regime: the
specified 20-sensor geometry, 81 times from 0 to 0.8, one Gaussian of width
0.08, wind (0.35,-0.15), D=0.005, zero noise and homogeneous IC/BC are shared
intentionally. Each inverse fit uses that case's sparse observations; dense
truth and source truth are evaluation-only, not fitting inputs.

Manifest SHA-256:
`53032fd0d06cb1ce5ebba0ca45e3bb9284aa110675980f0186fa67316800c149`.
Sensor-layout SHA-256:
`c0533d4c030bc4ce020eaaebf6531a19453d36b5e322e6c63eb9d27baef2ebd2`.
The manifest/input lock rejects modifications; all hashes were rechecked.

Reference qualification: the unchanged float64 solver used 161²/321²/641²
grids, with dt=0.0005/0.000125/0.00003125. The 641² fields are always the
reference. All 120 assessed field snapshots meet the 1% conditional Richardson
target (maximum 0.96018%). The pooled sensor-trajectory target is missed by
five retained cases:

| Scenario | Estimated sensor-trajectory relative error |
|---|---:|
| fresh_006 | 1.06705% |
| fresh_011 | 1.04452% |
| fresh_019 | 1.50382% |
| fresh_022 | 1.01303% |
| fresh_027 | 1.06284% |

All estimates were finite with positive observed convergence order. None was
silently dropped or regenerated. These are conditional three-grid estimates,
**not certified bounds**, individual-sensor bounds or a spacetime supremum.
Thus this is not a benchmark in which every sensor trajectory attained <1%.
See `results/fresh_blind_benchmark/reference_quality.csv` in the local-only archive.

## B. Frozen configurations and code

Both were unchanged. B_revised's configuration is byte-identical to its
Revision-4 archive. All original tracked Python source files were checked
against the frozen Git commit before and after fitting. All 71 source/script
files in the pre-fit execution snapshot also remained unchanged. B0 ran its
original 6,000 updates; B_revised ran its original 12,000 updates. Every update
was logged: 780,000 updates across 80 fits. Neither architecture, loss,
optimizer, source initialization, physical parameterization nor sampling changed.

| Item | SHA-256 |
|---|---|
| B_revised configuration | `3e400ef1195cf90f6481117ae5c395695cca23a12d47ab532ec6822e400c332e` |
| B0 configuration | `5f85e130900be9295bc48b3f152772a8f456b1a4b4b77f076258e1f6d8f30254` |
| B_revised recorded model-code bundle | `b055ec6659384ca0f499974ab3ec330bf4e2bf15e30b9678f3e472df7e80f476` |
| B0 recorded model-code bundle | `197f0eda01e65fa1bd3504f9d5f5fbe471751c4f96d800ab6579b0bbb142f2f9` |
| Reference solver | `171d63443cc789534402b08ad497b17f18fbb113e9530d38c6ffedd9ae77b8b8` |
| Protocol YAML | `a6d354ed70fee6e7ba146c4516c6375c2930a8d6a4b3d35517ab870b0604d18b` |

Model-code bundle hashes are hashes of canonical filename-to-SHA-256 mappings;
individual files are listed in
`results/fresh_blind_benchmark/model_code_hashes.json` in the local-only archive.
The complete frozen source inventory and environment/package versions are
preserved in preregistration and execution snapshots. This paired comparison
is of two frozen recipes, **not an equal-compute comparison**: revised uses
twice B0's updates, so gains cannot be attributed solely to hard constraints.

Historical-code caveat: a final extra byte-identity probe against pre-revision-4
commit `ce210e3862fbd34aa509a9fc139ab8698edfe88d` initially **failed** for
`training/forward_pinn.py`. Inspection found only the optional
`on_objective=None` argument and its guarded diagnostic callback, added before
the expected frozen commit. The strict B0 caller does not enable it. An
automated in-memory AST comparison verified that removing only this inactive
hook makes the complete module identical to the older version; all other six
recorded B0 code files and the B0 config are byte-identical. Therefore B0's
numerical fitting path is unchanged, but its whole file is not byte-identical
to the older commit. No code or result was patched, and this inactive branch
does not invalidate the benchmark. The failed probe and resolution are retained
in the append-only historical_B0_code_audit.json addendum, outside the earlier
artifact manifest. Older/current forward_pinn.py hashes are respectively
`d52be72501be658a69ff65b6dabbc66a46ec96fb90ca4fd406028e798e89378b` and
`e000b8f31e5704ad0f59f7af538da89b059bddac91020bcd636e02ba3382d727`.

## C. Fresh localization

All 30 primary results are included, with no missing metrics or discarded
outliers. SD is sample SD (ddof=1); quantiles use linear interpolation.

| Model | Mean | Median | SD | Q25 | Q75 | Q90 | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_revised | 0.029294 | 0.018111 | 0.035341 | 0.001583 | 0.043710 | 0.088225 | 0.124059 |
| B0 | 0.047754 | 0.042346 | 0.044077 | 0.005345 | 0.085022 | 0.102306 | 0.146798 |

The revised localization-only threshold was met in 26/30 cases. That is not
the joint recovery endpoint. Worst localization was fresh_011.

## D. Fresh Q recovery

Absolute relative Q-error distribution, in percent:

| Model | Mean | Median | SD | Q25 | Q75 | Q90 | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_revised | 21.0848 | 15.1819 | 22.4618 | 1.3832 | 32.9109 | 53.5076 | 79.1153 |
| B0 | 34.9121 | 30.5739 | 29.9894 | 4.0362 | 65.9304 | 73.6198 | 82.8481 |

B_revised: **30 underestimates, 0 overestimates, 0 approximately equal** using
the frozen equality tolerance. Mean signed Q error is **-0.240034** and median
**-0.171398** concentration/time. Signed relative mean/median are
**-21.0848%/-15.1819%**. The largest relative error is fresh_019:
Q_true=0.973155, Q_pred=0.203241. The most negative raw signed error is
fresh_011, **-0.812267** (Q_true=1.255570, Q_pred=0.443303).

## E. Concentration reconstruction

Metrics equally weight every node in all four 641² reference snapshots at
t=0.1,0.2,0.4,0.8. They are not preview-grid metrics, continuous spacetime
integrals or held-out-sensor forecasting scores. Negative predictions are not
clipped. Full-field and source metrics were independently recomputed for all
80 fits and matched the saved values (rtol=1e-12, atol=1e-14).

Relative L2 distribution, in percent:

| Model | Mean | Median | SD | Q25 | Q75 | Q90 | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_revised | 25.5381 | 19.1332 | 25.3051 | 3.2224 | 41.7620 | 68.2031 | 84.2932 |
| B0 | 38.1005 | 35.5269 | 30.4215 | 6.2181 | 70.1276 | 78.7439 | 84.6715 |

| Model / metric | Mean | Median | SD | Q90 | Worst |
|---|---:|---:|---:|---:|---:|
| B_revised RMSE | 0.013610 | 0.010452 | 0.013389 | 0.037256 | 0.044358 |
| B_revised MAE | 0.004119 | 0.003037 | 0.003805 | 0.010441 | 0.013133 |
| B0 RMSE | 0.020478 | 0.019021 | 0.016907 | 0.045304 | 0.049489 |
| B0 MAE | 0.006868 | 0.005948 | 0.004717 | 0.013270 | 0.015486 |

RMSE/MAE use synthetic concentration units. Full quantiles are in summary.csv.

## F. Unchanged secondary joint recovery endpoint

Localization <=0.08 **and** relative Q error <=20%:

- B_revised: **18/30 = 60.0%**, 95% Wilson interval **42.32–75.41%**.
- B0: **12/30 = 40.0%**, 95% Wilson interval **24.59–57.68%**.

There are six B0-failure/revised-success transitions and zero reverse
transitions; 12 cases fail both and 12 pass both. Revised failures are
fresh_003,004,006,011,012,019,021,022,027,028,029,030. All remain in outputs.
All 12 fail the Q criterion; four also fail localization. There are zero
computational failures, but **12 scientific recovery failures**.

## G. Physical validity

Fresh diagnostic random points use an independent stream, seed 2009202699:
8192 interior, 4096 initial and 1024 per boundary edge. These are independent
samples, not a continuous-domain residual bound. Dense negativity is evaluated
on the full saved fields.

| Diagnostic | B_revised mean / worst | B0 mean / worst |
|---|---:|---:|
| PDE residual RMSE | 0.008600 / 0.017462 | 0.007982 / 0.011934 |
| PDE residual MAE | 0.003321 / 0.005538 | 0.004730 / 0.006371 |
| BC RMSE | 0 / 0 | 0.003528 / 0.011590 |
| Maximum absolute BC violation | 0 / 0 | 0.029859 / 0.122212 |
| IC RMSE | 0 / 0 | 0.003315 / 0.007638 |
| Maximum absolute IC violation | 0 / 0 | 0.015776 / 0.049488 |
| Negative prediction fraction | 0 / 0 | 32.7144% / 57.5075% |

Global minimum predicted concentration: revised **0**, B0 **-0.014581**.
PDE RMSE median/Q90 for revised: **0.007391/0.015562**. Its hard envelope
enforces nonnegativity and exact homogeneous IC/BC, but the nonzero PDE
residual is not certified as acceptably small. Physical admissibility is not
proof of correct source recovery. Mean PDE RMSE is actually 7.75% above B0.

## H. Paired B0 versus B_revised evidence

Both recipes used byte-identical observations and reference fields for every
paired ID. Delta=revised-minus-B0, so negative favors revised. Ties were
defined prospectively as abs(delta)<=1e-8+1e-6*max(abs(a),abs(b)).

| Metric | Mean delta | 95% paired-bootstrap CI for mean delta | Median paired delta | Revised better / B0 better / tied |
|---|---:|---:|---:|---:|
| Localization | -0.018460 | [-0.029246,-0.008993] | -0.005075 | 19 / 11 / 0 |
| Relative Q error | -0.138273 | [-0.196258,-0.083394] | -0.048312 | 25 / 5 / 0 |
| Relative L2 | -0.125624 | [-0.187859,-0.068201] | -0.031727 | 26 / 4 / 0 |
| RMSE | -0.006868 | [-0.010323,-0.003675] | -0.001603 | 26 / 4 / 0 |
| MAE | -0.002750 | [-0.003761,-0.001808] | -0.001543 | 27 / 3 / 0 |
| PDE residual RMSE | +0.000619 | [-0.000763,+0.002125] | -0.000488 | 16 / 14 / 0 |

Relative-error deltas are fractions (e.g. -0.138273 is -13.8273 percentage
points). There are 30 complete pairs for each outcome. Mean localization,
Q error and L2 improve by 38.66%, 39.61% and 32.97% relative to B0; these are
relative changes of means, not paired median changes. The paired localization
median-delta CI is [-0.019608,+0.000460], including zero.

BC, IC and negativity improve in all 30 pairs; PDE MAE improves in 25/30.
PDE RMSE is mixed: median improves, mean and worst worsen. Its bootstrap
mean-delta interval includes zero, so the observed mean increase is **not**
strong evidence of population-level deterioration. It nevertheless fails the
frozen operational non-regression gate. Neither that gate nor its wording was
changed after seeing results.

Intervals use 10,000 paired resamples of scenario indices with replacement,
seed 2009202617, percentile 2.5/97.5 endpoints. These quantify source-distribution
variation conditional on this sensor layout, transport and primary seed;
they do not cover all optimizer, numerical or real-world uncertainty. No
p-value-only claim or general PINN superiority claim is made. These data also
cannot establish that improved hard-constraint validity caused better recovery.

## I. Systematic Q underestimation

**It persisted in 30/30 primary cases**: mean signed relative error -21.0848%,
median -15.1819%, worst -79.1153%. All 30 rows of the selected seed substudy
also underestimate Q. No calibration or changed Q initialization was applied.

## J. Prespecified optimizer-seed sensitivity

Exactly seeds 42,314,2718 were used for each index-selected case. Every row is
saved independently in seed_sensitivity.csv; primary results remain seed 42.
Below are min–max ranges, **not truth-selected estimates** or best-seed scores.

| Scenario | Localization range | Predicted Q range | L2 range | Recovered seeds / 3 |
|---|---:|---:|---:|---:|
| fresh_001 | 0.005519–0.006166 | 1.085364–1.089659 | 0.047800–0.061443 | 3 |
| fresh_004 | 0.020763–0.024435 | 0.750597–0.795604 | 0.293858–0.350332 | 0 |
| fresh_007 | 0.000201–0.001195 | 1.216956–1.221350 | 0.013726–0.019072 | 3 |
| fresh_010 | 0.000774–0.001839 | 1.083414–1.092793 | 0.013932–0.021060 | 3 |
| fresh_013 | 0.001978–0.002553 | 1.170133–1.171659 | 0.036232–0.041446 | 3 |
| fresh_016 | 0.017782–0.018157 | 0.920290–0.932290 | 0.188098–0.198205 | 3 |
| fresh_019 | 0.105397–0.110508 | 0.199467–0.210858 | 0.833623–0.849504 | 0 |
| fresh_022 | 0.030533–0.034978 | 0.526945–0.555562 | 0.383767–0.426543 | 0 |
| fresh_025 | 0.014360–0.018441 | 0.971412–1.018051 | 0.188578–0.200011 | 3 |
| fresh_028 | 0.050555–0.052746 | 0.663657–0.677242 | 0.409435–0.435452 | 0 |

All four failing primary cases in this subset fail all three seeds; all six
successful cases pass all three. Largest within-case range widths are
0.005110 localization, 0.046639 Q and 0.056473 L2. Largest sample SDs are
0.002555 localization, 0.025110 Q and 0.028237 L2 (full per-case SDs saved).
This supports persistence over these three stochastic initializations/sampling
streams, not universal optimizer robustness. Physical source initialization
remained fixed; alternate physical starting positions were not tested.

## K. Oracle fixed-Q identifiability

Predetermined fresh_015 has truth (0.296495,0.747487,Q=1.126001). Diagnostic A
fixes Q to that true value, explicitly an oracle. Its 21×21 location-grid
minimum is **(0.300,0.750)**, localization distance **0.004313**, sensor MSE
**2.638448e-6**. The minimum MSE beyond one source width from truth is
2.490748e-4, and beyond two widths 1.350473e-3. This grid therefore separates
the near-true region from those more distant locations in this single case.

The extra true-location oracle probe has MSE 2.249252e-7. The primary PINN
location (0.297348,0.737677) has fixed-true-Q MSE 1.280201e-5. These are post-fit
diagnostics, not fitting inputs. Fixing Q to truth alone does not establish
joint x,y,Q identifiability.

## L. Observation-profiled-Q identifiability

Diagnostic B uses Q_hat=max(0,<g,observations>/<g,g>) separately at each
location, with unit-strength forward response g and **no true-Q input**.
There is no upper Q bound. Its grid minimum is also **(0.300,0.750)**, with
Q_hat=**1.168623** and sensor MSE **5.232290e-7**. Minimum MSE beyond one/two
source widths is **5.547707e-5/4.820629e-4**. Profiling changes the landscape
and absorbs amplitude mismatch but does not remove location information in
this sampled case.

At the separately labeled true-location probe, Q_hat=1.137580 and MSE
6.014027e-8. At the primary PINN location, Q_hat=1.035924 and MSE 7.807003e-7;
the PINN itself estimated Q=1.006597. These discrepancies do not imply the
profiled parameters are exact truth: this diagnostic uses a 201² solver
against a 641² reference, plus a finite 0.025-spaced location grid.

Neither landscape is a posterior probability map, a continuous optimization
certificate, or evidence that all 30 cases (especially the worst failures)
are identifiable. fresh_015 was index-selected, not chosen for performance,
and happened to meet the primary recovery criterion.

## M. Exploratory error associations

Spearman correlations across all 30 fresh primary revised cases:

| Feature | Localization error | Relative Q error | Relative L2 |
|---|---:|---:|---:|
| True x | +0.721 | +0.610 | +0.698 |
| True y | -0.216 | -0.226 | -0.219 |
| True Q | +0.248 | +0.177 | +0.177 |
| Nearest-sensor distance | +0.308 | +0.477 | +0.329 |
| Distance to domain boundary | -0.495 | -0.499 | -0.490 |
| Observed sensor RMS | -0.809 | -0.876 | -0.838 |

Pearson results are also saved in exploratory_correlations.csv. Higher x,
weaker observed signal and smaller boundary distance are associated with
larger errors in this fixed layout; nearest-sensor distance has a weaker
association. These variables are mutually confounded. Sensor RMS is a joint
plume-capture/strength proxy, **not pure geometry**. This is exploratory,
unadjusted descriptive analysis of 30 cases, not causal evidence or a basis
for changing the frozen model.

## N. Three worst localization-ranked failures

All three finished 12,000 updates with finite outputs. All have zero BC/IC
violations and zero negative predictions, demonstrating that these constraints
alone do not ensure source recovery. All fail both recovery thresholds.

| Scenario | True (x,y,Q) | Predicted (x,y,Q) | Localization | Q error | L2 | RMSE | MAE |
|---|---|---|---:|---:|---:|---:|---:|
| fresh_011 | (0.732916,0.438796,1.255570) | (0.624307,0.378839,0.443303) | 0.124059 | 64.6931% | 77.1663% | 0.044358 | 0.013133 |
| fresh_029 | (0.719544,0.413643,1.192372) | (0.622787,0.365335,0.569182) | 0.108147 | 52.2647% | 67.6638% | 0.037232 | 0.011414 |
| fresh_019 | (0.736204,0.537531,0.973155) | (0.628336,0.532999,0.203241) | 0.107963 | 79.1153% | 84.2932% | 0.037474 | 0.010408 |

- **fresh_011:** PDE RMSE/MAE/max absolute residual
  0.014352/0.004842/0.369832; sensor Richardson estimate 1.04452%, flagged.
  Nearest sensor 0.109523; observed sensor RMS 0.015342. No extra seeds were
  prescribed for this index, so persistence across seeds is unknown.
- **fresh_029:** PDE RMSE/MAE/max 0.016875/0.005152/0.495496; maximum assessed
  reference estimate 0.91644%, target met. Nearest sensor 0.099004; observed
  sensor RMS 0.017551. No additional seeds were prescribed. This serious
  failure is not confined to reference-target misses.
- **fresh_019:** PDE RMSE/MAE/max 0.004795/0.001780/0.080481; sensor Richardson
  estimate 1.50382%, flagged. Nearest sensor 0.135879; observed sensor RMS
  0.009137. All three seeds fail, with Q approximately 0.199–0.211 and L2
  0.834–0.850. Its relatively small PDE residual coexists with the worst Q
  and relative concentration errors; residual size is not a recovery metric.

Mechanically selected illustration IDs are best **fresh_026**, lower-middle
median (rank 15/30) **fresh_016**, worst **fresh_011**. Each reconstruction
shows the t=0.8 reference, prediction, absolute error, both sources and sensors.
Selections use localization only; there was no attractive-example selection.

## O. Supported interpretation

**unreliable controlled recovery**.

The frozen strong category requires >=27/30 recoveries; only 18 recovered.
The conditional category's recovery and concentration/source-improvement
requirements are met, but its mean-PDE-RMSE non-regression condition is not:
0.008600 exceeds 0.007982. This gate is applied literally despite the paired
uncertainty interval crossing zero. The category is an operational predeclared
decision, not a claim that the mean PDE regression is statistically established.

The broader failure distribution reinforces caution: 40% joint failures,
Q underestimated in 30/30, Q-error Q90=53.51%, L2 Q90=68.20%, worst L2=84.29%,
and four persistently failing cases in the seed substudy. Improvement over B0
does not establish reliable inverse recovery. No threshold was revised to
obtain a more favorable label.

## P. Move to sensor sparsity/noise experiments?

**No under the preregistered progression rule.** This idealized zero-noise,
correctly specified baseline remains unresolved: systematic strength bias,
substantial recovery failures, and the failed mean-PDE non-regression gate.
No sparsity or noise experiments were run. Any later model-development work
would require separate authorization and another genuinely untouched test set;
these now-observed 30 cases must not be relabeled as untouched evidence.

## Q. Touch the real Almaty pipeline?

**No.** This study does not justify real source attribution or changes to the
Almaty pipeline. No AirKaz/ERA5 retrieval, real case study, changed wind,
multi-source inversion or sensor-count/noise sweep was performed. Even a
positive result here would support only single-Gaussian recovery under the
tested correctly specified synthetic conditions, not real-emitter discovery.

## R. Tests and integrity checks

- Targeted: **115 passed, 0 failed**. Recheck: 38.45 s.
- Full suite: **374 passed, 0 failed, 120 warnings**. Recheck: 123.18 s.
- Warnings: one NumPy/netCDF4 binary-compatibility RuntimeWarning and 119
  xarray/netCDF4 NumPy-shape DeprecationWarnings (6 inversion, 93 meteorology,
  20 real-case fixture warnings). Passing tests do not establish that the
  compatibility warning is harmless; no dependency was silently changed.
- No existing test was weakened, removed or changed. All benchmark code was
  tested before fresh fits; targeted then full-suite rechecks also passed
  after all primary fits while remaining seed runs executed separately.
- Integrity checks verified all 80 attempts, complete histories/checkpoints,
  exact 780,000 updates, immutable manifests and unchanged execution hashes.
- Independently recomputed all source/full-field metrics and recovery
  decisions for all 80 fits. No NaNs, computational failures or invalidation
  markers occurred. Scientific recovery failures remain in the raw tables.
- One extra historical byte-identity assertion (not a pytest test) failed;
  the documented investigation in B established an inactive instrumentation
  difference, not a numerical/model change. No failing test was weakened.

The new tests cover novelty/nonoverlap, config/code freezes, path/IO guards,
truth isolation, primary/additional seeds, unchanged recovery, failed-row
retention, manifest immutability, observation identity, paired statistics,
profiling linearity, interpretation gates, exact model checkpoint restoration
and figure generation. See saved test_results*.json and postfit_integrity_checks.json.

## S. Files created or modified

Modified only existing **README.md**. Added:

- `configs/fresh_blind_benchmark.yaml`
- `docs/fresh_blind_benchmark_protocol.md`
- `docs/fresh_blind_benchmark.md` (this report)
- `src/inverpinn/data/fresh_blind.py`
- `src/inverpinn/evaluation/fresh_blind.py`
- `src/inverpinn/visualization/fresh_blind.py`
- `scripts/generate_fresh_blind_benchmark.py`
- `scripts/run_fresh_blind_benchmark.py`
- `scripts/fresh_identifiability.py`
- `scripts/report_fresh_blind_benchmark.py`
- `tests/test_fresh_blind_benchmark.py`

Generated, Git-ignored outputs are under `results/fresh_blind_benchmark/`:
scenario_manifest.csv, protocol.yaml, B_revised_raw.csv, B0_raw.csv,
paired_results.csv, seed_sensitivity.csv, summary.json, summary.csv, all_runs.csv,
reference-quality and exploratory tables, locks/hashes/environment snapshots,
30 reference datasets, 80 fit directories and the two identifiability datasets.
All nine requested figures were saved as PNG and PDF:

- fresh_true_vs_predicted_sources
- fresh_localization_distribution
- fresh_Q_true_vs_predicted
- fresh_concentration_error_distribution
- fresh_B0_vs_Brevised_paired
- fresh_physical_diagnostics
- fresh_seed_sensitivity
- fresh_identifiability_fixed_Q
- fresh_identifiability_profiled_Q

Best/median/worst reconstruction PNG/PDFs and selection JSONs are separate.
The benchmark's artifact_hashes.json and report_complete.json lock the generated
results. This prose report was written afterward from those immutable artifacts.
No original model, physics, training, real-data module or model config changed.

## T. Git status and hard stop

Branch: **scientific-revision**. Frozen B_revised commit and current HEAD:
**9dc90c247b0a596b789cb0ee5f1c06f938db40c4**.
Uncommitted changes: README.md modified and the eleven added files listed in S;
generated artifacts remain ignored. No commit was created.

The benchmark is complete. **Stop here**: no tuning, rescue, new seeds,
sparsity/noise/wind/multiple-source experiments or real-data work follows.
