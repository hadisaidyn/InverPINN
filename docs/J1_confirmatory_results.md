# Revision 8: final fixed-hypothesis confirmation

## Decision

**J1 failed the preregistered final confirmation.** It recovered 23/30 cases
(76.67%; 95% Wilson interval 59.07–88.21%), below the required 27/30.
Every other confirmation gate passed. No `InverPINN_final.yaml` was created,
and no robustness scenarios or experiments were generated. The real
robustness entry gate was checked and rejected execution after this decision.
J1, B_revised and the classical inverse code were not modified.

Further small-tweak pure-PINN development stops here. For this controlled
problem, use the classical method as the recovery reference and frame the
research around the PINN's remaining limitations. A hybrid is a separate
possible research direction, not a method validated by this benchmark.
No new candidate, noise experiment, sensor sweep, wind experiment, or real
Almaty/AirKaz/ERA5 work was performed.

## Scientific history and independence

J1 was generated during development but was not promoted under Revision 7's
original selection rule: Q-error improvement was 19.83% and signed-bias
improvement 19.44%, both below the required 20%. Revision 8 evaluates J1 as
a newly frozen hypothesis, not a retroactive pass of that rule.

The initial repository was clean on `scientific-revision`, at
`33892895805d692c8c3960d8646c43d6a4f7536d`. The protocol was frozen at
2026-09-20 13:53:37.582055 UTC, before scenario generation. New seed
2009202608 produced 30 IID scenarios, `final_blind_v1_001` through `030`.
Scenario IDs, source hashes, source tuples and generation seeds were checked
against all 124 previous scenario-plan entries and historical configured
sources. There was no overlap, replacement, exclusion, or convenient redraw.
All 11,001 previous artifact files remained byte-identical.

The complete manifest was frozen at 14:41:56.627314 UTC. The descriptive
sensor-RMS split was frozen at 14:46:39.932597 UTC, before fitting. All 90
primary attempts completed before aggregate results were opened. Sparse-only
fitting interfaces and an I/O guard excluded source truth and consumed
outcomes; terminal attempt/checkpoint records preceded truth-based evaluation.
The guard protects against accidental access; it is not a security sandbox.

Each new scenario is fitted independently from its observations. This is
generalization of a fixed inference procedure to fresh scenarios, not
zero-shot transfer of one pretrained concentration network.

## Controlled problem and unchanged methods

The unit-square problem has one Gaussian source, sigma=0.08, wind
(u,v)=(0.35,-0.15), D=0.005, zero initial concentration and zero concentration
on all four boundaries. Source coordinates are uniform in [0.25,0.75] and
peak intensity Q is uniform in [0.8,1.4]. There are 20 unchanged fixed sensors,
81 observations per sensor at times 0 through 0.8, and no added noise.
Coordinates, Q and concentration use the declared synthetic units; these
numbers are not distances in kilometres or measured PM2.5 concentrations.
Q is peak Gaussian source intensity, not total integrated emissions.

J1 uses the exact executed Revision-7 recipe: three 64-wide tanh hidden layers,
the constrained field

`C = 16*x*(1-x)*y*(1-y)*(t/0.8)*softplus(f_theta)`,

sigmoid source x/y initialized at 0.5, and analytical Q projection in the PDE
loss. Q is a buffer, not an independent Adam parameter. At each PDE batch,

`R0 = C_t + 0.35*C_x - 0.15*C_y - 0.005*(C_xx+C_yy)`

and `Q = max(dtype.tiny, sum(G*R0)/sum(G*G))`, accumulated in float64 with
differentiable projection. There are exactly 12,000 Adam field/location updates
at learning rate 0.001; final Q-only profiling uses 8,192 fitting points.
Seed 42, batch seed 43, terminal-profile seed 44, loss weights, collocation,
constraints and initialization are unchanged. The inherited recipe's
`formulation.Q_parameterization` metadata is retained for exact recipe
comparison; J1 actually uses its separately specified profile buffer.

B_revised uses the original independent softplus/Adam Q parameter and its
unchanged 12,000-update recipe. The classical comparator uses the unchanged
201-square forward solver and three observation-selected starts from a fixed
21-square location bank, with bounded x/y optimization and observation-only
nonnegative Q profiling. It is not part of J1. All three methods use identical
observation hashes per scenario. There were no truth-selected retries,
extensions, per-case tuning, or best-of-seeds results. All 30 selected classical
optimizations reported convergence.

## Reference quality and evaluation

All references used 641-square grids, dt=0.00003125, with 161-square and
321-square refinements for Richardson error estimation. The combined explicit
stability numbers were 0.296, 0.276 and 0.266, respectively. Three scenarios
missed the approximate 1% reference target and were retained unchanged:

| Scenario suffix | Maximum estimated reference error |
|---|---:|
| 016 | 1.0631% |
| 017 | 1.1839% |
| 027 | 1.1264% |

These estimates are not certified mathematical bounds. All three are also
J1 recovery failures; no causal attribution is justified by that coincidence.
The three worst localization failures all met the reference target. No
reference flag changed a denominator or the decision.

Field errors equally weight all 641-square nodes at t=0.1,0.2,0.4,0.8; they
are not a continuous-time integral or a forecasting endpoint. Classical
201-square fields are bilinearly mapped onto the same reference grid, keeping
their numerical mismatch. Independent PINN physical diagnostics use 8,192
interior, 4,096 initial and 1,024-per-edge boundary points, seed 2009202808.
These are finite-point diagnostics, not global residual-error bounds.

## Preregistered gates

Every comparison used unrounded values. All gates were required.

| Gate | J1 outcome | Decision |
|---|---:|---|
| Joint recovery >=27/30; localization <=0.08 AND Q error <=20% | 23/30 | **Fail** |
| Median localization <=0.02 | 0.00557372 | Pass |
| Median absolute relative Q error <=10% | 5.97489% | Pass |
| Median field relative L2 <=15% | 7.42279% | Pass |
| Absolute median signed relative Q error <=10% | 5.97489% | Pass |
| No universal Q underestimation | 28 under, 2 over, 0 equal | Pass |
| Negative fraction =0 | 0 in every case | Pass |
| Maximum BC violation =0 | 0 in every case | Pass |
| Maximum IC violation =0 | 0 in every case | Pass |
| Median PDE RMS <=1.10 times paired B_revised median | Ratio 0.963762 | Pass |
| All 90 runs finite and evaluable | 90/90 | Pass |

Passing the median and physical gates does not override failing reliability.

## Full J1 distributions

SD is sample standard deviation (ddof=1). All 30 observations are included.
Relative errors below are percentages; absolute RMSE/MAE use synthetic
concentration units.

| Metric | Mean | Median | SD | Q25 | Q75 | Q90 | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| Localization | 0.0176055 | 0.00557372 | 0.0234573 | 0.00140951 | 0.0258768 | 0.0486069 | 0.0840810 |
| Absolute relative Q error (%) | 13.1006 | 5.97489 | 16.9018 | 0.810684 | 17.5067 | 34.7829 | 62.1828 |
| Concentration relative L2 (%) | 16.6601 | 7.42279 | 19.2635 | 2.33296 | 27.2187 | 47.5348 | 69.7244 |
| Concentration RMSE | 0.00839844 | 0.00388203 | 0.00998459 | 0.00111482 | 0.0155640 | 0.0211724 | 0.0409981 |
| Concentration MAE | 0.00265414 | 0.00137025 | 0.00286691 | 0.000511297 | 0.00484545 | 0.00632178 | 0.0115597 |

Signed Q error has mean -0.140638 and median -0.0544769. Signed relative Q
error has mean -13.0565% and median -5.97489%. With the frozen absolute equality
tolerance 1e-6, 28/30 are underestimates and 2/30 overestimates. Universal
underestimation disappeared in its literal sense, but a strong directional
bias remains. It would be misleading to call Q bias resolved.

J1 PDE RMS mean/median/worst are 0.00619746/0.00536071/0.0160324;
PDE MAE mean/median/worst are 0.00264281/0.00242715/0.00531549.
B_revised median PDE RMS is 0.00556228. Both PINNs have zero negative fraction,
minimum concentration zero, and exact zero IC/BC RMSE and maximum violation
in every evaluated case. Physical validity did not guarantee source recovery.

## Paired J1 versus B_revised

Differences are J1 minus B_revised; negative error differences favor J1.
Intervals are percentile paired-bootstrap intervals for the mean, using
10,000 resamples of the same scenario indices and seed 2009202818. A tie
requires absolute difference <=1e-8+1e-6*max(abs(a),abs(b)). No ties occurred.
The raw paired CSV and summary JSON also retain median differences and their
intervals. These intervals quantify scenario sampling, not optimizer-seed or
reference-discretization uncertainty. They are not a substitute for the gate.

| Metric | Mean paired difference [95% CI] | Median paired difference | J1 better / B_revised better |
|---|---|---:|---:|
| Localization | -0.00224736 [-0.00398102, -0.000765816] | -0.000229319 | 18 / 12 |
| Q error, percentage points | -3.09264 [-5.09282, -1.30178] | -0.862445 | 21 / 9 |
| L2, percentage points | -2.79808 [-4.65149, -1.19925] | -0.526978 | 21 / 9 |
| RMSE | -0.00144319 [-0.00239743, -0.000627678] | -0.000268147 | 21 / 9 |
| MAE | -0.000414340 [-0.000695332, -0.000179803] | -0.000143487 | 22 / 8 |
| PDE RMS | -0.000523230 [-0.00124102, 0.0000180964] | -0.000354888 | 20 / 10 |
| Fitting seconds | +2.09921 [0.636206, 3.58393] | +2.66552 | 8 / 22 |

Median localization changes from 0.00918550 to 0.00557372, median Q error from
9.31930% to 5.97489%, and median L2 from 11.0475% to 7.42279%. Recovery changes
from 21/30 to 23/30: two B_revised failures become recoveries, seven failures
persist, and no B_revised recovery is lost. The paired-median localization
difference interval is [-0.00232231,0.000142436], including zero; mean PDE
improvement's interval also includes zero. Improvement is heterogeneous and
insufficient for the required 90% reliability.

## Gap to classical inversion

Classical inversion recovered 30/30 (95% Wilson interval 88.65–100%). Its
median localization, Q error and L2 are 0.00111123, 1.21779% and 1.50665%.
Its worst values are 0.00364163, 3.53686% and 4.47401%, respectively.

| Metric | Mean paired J1-minus-classical gap [95% CI] | Median paired gap | J1 better / classical better |
|---|---|---:|---:|
| Localization | +0.0161671 [0.00862294, 0.0249489] | +0.00366625 | 6 / 24 |
| Q error, percentage points | +11.6362 [6.22179, 17.8735] | +4.65680 | 11 / 19 |
| L2, percentage points | +14.7550 [8.55751, 21.7511] | +5.29728 | 2 / 28 |

Mean/median fitting times are J1 93.4053/93.4479 seconds, B_revised
91.3061/91.4081 seconds, classical 64.9193/64.5720 seconds. The mean/median
paired J1/classical fitting-time ratios are 1.45095/1.46607. Classical's shared
441-response bank costs 214.845 wall seconds with two workers, or 408.093
summed forward seconds. Amortizing the latter over 30 cases adds 13.6031
seconds per classical case; the mean/median paired ratios then become
1.19638/1.21180. Timings include fitting/logging overhead but exclude shared
reference generation, evaluation and audit; they are hardware-specific.
Classical was more reliable here, not proven universally superior.

## Worst cases and weak-information association

The three worst examples were selected mechanically by descending localization
error, not appearance. All have exact IC/BC, nonnegative predictions, stable
finite-difference sensitivities and full local Jacobian rank 3.

| Scenario suffix | True (x,y) | Predicted (x,y) | Q true → predicted | Localization | Q error | L2 | Sensor RMS | Smallest scaled Jacobian singular value | PDE RMS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 030 | (0.711669,0.479887) | (0.630920,0.456452) | 1.269085 → 0.479933 | 0.0840810 | 62.1828% | 69.7244% | 0.0218562 | 0.0609344 | 0.0111420 |
| 008 | (0.658364,0.731122) | (0.582101,0.757987) | 0.968013 → 0.634683 | 0.0808564 | 34.4345% | 50.6798% | 0.0156058 | 0.161819 | 0.00996649 |
| 024 | (0.653230,0.608690) | (0.598141,0.612555) | 0.884656 → 0.549207 | 0.0552244 | 37.9186% | 47.1854% | 0.0313613 | 0.121303 | 0.00868334 |

Case 024 passes localization but fails Q. Cases 030 and 008 fail both. All
seven joint failures fail Q: 008,012,016,017,024,027,030. Only 008 and 030
also exceed the localization threshold. They are retained in full.

The predefined median signal threshold is 0.04957048018653265. The
lower-information group contains the 15 cases at or below it. This split
uses observations, not PINN performance. All seven J1 failures occur there.

| Method / signal group | Recovery | Median localization | Median Q error | Median L2 |
|---|---:|---:|---:|---:|
| J1, higher | 15/15 | 0.00139886 | 0.793034% | 2.77238% |
| J1, lower | 8/15 | 0.0251111 | 14.5908% | 27.5493% |
| B_revised, higher | 15/15 | 0.000982027 | 0.790614% | 2.01686% |
| B_revised, lower | 6/15 | 0.0352062 | 25.9641% | 38.6707% |
| Classical, higher | 15/15 | 0.00104126 | 1.24067% | 1.09310% |
| Classical, lower | 15/15 | 0.00147544 | 1.01577% | 2.09010% |

J1 improves the weaker group but does not resolve its fragility. It is not a
uniform improvement: the higher-signal group's typical errors are slightly
worse than B_revised. All 30 reference Jacobians have rank 3 and stable
finite-difference checks. The worst three cases have no sensors in the
previously defined plume corridor. Signal, geometry and singular-value
associations are descriptive, not causal; local rank is not a global
identifiability proof, and inverse-information proxies are not posterior
uncertainties. Classical recovery shows these observations were sufficient
for that method under this correctly specified problem.

## Supported claims and stopping boundary

J1 can recover many single Gaussian sources under these idealized synthetic
conditions, but has not met the predefined reliability threshold even with
correct wind, zero noise and 20 fixed sensors. Median performance conceals
a substantial low-signal tail. The physical constraints alone do not prevent
large source-amplitude and concentration errors.

No robustness boundary has been measured in Revision 8: fewer sensors,
measurement noise and wrong wind were not tested because the confirmation
gate failed. There is no support here for real-emitter identification,
arbitrary sources, multiple sources, unknown meteorology, real noise,
causal claims, or general PINN superiority. A 30-case, single-seed synthetic
benchmark is not real-world validation. The project is not ready to advance
this method to an exploratory Almaty case study under the stated gate.

The frozen failing hypothesis and all artifacts remain available. Do not
start Revision 9 with another small PINN tweak or select a rescue on these
consumed scenarios. No validated final core model is promoted.

## Audit, tests and artifacts

The independent audit recomputed source and dense-field metrics, physical
diagnostics, optimizer configuration and exact update counts. It checked
checkpoint/source consistency, classical start selection and an independent
classical forward reconstruction. All 90 runs passed; maximum numerical
metric discrepancy was 1.44329e-14. There were zero computational failures,
but seven J1 scientific recovery failures. These are distinct endpoints.
All 97 archived execution files and 11,001 historical artifacts were unchanged.

Final targeted suite: **184 passed, zero failures, zero warnings** (59.44 s).
Final full suite: **451 passed, zero failures, 120 warnings** (139.91 s).
The existing dependency warnings comprise one NumPy/netCDF4 binary-size
RuntimeWarning and 119 xarray/netCDF4 NumPy shape deprecation warnings from
synthetic inversion/meteorology/case-study fixtures. The warning is retained,
not suppressed or silently fixed; no real-data pipeline was run. Matplotlib
also reported an unwritable default cache and used temporary caches. No tests
were weakened, and no frozen model/dependency was changed to affect outcomes.

`results/final_confirmatory/` contains the manifest, protocol/locks, three raw
tables, paired tables, summaries, observability strata, worst cases, all
checkpoints, histories, gradients, per-run configs, reference arrays, provenance
and package archives, audit, JUnit reports, and seven required PNG/PDF figures.
Three mechanically selected worst-case reconstruction figures are inside
their J1 run directories. All failed recoveries and reference flags remain.
Large/generated artifacts remain local and git-ignored, following repository
conventions; source, configuration and this report are version-controlled.

The conditional robustness implementation is present and tested but was not
executed. `results/final_robustness/` and `configs/InverPINN_final.yaml` do not
exist. The reproducible commands are in the README; existing experiment outputs
are rejected rather than silently overwritten.

## SHA256 provenance

| Artifact | SHA256 |
|---|---|
| J1_confirmatory.yaml | `a963da9f9a3e1f4f11bea9b1ce9162d65223ec199e162724fe8da56ee73d3e0a` |
| B_revised.yaml | `3e400ef1195cf90f6481117ae5c395695cca23a12d47ab532ec6822e400c332e` |
| J1 training code | `99653e0446dad4215ea16ef1b320de4802c498e2bf4c038b90507531070e1893` |
| Profiled-source code | `d9a33cf02c4cd60fb90a2fa3f967f4dc0446691b2f6df60eedeef1a12e8e1ac7` |
| MLP code | `4efaed4291274f05164b908bc4617d8ad92204c75b8399906e11f9337a8027a6` |
| Constrained-concentration code | `f923f04e94d9466673aff172c228432cfafe1326b4c515eb93de31dae9abf6b4` |
| Classical observability/inverse code | `05020915a105d6ae821af14b49987f1dfd6e5f5db359a3f042098f23fbac1edd` |
| Reference solver code | `171d63443cc789534402b08ad497b17f18fbb113e9530d38c6ffedd9ae77b8b8` |
| Protocol object | `e1c3e2b575e9034d682504b21809ae29bf7daa69d6fd2363aa5612674ae01418` |
| Protocol document | `8eefe463e461a410ac4364ca28882093f5784ed2a6ab9b1c262989cebc515977` |
| Scenario plan | `65b8bdf9a4c0bece0d8afcec037d44771a9f3271ae6dcc2db9079b46e2a2866d` |
| Manifest CSV | `8fa5480ef5aabd50300e548c27ec355af8315568f52a6dfd5c640d3044ee74b9` |
| Sensor layout | `c0533d4c030bc4ce020eaaebf6531a19453d36b5e322e6c63eb9d27baef2ebd2` |
| Fitting source archive | `a6dc5072aeb377ec4408b8001020d9a662e64a077b3585ee2fb34ab1082a10e1` |
| J1_raw.csv | `17917bd23df4808e0285227f334ff574d797934b3d1f3f6c2e8b648a32f1890f` |
| B_revised_raw.csv | `f65eb96bfde8830049aff098873d4154e082f5c95e062ba1bf8c6b7234f2127a` |
| classical_raw.csv | `00253a7905c4ce07fb6f969449eeebaa308096bf2c6438e9248a40662c5e3fd8` |

The full per-file hash inventory, original Git revision, environment and
package versions are preserved with the experiment. The Revision-8 commit is
the commit introducing this completed report; it does not change the frozen
Revision-7 scientific code. Karpathy's coding guidelines informed the use of
small guarded adapters and explicit checks, rather than changes to the
already frozen trainers.
