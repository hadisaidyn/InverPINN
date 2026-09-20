# P0 revision 5: prospective fresh, blind, single-source validation

This is a confirmatory benchmark, not model selection. The initial repository
was clean on `scientific-revision` at
`9dc90c247b0a596b789cb0ee5f1c06f938db40c4`. The frozen B_revised file has SHA-256
`3e400ef1195cf90f6481117ae5c395695cca23a12d47ab532ec6822e400c332e`, identical to
the revision-4 archived copy. The original B0 config SHA-256 is
`5f85e130900be9295bc48b3f152772a8f456b1a4b4b77f076258e1f6d8f30254`.
No frozen source, model, optimizer, parameterization or training setting changes.

## Prospective freeze and independence

The protocol document and YAML, original executable source hashes, model
recipes, fixed sensors and package versions are archived before generating any
fresh source. New master seed **2009202605** is distinct from prior development
1909202601 and test 1909202602 and absent from the prior tracked configs/code.
NumPy SeedSequence spawns 30 child seeds; each child independently samples
uniform x,y in [.25,.75] and Q in [.8,1.4], in that order. Every draw is retained.
IDs are fresh_001 through fresh_030. New source hashes and child seeds must be
unique and disjoint from the prior deterministic source hashes/seeds.

The old hashes are reconstructed from frozen sampling rules in version-controlled
configuration, **not by opening prior data, truth, metrics or outcomes**. This
is a duplicate-exclusion check, not a selection procedure: collision stops the
benchmark rather than silently replacing a draw. A process-level path guard
denies reads/writes anywhere under results except this fresh benchmark. A fresh
ID/path allowlist rejects old IDs and symlink aliases. These guards prevent
accidental access, not a malicious OS-level attack.

The truth-bearing manifest includes all 30 source coordinates/strengths, seeds,
reference and observation hashes, solver commit/hash, grid/dt, sensor hash and
accuracy flags. A separate input-only manifest contains no source truth. Both
are hash-locked before fitting; validation refuses changed files. No scenario
is replaced, even if difficult or below the reference-accuracy target.

## Controlled physics, measurements and reference qualification

One peak Gaussian S=Q exp(-((x-xs)^2+(y-ys)^2)/(2*.08^2)); unit-square domain;
u=.35, v=-.15, D=.005; C initially zero and homogeneous Dirichlet values on all
four edges. Units remain synthetic length/time/concentration; Q is peak
concentration/time, not integrated emissions. No added measurement noise.
Use exactly the original 20 continuous sensor coordinates from
`configs/single_source_sensors.json`, and 81 times from 0 to .8 inclusive.

Use the unchanged validated CPU float64 upwind-advection/centered-diffusion,
forward-Euler reference pipeline, with 161²/321²/641² grids and respective
dt=.0005/.000125/.00003125 through t=.8. Always use 641², never outcome-dependent
grid selection. Retain four fields at t=.1,.2,.4,.8 and all sensor readings.
The three-grid Richardson target is <=1% at each retained field and the pooled
sensor trajectory, conditional on observed positive convergence order. Report
each component, null estimates and every target miss; this is not a certified
bound, individual-sensor bound or spacetime supremum. Nonfinite/unstable
references or undefined convergence require preserved failure records and
scientific review before fitting, not automatic replacement. A finite defined
target miss is flagged and retained; it does not silently exclude the case.

## Frozen models and blindness during inference

Run B0 and B_revised once each per case using primary optimization seed 42.
B0 uses its original 6000-update sparse-only API. B_revised uses the exact
revision-4 candidate fitting function, its hard nonnegative envelope, sigmoid
locations, softplus Q, fixed (.5,.5,1.1) initialization, configured scaling and
12000 Adam updates. Read recipes from frozen copies, not hand-reimplemented
optimizers. Hash all original model/physics/training dependencies against the
frozen Git commit before generation and before/after fitting. Archive benchmark
execution code before the first fit; no model change is permitted afterward.

Each fitting interface receives only sparse tensors, known transport, known
sigma and allowlisted training settings. While fitting, a Python I/O guard
denies fresh truth, scenario plans and truth-bearing manifests. Logging sees
only losses and estimated parameters. Each attempt saves exact config, every
update, checkpoint/optimizer, runtime and a terminal record. Evaluation may
open truth only after this terminal barrier. No epoch selection, extra updates,
per-case tuning or truth-selected retry is allowed. Additional seeds are
separate labeled runs and cannot overwrite the primary run.

All attempts, including failures, remain in raw output. Resume can run only
unstarted jobs; an interrupted existing attempt becomes a retained failure,
never a retry. Any bug potentially invalidating benchmark inference requires
preservation, explicit documentation and an invalidation decision; never patch
model code and quietly continue. Postprocessing implementation may complete
after reference generation, but must implement this pre-existing protocol and
cannot change it after seeing outcomes.

## Outcomes, numerical ties and uncertainty

Primary continuous outcomes are Euclidean localization error, absolute relative
Q error and concentration relative L2. Dense concentration errors equally weight
all nodes of all four 641² snapshots, matching revision 4, not plot previews.
Also report RMSE/MAE. No clipping. The **unchanged secondary** joint success
criterion is localization <=.08 AND relative Q error <=.20.

Independent physical diagnostics use RNG seed **2009202699**, independent of
training streams: 8192 interior, 4096 initial and 1024 points per edge; report
PDE RMSE/MAE, BC/IC RMSE and maxima, dense negative fraction and minimum C.
Independent random sampling is not a continuous-domain error bound or a
guarantee of zero coincident floating-point coordinates.

Each model reports mean, median, sample SD (ddof=1), q25/q75/q90 (linear
quantiles), worst and finite/missing counts. Failures remain in the denominator
30 for recovery and Wilson 95% intervals, never silently imputed as accurate
values. Continuous summaries report available-case statistics with missing
counts; failure prevents favorable interpretation. The paired table retains
all IDs and verifies identical observation hashes. Deltas = revised - B0.
Numerical ties satisfy |delta| <= 1e-8 + 1e-6*max(|a|,|b|); otherwise the smaller
error wins. Undefined pairs are missing, not tied. This tolerance is numerical,
not an assertion of scientifically equivalent error.

Use 10000 paired bootstrap resamples of scenario indices, seed **2009202617**,
for percentile 95% intervals of mean/median deltas. Report complete-pair counts;
no p-value headline or multiplicity-adjusted causal inference. All 30 scenarios
share sensor geometry, physics and training seed: intervals describe variation
over the specified source distribution conditional on those settings.

Q-bias: report signed Q_pred-Q_true in concentration/time and signed relative
error, mean/median, and under/equal/over counts using tolerance
1e-8+1e-6*|Q_true|. Include an identity-line scatter. Never recalibrate Q.

## Prespecified seed-sensitivity substudy

For fresh_001,004,007,010,013,016,019,022,025,028 only, B_revised also uses seeds
314 and 2718: **three total seeds including 42**, not three extra. Only neural
initialization and stochastic sampling change; physical source initialization
remains fixed. Report all 30 rows independently, within-case location/Q/L2
ranges and sample SD, and the number of seeds meeting the unchanged criterion.
The ten primary rows stay in primary analysis. No best-seed inference, headline
or truth-selected restart. This tests optimizer-seed sensitivity, not alternate
physical source starting positions.

## Two identifiability diagnostics, fresh_015 only

Predetermine a 21×21 location grid over [.25,.75]² and use the unchanged 201²
forward observation operator (dt=.00025), distinct from the 641² reference.
At each location compute the unit-Q sensor trajectory g. Linearity of the PDE,
homogeneous IC/BC and bilinear observations gives simulated readings Q*g.

A: mean((Q_true*g-y)^2), explicitly an **oracle fixed-true-Q** diagnostic.
B: Q_hat=max(0,<g,y>/<g,g>), then mean((Q_hat*g-y)^2), using only observations
and known physics. No upper prior bound is imposed; report Q_hat, including a
zero infimum if encountered. Zero/nonfinite <g,g> is a failure, not a fabricated
minimum. The profiled landscape is not seeded by or conditioned on true Q.

After both grids, separately evaluate the true location and primary PINN
location with both definitions, clearly labeling these post-fit point probes.
Report minima and mismatch outside one/two source widths relative to truth,
without using them to modify a fit. Discrete-grid minima and coarse-forward
model error limit interpretation. Neither landscape is a posterior probability
map or proof of global joint identifiability.

## Exploratory associations and figures

Compute Pearson/Spearman correlations of localization/Q/L2 errors with true
x,y,Q, nearest-sensor Euclidean distance, source-boundary distance, and observed
sensor RMS (a defensible plume-capture/strength proxy, not pure geometry).
Label associations exploratory and noncausal; no feature-based model changes.

Produce all nine requested aggregate/seed/landscape figures as PNG and vector
PDF. Best, median and worst B_revised examples use localization rank only,
ties by scenario ID, missing localization last; median is lower middle rank
15 of 30. No attractive-case selection. The three worst failures use this same
ranking, with full source, concentration, physical, reference and seed evidence.

## Frozen interpretation framework

The following are transparent operational reporting categories, not new
per-case success thresholds, physical laws or tuned scores. The .08/.20
secondary recovery criterion never changes. First require 30 complete finite
primary evaluations, zero negative predictions and exact zero IC/BC for the
revised model. Any violation prevents strong/conditional claims.

**Strong controlled recovery:** at least 27/30 joint recoveries (90%, failures
limited to <=3), median L2 <=.20, and mean PDE RMS no worse than paired B0.
Report outliers/maxima regardless; the category is not perfect recovery.

**Conditional controlled recovery:** if strong is not met, at least 18/30 joint
recoveries (60%, a clear majority), no worse mean/median localization/Q/L2 than
B0, at least 10% relative median improvement in both Q and L2, and no worse
mean PDE RMS. These conditions demand material paired aggregate improvement
alongside a majority of recoveries, not merely a better mean.

**Unreliable controlled recovery:** otherwise, including substantial failure
rates despite favorable mean changes. Undefined/seriously invalid references
are separately flagged as benchmark validity failures and cannot justify a
positive scientific claim. Maximum-case PDE regressions and persistent seed
failures are reported even when aggregate gates pass.

Proceeding to sparsity/noise requires strong or conditional recovery; otherwise
the controlled baseline remains unresolved. No outcome here justifies real
Almaty source attribution or modifying that pipeline. The maximum claim is
single-Gaussian recovery under these correctly specified synthetic conditions.

## Stop and preservation

Save protocols, source/environment snapshots, manifests, raw rows, paired
results, seeds, metrics, plots and checkpoints without silent overwrite.
No prior result directory is evaluated. No sensor/noise/wind/multiple-source
sweep, real-data work, model rescue or Git commit is part of this milestone.
Stop after the final A–T report; leave benchmark code/docs changes uncommitted.
