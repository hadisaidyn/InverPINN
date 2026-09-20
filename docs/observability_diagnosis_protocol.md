# Revision 6: prospective diagnostic_v2 protocol

This is diagnostic development, not another blind test or model selection.
Revision 5 was committed cleanly as
590c3faf0c6ec71677ca89b3093ea5a7c43116a7 before this work. B_revised and all
original physics/training code remain frozen at their recorded hashes. The
new procedures do not tune or replace any PINN, optimizer, loss or sensor.

## Design, exclusions and ordering

Before generation freeze this document and configs/observability_diagnosis.yaml.
Use new seed **2009202606**, absent from earlier tracked configs/code. A plain
NumPy Latin hypercube independently permutes 30 strata and jitters each stratum
for each of x,y,Q; rows remain in generated index order, with no design
optimization or performance filtering. Bounds are [.25,.75], [.25,.75], [.8,1.4].
IDs diagnostic_v2_001 through diagnostic_v2_030 are all retained. Per-case
reference RNG seeds come from SeedSequence(seed).spawn(30), independently of
the LHS generator. Compare source/seed hashes against all 70 consumed cases
reconstructed from frozen sampling rules, not their data/outcomes.

Preserve consumed results using before/after SHA-256 inventories; inventory
reads hash opaque bytes without parsing individual outcomes. Diagnostic
workers refuse old result paths. Joint classical/PINN fitting receives only
input tensors, known physics, known width and frozen settings, never source
truth. Explicit oracle routines are separately named and labeled. Freeze both
truth-bearing and input-only manifests before starting diagnostics.

Same sigma=.08, u=.35, v=-.15, D=.005, unit square, zero IC and all-edge zero
Dirichlet BC, 20 unchanged continuous sensor coordinates, 81 measurements over
[0,.8], and zero added noise. Q is peak concentration/time, not total emissions.
The unchanged three-grid 161²/321²/641² reference pipeline retains fields at
.1,.2,.4,.8. Always select 641²; flag finite Richardson misses rather than
replace cases. Undefined or nonfinite references require explicit review.
Richardson estimates are conditional, not certified bounds.

Complete conditional-Q, resolution, Jacobian, map, geometry, classical inverse,
prospective signal-correlation and reference-envelope diagnostics before any
B_revised fitting. Hash-lock that information-stage output. Only then run
exactly one frozen 12,000-update seed-42 B_revised fit on each new case.
No consumed-case fit, PINN multistart, rescue, calibration or new model occurs.

## Conditional Q and amplitude diagnostics

With fixed source location and zero IC/BC, the explicit PDE recurrence and
bilinear/time-linear observations are linear maps. S=Q*S_unit, hence y(Q)=Q*h.
All 1620 sensor/time entries have equal weight (including initial zeros).
For nonzero h, Q_hat=max(0,h^T*y/(h^T*h)) minimizes mean((Q*h-y)^2) for Q>=0.
There is no upper Q prior bound. The zero solution is an admissible limiting
case of positive strength, not a clipped neural parameter. Record analytic
and scalar numerical least-squares estimates, initialized at Q=1.1, and their
disagreement. No true Q enters either estimator; true x/y are oracle inputs.

At each true x/y/Q independently simulate the actual source, and compare this
direct forward result to Q_true*h to check numerical linearity. Save both
sensor vectors. Define amplitude ratio a=<y_ref,y_inference>/<y_ref,y_ref>:
a>1 means the inference prediction projects too strongly along the reference
vector; a<1 means too weak. Also report RMS ratio, relative observation L2,
mean signed concentration difference and normalized signed mean difference.
Neither ratio alone describes plume shape mismatch. Q_hat/Q_true is not
necessarily 1/a when shapes differ.

Compare independently run inference grids 101² (dt=.001), 201² (.00025) and
321² (.000125), all ending at .8 with the same physical sensors/times and
recomputed interpolation weights. Report signed/absolute Q errors, sensor
mismatch and runtime per resolution and case. No 641² matched-grid inverse
shortcut is used. The 321² reference-qualification run is never substituted
for an inference call. Positive/negative bias counts use 1e-8+1e-6*abs(Q_true).

## Scaled Jacobian and local information

Use CPU float64 forward outputs flattened in time-major, then sensor order.
theta=(x,y,Q), z_j=theta_j/s_j, s=(.5,.5,.6), and normalized concentration
y/.88. These fixed prior-range scales are not adapted to case truth/signal.
J_scaled=(d y/d theta)*diag(s)/.88. x/y columns use central differences with
normalized steps .001,.0005,.00025; Q derivative is exactly the unit response.
Keep .00025 prospectively. Compare J and its minimum singular value across
steps; flag relative J differences >.001 or smin differences >.01, without
silently changing the step. Source perturbations may extend outside the
source-sampling box but remain legitimate Gaussian forcing evaluations.

Report all singular values, rank, condition number, scaled column norms and
F=J_scaled^T J_scaled. Under hypothetical independent unit-variance noise in
normalized readings, F is an information diagnostic and inv(F) a local
covariance proxy in z units. No noise is added. Report eigenvalues, condition,
logdet and marginal standard-deviation proxies via SVD; mark rank deficiency
as undefined/infinite instead of adding a hidden ridge. Also report effective
Q information after projecting the Q column off x/y columns. Do not call any
quantity a real posterior or use unscaled condition numbers to compare units.
Temporal independence and local linearity are assumptions of the proxy, not
verified atmospheric facts. Absolute smin and relative conditioning answer
different questions: low signal can reduce information without poor condition.

## Observability maps and classical inverse

Use a fixed 21×21 location grid on [.25,.75]², representative Q=1.1, unchanged
201² forward solver and selected finite-difference step. Map sensor RMS,
Q/x/y sensitivity, scaled smin, condition and effective Q information. Every
map point uses candidate coordinates, never scenario source truth. Overlay
unchanged sensors and preserve unit responses/Jacobians in NPZ.

Classical joint inversion profiles Q analytically at each candidate location;
this variable projection is mathematically equivalent to joint least squares
under the stated linearity and Q>=0 constraint. Optimize location over [0,1]²,
not the tighter truth sampling box, with scipy.optimize.least_squares.
Predefine three starts: rank the fixed map's candidate locations by observation
MSE after profiling Q and greedily take three separated by >=.08. Thus the
candidate grid and rule are frozen; no source truth selects starts. Do not
count a truth-selected restart. Preserve all starts and choose only by the
smallest finite sensor MSE, with index tie-break. No optimality certificate is
claimed. Use max_nfev=60, ftol=1e-10, xtol=1e-8, gtol=1e-10. The location
residual derivative includes the derivative of profiled Q, with x/y unit
responses differentiated at the same fixed finite-difference step.

Record every start/status/estimate, source errors, sensor fit, runtime and
concentration errors. Compare classical fields bilinearly evaluated onto the
641² reference grid at the same four times; do not downsample reference truth.
The 201² solution is never compared to a reference generated by that same
numerical matrix. Source errors use the unchanged joint criterion .08/.20.

## Geometry and prospective associations

For sensor displacement d=sensor-source, define w=(u,v)/sqrt(u²+v²),
downwind a=d.w, crosswind b=abs(d.(-w_y,w_x)), and travel time tau=a/|wind|.
Report downwind count (a>=0), nearest Euclidean downwind distance, minimum
downwind crosswind distance, count with 0<=tau<=.8, and count also satisfying
b<=sqrt(sigma²+2D*tau). This is a one-standard-width kinematic corridor, not
an opaque optimized coverage score or a hard physical plume boundary.
Also report closest sensor, distance to domain edge, observed RMS/max, and x/y/Q.

Before PINN results, report Spearman correlations of reference sensor RMS
with classical location/Q error and scaled conditioning/smin. After frozen
PINN fits, relate its errors to the same metrics, effective Q information,
conditional-Q bias and classical outcome. Compare absolute correlations of
geometry and x/y with observability; this is descriptive, not out-of-sample
prediction selection. No multiplicity-adjusted causal conclusion is made.

## Envelope diagnostics, not a redesign

Use all 30 predetermined smooth reference solutions at all four retained
times, not selected failures. E=16*x*(1-x)*y*(1-y)*t/.8 and required
g=C/E=softplus(f) are evaluated only where E>0. Exclude, count and label the
zero-E edges; do not divide by an epsilon. Stable inverse softplus for g>0 is
f=g+log(-expm1(-g)); g=0 requires the -infinity limit. The target transform
gain dC/df=E*(1-exp(-g)). Report extrema/quantiles of g, f and gain, both over
all interior nodes and the predefined active mask C>=.01*max(C) per snapshot.
This separates background-driven tiny gradients from signal-carrying regions.
Plot all-case position/time trends and predetermined diagnostic_v2_015 maps.

After fitting, evaluate actual raw f, softplus(f), E*sigmoid(f), and spatial/
temporal gradients of f and C on a fixed 81² grid at all four times. Autograd
is diagnostic only; do not take optimizer steps or modify parameter gradients.
Compare required/actual latent scales and associations with errors. Finite
latent range does not prove approximation is easy; a small transform gain is
not proof of training failure. No envelope replacement is permitted.

## Evidence decomposition and limitations

Keep numerical diagnostics continuous. For descriptive relative-coverage flags
only, weak information means smin or effective Q information below its fixed
map's 25th percentile, computed before PINN results. These are map-relative
labels, not universal non-identifiability thresholds or recovery criteria.

Report nonexclusive evidence flags: A classical succeeds/PINN fails; B both
fail with relatively weak information; C localization passes but Q fails with
relatively weak Q information; D conditional Q error exceeds the existing
20% threshold at 201² and falls at 321²; E both succeed. Allow uncertain.
For primary failure attribution, A supports PINN-specific limitation; a
classical-only failure with D supports numerical mismatch for that comparator;
both-failure/weak-information cases remain mixed/uncertain, with possible
observation limitation, rather than asserting intrinsic impossibility. Report
fractions and overlaps without forcing causal identification from correlations.

Importantly, the PINN fits a continuous autograd PDE residual and does not
call the finite-difference inference solver. A biased 201² forward map could
explain classical-Q bias but cannot alone establish a causal path into PINN Q.
The Q_true -> conditional -> classical -> PINN table is an evidence comparison,
not automatically a causal chain. Classical success demonstrates recoverability
for that solver/optimizer and zero-noise observations, not robustness to noise.

Save exact config/code/environment/Git hashes, inputs, references, per-run
checkpoints/estimates, CSV/JSON metrics and PNG/PDF figures. Refuse silent
overwrite, retain numerical failures, and block PINN until information-stage
hashes are frozen. Test the contracts and mathematics, then run the full suite.
Report Observation, Hypothesis, Diagnostic, Evidence, Interpretation and
Limitation separately. End with the requested A–R report. No new model,
sensor/noise/wind sweep, new blind validation, sensor redesign or real-data work.
