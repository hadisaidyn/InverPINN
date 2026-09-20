# Revision 6: diagnostic_v2 identifiability and observability report

## Observation

The fixed observation system supports substantially more accurate recovery
than the frozen PINN achieved. Classical inversion recovered **30/30** new
diagnostic sources; B_revised recovered **21/30**. All nine PINN failures were
classical successes. B_revised underestimated Q in **30/30** cases; conditional
and classical inversion did not show universal underestimation.

This is a diagnostic development study, not another blind validation or a
model-selection cycle. It does not revise the frozen Revision-5 interpretation.
No model, loss, optimizer, initialization, update count, sensor, noise, wind or
real-data pipeline was changed. No failed recovery was rescued or omitted.

## Hypothesis

Revision 5's aggregate results motivated four distinguishable possibilities:
insufficient observation information; numerical forward-model amplitude bias;
limitations specific to the PINN formulation/training; and interactions with
the hard-constraint concentration envelope. These hypotheses and the diagnostic
procedures were fixed in [the protocol](observability_diagnosis_protocol.md)
before generating diagnostic_v2. Individual outcomes from all 70 consumed
development/held-out/fresh scenarios were excluded from this study.

## Diagnostic

### Design, ordering and provenance

The 30 new sources use a plain, non-optimized Latin hypercube with master seed
2009202606 over x,y in [0.25,0.75] and Q in [0.8,1.4]. Every generated case is
retained. Exactly one Gaussian source, sigma=0.08, u=0.35, v=-0.15, D=0.005,
zero IC/Dirichlet BC, the same 20 sensors, zero measurement noise, and 81
measurements from t=0 to 0.8 are used. Q is peak source intensity, not integrated
emissions. Coordinates, time and concentration use documented synthetic units.

The protocol was frozen at 05:21:26 UTC on 2026-09-20; the generated manifest
at 06:04:35 UTC. All information diagnostics, classical fits and associated
figures were completed/hash-locked at **06:37:16 UTC**. PINN execution began
at **06:38:05 UTC**, only after that gate. Each PINN used the unchanged seed-42
recipe for exactly 12,000 updates. Optimizer states and complete epoch histories
were independently checked for all 30 runs.

- Revision 5 was committed, with a clean tree, as
  `590c3faf0c6ec71677ca89b3093ea5a7c43116a7` before Revision 6 began.
- Frozen B_revised model revision:
  `9dc90c247b0a596b789cb0ee5f1c06f938db40c4`.
- B_revised YAML SHA-256:
  `3e400ef1195cf90f6481117ae5c395695cca23a12d47ab532ec6822e400c332e`.
- Protocol canonical SHA-256:
  `3037241fe6843e54998329b48e9d30b18664e21194497d19107c01cda32671c7`.
- Diagnostic manifest CSV SHA-256:
  `c28d82597cb8992983a14da3ecf1e9fcd82bea5cdbcf85e53cc8afb0a182a9df`.
- Identical information/PINN execution source-archive SHA-256:
  `77297118049af2d4bace37c3ddaf628fc719fd2bf1ac21c8286f38298dc3a096`.

All 5,201 historical artifact files retain their original hashes. Preservation
checks hash opaque bytes; they do not parse individual historical outcomes.
All new source truth hashes match the frozen plan. Every input reading equals
its saved noise-free reference reading exactly. Sensor coordinates and stored
measurement times are identical across all cases. The discrepancy from NumPy's
alternative linspace rounding is at most 1.11e-16, not interpolation or a
changed measurement schedule. Input-only fitting APIs and Python I/O guards
separate fitting from source/dense-field truth. Explicit oracle routines are
labeled separately. These guards are not an operating-system security sandbox.

### Reference qualification

All references use the unchanged 641² generator, dt=0.00003125, with the
161²/321²/641² refinement assessment. Cases 001, 006, 014 and 025 missed the
1% Richardson-estimate target: respectively **1.042%, 1.948%, 1.705%, 1.383%**.
All remain included and flagged. These are conditional numerical estimates,
including sensor-trajectory assessments, not certified mathematical bounds.
All four flagged cases were recovered by classical inversion; all four were
PINN failures, alongside five unflagged PINN failures.

### Mathematical diagnostics

For fixed location and homogeneous IC/BC, the discrete PDE and observation
operators are linear: y(Q)=Qh. Equal-weight least squares therefore gives
Q_hat=max(0,hᵀy/(hᵀh)). A separate scalar optimizer, initialized at 1.1,
checks this result. No true Q initializes either estimator. All 90 numerical
optimizations succeeded; the maximum analytic/numerical difference was
1.37e-10. Direct source-Q simulations verify linearity to floating-point
accuracy (maximum relative discrepancy at 201²: 5.44e-15).

The classical joint inverse profiles Q analytically and optimizes x,y over
[0,1]². Three starts are chosen by observation-only MSE on a predefined map
and a fixed separation rule. All 90 start records are saved; all terminated
successfully. The winner is selected only by sensor MSE, never by source error.
This is recoverability evidence, not a certified global optimum or an equal-
budget algorithm competition.

For local information, z=theta/[0.5,0.5,0.6] and observations are divided by
0.88. J_scaled=(dy/dtheta)diag([0.5,0.5,0.6])/0.88. Central x/y differences
at normalized steps 0.001, 0.0005, 0.00025 were checked; the preselected smallest
step was retained, and all 30 stability checks passed. Q derivatives are exact
unit-source responses. JᵀJ and covariance proxies assume hypothetical independent
unit-variance normalized noise. No noise is added and these are not posterior
uncertainties. The 441-point map fixes Q=1.1 and uses no scenario truth.

The envelope E=16x(1-x)y(1-y)t/0.8 is assessed using all four reference
snapshots, not selected failures. Required g=C/E and inverse-softplus raw f
are defined only at E>0, with one synthetic concentration unit as the output
scale. No epsilon substitutes for zero E. All-interior and active-plume
(C >= 1% of snapshot maximum) statistics are separate. Actual network outputs,
transform gain E*sigmoid(f), and coordinate gradients are evaluated without
optimizer steps or parameter-gradient modification.

## Evidence

### A–C: conditional strength recovery and numerical mismatch

At true x,y, 201² conditional Q has median relative error **0.899%**, worst
**3.427%**, with 11 underestimates and 19 overestimates. Mean signed error is
+0.000938 Q units; median +0.005955. Conditional recovery is accurate here.

| Inference grid | Median signed Q error | Median absolute Q error | Median relative Q error | Median observation L2 | Median seconds/case |
|---|---:|---:|---:|---:|---:|
| 101² | +0.014624 | 0.022182 | 2.166% | 2.910% | 0.169 |
| 201² | +0.005955 | 0.009133 | 0.899% | 1.213% | 1.544 |
| 321² | +0.002621 | 0.004147 | 0.411% | 0.556% | 6.787 |

Runtime includes independent unit-Q and true-Q forward solves plus the scalar
optimization, under the recorded two-worker execution. Relative-Q error falls
about 58% and then 54% in median. This is convergence toward the 641² reference,
not an estimate of formal convergence order or proof of exact continuous truth.

At 201² the projection amplitude ratio has median **0.994471**, mean 0.999351,
range 0.985139–1.035106: 11 cases too strong, 19 too weak along the reference
vector. Ordinary signed mean concentration bias has mean about +9e-6 and
median +1e-5 (17 positive, 13 negative). Shape changes explain why the two
amplitude summaries need not have the same sign. There is no uniform
over-amplitude mechanism explaining universal PINN Q underestimation.

### D–E: joint recovery and failure overlap

The unchanged criterion is localization <=0.08 AND relative Q error <=20%.

| Method / metric | Mean | Median | Sample SD | 90th percentile | Worst |
|---|---:|---:|---:|---:|---:|
| Classical localization | 0.001495 | 0.001331 | 0.000830 | 0.002841 | 0.003267 |
| Classical relative Q error | 1.599% | 1.193% | 1.141% | 3.308% | 4.607% |
| Classical concentration L2 | 1.812% | 1.352% | 1.143% | 3.098% | 5.843% |
| PINN localization | 0.024160 | 0.005214 | 0.035857 | 0.069958 | 0.130571 |
| PINN relative Q error | 18.431% | 4.338% | 25.340% | 63.729% | 84.023% |
| PINN concentration L2 | 21.489% | 5.490% | 27.225% | 70.090% | 88.475% |

Classical recovery: **30/30**; PINN recovery: **21/30**. Both succeed in 21;
classical succeeds/PINN fails in nine; classical-only failures and shared
failures are both zero. Nine PINN cases fail Q recovery; two also fail location.
Classical median fit runtime is 62.14 s (all starts). Mean sensor MSE is
1.16e-7 classical versus 1.08e-6 PINN. Mean concentration RMSE/MAE are
0.000942/0.000308 classical and 0.010927/0.003312 PINN. These dense metrics
equally weight the four fixed snapshots, not a continuous-time integral.

| PINN failure ID suffix | Localization | Relative Q error | Concentration L2 | Reference target met |
|---|---:|---:|---:|---|
| 001 | 0.078185 | 63.02% | 69.98% | no |
| 006 | 0.059859 | 70.11% | 71.11% | no |
| 009 | 0.055852 | 34.76% | 44.77% | yes |
| 014 | 0.121033 | 84.02% | 88.48% | no |
| 015 | 0.069044 | 32.27% | 46.52% | yes |
| 016 | 0.021209 | 21.52% | 25.12% | yes |
| 017 | 0.022181 | 27.05% | 27.03% | yes |
| 025 | 0.130571 | 80.88% | 87.16% | no |
| 029 | 0.052355 | 39.36% | 48.57% | yes |

PINN negativity fraction, minimum C, IC/BC RMSE and maximum violations are
exactly zero. Independent PDE-residual RMSE has mean 0.006744, median 0.005933,
worst 0.016325; PDE MAE mean 0.002925. These finite-point physical diagnostics
do not establish source recovery or an everywhere residual bound.

### F–I: local information, geography and prospective associations

All 30 case Jacobians and all 441 map Jacobians have rank three. Case median
scaled column norms are x=7.184, y=7.866, Q=1.177. Case minimum singular values
range 0.012405–1.701622 (median 0.405667); scaled conditions range
4.426–348.363 (median 22.383). Minimum information eigenvalues range
0.000154–2.895516. Median unit-normalized-noise physical standard-deviation
proxies are 0.1161 (x), 0.1278 (y), 1.4782 (Q); these are not actual error bars.

On the fixed-Q map, sensor RMS ranges 0.005533–0.090538; minimum singular
values span 0.008404–2.099656, about 250-fold. The weakest mapped region is
near x=0.75, especially y approximately 0.55–0.68, with additional lower-domain
weak coverage. The weakest point (0.75,0.575) has condition about 480.12.
Full rank does not imply uniform practical robustness or global uniqueness.

Prospective Spearman(sensor RMS, error): classical localization **-0.166**,
classical Q **-0.265**; PINN localization **-0.797**, PINN Q **-0.818** and
PINN L2 **-0.802**. RMS correlates with condition **-0.658**, minimum singular
value **+0.834**, and effective Q information **+0.829**. Thus weak signal
reappears as a strong PINN-error association, not as classical recovery failure.

The geometric definitions are transparent: with unit wind w, a=(sensor-source)·w,
b the absolute crosswind projection, and tau=a/|wind|, the corridor counts
sensors with 0<=tau<=0.8 and b<=sqrt(sigma²+2D*tau). This is a geometric proxy,
not a hard plume boundary; diffusion permits upwind influence.

PINN localization correlations: true x +0.807; true y +0.022; downwind count
-0.765; reachable-downwind count -0.759; corridor count -0.721; nearest-downwind
distance +0.721; minimum downwind crosswind distance +0.681. For local minimum
singular value, corridor count (+0.780) and nearest-downwind distance (-0.735)
are stronger associations than raw x (-0.689); they are not uniformly stronger
predictors of PINN error than x. These are descriptive, confounded associations,
not causal effects or a fitted out-of-sample predictive model.

### J: envelope and gradient evidence

Across all active reference regions/times, required C/E ranges about
0.00321–6.384; required raw f ranges -5.741–6.383. All-interior values include
near-zero background demands down to C/E=5.03e-29, raw f=-65.16. At t=0.8,
the median active-region q95 C/E is 0.4867 versus actual PINN softplus(f) q95
0.3796. Larger required q95 C/E correlates with Q error (+0.768) and source
x (+0.715), but both are confounded with sensor geometry.

At t=0.8 actual active q95 spatial input-gradient norms of f range
21.70–33.13; those of C range 0.419–2.929. Actual active q05 transform gains
range 0.00129–0.00737 (median 0.00500); the reference-required median is
0.00530. Actual gain correlates with Q error (-0.678), while required gain
has almost no association (+0.056). Raw input gradients do not uniformly
vanish. These measurements demonstrate uneven latent demands and attenuation,
but do **not** isolate a causal envelope-induced optimization failure.
Coordinate gradients are not parameter-update gradients. Reference quantiles
use 641²; network probes use the prospectively fixed 81² evaluation grid.

### K: where the systematic Q pattern appears

| Estimate | Under / over / approximately equal | Mean signed Q error | Median signed Q error |
|---|---|---:|---:|
| True Q | reference | 0 | 0 |
| Conditional Q, 201² | 11 / 19 / 0 | +0.000938 | +0.005955 |
| Classical joint Q | 5 / 25 / 0 | +0.010033 | +0.010825 |
| Frozen PINN Q | 30 / 0 / 0 | -0.201397 | -0.052233 |

Universal underestimation appears specifically in the PINN estimates. The PINN
does not call the 201² forward solver, so this comparison chain is not a causal
dataflow chain. Forward mismatch is measurable, but neither its signs nor its
conditional magnitude explain the observed universal, sometimes severe deficit.

## Interpretation

### L: failure evidence, not causal fractions

Among the nine PINN failures, the predefined primary evidence classification
is: observation/identifiability limitation 0/9; numerical mismatch 0/9;
PINN optimization/model limitation **9/9**; mixed/uncertain 0/9. This means all
nine exhibit classical success on the identical observations, **not** that
100% of their causal mechanism has been identified. Eight of nine are also
below the predefined map-relative information quartiles. Weak coverage may
amplify a PINN-specific limitation; that interaction remains unresolved.
Nonexclusive flags are A=9, B=0, C=6, D=0, E=21.

### M: single next intervention

**Improve the inverse/PINN formulation**, in a separately authorized,
development-only milestone. The observation-only classical procedure already
recovers every source, while the unchanged PINN has nine failures and 30/30
Q underestimates. The present study does not justify identifying the envelope,
a particular loss weight, optimizer, sampling rule or architecture as the sole
cause. No proposed intervention was implemented here.

### N–O: readiness

**No**: do not begin noise/sensor-count sweeps yet. First address the demonstrated
zero-noise PINN-specific recovery gap, while tracking weak-signal/Q-information
regions and numerical reference error. **No**: do not begin real Almaty work.
Unknown meteorology, real sensor noise, model misspecification, variable sources
and causal emitter attribution remain unvalidated.

## Limitation

This 30-case Latin-hypercube design is diagnostic development, not IID random
confirmatory validation. Saved Wilson calculations are descriptive only, not
design-valid confidence guarantees. No claim of global inverse uniqueness,
robustness to noise, arbitrary source discovery, real emitters, or general PINN
superiority follows. Local linear information is not a posterior. Correlations
do not partition causes. Multistart numerical success is not a global optimum
certificate. Four reference-target misses remain unresolved numerical caveats.
The frozen single-seed fits do not separate representation, optimization, loss
weighting, sampling, envelope effects and remaining continuum/reference mismatch.

## P: tests and independent audit

- Final targeted suite: **128 passed**, zero failures/warnings.
- Final full suite: **406 passed**, zero failures, **120 warnings**: one
  NumPy/netCDF4 binary-size RuntimeWarning and 119 xarray/netCDF4 NumPy-shape
  deprecations. No dependency/model changes suppressed these warnings.
- Experiment processes used temporary Matplotlib caches because the default
  user cache directory was not writable; figures were produced successfully.
- 32 new tests; no existing test weakened. JUnit XML and exact test commands
  are saved beside the experiment.
- Independent recomputation of all 60 source/field result records agreed to
  maximum absolute difference 2.61e-16. All checkpoint recipes, optimizer step
  counts, epoch histories, paired inputs and consumed-data hashes were verified.

## Q–R: files and repository state

New: `configs/observability_diagnosis.yaml`; this report and
`docs/observability_diagnosis_protocol.md`; `scripts/generate_diagnostic_v2.py`,
`scripts/run_observability_diagnosis.py`, `scripts/run_diagnostic_pinn.py`,
`scripts/report_observability_diagnosis.py`; `src/inverpinn/data/diagnostic_v2.py`,
`src/inverpinn/evaluation/observability.py`,
`src/inverpinn/visualization/observability.py`; `tests/test_observability_diagnosis.py`.
Modified: `README.md` only among previously tracked files.

Artifacts are under `results/observability_diagnosis/`, including all required
CSV/JSON tables and 12 required PNG figures (also vector PDFs), three additional
diagnostic figures, per-run checkpoints/configurations/logs, all classical
starts, reference and observation vectors, code/package provenance and integrity
reports. Large results remain ignored; none were force-added.

Branch: `scientific-revision`. Current commit:
`590c3faf0c6ec71677ca89b3093ea5a7c43116a7`. Revision-6 changes are intentionally
uncommitted. The frozen model configuration and all original scientific code
are unchanged. Work stops here: no tuning, new blind validation, sensor/noise
experiment, sensor redesign, AirKaz, ERA5 or Almaty changes follow this report.
