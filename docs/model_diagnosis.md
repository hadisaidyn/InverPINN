# P0 scientific revision 4 — development-only diagnosis

This report uses only the ten existing development cases. The old 30-case
held-out set is consumed: no old held-out truth, metrics, figures or shared
truth-bearing plan was loaded for this revision. No new held-out scenarios
were generated. Revision 3 was first committed on `scientific-revision` as
`ce210e3862fbd34aa509a9fc139ab8698edfe88d` (`Benchmark independent single-source
recovery`), with a clean working tree before revision-4 changes.

## Protocol and evidence boundaries

- B0 was actually rerun on all ten cases. All network/source checkpoint tensors
  and the requested final metrics match the old **development** fits exactly.
- The component-gradient observer was tested for bitwise optimizer/RNG
  invariance. Its optional hook is the only change to the old forward trainer;
  the original strict sparse-only inverse fitting API is unchanged.
- Each candidate uses the same observations, known transport, fixed source
  width, diagnostic points and dense reference fields. The optimization seed
  is 42. Each ablation runs on all ten cases. No case is dropped or selected
  for its favorable outcome.
- The candidate plan and selection rule were archived before any candidate
  result, after the complete B0 gradient audit. All fits use fixed budgets,
  not best-epoch or source-truth-based early stopping.
- Training sees only sparse positions/times/measurements and known physics.
  Source/dense truth is parsed only after a terminal fit record. Reference
  byte hashes are recorded before fitting and checked afterwards; hashing is
  provenance, not access to truth as training features.
- Numerical metrics equally weight four dense snapshots at t=0.1,0.2,0.4,0.8
  on the 641² reference grid. Saved previews use every fourth node; those
  previews are **not** used for metrics. Predictions are never clipped.
- Negative fraction and PDE diagnostics are finite sampling diagnostics, not
  a proof over continuous spacetime. The hard envelope separately guarantees
  nonnegativity and homogeneous IC/BC on the modeled domain.
- The reference remains approximate: development_004 missed the 1% conditional
  sensor-trajectory Richardson target (about 1.53%). It is retained for every
  candidate. Zero added noise is not exact continuum ground truth.
- These are development-selected results for one fixed 20-sensor geometry,
  source family, transport and seed. They are not an independent generalization
  estimate, posterior uncertainty analysis, or a causal source attribution.
- Per-fit times include diagnostics/checkpoint overhead. Two workers ran in
  parallel, and some test runs overlapped; wall times are descriptive resource
  records, not a controlled hardware-speed benchmark.

## Observation / Hypothesis / Experiment / Result / Conclusion

The following explicitly separates observations from explanations. Detailed
quantities, controls and limitations follow in sections A–R. The supplementary
term/parameter diagnostics were specified **after** B5 selection; they explain
existing results and must not be represented as preselection evidence. They
neither retrain nor rank candidates and leave all original artifacts intact.

| Observation | Hypothesis | Experiment | Result | Conclusion |
|---|---|---|---|---|
| Every B0 Q estimate is low | Softplus saturates | Signed gradient/Jacobian audit; exponential-Q control | Jacobians .1400–.7093; exp does not resolve failures | Numerical saturation is not supported as the main cause |
| Weak Q accompanies incorrect fields | Field/source compensation and amplitude shrinkage | Conditional PDE-optimal Q and joint amplitude-ray calculus | Conditional Q is near fitted Q; scalar shrinkage only .15–2.40% | Coupling is supported; scalar shrinkage alone is insufficient |
| Results vary with starts and budget | Finite-budget optimization contributes | Fixed Q and location starts; identical-prefix 12000-step controls | Higher starts/longer training improve some errors; difficult cases persist | Finite-budget dependence confirmed, not certified local minima or unique inverse solutions |
| B0 violates positivity and IC/BC | Soft constraints permit misleading sensor fits | Positive-only and hard-envelope controls; correlations | Hard constraints remove violations and improve B2 sensor fit | Architecture enables invalid fields; leakage as dominant Q-bias cause is unproven |
| Mixed-unit loss terms | Poor conditioning explains weak Q | Derive full scales, rescale losses, retain physical diagnostics | Scaling alone does not improve recovery | Objective was unit-dependent; this alone does not explain bias |
| Residual larger near estimated sources | Uniform sampling misses source support | Batch coverage and fixed residual maps | Counts agree with area expectation; residual concentration remains | Coverage failure not established; adaptive benefit untested |

## A. Why was Q systematically underestimated?

### Observed evidence

All ten B0 development estimates underestimate Q. The mean relative error is
34.24%, and the median is 32.25%. At all 110 sampled early checkpoints (updates
1..10 and 100 across ten cases), the physical PDE derivative dL/dQ is positive:
range 0.01137–0.02768. This initially pushes Q down. At the final update the
mean derivative is -0.000165, so a permanently downward gradient is **not**
the explanation.

Only the PDE term depends directly on source parameters. Data/IC/BC gradients
to Q, xs and ys are exactly zero structurally. Their influence is indirect,
through the neural concentration field's subsequent effect on PDE derivatives.
With r=A(C)-Qg, dL_PDE/dQ = -2 mean(r g). When the initial network's A(C) is
small relative to Qg, reducing Q is an immediate way to reduce residual loss.

Softplus's observed Jacobian ranges 0.1400–0.7093: it attenuates gradients but
does not numerically saturate. In the final 1000 updates the median weighted
NN gradient norms are data 0.06456, PDE 0.00598, BC 0.00845 and IC 0.00686.
Data/PDE gradients conflict at 47.86% of sampled checkpoints. These are
descriptive sampled gradients, not independent causal observations.

Exact final B0 raw_Q checkpoint values range from -1.81515 to .89172; the
corresponding Q Jacobians range .140017–.709245. Q is 37.1–38.0 decimal
orders above the float32 positive-normal floor: underflow is not near these
solutions. Intermediate raw logits reconstructed from physical CSV estimates
range -1.81516–.89182 and are explicitly labeled **inferred**, not exact saved
logits. Final x/y logits and Jacobians are also saved in
`diagnostic_supplement/final_source_parameters.csv`.

For a fair gradient-coordinate comparison use x/L,y/L,Q/Q*, with L=1 and
Q*=1.1. The across-case median of within-phase median
`||grad_(x/L,y/L) L_PDE|| / |grad_(Q/Q*) L_PDE|` is .473 initially, .724 early,
3.45 in updates 2900–3100, and 6.86 in updates 5001–6000 for B0. Location
gradients are relatively larger late, not at every stage; Q gradients are
nonzero, smaller and change sign. These parameter-scale-dependent ratios
are not Adam update ratios, and do not establish a sole cause of Q bias.

The fixed-field analytical control is particularly informative. Conditional
on each fitted B0 field and location, minimizing PDE residual over Q alone
gives values close to the learned strengths. For example, development_004 has
Q_fitted=0.15084 and conditional PDE-optimal Q=0.14814, whereas its true Q is
1.18107. The optimizer is not simply ignoring a large correct-strength signal;
the current field derivatives themselves support a weak source. The conditional
calculation uses no source truth; truth appears only in this comparison.

The two worst B0 strength cases also have the weakest observed sensor signals:
sensor RMS 0.01669 and 0.00947 in development_003/_004, compared with 0.07287 in
_001. Their training-sensor relative L2 errors are 11.94%/12.24%, despite tiny
absolute MSEs, while their dense concentration errors are 75.71%/89.44%.
Across the ten cases sensor RMS and Q error have Spearman correlation -0.915.
This supports a signal/geometry association, not a proof of non-identifiability
or a causal claim that a different sensor layout would fix the model. The
sensor layout and data are unchanged throughout the milestone.

### Mathematical mechanism and its measured limitation

For the diagnostic ray (C,Q) -> (aC,aQ), linearity of the PDE and homogeneous
IC/BC imply J(a)=w_data mean((a C_sensor-y)²)+a² P, where P is the weighted
physics penalty at a=1. The data-only and joint optimum scales have ratio
A/(A+P), with A=w_data mean(C_sensor²). Imperfect PDE/IC/BC satisfaction thus
creates an amplitude-shrinkage incentive. It vanishes for an exact physical
solution. In B0's terminal fields the measured ratio is 0.9761–0.9985: only
0.15–2.40% shrinkage relative to sensor-only rescaling. This is much too small
to explain the large terminal Q errors by itself. A scalar post-calibration
is not a demonstrated solution; field shape, location and optimization matter.

### Hypotheses versus tested interventions

Early source/field competition and nonzero parameter gradients are directly
observed. Numerical saturation is not supported as a sole cause. Initialization,
parameterization and convergence are isolated with fixed, all-case controls;
their completed results are summarized below. Finite-budget basin dependence
does not prove multiple exact inverse solutions. Non-identifiability, network
approximation error and optimization can remain confounded.

Changing only Q initialization from 0.5 to 1.1 to 1.5 yields median relative
Q errors of 32.25%, 31.90% and 19.98%, and median dense L2 errors of 37.05%,
35.82% and 22.12%. In development_004 the final Q values are 0.15084, 0.36773
and 0.55503, a range of 0.40418, with the same data and neural seed. The initial
guess therefore has a demonstrated finite-budget effect; the fitted strength
is not uniquely determined by the observed data *under this optimization
procedure*. All three runs still underestimate that case's Q. Higher starting
Q is not a validated correction: these unconstrained fields retain negative
concentrations and fail the predeclared physical-validity gate.

The controlled exponential-Q replacement (E1) changes mean Q error only from
34.24% to 33.46%, with median 31.86% rather than 32.25%. Development_004 remains
an 87.62%-strength-error failure. Its mean negative fraction increases to
40.47%. This does not support softplus saturation as the main cause or exp as
an overall remedy. Near small Q, softplus's derivative 1-exp(-Q) is already
approximately Q, similar to the exponential parameterization's derivative.

### Causal status

- **Confirmed finite-budget contributors/mechanisms:** the early PDE gradient
  pushes Q downward; changing only Q initialization or extending the identical
  optimizer trajectory changes terminal strength and field errors. The data
  loss has no direct source gradient. These facts do not uniquely apportion
  the final bias among mechanisms.
- **Supported hypotheses:** field-shape/source compensation and imperfect
  joint optimization; difficult, weakly observed plume geometry. Conditional
  Q supports coupling, while sensor signal versus Q error is an association,
  not a causal identifiability test.
- **Unresolved:** exact identifiability under these sparse observations,
  approximation limits versus optimizer failure, and whether another optimizer
  or sampler would remove the bias. Numerical Q-transform saturation and
  BC/IC leakage as the dominant explanation are not supported by these controls.

## B. Why were negative concentrations produced?

B0's linear output is unconstrained. Neither sparse data nor finite sampled
soft PDE/IC/BC penalties imply the maximum principle. Every development B0
prediction contains negative values; the mean negative fraction is 37.06%.
Thus the possibility is structural, while the actual violations also reflect
imperfect optimization/approximation. Longer training alone cannot provide a
nonnegativity guarantee.

B2 tests positive softplus output; B3 additionally multiplies by
E=16x(1-x)y(1-y)t/T. Both are nonnegative by construction. B3 imposes exactly
zero IC/BC. The whole product is differentiated, with analytic tests of first
and second derivatives. The envelope can suppress gradients near boundaries
and at small times, and softplus can attenuate gradients at negative logits.
These are reasons to measure reconstruction/PDE error, not to assume that
enforcing positivity automatically improves inverse inference.

## C. Were BC/IC violations contributing to apparently good sensor fits?

B0 mean BC RMS is 0.002527 and IC RMS 0.003028 (synthetic concentration units).
However, within B0, Spearman correlations of Q error with BC RMS, IC RMS and
negative fraction are **negative**: -0.406, -0.358 and -0.612 respectively.
Concentration L2 correlations with BC/IC RMS are -0.345/-0.321. Thus the ten
cases do not support the simplistic claim that larger violations explain the
worst recovery failures. Correlations are descriptive and do not establish
causation; hard-constrained constant-zero columns have undefined correlations,
which are left missing rather than reported as zero.

Sensor MSE correlations with BC RMS, IC RMS and negative fraction are
Spearman +.236, +.164 and +.442. Better sensor fit does not systematically
coincide with larger violations in these ten B0 cases. Case inspection also
shows development_003/_004 sensor relative errors around 11.94%/12.24% despite
dense-field errors 75.71%/89.44%: sparse agreement is not reliable reconstruction.

B2 -> B3 is the controlled addition of hard homogeneous constraints with all
other configured choices fixed. At 6000 updates this reduces mean sensor MSE
from 4.5324e-5 to 3.6013e-6 while making BC/IC exact, and reduces mean relative
L2 from 0.5148 to 0.3655. Good sensor fits therefore do **not** require violating
the conditions. But B3 still has mean physical PDE RMS 0.009496 versus B0's
0.007306, and median L2 0.3895 versus 0.3705. Hard constraints alone are not
accepted as a successful overall revision.

## D. Was dimensional/loss scaling incorrect or poorly conditioned?

The PDE signs and input-normalization derivatives are correct. The original
objective nevertheless combines C² losses with a (C/time)² residual without
explicit unit-compensating scales. Coordinates normalized inside the network
do not make that objective unit-invariant. With L=1, T=.8, Q*=1.1, C*=.88,
the correct dimensionless residual is (T/C*) times the physical residual.
The corresponding velocity, diffusion and source scales are 1.25, 1.25 and
1.1. See [the full derivation](nondimensionalization.md).

B1 changes only this scaling. Its median Q/L2 errors are 33.01%/38.37% versus
B0's 32.25%/37.05%; mean physical PDE RMS increases from .007306 to .008681.
Thus mixed-unit conditioning is a legitimate formulation issue, but correcting
it is not a demonstrated standalone explanation or remedy for the observed
bias. Supplementary physical and dimensionless term statistics are recorded
separately; every physical term has the same concentration/time units.

Final-checkpoint term RMS medians across the same ten cases and fixed 8192
independent spacetime points per case are:

| Physical term | B0 | B5 |
|---|---:|---:|
| C_t | .089171 | .098577 |
| u C_x | .094300 | .100409 |
| v C_y | .041338 | .047418 |
| D C_xx | .015443 | .016980 |
| D C_yy | .017986 | .020118 |
| S | .113198 | .124137 |
| r | .008215 | .005684 |

All six terms are significant; they span about 7.3-fold in these medians, not
orders of magnitude suggesting one obvious unit mistake. RMS hides signs,
so the supplement also saves signed values, means, MAE and maxima. Each
dimensionless term is its physical value multiplied by .8/.88. The residual
is the signed sum, not the sum of individual RMSs. This checks magnitudes
without claiming comparable parameter gradients or a small relative field
error from a small absolute residual.

## E. Was the baseline under-converged after 6000 Adam steps?

Yes, in the operational sense that doubling the *same* Adam trajectory changes
and improves its results. O1 extends B0 from 6000 to 12000 updates with the
same learning rate and samples; all first-6000 loss/source records match
exactly. Median localization/Q/L2 errors improve from 0.03646/32.25%/37.05%
to 0.01925/17.61%/20.79%. Mean independent physical PDE RMS decreases from
0.007306 to 0.004324; mean BC/IC RMS decreases to 0.001916/0.001842. Mean
independent objective decreases from about 1.15e-4 to 4.1e-5. Mean recorded
runtime increases from 38.0 to 75.4 seconds, with the timing caveats above.

This is not a complete remedy. O1 still recovers only 5/10 cases and has a mean
negative fraction of 36.01%. Development_004 retains 85.93% Q error and 88.61%
concentration L2 error. Longer optimization can improve reconstruction without
providing positivity or exact physical constraints.

B5 similarly extends B4's exact first-6000 trajectory. Its median Q/L2 errors
improve from 33.30%/41.38% to 21.22%/25.97%, and mean PDE RMS decreases from
0.009470 to 0.006958. However, difficult cases remain, and the median ratio of
last-1000 to preceding-1000 stochastic objective means is 0.984 (range
0.410–1.575). These variable-batch curves do not prove stationarity, global
optimality or further guaranteed improvement. B0's corresponding ratio is
0.797 (0.620–0.848), consistent with ongoing progress at its original cutoff.

Fixed-window, post-update parameter histories quantify this further. B0's
updates 5000→6000 move the estimated location by median .00403 (maximum
.01376) and Q by median +.02212 (range -.00263 to +.12258). Its identical-prefix
extension 6000→12000 moves location by median .00910 (maximum .02663), with Q
increasing in all ten cases by .00885–.29748 (median .05581). Relative Euclidean
network-parameter displacement over that extension is median 18.13%
(15.39–23.49%). This is substantial remaining movement in a number of cases,
not evidence of stationarity at 6000. Endpoint displacement, maximum excursion
and final single-step changes are all retained so oscillations are not hidden.

B4→B5 moves location by median .00613 (maximum .04863), Q by median +.01638
(range -.07568 to +.23247), and network parameters by median 17.47%. Q need
not rise during this constrained trajectory. These windows are descriptive,
not post-hoc stopping criteria; nonzero movement alone does not guarantee
future scientific improvement.

Only the predeclared Adam-6000 and Adam-12000 schedules were tested. No
post-hoc stopping rule, learning-rate search or L-BFGS stage was introduced.
L-BFGS would require an additional deterministic-objective/sampling design;
its possible benefit remains untested, not disproven.

## F. How sensitive was inference to Q initialization?

The all-case B0/Q11/Q15 comparison changes only Q_init=.5/1.1/1.5. Median Q
errors are 32.25%/31.90%/19.98%; median concentration L2 errors are
37.05%/35.82%/22.12%. Development_004's fitted Q ranges from .15084 to .55503
despite identical observations and neural seed. This is a strong finite-budget
effect in difficult cases, not proof that the highest start is always best.
All these unconstrained models retain negative concentrations. The known-prior
midpoint 1.1 is global, never substituted from a case's true Q. A bounded
prior transform was not added: it would restrict the physical hypothesis and
can mechanically limit strength error without solving field reconstruction.

## G. How sensitive was recovery to source-location initialization?

The five fixed locations are center (.5,.5), upper-left (.3,.7), upper-right
(.7,.7), lower-left (.3,.3), and lower-right (.7,.3). All use Q_init=.5,
identical observations, seed 42 and 6000 Adam updates. No true coordinate is
used to initialize any fit.

The median maximum pairwise separation of the five final source estimates is
0.00550, but development_003/_004 reach 0.30507/0.30091. Their lower-left starts
have localization errors 0.36727/0.39891. Recovery is therefore strongly
location-initialization-sensitive in these difficult cases, even though most
other cases have much smaller endpoint variation. Distinct finite-budget
endpoints are established; certified stationary local minima are not.

M0 selects the minimum common data-plus-physics objective, evaluated using all
observed sensor readings and the same independently sampled diagnostic points.
The selection function receives only start names and scalar objectives. It
does not receive true sources or dense-field errors. All five costs and the
selected checkpoint/hash are saved for each case.

This legitimate selection rule does **not** improve robustness here: M0 still
recovers 4/10 cases, with median localization 0.03646, Q error 32.43% and L2
37.23%, versus B0's 0.03646/32.25%/37.05%. Mean total training time rises to
186.3 seconds from 38.0 seconds. Mean negative fraction is 38.91%. Multistart
is not selected, and no truth-best restart result is advertised.

## H. Did collocation coverage contribute to failures?

Uniform sampling uses 512 new spacetime points per update. For an interior
source estimate, the expected count within radius sigma=.08 is 10.294, and
within 2 sigma is 41.177. Across 700 audited B0 batches the observed means are
10.473 and 41.263; the one-sigma count ranges 3–21, with no empty neighborhoods.
This does not support a literal failure to sample the source neighborhood.

Residual maps were predeclared for development_001, _005 and _010 at t=.2,.8.
Near-estimate (radius 2 sigma) RMS is 0.01598–0.02684, versus 0.00470–0.00931
away. Error is spatially concentrated, but this alone does not identify a
sampling cause rather than approximation/optimization error. Therefore no
adaptive sampling was added or claimed beneficial. Coverage is measured around
the estimate, not the true source; it does not certify all relevant plume
regions or all training updates. Edge residuals in maps are diagnostic only;
the PDE is trained in the open interior.

The post-selection supplement reuses those exact maps, excludes exact edges,
and adds an interior boundary band of width sigma=.08. This is a documented
geometric band, not a physical threshold. Median RMS across the six fixed
case/time maps (three cases, two times) is:

| Region | B0 | B5 |
|---|---:|---:|
| Within 2 sigma of estimated source | .021129 | .009757 |
| Away from estimated source | .007011 | .003948 |
| Interior boundary band | .004931 | .002001 |
| Global open interior | .009731 | .004768 |

These regions overlap, so their errors must not be summed. Maximum
boundary-band RMS across these maps rises from .009404 to .011894, despite
the lower median and exactly zero boundary **concentration**. This distinction
prevents hard boundary values from concealing adjacent interior PDE errors.
The selected three cases were fixed before fitting, not selected to represent
the worst residuals. No adaptive collocation candidate was tested.

## I. What candidate modifications were tested?

All rows use the same ten cases. Location, Q error and concentration L2 below
are **medians**. Negative fraction and PDE/BC/IC RMS are **means**; runtime is mean
seconds per case, including every start for M0. Q/L2/negative values are
percentages. BC/IC use synthetic concentration units. Raw results, means,
standard deviations and physical PDE diagnostics are also saved in CSV.

| Candidate | Change | Location | Q error % | L2 % | Negative % | PDE RMS | BC RMS | IC RMS | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | Frozen baseline | .03646 | 32.25 | 37.05 | 37.06 | .007306 | .002527 | .003028 | 38.0 |
| B1 | B0 + dimensionless loss scaling | .03692 | 33.01 | 38.37 | 38.35 | .008681 | .002483 | .002756 | 37.3 |
| B2 | B1 + positive output | .06421 | 50.99 | 51.53 | 0 | .017071 | .002878 | .005546 | 40.0 |
| B3 | B2 + exact IC/BC | .03601 | 31.93 | 38.95 | 0 | .009496 | 0 | 0 | 45.6 |
| B4 | B3 + Q_init=1.1 | .03677 | 33.30 | 41.38 | 0 | .009470 | 0 | 0 | 54.9 |
| B5 | B4 + Adam 12000 | .02229 | 21.22 | 25.97 | 0 | .006958 | 0 | 0 | 95.9 |
| Q11 | B0 + Q_init=1.1 | .03385 | 31.90 | 35.82 | 44.24 | .007236 | .002840 | .002775 | 36.3 |
| Q15 | B0 + Q_init=1.5 | .01955 | 19.98 | 22.12 | 39.46 | .007426 | .003321 | .002756 | 37.3 |
| E1 | B0 + exp(raw_Q) | .03479 | 31.86 | 35.75 | 40.47 | .007209 | .002503 | .003006 | 39.6 |
| O1 | B0 + Adam 12000 | .01925 | 17.61 | 20.79 | 36.01 | .004324 | .001916 | .001842 | 75.4 |
| L_UL | B0 + upper-left start | .04219 | 33.60 | 38.91 | 34.92 | .007516 | .002645 | .003168 | 37.8 |
| L_UR | B0 + upper-right start | .04275 | 33.76 | 38.42 | 33.56 | .007615 | .002598 | .003208 | 36.4 |
| L_LL | B0 + lower-left start | .03728 | 34.56 | 37.93 | 37.26 | .008869 | .002166 | .004285 | 36.7 |
| L_LR | B0 + lower-right start | .04262 | 34.00 | 38.55 | 33.41 | .008046 | .002596 | .003658 | 37.3 |
| M0 | Truth-independent five-start selection | .03646 | 32.43 | 37.23 | 38.91 | .007082 | .002164 | .002799 | 186.3 |

There were **140 actual fits, 960,000 Adam updates, zero computational failures**
and ten derived multistart selections. The raw table has 150 rows, not 150
independent fitted models. All failed *scientific recoveries* remain included.

## J. Which interventions helped?

Only B5 passes the complete predeclared aggregate eligibility criteria. Its
three primary means and medians improve relative to B0, its negative/IC/BC
violations vanish, and mean independent PDE RMS decreases. This is a result
for the **tested combination**, not proof that each constituent change is
independently beneficial. Scaling alone does not help; positive output alone
worsens recovery; hard IC/BC alone improves B2 but does not pass all B0
non-regression checks. Higher Q initialization and longer unconstrained
training can improve primary metrics, but remain physically invalid under the
nonnegativity gate. Exp-Q and multistart are not demonstrated remedies.

B5 does not dominate O1 on reconstruction, and its maximum PDE RMS is worse
than B0. These tradeoffs are not hidden by a weighted score. The mean-PDE gate
and the other thresholds were declared before candidate results; they are not
a universal certificate of physical validity or reliable source identification.

- **Clearly beneficial on these development data:** longer fixed Adam budget;
  hard IC/BC relative to positive-only B2; the complete B5 combination relative
  to B0's predeclared aggregate gates. Positivity is a guarantee, not an error
  improvement by itself.
- **Neutral/not a demonstrated remedy:** exp-Q; truth-independent multistart
  (almost unchanged primary errors for much greater cost).
- **Harmful in the tested control:** positive-only B2 worsens Q, field and PDE
  error; scaling-only B1 also regresses aggregate accuracy. B3→B4's midpoint Q
  change is not independently beneficial, despite being part of selected B5.
- **Uncertain:** L-BFGS, adaptive collocation, B5 initialization robustness,
  larger networks, exact identifiability and generalization. None was inferred
  from results of a different configuration.

## K. What exactly is B_revised?

**B_revised = B5**, frozen in `configs/B_revised.yaml` and byte-identically in
`results/model_diagnosis/B_revised.yaml`.

- Architecture: input x,y,t; three hidden layers of width 64 with tanh; scalar
  linear raw head f. Inputs normalize to [-1,1]. There are 8641 NN parameters.
- Physical output: C=16x(1-x)y(1-y)(t/.8) softplus(f), with one synthetic
  concentration unit as output scale. No clipping is applied.
- One peak Gaussian, fixed sigma=.08. xs=sigmoid(raw_x), ys=sigmoid(raw_y),
  Q=softplus(raw_Q)+float32 tiny; all three raw source parameters train jointly
  with the network. Single fixed initialization: xs=ys=.5, Q=1.1.
- Dimensionless scales L=1, T=.8, C*=.88. Preference weights are data=10,
  PDE=1, IC=1, BC=1; physical MSE factors are 1/C*² for data/IC/BC and
  T²/C*² for PDE. IC/BC terms are structurally zero, not silently disabled.
- Adam, 12000 updates, constant lr=.001, betas=(.9,.999), eps=1e-8, no weight
  decay, AMSGrad, maximization, fused or differentiable optimizer. All defaults
  are explicitly archived in the frozen configuration and optimizer checkpoint.
- CPU float32; one Torch thread; deterministic algorithms; seed 42. Local
  sampling generator seed 43. Each update uses 512 sensor rows sampled with
  replacement, 512 uniform interior points, 128 uniform initial points and
  64 points per edge. All collocation/IC/BC points resample each update.
- Domain [0,1]² x [0,.8], u=.35, v=-.15, D=.005. Uniform collocation only;
  no adaptive sampling, multistart, truth-based initialization, checkpoint
  selection, learning-rate schedule or L-BFGS is part of B_revised.

The configuration records the frozen candidate source-archive hash and
selection-rule hash. Packages and actual working code are archived separately
from the git commit, so a dirty working tree is not mistaken for committed
reproducibility.

The complete machine-readable configuration is
[`configs/B_revised.yaml`](../configs/B_revised.yaml), including every Adam
flag. The candidate-selection procedure was frozen before candidate fits:
require all ten successful finite runs, zero negative fraction, no worse
mean/max BC or IC RMS, no worse **mean** physical PDE RMS, and no worse means
or medians for localization, Q error and concentration L2 than B0. Pareto-filter
the three primary medians, then use the declared tie-break order Q median,
concentration median, localization median, runtime. B5 is the only eligible
candidate; no weighted outcome score is introduced. The maximum-case PDE RMS
is **not** protected by that aggregate gate and its regression remains explicit.

## L. How does B_revised compare with B0 on development scenarios?

| Diagnostic | B0 | B_revised |
|---|---:|---:|
| Median localization | .036460 | .022290 |
| Mean localization | .048464 | .038019 |
| Median relative Q error | 32.25% | 21.22% |
| Mean relative Q error | 34.24% | 27.22% |
| Median concentration L2 | 37.05% | 25.97% |
| Mean concentration L2 | 37.72% | 31.38% |
| Mean concentration RMSE | .020072 | .016562 |
| Mean concentration MAE | .006812 | .004920 |
| Joint recovery criterion | 4/10 | 5/10 |
| Mean runtime, seconds | 38.0 | 95.9 |

Median localization/Q/L2 errors decrease by 38.87%/34.21%/29.89% relative to
B0. Location improves in 7/10 cases; Q and concentration L2 improve in 8/10.
Mean location/Q/L2 errors decrease by 21.55%/20.51%/16.81%; mean RMSE/MAE by
17.49%/27.78%. Recorded mean runtime increases by 152.08% (2.52 times).
All ten revised Q estimates still underestimate truth. The worst revised Q
error is 83.95%, and worst concentration L2 is 89.21%. Thus systematic bias
and serious failures are **reduced, not solved**. These selected development
differences must not be reported as confirmed unseen-case improvements.

Initialization sensitivity is **not established for B5**: it was fitted only
from (.5,.5,1.1). The strong B0 sensitivity in sections F/G cannot certify or
quantify B5 robustness. No additional starts were introduced after selection.

## M. Is B_revised physically valid enough for a fresh benchmark?

**Yes, enough to test, not enough to certify recovery:** it has guaranteed
nonnegativity and exact homogeneous IC/BC, and improves mean
independent residual, but not uniformly across PDE diagnostics:

| Diagnostic | B0 | B_revised |
|---|---:|---:|
| Mean negative fraction | 37.06% | 0 |
| Mean BC RMS | .002527 | 0 |
| Mean IC RMS | .003028 | 0 |
| Mean PDE RMS | .007306 | .006958 |
| Maximum case PDE RMS | .009335 | .014324 |

Mean physical PDE RMS improves about 4.76%; it improves in only 5/10 cases.
The maximum increases about 53.4%, at development_006. Exact boundary/initial
constraints are not an exact interior PDE solution. The maximum-residual
regression is a material limitation for fresh validation.

## N. Is there sufficient evidence for ONE untouched held-out benchmark next?

**Yes, narrowly:** B5 passes the predeclared development selection gates and
has a frozen, physically constrained formulation worth independently testing.
This is justification to **test**, not to claim reliable recovery. The 5/10
development recovery rate, remaining Q underestimation, worst-case PDE
regression and selection on only ten cases require explicit scrutiny in that
future test. No old held-out evaluation, replacement held-out generation,
sensor/noise/wind sweep or Almaty analysis was performed in this milestone.

## O. Did previous held-out scenarios influence model selection?

**NO.** The user's supplied historical aggregate failure motivated the
diagnosis, but no old held-out scenario, truth, metric or plot was loaded to
design or select candidates. All model decisions used existing development
data only. No new held-out scenarios were generated. The path allowlist and
Python open-audit hook are accidental-access guards, not an OS security audit.

## P. Tests

Final targeted command:

```bash
.venv/bin/python -m pytest tests/test_model_diagnosis.py tests/test_single_source_benchmark.py tests/test_diagnosis_supplement.py -q
```

**74 passed in 30.28 seconds.** The 32 original revision-4 tests cover
development-only access, held-out/shared-manifest guards, read-only references,
positive and hard constraints, exact envelope product derivatives, Q transforms
and signed physical gradients, nondimensional residual equivalence,
deterministic fits, truth-independent starts/selection and amplitude-ray
calculus. The original 23 benchmark tests are preserved. The 19 supplemental
tests cover each analytical PDE term, finite-value checks, region masks,
post-update motion indexing, inferred logits, dimensionless parameter-gradient
chain rules and refusal to overwrite or read outside a frozen report manifest.

Final full suite: **341 passed, 120 warnings in 103.58 seconds**. There were
zero test failures. No previous test was weakened, and no failing scientific
test was silently repaired. Existing test-only fixtures are not new research
benchmarks or an Almaty analysis.

Warnings are reported separately: one NumPy/netCDF4 binary-compatibility
RuntimeWarning (expected ndarray header size 16, obtained 96), and 119
xarray/netCDF4 NumPy-2.5 shape-setting DeprecationWarnings (6 inversion, 93
meteorology, 20 real-case fixture warnings). They predate this change; passing
tests do not establish that the dependency compatibility warning is harmless
for every file. No real-data/dependency change was made in this milestone.
Matplotlib's unwritable default cache also caused a temporary-cache/font-cache
notice during plotting, not a numerical/test failure.

The experiment verifies 220 unchanged development data files, all 140 terminal
fits, exact B0 checkpoints/metrics and exact 6000-update optimizer prefixes.
All requested figures and machine-readable outputs are retained. Post-run
verification checked all 2064 original artifact hashes and decoded all 328
original PNGs. Independently reloading all ten B_revised checkpoints and
recomputing full 641² concentration metrics and physical diagnostics produced
exactly zero difference from saved metrics. The supplement rechecked all
original artifact/reference hashes and uses its own manifest, without altering
the original report. The new term plot was also inspected visually.
All 33 supplementary artifact hashes and the required summary hash were
verified, and supplementary residual RMSs match the 20 original B0/B5 metrics
with exactly zero difference. The tracked and archived B_revised configurations
are byte-identical. `git diff --check` passes.

## Q. Files modified/created

Modified:

- `README.md`
- `src/inverpinn/training/forward_pinn.py` (optional read-only observer hook)

Added:

- `configs/model_diagnosis.yaml`
- `configs/model_diagnosis_candidates.yaml`
- `configs/B_revised.yaml`
- `configs/model_diagnosis_supplement.yaml`
- `docs/model_diagnosis.md`
- `docs/model_diagnosis_initial_audit.md`
- `docs/model_diagnosis_selection.md`
- `docs/nondimensionalization.md`
- `scripts/run_model_diagnosis.py`
- `scripts/report_model_diagnosis.py`
- `scripts/supplement_model_diagnosis.py`
- `src/inverpinn/data/development_only.py`
- `src/inverpinn/evaluation/model_diagnosis.py`
- `src/inverpinn/evaluation/diagnosis_supplement.py`
- `src/inverpinn/models/constrained_concentration.py`
- `src/inverpinn/physics/diagnostic_source.py`
- `src/inverpinn/physics/nondimensional.py`
- `src/inverpinn/training/development_candidates.py`
- `src/inverpinn/training/objective_diagnostics.py`
- `src/inverpinn/visualization/model_diagnosis.py`
- `tests/test_model_diagnosis.py`
- `tests/test_diagnosis_supplement.py`

Generated artifacts remain ignored under `results/model_diagnosis/`. No
original reference or real-data file was modified. The coding-guidelines skill
influenced the narrow legacy-code change and the explicit identity-control
tests; physics transformations remain separate from the neural-network code.

## R. Git status

Branch: `scientific-revision`.
Preserved benchmark commit: `ce210e3862fbd34aa509a9fc139ab8698edfe88d`,
`Benchmark independent single-source recovery`.
Revision-4 commit message: `Diagnose and revise inverse PINN training`.
The resulting commit hash and final clean-tree check are reported in the
completion message and can be retrieved with `git log -1` / `git status`.
A commit cannot embed its own hash in this tracked report without changing
that hash. Ignored generated artifacts are not force-added, and the original
benchmark commit/history are not rewritten. The supplementary analysis reused
the completed revision-4 fits instead of mislabeling them as the old benchmark.

## Artifact map

`results/model_diagnosis/` retains B0 audit CSVs, per-candidate/per-case
checkpoints, exact recipes, losses/source estimates at every update, gradient
audits, preview grids, physical metrics and PNG/PDF figures. The complete
machine-readable report includes `raw_results.csv`, `candidate_summary.csv`,
`paired_changes.csv`, `violation_correlations.csv`,
`source_initialization_diagnostics.csv`, `convergence_diagnostics.csv` and
`amplitude_diagnostics.csv`. Selection and multistart decisions are recorded
separately. Code/environment snapshots distinguish baseline execution,
candidate execution and later postprocessing. An artifact hash manifest and
unchanged-reference check accompany the final report.
`diagnosis_summary.json` adds machine-readable evidence, selection and complete
configuration; `diagnostic_supplement/` contains its own frozen protocol,
source/package/git snapshot, seven CSVs, 20 NPZ term fields, PNG/PDF term plot
and completion/hash manifests. It reads checkpoints/histories only, with no
new fit or re-selection. The previous completed report and all its figures,
raw results, candidate summary and selection remain byte-identical.
Zero values in per-run logarithmic loss plots use a 1e-30 display floor; CSV
values and all physical diagnostics retain exact zeros. No concentration or
source estimate is clipped for evaluation.
