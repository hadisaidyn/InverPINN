# Revision 3: preregistered single-source recovery protocol

This document and `configs/single_source_benchmark.yaml` are written before
held-out generation/training. The runner freezes their bytes and SHA-256 hashes
before making references; it also archives executed source files and environment
versions. The completed reference manifest is locked before any held-out fit.
No result-dependent changes to this protocol are permitted in this revision.

## Question and scope

Can the existing inverse PINN recover a single unseen Gaussian source, conditional
on the correct PDE, source family/width, transport, initial/boundary conditions,
noise-free observations, and this one fixed sensor geometry? Each scenario fits a
fresh model; "held-out" means excluded from protocol/hyperparameter development,
not withheld sensor readings within that scenario. All its sparse readings are
training data; dense off-sensor fields and source parameters are evaluation-only.
This is not forecasting, an atmospheric source discovery claim, or a comparison
with another estimator.

## Audit of the pre-revision implementation

`train_inverse_pinn.py` explicitly loads dense C/source truth only after fitting.
Its sparse loader nevertheless opens a mixed archive containing truth, reads
dense-grid axes, and passes a broad configuration to the optimizer. Thus existing
tests show non-use of selected truth fields, not a narrow structural interface.
The configured initial guess (0.5,0.5,0.5) is constant, not calculated from truth;
there is no checkpoint selection by dense error. This audit found no direct
source-truth use in the inspected optimization path. It cannot establish how old
hyperparameters were historically selected, nor certify old experiment claims.

The new entry point takes only copied sparse tensors, scalar u,v,D, known sigma,
a strict training-settings record, and an optimization seed. No scenario path,
ID, manifest, generation seed/ranges, source truth, dense arrays, or evaluation
grid is passed to that function. Unexpected settings/physics fields are rejected.
Generation, fitting and evaluation are separate stages. Evaluation refuses to
load truth until the training attempt has terminated and its status is saved.
This is an application-level dataflow boundary, not an operating-system security
sandbox against deliberately malicious code with filesystem privileges.

The existing source uses sigmoid coordinates and softplus strength (plus the
smallest normal float), with fixed sigma. The pointwise tanh MLP permits negative
outputs. Autograd includes coordinate-scaling chain-rule factors. The residual is
`C_t + u*C_x + v*C_y - D*(C_xx+C_yy) - S`; its sign agrees with the finite-difference
solver. All four losses are unnormalized MSEs. PDE MSE has units C²/time², so the
fixed weights are numerical choices in these synthetic units, not universal
dimensionless physical constants. No equations, activation, positivity treatment,
stencil, optimizer or loss definition is changed to improve recovery.

## Frozen physical and observation design

- Domain: x,y in [0,1], t in [0,0.8], in synthetic length/time units.
- PDE: `C_t + 0.35 C_x - 0.15 C_y = 0.005 (C_xx+C_yy) + S`.
- Source: `S=Q exp(-((x-xs)^2+(y-ys)^2)/(2*0.08^2))`, time-independent.
  Q is the peak concentration/time rate, not integrated mass emission.
- Zero initial C; zero Dirichlet C on all four edges. These are absorbing clean
  boundaries, not open outflow or impermeable walls. No atmospheric realism is
  implied. Source margins limit direct source truncation, not all plume-boundary
  interaction under wind.
- Independently draw xs,ys uniformly in [0.25,0.75] and Q uniformly in [0.8,1.4].
  The 0.25 margin is 3.125 source widths. There is no rejection based on recovery.
- Development: 10 cases, master seed 1909202601. Held-out: 30 cases, seed
  1909202602. NumPy SeedSequence spawns one independent integer seed per case;
  default_rng of that seed draws the three independent scalars. Freeze all IDs,
  seeds and source values once before references. Assert distinct IDs/parameters.
- Exactly 20 continuous sensor coordinates in `configs/single_source_sensors.json`.
  These reproduce revision 2's seed-42 layout; they are not moved to cover the
  sampled sources. The exact coordinate file is copied once and hashed. Individual
  self-contained input archives duplicate its values; bitwise identity is checked.
  This geometry is uneven and conditions the inference claim.
- 81 measurement times `0,0.01,...,0.8`; spatial bilinear and native/linear temporal
  reference sampling. No snapping and no noise draw: inputs equal clean readings
  bit-for-bit (zero noise, including t=0).
- References: fixed 641² grid, dx=dy=0.0015625, dt=0.00003125, 25,600 steps;
  same revision-2 explicit scheme. Save only 4 dense snapshots at 0.1,0.2,0.4,0.8
  and the sparse time series. Every scenario also runs 161² and 321² grids to
  calculate conditional Richardson estimates as in revision 2, target <1% for
  each assessed field and pooled sensor trajectory. No certified bound is claimed.
  A reference failing the estimated target remains in the manifest and results
  with its accuracy flag; it is neither silently dropped nor relabeled qualified.

## Frozen fit and stopping rule

Retain the existing inverse config: 3 hidden layers of width 64, tanh activation,
linear scalar output, coordinate scaling to [-1,1] inside the MLP; CPU float32.
Adam, constant learning rate 0.001, exactly 6,000 updates, no scheduler, no early
stopping or best-checkpoint selection. One update is the logged "epoch".
Per update: 512 sensor readings sampled with replacement; 512 independent uniform
interior collocation points; 128 uniform IC points; 64 uniform points on each of
four edges. Points resample each update. Weights: data 10, PDE 1, IC 1, BC 1.
Source initialization is always xs=ys=0.5, Q=0.5, sigma=0.08. It does not consult
truth; coincidental proximity is possible and must not cause resampling.
Fail on nonfinite predictions, losses, gradients, parameters or evaluation values.
Do not clip negative C or raw source parameters.

All 10 development and 30 held-out cases use optimization seed 42. A reduced
optimizer-variation design is chosen in advance to bound CPU cost: heldout_001,
007,013,019,025 (every sixth ID, independent of their sampled parameters/results)
also use 314 and 2718, for 50 fits total. Seeds affect network initialization and
training samples, not the fixed physical source guess. The subset is an ID-spaced
sample, not guaranteed spatial stratification. It is not a full 30×3 study.
Use 1 PyTorch thread per fit/reference; at most 2 independent worker processes.
Same-environment deterministic algorithms/seeds; do not claim cross-platform
bitwise reproducibility. Runtime, timestamps and serialized checkpoint ZIP bytes
are not deterministic scientific endpoints.

## Endpoints and diagnostic evaluation

Primary endpoint: continuous Euclidean source-location error in domain-length
units. Also report |Qhat-Q|/Q, and dense concentration relative L2, RMSE and MAE.
Dense metrics pool all 641² nodes of the 4 explicitly selected snapshots with
equal weight; they are a discrete snapshot-ensemble metric, **not** an integral
over the nonuniform time axis. Save per-snapshot metrics too. No t=0 dilution.
Report true/predicted xs,ys,Q for every run, without clamping C.

Secondary descriptive binary criterion: location error <= one source width
(0.08) AND relative Q error <=20%. One width is geometrically interpretable;
20% is a declared amplitude tolerance, not a scientifically validated universal
cutoff. Continuous outcomes remain primary. This is parameter recovery, not a
certificate of physical validity. `computational_success` (completed finite fit,
valid finite constrained parameters) and `recovery_success` are separate fields.
Training/evaluation/reference failures are retained, counted, and recorded with
null unavailable metrics; failed training cannot count as recovery.

Diagnostics use fixed independent seed 1909202699, never the training generator:
8,192 new interior points; 4,096 new IC points; 1,024 points per edge. Report PDE
RMSE/MAE, IC RMSE/MAE/maxabs, BC RMSE/MAE/maxabs, fraction of all dense predictions
strictly <0, and minimum C. Independent random streams provide disjoint continuous
sample designs almost surely; no training-point replay. No diagnostic tunes fits.
These finite pointwise checks are not a proof of global physical admissibility.

Aggregate **primary-seed held-out** cases, not pseudo-independent pooled restarts:
count, missing count, mean, median, sample std (ddof=1), q25,q75,q90,max for each
endpoint. Recovery rate uses all 30 as denominator with a 95% Wilson interval;
this describes scenario variation under the sampling distribution, not sensor
noise or atmospheric uncertainty. Finite-case continuous summaries explicitly
give denominators; do not hide missing/failed cases. Additional seeds are reported
separately as within-scenario ranges/std and changes in recovery status. Do not
select best seeds. Report the development results separately, never pool them.

## Figures and identifiability

Plot every held-out primary true/predicted position with connecting segments;
localization distribution; Qhat versus Q with identity; physical diagnostics.
Mechanically select best/lower-median/worst by localization rank (tie: scenario
ID), saving truth, prediction and absolute-error maps at t=0.8 with sensors and
true/predicted sources. If a failed fit exists it remains highlighted in the
results/failure accounting and cannot become an attractive substitute example.

Identifiability: predetermined heldout_001, 21×21 candidate locations over
[0.25,0.75]² (spacing 0.025), fixed **true** Q and known sigma/physics for this
evaluation-only controlled diagnostic. Calculate sensor MSE with the separate
201² inference solver, dt=0.00025, 3,200 steps, at the exact same physical sensor
coordinates/times. Also evaluate the exact true location and PINN location.
Mark truth and PINN on the map and report discretization mismatch at truth.
This controls strength to isolate location sensitivity and can overstate joint
identifiability of (xs,ys,Q). A coarse finite grid and one scenario cannot prove
global uniqueness. No Gaussian likelihood/noise scale/prior is specified: the
surface is **not** a posterior/probability/source-likelihood map. Compare grid
minimum location and outside-one-/two-width objective minima descriptively;
do not invent confidence regions from an arbitrary mismatch threshold.

## Execution, failure and preservation

`python scripts/run_single_source_benchmark.py` creates a new frozen run, generates
all references, runs the fixed fits, evaluates, and writes reports/landscape.
An existing output raises; explicit `--resume` validates hashes and skips only
already recorded attempts. Failed attempts are not retried with altered seeds.
Each run saves config, sparse-input hash, checkpoint when completed, per-update
loss/source logs, plots, metrics, attempt status and evaluation status. The root
saves manifest, protocol, code archive/hashes, git commit, package versions, CSV/
JSON summaries and publication-resolution figures. Generation source truth lives
outside training inputs. No large generated fields enter Git. The final report
must classify evidence conservatively and include worst failures and limitations.
