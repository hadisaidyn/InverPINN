# InverPINN

An incremental research project for inverse two-dimensional advection--diffusion using physics-informed neural networks to infer PM2.5 pollution sources from sparse observations.

The project provides a physics residual, Gaussian emission sources, and an explicit finite-difference forward solver with reproducible visualizations. A PINN and real-data pipeline remain future stages.

## Layout

- `configs/`: version-controlled settings for future synthetic, PINN, and Almaty stages.
  - `synthetic.yaml`: Gaussian source parameters, plotting grid, random seed, and output directory.
  - `pinn.yaml`: placeholder settings for a future PINN-training stage.
  - `almaty.yaml`: placeholder settings for a future Almaty PM2.5 and ERA5 integration stage.
- `data/`: data separated by lifecycle and provenance.
  - `raw/`: immutable source measurements and downloads; local data is ignored by Git.
  - `processed/`: derived, reproducible data products; local data is ignored by Git.
  - `synthetic/`: future synthetic datasets with exactly known ground truth.
- `src/inverpinn/`: installable Python package.
  - `physics/`: advection--diffusion residual and Gaussian source definitions.
  - `data/`: future synthetic-data and measurement-loading code.
  - `models/`: future PyTorch neural-network components only.
  - `training/`: future deterministic training orchestration and checkpoint handling.
  - `evaluation/`: future comparison metrics and inverse-source evaluation.
  - `visualization/`: future plot-generation code.
- `scripts/`: small command-line entry points for reproducible workflows.
- `notebooks/`: exploratory, non-authoritative analyses; reusable logic belongs in `src/`.
- `experiments/`: version-controlled experiment notes, manifests, and launch definitions.
- `tests/`: automated tests for every implemented stage.
- `results/`: generated checkpoints, metrics, figures, and resolved configurations; ignored by Git.

## Development

Create an environment with Python 3.11 or newer, install the project, then run the test suite:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

Read [AGENTS.md](AGENTS.md) before adding any research capability.

## Gaussian source visualization

```bash
.venv/bin/python scripts/visualize_source.py --config configs/synthetic.yaml
```

The source is `S(x,y) = Q exp(-((x-x_s)^2 + (y-y_s)^2)/(2 sigma^2))`.
The default center is `(0.30, 0.70)`, peak strength `Q = 1`, and width
`sigma = 0.08` on the unit square. Q is the peak rate, not the integrated
emission rate. The heatmap shows emissions, not transported concentration.

The script saves `results/synthetic_source.png`, the resolved configuration,
metrics, and a `.pt` checkpoint containing the evaluated field (no trained
model). All computations use a fixed grid and the configured seed.

## Forward simulation

```bash
.venv/bin/python scripts/run_forward_simulation.py --config configs/forward.yaml
# Optional GIF:
.venv/bin/python scripts/run_forward_simulation.py --config configs/forward.yaml --animate
```

`configs/forward.yaml` specifies constant u, v, D, timestep, grid dimensions,
steps, and a list of Gaussian sources (an empty list gives no emissions).
The unit-square grid starts with zero concentration. All four edges remain
at zero: ideal clean-air boundaries, not a closed or no-flux box. Boundary
nodes receive no emissions. Plumes near edges may not represent unbounded
atmospheric transport.

Forward Euler uses first-order upwind advection and centered diffusion. The
solver rejects `dt*(abs(u)/dx + abs(v)/dy + 2*D/dx**2 + 2*D/dy**2) > 1`.
Upwinding introduces numerical diffusion; this is a simple baseline solver.

Outputs are `results/forward_simulation_final.png`, metrics, resolved config,
and `forward_simulation.pt`, containing `C[t,y,x]` and coordinates `t`, `y`,
`x`. The history includes initial zeros and every timestep. An optional
`forward_simulation.gif` uses a fixed color scale. No neural network is involved.

## Sparse sensors

```bash
.venv/bin/python scripts/generate_sensor_datasets.py --config configs/sensors.yaml
```

Run the forward simulation first. This command samples its saved field at
5, 10, 20, and 50 fixed, uniformly random locations, using seed 42 and additive
Gaussian noise with absolute standard deviation 0.005 concentration units.
Settings are configurable. Sensor coordinates are nested across these counts;
noise draws are reproducible per dataset but not shared across counts.

`inverpinn.data.sensors.sample_sensors(C, x, y, n_sensors=N, seed=42)` returns
positions `[N,2]` in `(x,y)` order and measurements `[T,N]`. For manual placement,
replace `n_sensors` with `positions=torch.tensor([[0.3, 0.7], [0.5, 0.5]])`.
Set `noise_std=0` for noise-free data. Off-grid points use bilinear interpolation
on strictly increasing coordinate vectors. All stored times are sampled, and
negative noisy readings are retained to preserve the Gaussian noise model.

Sparse `.pt` checkpoints and CSV tables are saved in `data/synthetic/sensors/`;
they contain positions, times, and observations, without the complete field.
Overlays, per-dataset metrics, and resolved configs are in `results/sensors/`.
Provenance includes the forward configuration and checkpoint SHA-256. The
finite-difference field is numerical reference data, not an exact analytical
PDE solution; interpolation is tested separately with exact bilinear fields.

## Complete synthetic dataset in one command

With the installed environment activated (`source .venv/bin/activate`), run:

```bash
python scripts/generate_synthetic_dataset.py
```

This independently runs the forward solver and samples 5, 10, 20, and 50
sensors using `configs/dataset.yaml`; no prior checkpoint is required.
Change experiments with `--config path/to/config.yaml`. Use
`--output-dir data/synthetic/another_run` to keep a separate simulation;
existing output directories are never overwritten.

Each run saves these files together in `data/synthetic/dataset/` by default:

- `dataset.npz`: compressed numerical arrays, loadable without pickle.
  Contains `C[time,y,x]`, `t`, `x`, `y`, `source_positions[source,xy]`,
  `source_Q`, `source_sigma`, scalar `u`, `v`, `D`, `seed`, and `noise_std`.
  For each N, `sensor_positions_N[N,2]`, `measurements_N[time,N]`, and
  `clean_measurements_N[time,N]` store sensor coordinates, noisy readings,
  and noise-free readings for evaluation. Keep truth arrays out of inverse
  model training inputs unless the experiment explicitly calls for them.
- `metadata.json`: schema, dimensions, units, initial/boundary conditions,
  source/noise conventions, full configuration, software versions, code hashes,
  and the NPZ SHA-256 checksum.
- `config.yaml`: resolved experiment configuration.
- `metrics.json`: stability number, concentration range, and noise errors.
- `sensor_positions.png`: sensor overlays on the final concentration field.

The NPZ is also the complete simulation checkpoint. Multidimensional arrays
are not flattened into CSV. Ground truth here is the numerical forward
solution with exactly known source parameters, not an analytical PDE solution.
Reproducibility means equal arrays with the same config and software environment.

```python
import numpy as np
with np.load("data/synthetic/dataset/dataset.npz", allow_pickle=False) as data:
    concentration = data["C"]
    observations = data["measurements_20"]
```

## Data-only neural baseline

```bash
source .venv/bin/activate
python scripts/train_data_baseline.py
```

Configure `configs/data_baseline.yaml` to select the dataset, sensor count,
seed, network width/depth, optimizer updates, learning rate, and minibatch size.
The default uses 20 sensors and three tanh hidden layers of 64 neurons.
Inputs `[x,y,t]` are scaled to `[-1,1]`; output is scalar concentration.
Targets are RMS-scaled using only noisy sensor observations. Training minimizes
sensor MSE, with no PDE, boundary, initial-condition, or dense-field loss.
Each configured epoch is one randomly sampled minibatch optimizer update.

`results/data_baseline/` contains `model.pt`, resolved `config.yaml`,
`metrics.json`, `training_curve.png`, `predicted_concentration.png`, and
`predictions.npz` with full-grid predictions and sensor-loss history.
Choose a new output directory for repeat experiments. The checkpoint stores
input bounds and target scale: physical predictions are `model(points) *
checkpoint['target_scale']` after restoring the model state.

The numerical reference is loaded only after training, for evaluation.
Relative L2 is `||prediction-reference||₂ / ||reference||₂`, reported over
the entire space-time grid and separately at the final time. This includes
training sensor locations and is not an independent simulation test set.
For an all-zero reference the metric is undefined and saved as JSON null.
Predictions can be negative because the linear output is unconstrained.

## Automatic PDE differentiation

`physics/autodiff.py` computes coordinate derivatives without adding training:

```python
import torch
from inverpinn.physics.autodiff import concentration_derivatives, autograd_pde_residual

coordinates = torch.tensor([[0.3, 0.7, 0.2]], requires_grad=True)
C = model(coordinates)  # model must evaluate each row independently
derivatives = concentration_derivatives(C, coordinates)
r = autograd_pde_residual(C, coordinates, u=0.35, v=-0.15, D=0.005, S=0.0)
```

The returned derivative keys are `C_t`, `C_x`, `C_y`, `C_xx`, `C_yy`, each
of shape `[N,1]`. The residual is `C_t + u*C_x + v*C_y - D*(C_xx+C_yy) - S`,
consistent with the forward solver. Pass the evaluated source rate as `S`.
Coordinates must require gradients before evaluating the model; do not detach
outputs. For the data baseline checkpoint, multiply model output by its saved
target scale before differentiating to obtain physical concentration derivatives.

The implementation assumes independent samples (no cross-sample attention or
training-mode batch normalization). It retains higher-order graphs for future
parameter gradients. Affine and explicitly connected constant fields have zero
curvature; detached outputs raise errors. This differentiation module performs no training.

## Modular PINN losses

`training/losses.py` supplies `data_loss`, `pde_loss`,
`initial_condition_loss`, and `boundary_condition_loss`. Each is a mean squared
error; the PDE function squares precomputed residuals against zero. Initial and
boundary targets are explicit. The boundary loss is Dirichlet concentration MSE,
matching the forward solver, not a flux condition. Callers must evaluate models
at the correct sensor, interior, initial-time, and boundary coordinates.

```python
import logging
import yaml
from inverpinn.training.losses import combine_losses, log_losses

with open("configs/pinn.yaml") as stream:
    weights = yaml.safe_load(stream)["loss_weights"]
# components contains scalar tensors keyed by data, pde, initial, boundary.
total = combine_losses(components, weights)
logger = logging.getLogger("inverpinn.losses")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("pinn_losses.log"))
record = log_losses(logger, step=0, losses=components, weights=weights)
```

Every weight must be supplied explicitly in configuration, even a zero weight.
The logger emits raw losses, weights, weighted contributions, and total; its
returned detached dictionary can be saved as CSV or JSONL. The total retains
autograd gradients. Shape mismatches, empty batches, nonfinite values, and
invalid weights fail explicitly rather than silently broadcasting or hiding errors.
PDE MSE has units concentration²/time²; other losses have concentration², so
weights must reflect the chosen scaling. The configured equal weights are an
initial experimental choice.

## Forward PINN experiment (known source)

```bash
source .venv/bin/activate
python scripts/train_forward_pinn.py
```

The experiment reads `configs/pinn.yaml` and the unified synthetic dataset.
Source coordinates, strengths, widths, wind, and diffusion are known and fixed.
Only MLP weights are optimized. Training uses noisy sensor readings, interior
PDE residuals, zero initial concentration and zero concentration on all four
boundaries. Each Adam step resamples interior, initial and boundary points;
every edge receives the same number of points. Loss weights and sample counts
come from configuration. The physical source is evaluated continuously, while
the reference solver discretizes it and introduces numerical diffusion.

The default uses CPU float32, 20 sensors, three 64-neuron tanh layers and 3000
Adam updates. No dense concentration values enter optimization. Full-grid
reference data is loaded after training for relative L2, MAE and RMSE, both
over all space-time points and at the final time. These are reconstruction
metrics, not a held-out simulation score. Relative L2 is null for zero truth.

`results/forward_pinn/` stores `ground_truth.png`, `pinn_prediction.png`,
`absolute_error.png`, `training_losses.png`, `metrics.json`, `predictions.npz`,
`model.pt` (model and optimizer), resolved `config.yaml`, detailed `training.log`
and per-step `training_history.jsonl`. Plots show the final time; truth and
prediction share a color scale. Loss curves are raw component MSEs plus the
weighted total, recorded before each update on its sampled batch.

NaN/Inf inputs, predictions, residuals, losses, gradients or updated weights
abort with context. Failures within the run are saved in `failure.txt` and
re-raised, producing a nonzero exit status. No clipping or silent recovery is
used. Existing output directories are protected from overwriting.

## Inverse PINN: unknown source location and strength

```bash
source .venv/bin/activate
python scripts/train_inverse_pinn.py
```

`configs/inverse_pinn.yaml` initializes one source at `(0.5,0.5)` with `Q=0.5`.
The source width `sigma=0.08`, constant wind and diffusion are assumed known.
The optimizer jointly updates the MLP and three raw source parameters through
the sensor and physics constraints. True source coordinates, strength, and
dense concentration are not read until evaluation after training. The single
source assumption is part of this experiment, not an inferred source count.

`TrainableGaussianSource` maps raw locations through sigmoid and raw strength
through softplus plus a dtype-sized positive floor, without optimizer clipping.
Coordinates remain in `[0,1]` and Q remains positive. Initial locations must
be strictly interior for finite logits. Saturation near edges can reduce
gradients; constraints do not guarantee identification from sparse data.

The existing PINN trainer accepts this source module instead of known source
parameters. Each epoch is one Adam step. Post-update `x_s`, `y_s`, and `Q` are
recorded after every epoch in `training_history.jsonl` and `training.log`;
component losses in the same record are pre-update sampled-batch values.
All weights and initial guesses come from config. Defaults use 6000 updates
and a data weight of 10; no reference-field-based checkpoint selection occurs.

`results/inverse_pinn/` contains the network/source/optimizer checkpoint,
resolved configuration, source estimates, reconstruction/source-error metrics,
full-grid predictions, concentration heatmap, loss curves and parameter-history
plots. Restore `source_state_dict` to recover raw parameters and fixed width.
NaN/Inf checks apply to source gradients and updated raw parameters as well
as network computations. Existing runs are never overwritten.

## Source localization evaluation

```bash
python scripts/evaluate_inverse_pinn.py --run-dir results/inverse_pinn
```

Reusable functions in `evaluation/metrics.py` calculate Euclidean source
location error, absolute relative source-strength error, concentration relative
L2, MAE and RMSE. The evaluator checks the dataset checksum and exact grid/time
alignment. It supports equal-sized source sets via minimum-total-distance
Hungarian matching. Relative metrics with zero reference denominators are undefined
and saved as JSON null. Nonfinite values and mismatched shapes are rejected.

The command creates `evaluation/report.md`, `evaluation/metrics.json` and
`evaluation/source_localization.png` inside the run directory. True and
predicted locations share the same map over the final reference concentration.
Reports show full space-time and final-time concentration metrics and identify
units and evaluation scope. Relative errors are fractions in JSON and
percentages in the human-readable report. Use `--output-dir` for another report
location; existing evaluations are protected from overwriting.

## Sensor-count experiment runner

```bash
python scripts/run_sensor_sweep.py --config configs/sensor_sweep.yaml
```

The default sweep runs 5, 10, 20, and 50 sensors with five independent seeds
`[0,1,2,3,4]`, for 20 inverse-PINN experiments. Every run uses the same settings
from `configs/inverse_pinn.yaml` (6000 updates by default). Each seed regenerates
sensor positions and Gaussian noise and reseeds network initialization and
collocation sampling. The numerical reference field stays fixed. Sensor counts
are paired within a seed, with nested positions; noise is not shared across
counts. The experiment measures combined placement/noise/training variability,
not variability across different physical pollution scenarios.

Two independent training subprocesses run concurrently by default; configure
`workers` to change concurrency. Per-run files include config, checkpoint,
loss/parameter logs, predictions and plots. Seed-specific NPZ datasets and
the resolved sweep config preserve provenance. Outputs are under
`results/sensor_sweep/`:

- `raw_results.csv`: one row per count/seed, including all errors, estimated
  and true source parameters, status, and run directory; saved after each run.
- `summary.csv`: means and sample standard deviations (`ddof=1`) over five seeds.
- `report.md`: readable table and experiment definitions.
- `error_vs_sensor_count.png`: three error-bar panels with individual runs.

Source-strength error means `abs(Q_pred-Q_true)/abs(Q_true)` and concentration
L2 uses the full space-time grid. CSV relative errors are fractions; report and
plot display percentages. No failed runs or outliers are silently dropped.
Any failure leaves raw results and logs, produces a nonzero exit code, and
prevents an incomplete summary. Existing sweep directories are not overwritten.

## Measurement-noise robustness

```bash
python scripts/run_noise_sweep.py --config configs/noise_sweep.yaml
```

Runs Gaussian noise levels 0%, 2%, 5%, 10% and 20% with five seeds each, using
20 sensors and the same 6000-step inverse-PINN configuration. Noise percentage
is explicitly `100 * noise_std / RMS(clean_sensor_readings)` over all sensor
locations and stored times, including initial zeros. This is constant additive
Gaussian noise within a run, not a pointwise relative perturbation. Each dataset
stores its percentage, clean RMS, and actual noise standard deviation.

Across noise levels, each seed holds sensor positions, standard-normal draws,
network initialization and collocation sequences fixed. Across seeds, these
factors vary independently. No negative readings are clipped. All runs and
their checkpoints, plots, configurations and logs are retained, including poor
fits; the numerical reference is fixed and only used for generating observations
and evaluating final estimates.

`results/noise_sweep/` contains `raw_results.csv`, `summary.csv`, `report.md`
and `localization_error_vs_noise.png`. Error bars show mean ± sample standard
deviation (ddof=1) across five seeds, not confidence intervals. Faint lines show
individual paired seeds. Sensor placement and optimization variability can
dominate the noise effect; zero measurement noise need not imply zero error.

## Controlled wind experiments

```bash
python scripts/run_wind_sweep.py --config configs/wind_sweep.yaml
```

The magnitude comparison uses speeds 0, 0.2, 0.4 at 0°; the direction comparison
uses 0°, 90°, 180°, 270° at speed 0.4. The shared reference (speed 0.4, 0°)
is trained once, giving six unique conditions and 30 runs with five paired seeds.
Angles mean flow **toward**, counterclockwise from +x: 90° is +y. Calm wind
has no direction and is blank in the tidy CSV direction column.

Every condition gets a new forward concentration field while retaining source
parameters, diffusion, grid, timestep, time horizon, initial/boundary conditions
and sensor count. A common timestep must satisfy CFL for every wind; it is not
silently adjusted. Per seed, sensor positions, additive absolute noise draws,
network initialization and collocation samples remain fixed across conditions.
Noise std is 0.005 concentration units, so SNR can change with the plume.
Wind remains known to the inverse model; only source location/strength and the
concentration network are trained with identical settings.

`results/wind_sweep/raw_results.csv` is tidy: one row per condition and seed,
with wind speed/direction, u/v, estimated and true source parameters, errors,
status, and a paired localization-error change versus the reference wind.
`summary.csv` gives mean and sample SD (ddof=1), including paired changes;
`conditions.csv` records the design and stability numbers. `report.md` and
`localization_error_vs_wind.png` summarize the comparisons. All reference
fields, configurations, raw logs, predictions and checkpoints are retained.

These are controlled comparisons in one finite domain, not general atmospheric
claims. Source placement relative to the boundaries and direction-dependent
upwind numerical diffusion also affect the results. Error bars represent seed
variability, not confidence intervals; failed runs are not silently discarded.

## Two-source inverse problem and permutation symmetry

```bash
python scripts/generate_synthetic_dataset.py --config configs/two_sources.yaml
python scripts/train_inverse_pinn.py --config configs/inverse_two_sources.yaml
python scripts/evaluate_inverse_pinn.py --run-dir results/inverse_two_sources
```

`TrainableGaussianMixture` sums K constrained Gaussian components. A list under
`source_initial` configures K; the original dictionary remains supported for
single-source runs. Every component has sigmoid x/y and softplus Q; widths and
K remain known. All 3K raw parameters and the concentration MLP train jointly.
No ordering or repulsion is imposed. Distinct initial guesses avoid exact
initial symmetry, but coincident/nearby source estimates can still be a local
minimum or reflect non-identifiability. The source-list order is not physical.

Evaluation uses `matched_source_metrics` to minimize total Euclidean distance
over a one-to-one assignment. Per-source localization and strength errors use
that same assignment and are averaged for summaries. Unequal source counts
are rejected; missing/extra-source penalties are not yet defined. Coincident
sources or equal-cost assignments have no uniquely identifiable correspondence.
The report map draws the assigned pairs. Per-epoch logs use `source/0/x_s`,
`source/0/y_s`, `source/0/Q`, etc.; checkpoints retain all component states.

The initial two-source experiment uses 50 sensors and 6000 updates. Its two
estimates cluster around the stronger source instead of recovering both sources;
the saved report records this failure. Generalized model support does not imply
reliable source-count recovery or convergence for every initialization.

## Fixed spatial interpolation baselines

```bash
python scripts/evaluate_spatial_baselines.py --config configs/spatial_baselines.yaml
```

IDW (power 2), thin-plate RBF (degree 1, zero smoothing), and ordinary kriging
(unit-sill exponential covariance, range 0.2, nugget 0) independently reconstruct
each observed time slice. Settings are prescribed in configuration before
evaluation, with no dense-truth tuning, boundary pseudo-observations or clipping.
The simple kriging covariance is not fitted or claimed to yield calibrated
uncertainty. Duplicate sensors are rejected; RBF needs noncollinear positions.

All methods read the exact same saved noisy observations and sensor positions
as the two-source PINN (50 sensors), not regenerated measurements. Dataset,
position and observation hashes document this. PINNs additionally receive
physics constraints, so information and compute budgets differ.
`results/spatial_baselines_two_sources/` saves a report, metrics CSV, final-time
comparison figure, full prediction arrays, config and an interpolation checkpoint
containing observations/positions. Full-grid and final-time L2, MAE, RMSE are
reported; they include extrapolation outside the sensor convex hull.

## Classical PDE-constrained inverse baseline

```bash
python scripts/fit_classical_inverse.py --config configs/classical_inverse.yaml
```

This single-source baseline calls the existing finite-difference solver for
every trial `(x_s, y_s, Q)` and minimizes the sum of squared sensor residuals
using SciPy bounded trust-region least squares. It uses the same bilinear
observation operator, all saved noisy readings, and no dense truth during
optimization. Wind, diffusion, Gaussian width, zero initial concentration and
zero Dirichlet boundaries are known. Coordinates stay in `[0,1]`; strength has
a tiny positive floor of `1e-12` and no upper bound. Q is peak source intensity.

The default uses exactly the original single-source PINN's 20 sensors, width
0.08 and initial guess `(0.5,0.5,0.5)`. Optimization is deterministic, with no
resampling or truth-based tuning. Configuration records the dataset and sensor
hashes, numerical grid, optimizer settings and package versions. Outputs in
`results/classical_inverse/` include metrics/report, prediction arrays,
parameter-and-observation checkpoint, comparison plot and objective-call CSV.
Calls include finite-difference Jacobian probes and rejected trials; they are
not optimizer iterations. Nonconvergence is saved and raises an error.

This is a local optimizer, not a guarantee of global recovery; sparse sensors
or weak signals can leave the source poorly identifiable. It currently fits
one source, not K sources. Synthetic generation and inversion use the same
discrete solver (an **inverse crime**), making this an intentionally optimistic
implementation baseline. It does not establish performance under discretization
error, model mismatch, or real atmospheric conditions. Compare with the PINN's
continuous-PDE formulation with that distinction and compute cost in mind.

## Paired PINN loss ablations

```bash
python scripts/run_ablation.py --config configs/ablation.yaml
```

Runs A (full), B (PDE weight zero), C (boundary weight zero), D (initial
weight zero), and E (configured PDE weights 0.1 and 10 versus full weight 1).
All other settings remain fixed: 6000 updates, architecture, optimizer, source
initial guess, stored sensor positions/noisy readings, and paired initialization
and sampling seeds `[0,1,2]`. The dataset list supports multiple numerical
scenarios; the initial run uses the original one-source, 20-sensor dataset.
No dense-truth selection or retraining of unfavorable outcomes is performed.

Results include per-run plots, checkpoints, full loss histories and configurations,
`raw_results.csv`, `summary.csv` (mean, sample SD and paired changes from A),
`report.md` and `ablation_comparison.png` under `results/ablation/`.
SD measures training-seed variability on fixed data, not independent datasets
or confidence intervals. In B the source is structurally unidentifiable and
stays at its initial guess: the source enters only the removed PDE term. B
still retains IC/BC losses. Removing IC/BC penalties does not remove relevant
sensor observations, including t=0. Disabled terms remain logged/computed to
preserve identical sampling streams; this is not a computation-speed ablation.

## AirKaz access research

See [historical AirKaz access and restrictions](docs/airkaz_data_access.md),
checked 19 September 2026. An official daily-data XLSX link is verified, but
reuse consent, unambiguous historical dates/timezones and coordinate QA remain
unresolved. No public API was verified and no scraping/ingestion code was added.

## ERA5 ingestion, coordinates and modeling dataset

```bash
python scripts/run_meteorology_demo.py
# After supplying authorized local inputs and editing the configuration:
python scripts/prepare_modeling_dataset.py --config configs/almaty.yaml
```

The [pipeline documentation](docs/meteorology_pipeline.md) specifies input fields,
UTC timestamp rules, WGS84 / UTM 43N (EPSG:32643) transforms, wind-vector basis,
units, missing-data policy and leakage limits. The pipeline processes local
hourly ERA5 NetCDF, bilinearly samples meteorology at sensor coordinates, and
uses only prior/current valid times within a configured maximum age. No missing
hour/cell is filled. Longitude/latitude and original observations are retained.

Raw content-addressed caches live separately from processed NetCDF datasets.
`modeling.nc`, QC summaries/plot, resolved configuration and provenance form a
reproducible preprocessing result. The synthetic demo saves under
`data/processed/modeling_demo/`. Real-time forecasting is explicitly unsupported:
ERA5 reanalysis availability and assimilation are not causal at observation time.
No AirKaz scraper or ERA5 downloader is included.

## Exploratory real-data winter case-study pipeline

```bash
python scripts/run_real_case_study.py --config configs/real_case_study.yaml
```

Requires verified real `modeling.nc`, `meteorology.nc`, paired provenance and
explicit usage-permission confirmation. **Only synthetic demo inputs currently
exist; no real episode/result is claimed.** The default command therefore fails
with a missing-input message until the configuration is completed.

The [case-study protocol](docs/real_case_study.md) defines the winter selection
rule, held-out sensors, physical scaling, hourly regional wind approximation,
initial/boundary priors and uncertainty limits. Outputs are source-intensity
maps (not calibrated likelihoods or emitter inventories), sensor validation,
checkpoints and a report separating measured facts, model estimates and
interpretation. Synthetic tests cannot be relabeled as a real Almaty study.

## Thermal inversion diagnostic

```bash
# Reproducible manufactured-profile verification (not Almaty observations):
python scripts/diagnose_inversions.py --config configs/inversions_demo.yaml
# After supplying vertical profiles, confirmed study dates and height domain:
python scripts/diagnose_inversions.py --config configs/inversions.yaml
```

The [scientific definition and references](docs/thermal_inversions.md) were
researched before implementation. The diagnostic detects increasing actual
temperature with height, without an arbitrary strength/depth threshold. It
reports layer geometry, temperature rise, unresolved boundaries, coverage and
consecutive candidate-event samples, without filling missing values.

Outputs include `inversion_events.csv`, `inversion_layers.csv`,
`profile_coverage.csv`, a diagnostic plot, NetCDF input checkpoint, configuration,
hashes, summary and interpretation report. CSV rows retain a real/synthetic label.
The method check is saved in `results/inversions_method_check/`.

**The observed study-period table remains unavailable:** current single-level
meteorology cannot establish a vertical temperature inversion, and no confirmed
study interval or real vertical profiles have been supplied. The default run
writes `results/inversions_almaty/report.md` with `not_assessed` status and fails,
rather than reporting a misleading zero-event table. Output directories must be
new; choose a different directory in the configuration to rerun.

## Reproduce paper figures and tables

```bash
python scripts/reproduce_paper.py
```

The [paper manifest](configs/paper.yaml) orchestrates all declared synthetic
experiments with their existing training budgets and seed lists. Results go to
a **new** `results/paper/` directory: final 300-dpi PNG/vector PDF figures and
CSV/JSON tables in `paper/`, original runs in `experiments/`, plus package
versions, Git status/commit, source snapshots, saved configs and artifact hashes.
No existing run is overwritten. Failures preserve logs and stop dependent stages.

Real-data/inversion stages are explicitly excluded until verified inputs exist.
This workspace currently has no Git commit; that is recorded as unavailable.
Set `require_git_commit: true` in the manifest for a committed archival release.

For a quick pipeline check, **not paper results**:

```bash
python scripts/reproduce_paper.py --config configs/paper_smoke.yaml
```

Use `--output-dir results/paper_new_run` for another run. See the
[reproduction protocol](docs/reproduce_paper.md) for configuration, determinism,
artifact selection and publication-format limitations.
