# P0 revision 2: selective synthetic reference generation

## Inspection completed before implementation

Revision 1 was committed on `scientific-revision` as
`12b12bc868aae9837a4d1888f6a4ec95d49b45fa` before this work. Generated
artifacts remained ignored; the working tree was clean after the commit.

### Data actually required

| Consumer | Required quantities | Unnecessary current coupling |
| --- | --- | --- |
| Data-only/inverse PINN training | Sensor x,y; measurement times; noisy C; domain; known u,v,D for PINN | Scripts obtain time/domain coordinates from the same NPZ as dense truth |
| Known-source forward PINN | Same, plus source parameters as explicitly known physics | Truth parameters are intentionally inputs only for this forward task |
| Source localization evaluation | True source coordinates, Q, width; estimates | Truth lives beside inputs in the legacy NPZ |
| Concentration evaluation and heatmaps | Full fields at explicitly selected evaluation times; final field | Training/evaluation scripts allocate predictions matching all stored truth times |
| Spatial interpolation baselines | Sparse readings; query coordinates; truth at chosen evaluation times | `evaluate_spatial_baselines.py` assumes observations and dense truth share one time axis |
| Sensor/noise sweeps | A fixed pool of positions and clean readings at requested observation times; shared noise draws | `run_sensor_sweep.py` loads/copies all C and resamples it; noise sweep invokes it |
| Wind sweep | A selective reference per wind setting | `run_wind_sweep.py` creates/saves all integration-time fields |
| Classical inverse | Sparse readings at physical times; independent inference discretization | `fit_classical_inverse.py` derives nx,ny,dt,steps directly from reference x,y,t; `fit_source` requires steps+1 observations and returns a full field history |
| Animation/debugging | A modest sequence of frames | All internal steps are unnecessary unless deliberately debugging the integrator |

Specific full-history assumptions occur in `generate_synthetic_dataset.py`,
`generate_sensor_datasets.py`, `run_sensor_sweep.py`, `run_wind_sweep.py`,
`train_data_baseline.py`, `train_forward_pinn.py`, `train_inverse_pinn.py`,
`evaluate_inverse_pinn.py`, `evaluate_spatial_baselines.py`, and
`physics/inverse_source.py`. Their current small-grid API remains supported.
No scientific objective requires a dense field at every integration step.
Initial concentration and boundary values are analytically known zeros.

## Scope and storage contract

Keep the validated forward-Euler/upwind/centered-diffusion stencil unchanged.
Add explicit `full_history`, `final_only`, `selected_times`, and `sensor_only`
outputs to the same solver. The default still returns the legacy full tensor;
`return_history=False` remains a final-only compatibility alias.

`selected_times` may collect sparse sensor observations in the **same** pass.
Both selective modes keep two working fields, not a dense integration history.
Requested field snapshots are copied so later updates cannot mutate them.
The final field is one selected snapshot, not a second duplicate array.

Physical output times need not coincide with internal steps: explicitly use
piecewise-linear interpolation between the two bracketing Euler states, never
round arbitrary times silently or change dt to hit output times. Exact native
times use the stored native field. Out-of-domain/nonfinite/unsorted/duplicate
times fail. Spatial point observations use the existing tested bilinear sampler
on each grid; no nearest-node snapping and no out-of-domain extrapolation.

Storage scales as O(nx*ny + n_field_times*nx*ny + n_measurement_times*N).
For sensor-only output it is O(nx*ny + n_measurement_times*N), independent of
the number of integration steps except for requested sparse output counts.

## Planned verification before pilot generation

1. Prove full-history/selective equality at native times, verify explicit
   interpolation off the native lattice, and guard against history allocation.
2. Benchmark the same single Gaussian on 161, 321 and 641 grids with fixed
   physics, common physical observation/evaluation times, and stable dt.
3. Distinguish measured inter-grid differences from Richardson estimates; no
   Gaussian result is an analytic error or certified bound. Inspect selected
   field times and pooled sensor trajectories, not only the final frame.
4. Separate source/evaluation truth from the model-input loader, record complete
   configuration/provenance, and test independent inference discretization.
5. Only after tests, generate three single-source pilot scenarios; no training,
   parameter-recovery comparison, suite rerun, or historical-data replacement.

## API and numerical meaning

```python
result = solve_advection_diffusion(
    **solver, sources=[source], output_mode="selected_times",
    output_times=[0.1, 0.2, 0.4, 0.8],
    sensor_positions=positions, measurement_times=measurement_times,
)
# result["fields"] is [4, ny, nx], not [steps+1, ny, nx].
# result["measurements"] is [number of measurement times, number of sensors].
```

`final_only` returns a 2D tensor; `full_history` returns the legacy 3D tensor.
`sensor_only` returns the same dictionary as `selected_times`, with an empty
field axis. The dictionary contains coordinates/times for every returned array.
No measurement noise is added in the solver. Generation adds independent
Gaussian noise with explicit absolute standard deviation and a separate local
noise seed; negative readings are not clipped. Native output-time equivalence
tests use exact floating-point equality. Off-step samples are compared to an
independently assembled interpolation of the full history with tight tolerance.

Time interpolation is an approximation to the same discrete trajectory, not
an assertion of exact continuous-PDE values. Refinement estimates below include
the configured discrete observation map. These synthetic coordinates and times
are nondimensional/model units, not metres, seconds, or real PM2.5 measurements.

## Reference versus inference: enforceable minimum separation

Reference candidates are 161/321/641 nodes per axis with dt=0.0005/0.000125/
0.00003125, all ending at t=0.8. The **separate**
`configs/reference_inference.yaml` specifies 201x201 and dt=0.00025 (3,200
steps). Its spatial grid is not nested with 641x641. Known u,v,D come from the
source-free model inputs, not the source-truth config. Source trial parameters
must be supplied explicitly to `inference_observations`; there is no optimizer
and it never opens truth files.

Both use bilinear point observations in space and linear interpolation in
physical time, but the cell indices, interpolation weights, and time brackets
are computed separately on each grid/time lattice. The same **physical sensor
coordinates and observation times** are used for all methods, with no snapping.
The separation validator rejects identical grids, source-bearing inference
configuration, mismatched domains/time intervals, or an unstable inference dt.
Regression tests show nonzero reference/inference sensor mismatch at the same
source parameters; the coarse operator cannot simply reproduce its generating
discrete state. This is a mitigation of the same-grid inverse crime, **not**
independent validation of the shared PDE, boundary conditions, or stencil family.

The legacy classical fitter is **unchanged** and still has the old same-grid
assumption. It must not be used as a schema-2 benchmark by relabeling coordinates
or converting selected fields back into an apparent full history. Future
adaptation must consume the sparse input loader, explicitly configured trial
operator, and evaluation-only snapshots. No baseline comparison was run here.

## Schema 2 and anti-leakage boundary

Each pilot scenario contains:

```
inputs/
  observations.npz       # positions[N,2], times[M], noisy measurements[M,N]
  physics.json           # known u,v,D; domain; units; initial/boundary conditions
truth/
  source.json            # true x_s,y_s,Q,sigma and Q interpretation
  grid_<n>/checkpoint.npz # fields[K,ny,nx], x,y, field_times; clean sensor samples
  grid_<n>/config.yaml   # exact per-grid integration/observation settings
  grid_<n>/metrics.json  # runtime, peak RSS, retained bytes, CFL, checkpoint hash
  ...                    # three grids of convergence evidence, not duplicate fields
  reference_benchmark.csv
  reference_summary.json
  config.yaml, provenance.json, environment.json, source_snapshot.zip
config.yaml              # resolved scenario/validation configuration; not inputs
metadata.json            # schema, seeds, accuracy, provenance, truth references
metrics.json
pilot.png                # evaluation-only final field, sensors and true source
```

Temporary worker-request YAMLs are retained next to the per-grid folders for
auditability. The final field is `fields[-1]` at the selected reference path in
metadata; it is **not saved again** under a separate final-C file. Lower-grid
checkpoints are distinct convergence evidence. Clean sensor samples live only
in truth checkpoints, while noisy observations live only in inputs.

`load_model_inputs(scenario)` opens only `inputs/observations.npz` and
`inputs/physics.json`. It rejects extra NPZ keys and unexpected physics keys.
Its feature matrix contains **only x,y,t** and its targets contain only noisy
C. It does not open metadata, source truth, generation config or dense fields.
The file boundary prevents accidental loader leakage, not deliberate access
to evaluation files. Known-source forward experiments would require a separate
explicit API, not a hidden exception in this inverse-input loader.

Metadata records source-truth file paths, full physical/configuration settings,
grid/dt/steps, source convention, seed/noise seed, exact sensor coordinates and
measurement/evaluation times, units, Git commit, dirty-state source hashes and
a code archive, UTC generation time, package versions, checkpoint hashes, and
conditional reference-error estimates. Arrays are float64 in NPZ files.

## Accuracy and resource accounting

The benchmark compares all three grids at identical observation coordinates
and times. At each requested field time, all three fields are restricted to the
coarsest common nodes without interpolation. Write d0=||C_h-C_h/2|| and
d1=||C_h/2-C_h/4||. For decreasing, nonzero differences, estimate
`p=log2(d0/d1)` and `C_extrap=C_fine+(C_fine-C_medium)/(2^p-1)`.
Every candidate is compared with this extrapolation. Sensor-trajectory norms
pool all requested times and sensors; they are not per-reading relative errors
near zero. Additional rows give final-field differences on each pair's native
coarse grid. These are labeled **measured differences**, not exact errors.

Choose the lowest-runtime tested grid whose estimates are defined and <=1%
for **every selected field time and the pooled sensor trajectory**. Undefined
estimates never count as zero or a pass. The 1% objective remains an estimated
acceptance test, not a mathematical bound. Common-grid quadrature does not
certify fine-grid maxima or unrequested times; pooled sensor norms do not
guarantee pointwise accuracy for low-concentration readings.

Each grid is executed in a fresh subprocess. Peak RSS is the process high-water
mark, including Python/PyTorch, not just tensor storage. Pre-solve high-water
RSS and its increase are also saved (the increase is not an exact allocation
peak). The RSS reading is taken immediately after integration: it is a
**solver-phase process peak**, not an end-to-end generation peak including
subsequent NPZ compression, plotting, or the parent process. Analytic
retained-array bytes and hypothetical full-history bytes are
reported separately. Wall runtime covers integration and selective sampling,
excluding process startup, compression and plotting. No result is relabeled
as a model-recovery benchmark.

## Commands and restrictions

```bash
python scripts/benchmark_reference.py --config configs/reference_benchmark.yaml
python scripts/generate_reference_pilot.py --config configs/reference_pilot.yaml
```

All output directories must be new. Both commands accept `--output-dir`.
The pilot command allows at most three safe, unique scenario IDs with distinct
seeds. Every scenario repeats its own three-grid validation. If no candidate
meets the desired estimated target, it is explicitly marked unqualified and
the finest result is retained only as a diagnostic pilot; the target is never
weakened. All pilots set `benchmark_release_ready=false` regardless of results.
No old dataset is overwritten or regenerated, and no PINN, inverse optimizer,
sensor/noise/wind suite, multi-source inference, or real-data pipeline is run.

## Representative benchmark results (2026-09-19)

Known Gaussian: x_s=0.30, y_s=0.70, Q=1, sigma=0.08; u=0.35, v=-0.15,
D=0.005, final time 0.8. Fixed placement seed 42, 20 sensors, 81 measurement
times (0 to 0.8), and four field times (0.1, 0.2, 0.4, 0.8).

| Grid | dx=dy | dt | Steps | Integration + sampling (s) | Solver-phase peak RSS (MB) | Max assessed Richardson relative L2 estimate |
| --- | --- | --- | --- | --- | --- | --- |
| 161x161 | 0.00625 | 0.0005 | 1,600 | 0.187 | 273.7 | 2.8674% |
| 321x321 | 0.003125 | 0.000125 | 6,400 | 2.214 | 288.6 | 1.4959% |
| 641x641 | 0.0015625 | 0.00003125 | 25,600 | 85.004 | 331.8 | 0.7803% |

MB denotes decimal 1,000,000 bytes. On this machine 641 is the **cheapest
tested qualifying configuration**, not a proven globally optimal grid.
The measured final-field inter-grid differences are 1.3848% (161 vs 321)
and 0.7191% (321 vs 641). There is no next-finer run for 641, so its inter-grid
difference is **null**, not zero. Its remaining error is estimated from the
three-grid Richardson analysis, with final-field p=0.938874.

At 641, field-time error estimates are 0.2076%, 0.3702%, 0.5903%, 0.7803%;
pooled sensor-trajectory error is 0.6900% (p=0.970737). These all meet the
desired **estimated** 1% target for this representative design. No pointwise,
all-time, independent-solver or certified error claim follows.

Four 641-square snapshots occupy 13,148,192 bytes; 81x20 sensor readings
occupy 12,960 bytes. Hypothetical full-history storage is 84,151,715,848 bytes.
The measured pre-solve process high-water RSS is 279.4 MB and its increase is
52.4 MB; absolute process peak is 331.8 MB. Integration is practical on this
machine without allocating tens of gigabytes.

Saved `storage_equivalence.json` reports maximum absolute differences of
**0.0** for final fields, selected native-time fields, sensor-only readings,
and combined selected-mode readings relative to full history. A separate
comparison against committed revision 1 on the legacy 81-grid configuration
also produced bitwise identical full histories, maximum difference **0.0**.
For an additional off-step check at t=0.002,0.017,0.063,0.099 on a 17x13
grid with dt=0.005, explicit interpolation of full history differed from
selective fields by **0.0**, and sensor readings by **1.38778e-17**. The latter
is roundoff from commuting the spatial and temporal linear interpolations,
not a change in PDE evolution.

The old 81-grid datasets remain historical implementation/debugging fixtures,
**not valid <=1% numerical benchmark ground truth**. They have not been deleted,
overwritten or relabeled as high-accuracy references.

## Regression evidence and remaining defects/limitations

Tests executed before production pilot generation:

- New selective-output/reference infrastructure tests: **30 passed**, 24.39 s.
- Solver, numerical-validation, sensor and selective-output regression group:
  **65 passed**, 2.06 s (overlaps the new group; do not add counts together).
- Full suite: **267 passed, zero failures**, 86.80 s; **120 warnings** retained.
- Warnings are one NumPy/netCDF4 binary-compatibility warning and 119 existing
  xarray/netCDF4 NumPy shape-assignment deprecation warnings. CLI plotting also
  reports that the normal Matplotlib cache is not writable and uses a temporary
  cache. No warning is hidden or used to change a scientific assertion.

Ranked findings:

1. **High, legacy reference limitation:** 81x81 numerical fields are not
   accuracy-qualified ground truth. The new route supplies refinement evidence;
   historical datasets/runners are preserved, not silently upgraded.
2. **High, legacy inverse-crime coupling:** the existing classical runner
   derives its grid/time integration from the dataset. This milestone supplies
   and tests an independent inference configuration/forward observation map,
   but deliberately does not modify or run the optimizer. Consumer migration
   remains required before a fair final comparison.
3. **Medium, fixed for the new route:** all-times dense storage was an API/data
   layout assumption, not a mathematical requirement. Selective modes remove
   it and reject invalid physical-time requests. Legacy full-history mode stays
   available explicitly for small debugging runs.
4. **Scientific qualification limit, not a certified defect:** three-grid
   Richardson estimates assume asymptotic power-law error. They do not bound
   arbitrary source positions, every sensor reading, unrequested times or
   alternative boundary physics. Each pilot therefore repeats the check and
   remains an infrastructure pilot even when the estimated target is met.

No new PDE-sign/stencil/evolution defect was found in tested valid modes. No
failing test was weakened and no PINN architecture, optimization system,
classical fitter, real-data pipeline or historical dataset was changed.

## Three-scenario pilot results

All three independent scenarios completed at 641x641 with the configured
dt=0.00003125. Each has 20 fixed sensors, 81 measurement times (1,620 noisy
readings), and four selected field snapshots. Independent local placement and
noise seeds are recorded; Gaussian absolute noise std is 0.005 throughout.

| Scenario | Seed | True (x_s,y_s,Q) | Max assessed Richardson relative L2 estimate | 641 integration/sampling (s) | Solver-phase peak RSS (MB) |
| --- | --- | --- | --- | --- | --- |
| pilot_001 | 101 | (0.25, 0.65, 0.8) | 0.8266% | 94.006 | 348.7 |
| pilot_002 | 202 | (0.50, 0.35, 1.1) | 0.7743% | 98.507 | 321.4 |
| pilot_003 | 303 | (0.65, 0.60, 1.4) | 0.7856% | 97.367 | 347.3 |

Each scenario occupies approximately 16.8 MB on disk **including all three
refinement checkpoints**, metadata, plot and code provenance. The pilot total
is about 50.4 MB (48 MiB). No large binary was added to Git. These are
infrastructure pilots, not three successful localization/recovery experiments;
every scenario retains `benchmark_release_ready=false`.

Post-generation verification confirmed every reference array is finite,
reference shape is [4,641,641], clean/noisy observations have shape [81,20],
model features have shape [1620,3] with columns only x,y,t, stored noisy readings
are exactly reproduced by the recorded noise seeds, and all input/reference/
source-archive hashes match. Source truth remains in evaluation-only files.

## File inventory and Git handoff

[The complete file inventory](synthetic_reference_files.txt) enumerates every
modified/created task file: 13 source/documentation files and 100 ignored
generated artifacts, each with its absolute path. The representative benchmark
is in `results/reference_benchmark/`; the three pilots are in
`data/synthetic/reference_pilot/`. No generated artifact was force-added.

Branch: `scientific-revision`. Latest committed milestone:
`12b12bc868aae9837a4d1888f6a4ec95d49b45fa`,
`Validate finite-difference numerical convergence`. The current revision-2
implementation, tests, configs and documentation remain uncommitted for review.
The Karpathy skill's narrow-change guidance kept the numerical stencil and
training/real-data systems unchanged; only storage, reference validation,
input separation and related provenance infrastructure were added.

**Stop point:** reference infrastructure and three pilots only. No PINN
retraining, single-source recovery benchmark, full sweep, real-data revision,
or multiple-source inference was started.
