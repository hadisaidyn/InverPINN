# P0 revision 1: numerical accuracy validation

## Inspection before scientific code changes

The original implementation is preserved by commit
`38230c254f4d03e031b2431714ddfcb94c053062` (`Initial research snapshot`).
Inspection of `src/inverpinn/physics/finite_difference.py` found:

- Uniform nodal grid on `[0,1]^2`, including both endpoints.
- Forward Euler in time; first-order, sign-dependent upwind advection;
  centered three-point second differences for diffusion in each direction.
- PDE: `C_t + u C_x + v C_y = D(C_xx + C_yy) + S`.
  The implemented advection, diffusion, and source signs agree with this PDE.
- Zero initial concentration; homogeneous Dirichlet values on **all four**
  edges for all times. These are absorbing/clean-air boundaries, not no-flux
  or general open-outflow conditions. Only interior nodes receive forcing.
- Time-independent, additive Gaussians. `Q` is the peak concentration/time
  rate, not the integrated emission rate; `sigma` is a length.
- The combined monotonicity bound is
  `nu = dt (|u|/dx + |v|/dy + 2D/dx^2 + 2D/dy^2) <= 1`.
  The solver rejects violations before allocating the solution history.
- Expected truncation error for smooth solutions is
  `O(dt + |u| dx + |v| dy + D dx^2 + D dy^2)`.
  Thus expect first-order spatial convergence with nonzero wind, second-order
  spatial convergence without wind, and first-order time convergence.
- No continuous-solution convergence tests existed. Transport-direction,
  positivity, superposition and limiting-case tests are not accuracy evidence.
- Full history costs `8 (steps+1) nx ny` bytes: about 5.28 GB for a 321-square
  grid, 6,400 steps. A final-state-only option is needed for affordable studies.

This inspection was completed before changing the stencil or its interface.
The validation extension retains the original stencil, zero initial/boundary
conditions and default full-history return. It adds optional signed,
time-dependent manufactured forcing, evaluated at the **old** Euler time, and
optional final-state-only storage. Neither changes the Gaussian-data defaults.

## Manufactured solutions and units

All quantities here are in consistent **synthetic domain, time, and
concentration units**, not metres, seconds or measured PM2.5 units. Wind has
length/time units, D length squared/time, and S concentration/time.
The formulas below use nondimensional coordinates x/L0, y/L0, t/T0 and C/C0,
with the reference scales numerically set to one. In particular, `exp(t)`
means exp(t/T0), not the exponential of a dimensional time. Dimensional
transport coefficients become `u*T0/L0` and `D*T0/L0**2` in this formulation.
Manufactured forcing can be negative: this is an analytic test, not a claim
that real pollution emissions are negative.

For spatial refinement use `P = sin(pi x) sin(pi y)`, `C = t P`:

```
C_t = P
C_x = t pi cos(pi x) sin(pi y)
C_y = t pi sin(pi x) cos(pi y)
C_xx + C_yy = -2 pi^2 t P
S = P + t [u pi cos(pi x) sin(pi y)
           + v pi sin(pi x) cos(pi y) + 2 D pi^2 P].
```

This is zero at t=0 and at every boundary, exactly the solver's conditions.
All grids use the same wind, diffusion and final time. Since `C_tt = 0`, the
exact sampled solution has no forward-Euler time truncation term. The
numerical spatial error still evolves in time: compare every run with dt/2
to check that time stepping does not materially change the spatial result.
Use dt proportional to h squared, satisfying the diffusion restriction.
A separate wind-free sine study checks the diffusion stencil's second order.

For temporal refinement use `P = x(1-x)y(1-y)` and
`C = (exp(t)-1) P`, with u=v=0 and D>0:

```
C_t = exp(t) P
P_xx + P_yy = -2 [x(1-x) + y(1-y)]
S = exp(t) P + 2D (exp(t)-1) [x(1-x) + y(1-y)].
```

The centered second difference differentiates this quadratic polynomial
**exactly at every interior node**, including nodes next to the boundaries.
Therefore the sampled analytic solution solves the semidiscrete ODE exactly;
spatial truncation error cannot create a temporal error floor. This is more
controlled than merely assuming a fine mesh removes spatial error. The
temporal experiment verifies forward Euler with nonzero diffusion, not a
second-order advection method. It also has zero initial/boundary values.

## Error definitions and Gaussian resolution check

For e = C_numerical - C_exact at the final time, use the discrete integral
norm `L2 = sqrt(dx dy sum_ij w_i w_j e_ij^2)`, with trapezoidal endpoint weights
1/2 (interior weights 1), and `Linf = max |e_ij|`. Relative L2 divides by the
same norm of the reference field. L2 units are concentration times length;
Linf units are concentration. Relative errors and empirical orders are
dimensionless. Here dx=dy=h. Report both L2 and Linf measured orders.

The Gaussian has no known closed-form solution with these finite-domain
boundaries. Do **not** transfer a manufactured solution's absolute error to
it. Instead refine the original Gaussian problem, fixing the source, wind,
diffusion, final time and boundary conditions. Compare finer fields at
coincident coarse nodes (no interpolation), and estimate the remaining error
with Richardson extrapolation. Its asymptotic-order assumption is explicitly
an estimate, not exact ground truth or a confidence interval. Halving dt on
the current and finest Gaussian meshes separates the time-step sensitivity
from the dominant spatial error. These checks do not regenerate any dataset.

For successive grids h, h/2, h/4, let delta_h be the integral L2 norm of
`C_h - C_(h/2)` at coincident nodes. Estimate
`p = log(delta_h/delta_(h/2))/log(2)`. From the finest two fields form
`C_extrapolated = C_fine + (C_fine-C_coarse)/(2^p-1)`. The current/fine error
estimates compare each with this extrapolation on the current 81x81 nodes.
This restriction is explicit: it avoids interpolation but is not a guarantee
on unsampled fine-grid peaks. The NPZ retains the fine fields and the
extrapolated comparison field for independent reanalysis.

Stability means bounded/non-amplifying discrete evolution under the CFL
restriction; convergence means errors decrease toward the continuous
solution under refinement; sufficient accuracy means the residual error is
small enough for an explicitly stated scientific tolerance. None implies the
next automatically. No project-wide acceptance tolerance has been supplied.

## Reproduction

Run from the repository root with the installed project environment:

```bash
python scripts/validate_numerics.py --config configs/numerical_validation.yaml
```

The output directory must not already exist. Use `--output-dir` for a new run;
existing outputs are never silently replaced. The runner saves CSV/JSON
metrics, PNG and vector PDF plots, final-field checkpoints (NPZ), resolved
configuration (including the exact synthetic solver settings), package
versions, Git provenance, and a source archive covering uncommitted code.
No neural network is trained and no existing dataset/configuration is changed.

### Storage limitation before future regeneration

The validation runs retain only final fields. This is **not** equivalent to
making the dataset generator memory-safe: its unchanged default still requests
the entire time history. At 641x641, dt=0.00003125 and final time 0.8, full
float64 history alone needs approximately 84 GB. Before regenerating datasets
at that integration resolution, a separate milestone must select and stream
the required observation/output times. Do not blindly substitute the fine-grid
configuration into the existing full-history generator. This milestone does
not implement output decimation or alter any training/observation times.

## Measured results: 2026-09-19

Artifacts are in `results/numerical_validation/`. All final times are 0.8.
The numerical stencil was **not changed**: a direct comparison with the
pre-revision Git version on `configs/dataset.yaml` produced bitwise identical
complete concentration histories (maximum absolute difference zero).

### Exact-solution spatial study

| Grid | h | dt | Integral L2 error, wind + diffusion | Linf error | L2 order |
| --- | --- | --- | --- | --- | --- |
| 41x41 | 0.025 | 0.008 | 0.00913140 | 0.018286 | — |
| 81x81 | 0.0125 | 0.002 | 0.00463227 | 0.009277 | 0.979118 |
| 161x161 | 0.00625 | 0.0005 | 0.00232964 | 0.004666 | 0.991610 |
| 321x321 | 0.003125 | 0.000125 | 0.00116780 | 0.002339 | 0.996316 |

Linf orders are 0.978949, 0.991502, 0.996408. Diffusion-only L2/Linf orders
are 1.989460, 1.997378, 1.999346. The measured behavior agrees with first-order
upwind advection and second-order centered diffusion, not second-order mixed
transport. Halving dt changes mixed-transport L2 error by 0.3874%, 0.0976%,
0.0246%, 0.00616%, respectively: time error is not driving the spatial slopes.

### Exact-solution temporal study

With an 81x81 grid, zero wind, D=0.005 and the spatially exact polynomial,
dt=0.004, 0.002, 0.001, 0.0005 gives L2 orders **0.999657, 0.999829,
0.999914**. Linf orders are 0.999634, 0.999817, 0.999909. This verifies
first-order Euler convergence without a spatial error floor.

### Current Gaussian configuration

`configs/dataset.yaml` uses 81x81, h=0.0125, dt=0.002, 400 steps,
u=0.35, v=-0.15, D=0.005, and one Gaussian at (0.30,0.70), Q=1, sigma=0.08.
It is **stable** (combined CFL=0.336), and the refinement sequence is
**convergent**, but this particular resolution is **not sufficiently accurate
for a provisional 1% final-field relative-L2 target**.

| Compared grids | Relative L2 difference (finer field denominator) |
| --- | --- |
| 41 to 81 | 4.52695% |
| 81 to 161 | 2.57806% |
| 161 to 321 | 1.38478% |
| 321 to 641 | 0.719109% |

Gaussian difference orders approach one: 0.789063, 0.884100, 0.938874.
Richardson extrapolation from 321/641 using the last measured order estimates
the current 81-grid error at **5.39699% relative L2**, integral L2=0.00419869,
Linf=0.0255686. This is not an exact error bound. At 641x641 the remaining
estimated relative L2 error is **0.780304%** on the same comparison nodes.
Halving dt changes the actual Gaussian field by 0.098532% on the 81 grid,
and 0.00171366% on the 641 grid, relative to each original-dt field. The spatial
discretization, not merely the time step, must be refined.

For a **provisional** 1% final-field relative-L2 objective, recommend numerical
integration on **641x641, dx=dy=0.0015625, dt=0.00003125**, 25,600 steps to
t=0.8 (CFL=0.266). This is an estimated tolerance attainment, not a certified
bound, and does not establish accuracy at every observation time or for other
source/transport configurations. Full space-time/sensor-level accuracy and a
scientifically justified acceptance tolerance must be checked before release.
Recommend regenerating accuracy-sensitive synthetic benchmarks once those
conditions and output-time storage are addressed. **No existing datasets or
synthetic generation settings were modified or regenerated in this revision.**

### Defects and limitations, ranked

1. **High: unquantified reference discretization error.** Treating the current
   stable 81-grid field as exact ground truth is unjustified at approximately
   5.4% estimated final-field L2 error. Verification and quantitative reporting
   are now present; the existing datasets are intentionally not replaced.
2. **Medium: full-history memory scaling obstructs fine-grid validation.**
   Final-only storage now permits these checks. Dataset generation still needs
   a separate selective-output implementation before the recommended refinement.
3. **No PDE-sign, stencil, CFL, initial-condition or boundary-enforcement
   defect found in the tested cases.** Zero-Dirichlet boundaries are validated
   as implemented, not endorsed as an atmospheric open-boundary model.

### Tests and retained warnings

- Before extension: 14 original finite-difference tests passed.
- Targeted validation: 35 tests passed, zero failures.
- Full suite: **237 passed, 120 warnings**, zero failures (113.02 seconds).
- Warnings: one NumPy/netCDF4 binary-compatibility warning and 119 xarray/netCDF4
  NumPy shape-assignment deprecation warnings in existing real-data tests.
  These were not suppressed or fixed in this numerical-only milestone.
- The runner test also verifies numerical repeatability, artifact completeness,
  preservation of synthetic configuration inputs, and refusal to overwrite.
- No failing test was weakened. No PINN, baseline, real-data or case-study
  implementation was modified. Runtime columns are observed wall times,
  not hardware benchmarks (the full suite ran during the final dt/2 check).

Git branch: `scientific-revision`. Pre-revision commit: `38230c2` (full hash
above), also preserved on `main`. Scientific changes remain uncommitted.
The initial commit honored existing ignore rules: generated data/results and
the virtual environment are not in Git and were preserved locally.
