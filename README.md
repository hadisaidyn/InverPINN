# InverPINN

**Controlled research complete. J1 failed final confirmation.**

How reliably can a physics-informed neural network recover the location and strength of an unknown pollution source from sparse concentration measurements, and what limits its performance relative to classical PDE-constrained inversion?

InverPINN investigates that question using PyTorch, a validated 2D advection–diffusion simulator, sparse synthetic sensors, physical constraints, analytical source-amplitude profiling and independent blind evaluation. Almaty motivates the application; this work does **not** validate real Almaty emitter identification.

## Main result

J1 improved localization and source-strength recovery relative to the frozen revised PINN but achieved **23/30 joint recoveries (76.67%)**, below the **preregistered 27/30 (90%) reliability threshold**. It therefore **failed final confirmation**. Classical inversion recovered **30/30** on exactly the same observations.

| Final method | Recovery | Median localization | Median relative Q error | Median concentration L2 |
| --- | --- | --- | --- | --- |
| B_revised | 21/30 | 0.009186 | 9.319% | 11.048% |
| J1: variable-projection PINN | 23/30 | 0.005574 | 5.975% | 7.423% |
| Classical PDE-constrained inversion | 30/30 | 0.001111 | 1.218% | 1.507% |

J1 had zero negative predictions and exact homogeneous initial/boundary conditions, yet underestimated source strength in **28/30** cases. Its recovery Wilson 95% interval is **59.07–88.21%**. Physical admissibility did not guarantee inverse accuracy. Classical inversion substantially outperformed J1 in this particular correctly specified, zero-noise problem—not necessarily in other inverse problems.

Read the [complete manuscript](paper/manuscript.md), [technical/plain-language summary](docs/project_summary.md), [reproduction guide](docs/reproducibility.md) and [scientific audit](docs/final_scientific_audit.md).

## What was tested

The equation is `C_t + u C_x + v C_y = D(C_xx + C_yy) + Q G(x,y)` for one stationary Gaussian of known width. Wind and diffusion are known; 20 sensors observe 81 times; added noise is zero. A 641² numerical reference follows manufactured-solution and refinement checks. Q is peak source intensity, not integrated emissions. Units are synthetic.

J1 uses a three-layer, 64-unit tanh concentration network, hard positivity/IC/BC constraints, sigmoid locations and analytical Q profiling of the PDE residual. The classical control instead optimizes source parameters through a numerical forward model. Both see the same sparse data. The final 30 scenarios were excluded from model development; every failure is retained.

## Installation

Use Python 3.11 or newer (archived research environment: Python 3.13.3). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Dependencies include PyTorch, NumPy, SciPy, pandas, matplotlib, xarray, netCDF4, scikit-learn, PyYAML, pyproj and pytest. The lightweight **report-only** installation is:

```bash
python -m pip install -r paper/requirements-report.txt
```

Recorded package versions are provenance, not a guarantee of cross-platform bitwise training reproduction. See the guide for the original environment and archive requirements.

## Reproduce the final paper without training

The committed compact evidence bundle is enough; the large ignored results tree is not required.

```bash
python scripts/build_paper.py --verify-only
python scripts/build_paper.py
python -m pytest tests/test_paper_reporting.py -q
```

The default build writes a **new** `paper/reproduced/` directory with eight main figures and one supplementary figure (300-dpi PNG and PDF), four main tables, raw/distribution/paired CSVs, JSON metrics, hashes and environment provenance. It refuses overwrite. For another build, choose a new directory:

```bash
python scripts/build_paper.py --output-dir paper/reproduced_second
```

This is evidence-to-paper reproduction: **no model fitting, checkpoint inference, new simulation, tuning or network download**. Committed canonical outputs are in [paper/generated](paper/generated/README.md). Do not use the historical `scripts/reproduce_paper.py` as the final-paper command: that older runner launches training and exploratory sweeps outside the completed study's reporting scope.

## Repository structure

| Path | Purpose |
| --- | --- |
| `paper/` | Manuscript, supplement, compact evidence and final generated figures/tables |
| `configs/` | Version-controlled experimental protocols, physical settings and frozen model recipes |
| `src/inverpinn/physics/` | PDE, derivatives, Gaussian sources, finite differences and amplitude profiling |
| `src/inverpinn/models/` | Neural concentration representations and interpolation models |
| `src/inverpinn/data/` | Synthetic/reference generation, sensor sampling and separate future real-data utilities |
| `src/inverpinn/training/` | Frozen neural fitting code; not executed by the reporting pipeline |
| `src/inverpinn/evaluation/` | Recovery, reconstruction, physical and observability diagnostics |
| `src/inverpinn/visualization/` | Historical experiment plotting utilities |
| `scripts/` | Command-line entry points; `build_paper.py` is the final report-only entry point |
| `tests/` | Scientific regression tests and isolated report-only checks |
| `docs/` | Methods, protocols, historical reports, final explanation and reproduction guide |
| `data/raw/` | Unmodified input caches; not a claim that real observations are available |
| `data/processed/` | Derived modeling inputs, kept separate from originals |
| `data/synthetic/` | Generated synthetic datasets and metadata |
| `results/` | Ignored full experimental artifacts, including frozen final confirmation |
| `artifacts/` | Ignored audit/execution artifacts, including the finalization preservation inventory |
| `experiments/` | Reserved experiment organization area; completed outputs live under `results/` |
| `notebooks/` | Optional exploration; not required to rebuild the paper |

## Provenance and limits

The frozen scientific state is commit `eb18380f1488f5c679a0981da75e667fa3ba1b51` on `scientific-revision`. [Evidence manifest](paper/evidence/manifest.json) entries identify each original file, SHA-256 and byte-identical bundle copy. Final manifests, configs, decision, raw results and two mechanically selected saved field previews are included. Full training histories, checkpoints and reference datasets remain local ignored artifacts; no public archive DOI is claimed.

This is one Gaussian, known transport, one fixed network, simplified 2D physics and zero added noise. It does not establish noisy-sensor performance, uncertain-weather performance, multiple-source recovery, terrain-aware transport or real-city causal attribution. Three final numerical-reference estimates exceeded the 1% target and were retained. Numerical error estimates are not certified bounds.

The core research phase is **complete**. No new PINN candidate, post-failure rescue or robustness campaign is part of this finalization. Optional next steps are manuscript formatting/submission, a poster or a presentation; any real-data extension requires a separately authorized and independently validated study.
