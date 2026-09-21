# InverPINN Explorer

Explore frozen synthetic blind-test scenarios and compare PINN source inversion against classical PDE-constrained inversion. **This is a result viewer, not a live scientific inference engine.**

All scenarios shown here are synthetic controlled experiments from the frozen research benchmark. This demo does not identify real pollution emitters in Almaty or provide operational environmental monitoring.

## Run locally

From the repository root, use Python 3.13 and an isolated environment (do not change the research environment):

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Open the local URL printed by Streamlit; stop with Ctrl-C. On Windows use `.venv-demo/Scripts/Activate.ps1`. No project installation, PyTorch, checkpoints, API credentials or research archive is needed. Dependency installation needs internet; subsequent viewing uses local files. Usage telemetry is disabled in the root Streamlit configuration.

## What is shown

- Exactly `final_blind_v1_001` through `final_blind_v1_030`; no development or earlier blind sets.
- Reference, J1 prediction, absolute error and classical reconstructed fields at saved times 0.1, 0.2, 0.4 and 0.8. These are not the complete 81-time sensor schedule.
- True source (star), J1 estimate (×), classical estimate (diamond), and 20 sensors (small circles). Marker shapes distinguish methods independently of color; legend clicks help reveal coincident markers.
- Frozen localization, relative Q error and concentration L2, recovery labels, full-precision details, observability diagnostics and reference-quality flags.
- Global Q bias and paired-location plots, plus all seven retained J1 failures. Global plots always use all 30 cases, regardless of the scenario filter.

The default is lowest-number scenario 001. The paper's representative success is 003, the median localization rank among J1's 23 recovered cases. The worst localization case is 030. A single success/median shortcut deliberately represents the paper's one existing selection; no new attractive example or best case is chosen.

Joint recovery is localization ≤ 0.08 **and** relative Q error ≤ 20%. J1 recovered 23/30, missing the preregistered 27/30 requirement; classical inversion recovered 30/30. J1 underestimated Q in 28 cases and overestimated it in two. The viewer prominently retains this failed confirmation. See the [canonical summary](../paper/generated/summary.json) and [paper](../paper/InverPINN_Paper.pdf).

Coordinates, time and concentration use synthetic units; this is not a geographic map. Q is Gaussian peak source intensity, not integrated emissions. The plots do not represent posterior probabilities. No real Almaty conclusions follow from this benchmark.

## Files

```text
demo/
├── app.py                Streamlit interactions; cached read-only loading
├── evidence.py           Evidence/hash checks and lossless metric extraction
├── plots.py              Plots of stored arrays and source coordinates
├── requirements.txt      Viewer-only pinned dependencies
├── data/
│   ├── final_metrics.json  All final metrics, sensors and scenario metadata
│   ├── manifest.json      Input/output hashes and conversion provenance
│   └── fields/            One compact NPZ per final scenario
├── README.md
├── QA.md                  Local verification record
├── demo_preview.png       Screenshot of the actual default scenario
└── DEPLOYMENT.md          Optional instructions; no deployment performed
```

## Derived data and scientific preservation

[build_demo_data.py](../scripts/build_demo_data.py) copies metrics from the sealed [final evidence](../paper/evidence/final) and checks canonical medians and recovery labels. Scientific evidence is anchored at commit `eb18380f1488f5c679a0981da75e667fa3ba1b51`; the viewer work started from final-delivery commit `2badcf1639faeb136a29567552109b81aade8ade`.

All 60 archived J1/classical prediction NPZ files were checked against the original SHA-256 preservation inventory, itself hash-anchored in the committed paper evidence. Paired references, coordinates and times must match exactly. The [demo manifest](data/manifest.json) records these input hashes, every derived-file hash, array formats and per-case conversion errors. This is integrity/provenance checking, not an independent rerun of the science.

The saved preview fields were already 161×161 (stride 4 of the 641×641 evaluation grid). The builder keeps every second preview node, including boundaries: 81×81, effective stride 8. There is **no spatial interpolation**. Values are converted to float32; the largest measured absolute conversion difference is `2.9300541326549023e-08` synthetic concentration units. Downsampling omits intervening nodes; it is not a claim that the complete field has that approximation error. Axes and times retain float64. The bundle is approximately 8.4 MiB.

Display absolute error subtracts the saved J1/reference previews. All reported scientific metrics remain the original full-resolution CSV values, never estimates from the display grid. All concentration modes use the same color limits across methods and saved times within a scenario; error has a separate scale. No metrics, source estimates or scientific thresholds are changed.

### Verify a clean checkout

```bash
python scripts/build_demo_data.py --verify-only
python -m pip install pytest==9.1.1
python -m pytest tests/test_demo.py -q
```

Startup verifies bundle hashes and its metadata against the sealed evidence. A mismatch stops the viewer. Per-scenario cached fields also undergo hash, shape and finite-value checks. Restart the app after intentionally rebuilding any presentation data; the cache assumes files remain immutable during a viewing session.

### Regenerate from the preserved archive

The complete 30-case display bundle is committed. **Full regeneration requires the original local research archive**, which is not included in a clean clone. Do not generate new predictions to replace missing archived files. The two historical paper previews alone cannot regenerate the other 28 cases.

With the preserved archive at its original locations:

```bash
python scripts/build_demo_data.py --output-dir artifacts/demo_rebuild_check
```

For a restored archive, use `--archive` for the directory containing `runs/J1/` and `runs/classical/`, and `--inventory` for the preserved `protected_before.json`. Its hash must match the paper evidence. Every input is verified before output creation. Choose a new output directory each time; the builder refuses to overwrite. Output is restricted away from scientific directories. The full-archive regeneration test skips explicitly when those original inputs are unavailable; all committed-bundle checks still run.

The builder never imports scientific modules, trains, solves a PDE or evaluates a checkpoint. ZIP metadata is deterministic; tests compare rebuilt hashes to the committed bundle. Source files are only read.

## Supporting material

The app serves the existing paper PDF and reproducibility guide locally and shows the existing figure gallery. No GitHub URL is invented. [Deployment instructions](DEPLOYMENT.md) describe a possible later public setup; nothing is published by running this viewer.

[Actual screenshot](demo_preview.png) · [Local test and visual QA record](QA.md).
