# Public deployment audit

Deployment engineering only. Starting commit: `6bdca71254e1f59ea16e3fa5f0b68e98ee82070d`, clean `scientific-revision` branch. No GitHub remote existed. Frozen science, models, thresholds and display-bundle values are not to be changed.

## App dependency inventory before changes

- Entry point: `demo/app.py`; command from the repository root: `streamlit run demo/app.py`.
- Runtime modules: `demo/evidence.py`, `demo/plots.py`; standard-library path/JSON/CSV/hash/statistics helpers. External imports: Streamlit, NumPy, pandas, Plotly and PyYAML.
- Every asset path is anchored to `Path(__file__).resolve()` and the repository root, not the current working directory or a user's home directory.
- `demo/data/manifest.json` and `final_metrics.json`, plus 30 display-field NPZ files, are tracked. The archived scientific results are not loaded by the viewer.
- Integrity inputs: seven files under `paper/evidence/` plus its manifest, and `paper/generated/summary.json` with `artifact_hashes.json`.
- Public assets: existing `paper/InverPINN_Paper.pdf`, eight PNGs under `paper/generated/figures/`, and `docs/reproducibility.md`.
- Caching: verified metadata and one field bundle per selected case; field cache capped at 30 entries. No fitting, checkpoint inference, dataset generation or solver imports.
- Streamlit config: light theme, headless server, telemetry disabled, viewer toolbar. No committed port or localhost-only binding.
- Documentation mentions localhost only as a local-run address. Sealed historical provenance remains untouched. The bundle regeneration script needs ignored archives, but it is not imported or executed at runtime.
- Existing GitHub Actions only checks/rebuilds reports from saved evidence; it does not train or simulate.

## Runtime choice

Recommend **Python 3.13** in Community Cloud's Advanced settings. Exact local validation uses Python 3.13.3. Installed package metadata requires Python ≥3.12 for NumPy, ≥3.11 for pandas, ≥3.10 for Streamlit and ≥3.8 for Plotly/PyYAML. Python 3.13 meets all constraints and preserves the already tested viewer environment without upgrading dependencies. This is a tested compatibility choice, not an assumption that the newest Python is preferable.

Streamlit documents that Community Cloud supports maintained Python versions, defaults to 3.12, and lets users select Python in the deployment UI. Therefore no unsupported `runtime.txt` or hard-coded server binding is introduced. The platform manages the Python patch release. [Official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Verification results

- All five unchanged pins installed into a fresh Python 3.13.3 virtual environment. PyTorch, SciPy and the InverPINN scientific package were absent; `pip check` passed. The app rendered and selected all 30 cases using only viewer dependencies, before adding any test runner.
- Python 3.13 CPython x86-64 Linux wheel resolution succeeded for every direct/transitive dependency using manylinux 2.28/2014 targets. This is a compatibility/resolution check, not a claim that a Linux server was run locally.
- Deployment/demo tests: **62 passed**. Existing reporting/documentation/final-delivery tests: **68 passed**. No pytest warnings or failures in the completed runs. AppTest logs an expected bare-mode ScriptRunContext warning; pip disables an unwritable sandbox cache. Neither affects app operation or is hidden by test changes.
- The check script verifies all 58 runtime assets are tracked, readable and portable, every final ID loads, and canonical recovery/median/Q-bias values agree. The full scientific suite was not run; no training or simulation occurred.
- Prepublication credential-pattern scan: no findings in the publishable checkout and **392 historical blobs** reachable from the starting HEAD. No private-network service URL found in public tracked text. Sealed provenance was preserved. These are bounded heuristic scans, not guarantees about every possible secret representation; authentication remains in the system keyring, outside the repository.
- Confirmed author: **Khadis Aidyn** in CITATION, README, paper PDF metadata and first-page byline, presentation metadata and latest Git authorship. Earlier historical Git names are preserved; no history rewrite or invented identity fields.
- Original science, `demo/data/`, plotted source positions and `demo/evidence.py`/`plots.py` remain byte-identical to the starting commit. Changes are public wording, deployment checks, documentation, secrets-file exclusion and a viewer-only CI job.

## Performance and memory

The first-ever AppTest render in the freshly installed environment took **12.493 s**, above the few-seconds target. A subsequent process in the same environment took **1.377 s**. These are Python/Streamlit render measurements, not browser-paint or hosted-network measurements; retain the slower first initialization as a cold-start limitation.

Across all 30 selections, median switching was **0.429–0.469 s**, maximum **0.520–0.889 s** in those two runs. Peak process RSS was **318,472,192–341,262,336 bytes** (approximately 304–325 MiB), including the app test harness and dependencies, not a claimed cloud memory quota or per-user bound.

The complete bundle is **8,759,214 bytes** (8.35 MiB); all 30 decoded field-array sets total **9,487,680 bytes** (9.05 MiB). Cache and plotting/serialization overhead add memory beyond the arrays. No gigabyte archive is loaded. Display downsampling is unchanged: retained nodes have exact coordinate alignment, common per-scenario color limits describe the display arrays, and omitted fine-grid extrema are not asserted to be preserved. Scientific metrics retain their original full-resolution values.

## Publication gate

GitHub CLI authentication was confirmed as `hadisaidyn` using the system keyring outside sandbox restrictions. `hadisaidyn/InverPINN` did not exist at the prepublication check. Existing `main` is an ancestor of `scientific-revision`; a fast-forward preserves all commits and the scientific branch.

The available Streamlit browser session shows **Continue to sign-in**, not an authenticated workspace. No credentials were requested or read. Public hosting therefore still requires the owner's Streamlit login and the exact UI steps in the [deployment guide](DEPLOYMENT.md). No live URL is fabricated. No explicit reuse license is currently granted.
