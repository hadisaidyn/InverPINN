# Reproducing the paper experiments

From the repository root with the project installed in the active environment:

```bash
python scripts/reproduce_paper.py
```

This executes `configs/paper.yaml` in its listed order. It generates numerical
datasets afresh, trains the data-only/forward/inverse models, evaluates the
classical and interpolation baselines, runs sensor/noise/wind/ablation sweeps,
and repeats the two-source experiments with permutation-aware evaluation.
The full configuration retains existing training budgets and seed lists; it can
take substantial time. Completion of a smoke test is not full scientific validation.

```bash
python scripts/reproduce_paper.py --config configs/paper_smoke.yaml
python scripts/reproduce_paper.py --output-dir results/paper_second_run
```

Output directory creation is exclusive, including for smoke runs. No force,
silent reuse, automatic deletion or resume option exists. After a failure,
inspect `run.json` and `logs/<stage>.log`, correct the cause, and choose a new
directory. Previously generated runs remain untouched.

## Configuration and dependencies

Each stage has an ID, supported experiment kind, optional source config, explicit
recursive overrides and required artifact paths. `${stage_id}` refers to the
output directory of an earlier enabled stage. Forward references, duplicate IDs,
unknown runners, path-escaping artifact names and unexplained exclusions fail.
Output directories are owned by the paper runner, regardless of values in the
source experiment configs. Nested `inverse_config` and `forward_config` files
are snapshotted and rewritten to the run's configuration directory. No source
configuration is modified. Relative config/input paths use the repository root.

Real-data and inversion stages are **explicitly excluded** in the default paper
manifest because verified inputs are absent. `run.json` retains these reasons.
Enable them only after completing their input configs and permission checks.
No real-data figures or zero-event claims are manufactured. A complete run means
all enabled, declared experiments succeeded, not that excluded results exist.

## Outputs and provenance

- `config.yaml`, `configs/`: master manifest and fully resolved stage/dependency
  configurations; each experiment also saves its own `config.yaml`.
- `experiments/`: all metrics (CSV/JSON), full training histories, checkpoints,
  predictions, figures and reports from existing runners, including nested sweeps.
- `paper/figures/`: selected top-level PNG and PDF figures, prefixed with stage ID.
- `paper/tables/`: stage-level CSV tables and JSON metrics/summaries. Original
  metric schemas and statistical definitions are preserved, not mixed across
  experiments or silently re-aggregated. Nested raw metrics remain in experiments/.
- `artifacts.csv` / `artifacts.json`: complete index of original experiment files,
  sizes and SHA-256 hashes. Source metrics are machine-readable even when a
  runner's accompanying human report is Markdown.
- `environment.json`: installed package versions, Python executable/version,
  operating system, architecture and numerical process settings. No credentials
  or unrelated environment variables are captured.
- `git.json`: HEAD commit, working-tree status and tracked diff when available.
  Non-Git workspaces explicitly record `commit: null` and the reason. Set
  `require_git_commit: true` for a release that must have a commit.
- `source_snapshot/`, `source_hashes.json`: Python sources, YAML configs and
  dependency declarations, including uncommitted code. A dirty commit hash alone
  is not sufficient to identify the executed source.
- `paper_provenance.json` beside each successful experiment's saved config,
  including nested training runs: environment, Git state, source hashes and
  master seed. Experiment configs retain their own independent seed lists.
- `run.json`: pending/running/complete/failed/excluded states. Failed subprocesses
  and missing required artifacts stop the run with a nonzero exit code.

## Reproducibility and figure policy

Python hash seed and BLAS/OpenMP thread limits are set before child startup.
Python, NumPy and PyTorch get a deterministic master seed. Existing scientific
runners still apply their configured experiment seeds, local generators and
paired-seed designs. CPU execution and deterministic PyTorch algorithms are
requested; unsupported operations raise. Fixed software/hardware should reproduce
numerical results, but different PyTorch/BLAS/platform versions need not be
bitwise equal. Runtime metrics, absolute paths and some archive metadata can differ.
This is an environment record, not a universally portable package lockfile.

Top-level final figures are rendered directly from Matplotlib artists at at least
300 dpi PNG plus PDF with embedded TrueType fonts and tight bounding boxes.
This is not upscaling of old raster images. Existing labels, shared colour scales
and sample-standard-deviation error bars remain unchanged; no confidence
interpretation is added. Nested per-seed diagnostic plots retain their existing
format/resolution and are not selected as final figures. Journal-specific physical
column sizes and caption editing still require the target journal specification.

The plot wrapper is confined to a dedicated child process and never changes
standalone experiment behavior. No new physical equations, estimator tuning or
training-budget changes are introduced by the production manifest.
