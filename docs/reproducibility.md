# Reproducing the final InverPINN report

## Scope: reproduce the paper, not a new experiment

The controlled research is complete. J1 **failed: 23/30 recoveries, required 27/30**. These commands read saved evidence; they never train a model, load a model checkpoint, generate a source, solve the PDE or use the network.

From the repository root, after the [README report-only setup](../README.md#quick-start):

```bash
python scripts/build_paper.py --verify-only
python scripts/build_paper.py
python -m pytest tests/test_paper_reporting.py tests/test_repository_docs.py -q
```

The build normally takes seconds (canonical rendering about 3.8 s, excluding first-use font-cache startup). It creates `paper/reproduced/`; if that exists, choose a new `--output-dir`. No overwrite/force/resume option exists. Destinations inside frozen `results/`, `data/`, `src/`, `configs/` or the evidence bundle are rejected. A partial reporting failure remains visible in its new directory; fix reporting only and choose another output path. Never patch frozen results to make a report pass.

On a host with an unwritable home font cache, set `MPLCONFIGDIR` to a writable temporary directory before the command. This changes rendering cache location, not scientific values. PNG/font rendering can differ across systems; metric JSON and CSV should reproduce numerically under recorded versions. Git status, runtime and platform metadata intentionally differ between builds.

## Inputs and outputs

The committed [evidence manifest](../paper/evidence/manifest.json) contains 41 byte-identical copies from the existing archive, original relative paths, sizes and SHA-256 hashes. It includes all 90 final per-case records; final manifest/protocol/configs; failed decision; observation hashes; historical numerical/diagnostic/development summaries; observability grid; fixed sensors; two mechanically selected field-preview NPZ files; and prior test reports. No pickle or model checkpoint is used. Git attributes preserve the archive's original bytes, including CSV CRLF line endings, across checkout platforms.

The reporter verifies every bundled file, source metric calculations, all 30 IDs per method, primary seed, paired observation/reference hashes and the frozen decision. It independently recomputes distributions, recovery intervals and paired mean effects. It refuses changed or incomplete evidence rather than omitting cases. The bundle supports **reproduction of reported analysis**, not independent reconstruction of original fitting trajectories.

Outputs: eight main figures and one supplement in 300-dpi PNG plus PDF with embedded fonts; four main tables in CSV/Markdown; raw, distribution and paired CSVs; `summary.json`; `provenance.json`; and artifact hashes. See [canonical outputs](../paper/generated/README.md) and [selection/statistical rules](../paper/supplement.md). The bundle and plots occupy only a few megabytes; no 11-GB results archive is needed for these commands.

## Frozen scientific state and hashes

The starting branch was `scientific-revision`, clean, at **eb18380f1488f5c679a0981da75e667fa3ba1b51**, message `Run final confirmatory InverPINN validation`. Reporting changes are in a later documentation commit; the scientific commit remains the provenance anchor.

| Artifact | SHA-256 / commit |
| --- | --- |
| B_revised freeze commit | `9dc90c247b0a596b789cb0ee5f1c06f938db40c4` |
| Observability diagnosis commit | `199559f901bd21981f155d70550e3c82d06dde70` |
| Formulation development commit | `33892895805d692c8c3960d8646c43d6a4f7536d` |
| J1 config | `a963da9f9a3e1f4f11bea9b1ce9162d65223ec199e162724fe8da56ee73d3e0a` |
| B_revised config | `3e400ef1195cf90f6481117ae5c395695cca23a12d47ab532ec6822e400c332e` |
| J1 fitting code | `99653e0446dad4215ea16ef1b320de4802c498e2bf4c038b90507531070e1893` |
| Q profiling code | `d9a33cf02c4cd60fb90a2fa3f967f4dc0446691b2f6df60eedeef1a12e8e1ac7` |
| Reference solver | `171d63443cc789534402b08ad497b17f18fbb113e9530d38c6ffedd9ae77b8b8` |
| Classical implementation | `05020915a105d6ae821af14b49987f1dfd6e5f5db359a3f042098f23fbac1edd` |
| Final scenario manifest | `8fa5480ef5aabd50300e548c27ec355af8315568f52a6dfd5c640d3044ee74b9` |
| Final J1 raw CSV | `17917bd23df4808e0285227f334ff574d797934b3d1f3f6c2e8b648a32f1890f` |

The complete preregistered code/config inventory is in [preregistration.json](../paper/evidence/final/preregistration.json), with locks and independence/integrity records alongside it. The reference generator commit is 3389289…; the later scientific finalization commit records completed results. These are different provenance meanings.

## Environment and deterministic seeds

The [archived environment](../paper/evidence/final/environment.json) records Python 3.13.3, macOS 15.0.1 ARM64, PyTorch 2.14.0, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, PyYAML 6.0.3 and pytest 9.1.1. The report-only [pinned requirements](../paper/requirements-report.txt) exclude PyTorch. Full-project requirements remain unchanged. This is not a portable lock for every transitive dependency, hardware platform or future package availability.

| Purpose | Seed/settings |
| --- | --- |
| Final source generation | 2009202608; 30 IID cases |
| Neural initialization | 42; one fixed primary fit per case |
| Training resampling | 43 |
| J1 terminal fitting profile | 44; 8192 points |
| Independent physical diagnostics | 2009202808 |
| Paired bootstrap | 2009202818; 10000 paired resamples |
| Neural execution | CPU float32; one PyTorch thread; deterministic algorithms |

Per-scenario seeds, grids, dt, reference flags, truth and observation/reference hashes are in [manifest.csv](../paper/evidence/final/manifest.csv). The protocol preceded generation; every scenario and quality flag remained. The [independence check](../paper/evidence/final/independence_check.json) documents exclusion of consumed scenarios. Repeated-training results did not replace primary fits.

## Reference and full-archive boundary

Reference: 641², Δt=0.00003125, 25,600 steps, with 161² and 321² checks. All methods use the same 20 sensors and 81 times; dense evaluation uses four saved times. Classical inference: 201² and Δt=0.00025. The [manuscript](../paper/manuscript.md) documents units, boundary conditions, uncertainty and profiling.

The local research archive is approximately 11 GB under `results/` plus 89 MB under `data/`, including historical runs, ignored by Git. A clean clone contains the compact bundle but **not** all observations, full reference fields, training histories and checkpoints needed to independently repeat every experiment. Complete experiment reproduction requires obtaining/depositing that archive and a compatible environment. No public archive/DOI is asserted. Historical scripts/protocols document execution, but some enforce historical HEAD/config/consumed-data locks and cannot simply run on the reporting commit.

Measured per-case medians were 93.45 s J1, 91.41 s B_revised and 64.57 s classical, plus response-bank setup and substantial reference-generation/evaluation work. These fitting runtimes are not end-to-end costs or guarantees on other hardware. The [old orchestration guide](reproduce_paper.md) launches training and unrelated sweeps; it is not the final paper command.

## Preservation and tests

Finalization inventoried 13,399 existing data/result/source/config/original-script/original-test files. The complete local inventory is `artifacts/finalization_audit/protected_before.json`; its digest/count are committed in the evidence manifest. On the original archive only:

```bash
python scripts/package_paper_evidence.py --verify-preservation
```

The packager is an archive-maintenance utility, not needed for a reader's build. It refuses overwrite and never fits or simulates.

This historical inventory also includes ignored `src/inverpinn.egg-info/` build
metadata. An editable package installation can regenerate those files and make
the strict archive check fail even when every scientific file is unchanged.
GitHub-polish verification found exactly three such changes; see the
[polish audit](github_polish_audit.md). The inventory and checker were preserved,
not weakened or rewritten to conceal that difference.

Only report/document tests are executed during GitHub polish. The full scientific suite includes tiny model-training tests, so rerunning it would violate the no-training instruction. Its frozen report (**451 passed**) is preserved and explicitly **historical**, not represented as a new run. Reporting tests cover integrity, numerical aggregation, no-training boundaries, complete rendering, no overwrite and short-description word counts. Documentation tests cover clean-checkout links, citation/workflow metadata and README numbers against the sealed raw results.

## Lightweight CI and documentation checks

The [workflow](../.github/workflows/ci.yml) uses Python 3.13, the unchanged report
dependency pins and a dependency-free package installation. It runs an import
smoke check, evidence verification, `scripts/check_docs.py`, the two report/docs
test files above, and a build in a fresh output directory. It needs no ignored
archive, private data, PyTorch, GPU or training. Hosted CI status is not claimed
until the workflow actually runs on GitHub.

The standard-library link checker examines tracked and new nonignored Markdown
and validates relative targets/images, heading anchors and machine-local paths.
An existing but ignored local output does not count as a publishable link.
Sealed `paper/evidence/` documents are excluded as immutable historical provenance.
External URLs are inventoried, not network-availability or usage-permission checks.
The public historical file inventory uses repository-relative paths; original
execution paths inside the sealed numerical configuration remain unchanged.
