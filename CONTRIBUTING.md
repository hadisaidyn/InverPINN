# Contributing

The controlled research is complete. Read the [README](README.md),
[engineering rules](AGENTS.md) and [scientific audit](docs/final_scientific_audit.md)
before proposing changes. Documentation, reporting defects and reproducibility
questions are welcome; the failed confirmation must remain explicit.

## Setup and lightweight checks

Use the Python 3.13 [report-only setup](README.md#quick-start), then run:

```bash
python scripts/check_docs.py
python -m pytest tests/test_paper_reporting.py tests/test_repository_docs.py -q
python scripts/build_paper.py --output-dir paper/reproduced_contribution
```

Choose a new output path if it exists. These commands need only tracked evidence,
not the full research archive. The [CI workflow](.github/workflows/ci.yml) runs
this class of checks; it does not train models. The full suite contains training
fixtures and is deliberately not part of documentation CI.

## Scientific-change expectations

- Do not rewrite frozen protocols, manifests, model configurations or result files.
  Report a suspected scientific bug with evidence; preserve the original run.
- Any renewed model development requires explicit authorization, a new protocol
  and new scenarios. Previously consumed sets cannot become fresh validation data.
- Preserve seeds, resolved configuration, environment/package versions, Git commit,
  numerical diagnostics, raw outcomes and failed cases for any authorized experiment.
- Explain the mathematical reason for failing tests. Do not weaken thresholds or
  change expected results simply to obtain a pass.
- Keep physical equations separate from model code; prefer small changes with tests.
- Never commit credentials, environments, large generated archives or checkpoints.

Use a short issue to distinguish a reporting problem from a scientific question.
In a PR, state scope, verification and whether any frozen file is touched. No
license choice is established; contribution discussions do not establish reuse
permission on the owner's behalf.
