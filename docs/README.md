# Documentation

Start with the [project overview](../README.md), [manuscript](../paper/manuscript.md)
or [plain-language summary](project_summary.md). **Final J1 confirmation failed:
23/30 recoveries, with 27/30 required.** Classical inversion recovered 30/30.

## Scientific methodology

The order below follows the research, not a list of validated product features.
Earlier development and diagnostic outcomes are historical, not additional final
test cases. Their original protocols and quantitative conclusions are preserved.

| Stage | Explanation / results | Frozen protocol or supporting detail |
| --- | --- | --- |
| Numerical foundations | [Finite-difference validation](numerical_validation.md) | [Units and scaling](nondimensionalization.md) |
| High-accuracy reference | [Synthetic reference](synthetic_reference.md) | [Historical file inventory](synthetic_reference_files.txt) |
| Initial controlled inversion | [Single-source benchmark](single_source_benchmark.md) | [Protocol](single_source_benchmark_protocol.md) |
| Physical/optimization diagnosis | [Model diagnosis](model_diagnosis.md) | [Initial audit](model_diagnosis_initial_audit.md), [selection rule](model_diagnosis_selection.md) |
| Revised-model blind test | [Fresh blind results](fresh_blind_benchmark.md) | [Protocol](fresh_blind_benchmark_protocol.md) |
| Observability and classical control | [Diagnosis](observability_diagnosis.md) | [Protocol](observability_diagnosis_protocol.md) |
| Inverse-formulation development | [Candidate results](inverse_formulation_revision.md) | [Protocol](inverse_formulation_protocol.md), [field/source coupling](source_field_coupling.md) |
| Final J1 confirmation | [Final results](J1_confirmatory_results.md) | [Preregistered protocol](J1_confirmatory_protocol.md) |

The final protocol's conditional robustness stage was **not activated**, because
confirmation failed. Do not infer that an implemented utility or a historical
plan establishes a completed or validated experiment.

## Reproducibility

- [Report-only guide](reproducibility.md): setup, expected cost, environment and archive limits.
- [Pinned report dependencies](../paper/requirements-report.txt): no neural training dependencies.
- [Frozen environment](../paper/evidence/final/environment.json) and [evidence manifest](../paper/evidence/manifest.json): historical provenance and SHA-256 hashes.
- [Contribution and verification guide](../CONTRIBUTING.md): lightweight tests and scope discipline.
- [GitHub-polish audit](github_polish_audit.md): verified commands and preservation caveats.
- [Historical experiment runner](reproduce_paper.md): **not** the final paper command; launches training/sweeps.

Public links point to files in a clean checkout. References to ignored `results/`
and `data/` archives are explicitly labeled as local-only. Sealed evidence can
retain original execution paths: these are immutable provenance, not setup instructions.

## Communication

- [Technical and plain-language project summary](project_summary.md).
- [Short application descriptions](application_description.md).
- [Final scientific claim audit](final_scientific_audit.md).
- [Research freeze and presentation changelog](../CHANGELOG.md).

## Paper

- [Manuscript](../paper/manuscript.md) and [supplement](../paper/supplement.md).
- [Figure gallery](figures.md): eight existing study figures with context.
- [Figure/table index](../paper/generated/README.md), [final benchmark table](../paper/generated/tables/table3_final_benchmark.md) and [machine-readable summary](../paper/generated/summary.json).
- [Citation metadata](../CITATION.cff).

## Separate, unvalidated real-data work

These historical access notes and utility guides are **outside the completed
controlled validation**. They are not instructions to initiate a new study or
claims of real emitter identification. Access and usage notes are historical,
not a fresh verification of availability or permission.

- [AirKaz access research](airkaz_data_access.md).
- [Meteorology and coordinate/data processing](meteorology_pipeline.md).
- [Thermal-inversion definitions](thermal_inversions.md).
- [Real-case-study assumptions](real_case_study.md).
