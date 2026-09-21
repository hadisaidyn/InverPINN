# InverPINN v1.0.0 — Research Freeze

**Prepared release notes; no tag, GitHub release or publication has been created.**
The research release label is v1.0.0. Python package metadata remains the historical
0.0.0 to avoid confusing the frozen research environment with a new scientific model.

## Purpose and evidence

InverPINN studies recovery of a single Gaussian pollution source from sparse
concentration observations, comparing physics-informed neural networks with
classical PDE-constrained inversion.

The controlled workflow includes numerical derivative/convergence checks,
641×641 references, sparse observations, constrained PINN fields, source-strength
profiling, observability diagnostics and independent protocol-frozen scenarios.

## Final result

J1 recovered **23/30 (76.67%)**, below the required **27/30 (90%)**.
It **failed preregistered confirmation**. Its median localization, Q and field L2
errors were **0.005574, 5.975%, 7.423%**. Q remained low in 28/30 cases (two high).
Exact positivity and zero initial/boundary conditions did not guarantee recovery.

Classical inversion recovered **30/30**, with medians **0.001111, 1.218%, 1.507%**,
on identical observations. The release does not promote J1 as a validated production model.

## Deliverables and reproduction

- [Paper PDF](../paper/InverPINN_Paper.pdf) and [source manuscript](../paper/manuscript.md).
- [A1 poster](../presentation/InverPINN_Poster.pdf).
- [Editable presentation](../presentation/InverPINN_Presentation.pptx),
  [presentation PDF](../presentation/InverPINN_Presentation.pdf) and
  [speaker notes](../presentation/speaker_notes.md).
- Sealed evidence, machine-readable tables and [local build instructions](reproducibility.md).
- [Communication materials and complete inventory](FINAL_DELIVERABLES.md).

## Limits and manual choices

The evidence covers one source family, known constant transport, fixed sensors
and zero noise. It does not establish real Almaty attribution or robustness to
noise or uncertain meteorology. Three reference quality flags remain; estimates
are not mathematical bounds. Full experimental archives are not publicly deposited.

Public author: Khadis Aidyn. No license has been chosen. Follow the
[manual publishing checklist](PUBLISHING_CHECKLIST.md) before releasing.
