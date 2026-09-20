# Final scientific consistency audit

This is a **reporting-only audit**, anchored to scientific commit `eb18380f1488f5c679a0981da75e667fa3ba1b51`. The starting branch, commit/message and clean working tree matched the requested state. No model was trained, no new source/PDE experiment was run and no frozen result/config/scientific implementation was changed.

## Claims checked and qualified

| Possible overstatement | Final evidence-based wording |
| --- | --- |
| Successful final PINN validation | J1 failed the preregistered 27/30 gate: 23/30, 76.67% |
| Superior pollution-source PINN | Classical recovered 30/30 and had lower median errors in this setting |
| Strength bias solved | Universal underestimation ended, but 28/30 remained underestimates |
| Positivity/BC/IC imply correct inference | All were exact; seven joint source recoveries still failed |
| Simultaneous fitting proven to cause bias | Field/source inconsistency and intervention evidence support a mechanism, not unique causation |
| Sensor information explains all failures | Classical/oracle results rule out insufficient information alone for the tested cases |
| J1 previously promoted | Development relative-improvement gates were missed; final testing was a newly frozen hypothesis |
| Reference truth is exact / certified | Source truth exact; concentration numerical; Richardson estimates are not bounds; three flags retained |
| Real Almaty emitter discovery | Almaty motivation only; no field-validated source attribution |
| General PINN or classical superiority | Conclusion restricted to one known-width source, known physics, zero noise and this network |
| Fully reproducible training from a clean clone | Compact bundle rebuilds paper; complete training reruns require the separate ignored archive/environment |

The root README was rewritten from a stage-by-stage feature diary to the final scientific question and evidence. The legacy reproduction guide now warns that its training/sweep command is not the final-paper workflow. Frozen historical protocols/results were preserved, not edited to conceal earlier findings.

## Numeric audit

All 90 final raw rows are included and paired by ID plus observation/reference hashes. Source distances and Q errors are recalculated from predictions/truth; recovery is recomputed using ≤0.08 and ≤20%. Distributions are independently checked against the archived final summary. J1's 23/30, B_revised's 21/30 and classical's 30/30 counts, headline medians, signed bias, under/over counts, zero physical violations and reference flags are explicitly tested. Numerical, conditional-Q and development claims are traced to their respective bundled historical evidence rather than pooled with confirmation.

The reporter preserves missing classical continuous-PDE metrics. Bootstrap intervals are reporting-only paired scenario intervals, not new tests or a changed decision. Failed cases are not removed; no truth-selected seed or attractive manual example is substituted.

## Language and reference audit

The audit searches README, Markdown documentation and the manuscript for “successfully identifies pollution sources”, “outperforms classical methods”, “proves”, “real-world emitter”, “first ever”, “highly accurate” and “robust”. Occurrences are inspected in context: explicit negations, limitations, historical protocol terms and quoted claims in this audit are not affirmative performance claims. No unsupported affirmative statement is retained in the final report. A negative result is not softened into success.

External references were checked using the required gstack browser. Access-blocked publisher pages were not bypassed; publisher-deposited DOI metadata and author arXiv records are identified in the [bibliographic audit](../paper/supplement.md). No real sensor data were accessed, scraped or used.

## Verification record

The dedicated reporting suite checks no scientific imports, frozen hashes, corruption rejection, matching observations/seeds, failure retention, original failed decision, headline/historical metrics, case selection, exact short-description word counts, links, complete rendering and overwrite protection. It also checks the archived full-suite report as historical evidence, not a new run.

The scientific full suite was **not rerun**, because it includes small model-training tests and this milestone explicitly prohibits any training. Its preserved final-confirmation report records **451 passed, zero failures/errors**. No existing test was weakened or modified.

The dedicated reporting suite passed **27 tests, zero failures, zero pytest warnings**, with the final run taking 5.01 s; see [JUnit report](../paper/reporting_tests.xml). The complete original archive preservation comparison passed for **all 13,399 files**; its count and SHA-256 are recorded in the evidence manifest. Scientific source/config hashes, paired observations and every frozen outcome remain unchanged. A staged whitespace check initially flagged original CSV CRLF endings. They were preserved byte-for-byte, not normalized; `paper/.gitattributes` recognizes CR-at-EOL and disables checkout normalization for evidence. With that explicit archival policy, the staged whitespace check passes. A visual pass identified crowded logarithmic tick labels in the first draft convergence figure; only reporting labels/layout were corrected, with the first draft preserved in the local audit directory. All source data and scientific values remained unchanged. The first plotting invocation emitted an unwritable-home font-cache notice; using a writable temporary Matplotlib cache removed it. No mathematical test failure occurred and no existing test was altered. No claim is made that this documentation audit constitutes independent scientific peer review.
