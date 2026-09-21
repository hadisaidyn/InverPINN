# Local verification record

Checked on 2026-09-21. This is presentation QA, not a new scientific experiment.

## Environment and tests

- Isolated Python 3.13.3 environment with the exact [viewer requirements](requirements.txt); pytest 9.1.1. No PyTorch or scientific solver installed in this environment. Dependency consistency check passed.
- `python -m pytest tests/test_demo.py -q`: **47 passed**, no failures, no pytest warning summary. Includes all 30 cases, copied metrics/coordinates, fixed recovery labels, unchanged scientific paths, import guards, runtime interactions and full archived-bundle regeneration.
- Existing report/delivery environment: `python -m pytest tests/test_paper_reporting.py tests/test_repository_docs.py tests/test_final_delivery.py -q`: **68 passed**, no failures, no pytest warning summary.
- `python scripts/build_demo_data.py --verify-only`: passed. All 60 original prediction files matched their preserved inventory hashes during regeneration; the rebuilt manifest and every derived-file hash matched the committed demo bundle.
- Documentation link check passed. No historical test or scientific implementation was changed. The full scientific suite was deliberately not run because it includes fitting/simulation, outside this read-only task.

The first new-demo test run had 33 failures caused by one over-strict axis check, with 14 checks passing. Archived grid arithmetic differs from a new `linspace` by at most `1.1102230246251565e-16`, one float64 rounding unit. The new viewer validation was corrected to use absolute tolerance of two machine epsilons, with zero relative tolerance and exact endpoint/shape checks. File-hash equality remains exact. No data, existing tests or scientific thresholds were changed. The same demo tests then passed.

AppTest emits harmless bare-mode Streamlit log warnings about missing runtime/ScriptRunContext when importing outside a running server; these are not failed tests. A shell SHA utility encountered a host locale error, so existing Python SHA-256 utilities verified the downloaded files successfully. Pip also noted its unwritable sandbox cache and disabled caching; dependencies were consistent. No warning was suppressed by changing tests.

## Browser inspection

Used gstack browse with actual local Streamlit server and Chromium 145.0.7632.6; screenshots were opened and visually inspected. The frontend-design guidance was applied to a light, restrained scientific layout, not decorative generated imagery.

- Desktop 1360×1100 and tablet 768×1024: plots, legends and metric tables readable; document and main-container widths did not exceed viewport width. The wide failures table scrolls internally on narrow screens.
- Main map: saved field, sensor circles, true star, J1 × and classical diamond present. Both normalized axes remain exactly [0,1]. A presentation-only Plotly constraint was adjusted to shrink the plotting domain instead of extending its coordinate range.
- Scenario changes, worst-case 030 shortcut and the seven-case J1-failure filter worked; case 030 showed localization 0.084081, Q error 62.183% and L2 69.724%, retained as a failure. AppTest additionally covered all four field modes, saved-time changes, the paper's median-success shortcut 003 and the retained reference-quality flag for 016.
- Global Q plot showed both methods, all 30 estimates each and the identity line, with 28/2 J1 under/over counts. The classical-location toggle produced 30 additional classical markers and preserved the paired true/J1 markers. The retained-failure table was visible.
- Existing eight-image gallery loaded locally. Paper and reproducibility-guide download buttons worked; downloaded bytes had identical SHA-256 hashes to the committed originals.
- The failed-confirmation banner, frozen criterion, synthetic-only disclaimer and non-operational warning remained visible. No application JavaScript errors were observed; earlier console messages from the separate official documentation tabs were not app errors.
- [Screenshot](demo_preview.png) comes from the running app at default scenario 001, reference view, time 0.8. The capture viewport was enlarged vertically to include the complete page; no plotted values, markers or metrics were edited for the image.

## Performance and limits

A single local AppTest timing measurement gave **1.479 s** for initial app rendering and **0.505 s** for a scenario switch. These measure Python/Streamlit execution, not complete browser paint or public-network latency. Local browser startup and switching also worked interactively; no hosted timing claim is made. Cached data contains approximately 8.4 MiB, not full research archives.

No public deployment, training, inference, PDE simulation or new scenario generation occurred. Mobile phones and hosted service behavior were not tested. Overlapping estimates can obscure one another visually; distinct marker shapes, hover and legend toggles are available, and exact coordinates remain in the full-precision details.
