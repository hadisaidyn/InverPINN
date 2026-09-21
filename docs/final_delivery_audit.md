# Final delivery audit — 2026-09-21

## Scope and identity

Delivery began on clean `scientific-revision` at
`1abae14e6cd4958aacc65cb2508f9a21d36feb5c` (`Polish GitHub repository for public research release`).
No remote or license was present. The user explicitly confirmed **Khadis Aidyn**
as the public name. This replaces the previously unverified, automatically derived
“Hadis” in current citation/display metadata, not in historical Git records or
sealed provenance. No institution, affiliation, public email, ORCID or DOI was invented.

Git had no explicitly configured `user.name` or `user.email`. The final commit
uses the confirmed name as a command-local override without changing global
configuration or inventing a public email. Public Git identity remains a manual
check before pushing. Personal implementation/derivation contributions were not
confirmed; first-person application copy explicitly requires that review and
acknowledges AI assistance. Research-release v1.0.0 notes are a draft, not a tag
or published release. Package version remains historical **0.0.0**.

## Scientific and claim preservation

No training, optimization, model inference, PDE simulation or new scientific
experiment ran. Scientific implementations, frozen configurations, original
manuscript/supplement, sealed evidence and canonical figures/tables were not edited.
The artifact builders only read existing evidence and typeset presentation copies.
The checker verifies all **41 evidence files and 33 canonical report files** by
SHA-256. Scientific paths are also compared against the starting Git commit.

The additional strict original-archive check was rerun and remains nonzero for
the **same three ignored generated `src/inverpinn.egg-info/` files** documented
during GitHub polish: `PKG-INFO`, `SOURCES.txt` and `requires.txt`. It found no
additional differences or added files: **13,396/13,399** inventory entries remain
byte-identical. This is not reported as an unqualified preservation-check pass.
The original inventory and checker were not modified; see the
[previous caveat](github_polish_audit.md#strict-archive-check-caveat).

The final result appears prominently in the abstract, poster and slide 9:
**J1 23/30 = 76.67%, below required 27/30 = 90%; confirmation FAILED.**
The classical comparison remains **30/30**. Median localization/Q/L2 remain
0.005574/5.975%/7.423% for J1 and 0.001111/1.218%/1.507% for classical inversion.
J1's **28 underestimates and two overestimates** remain explicit. Exact
nonnegativity and IC/BC are not presented as proof of source recovery.

Historical B0 results and development/observability results are labeled as
separate experiments, not pooled with final results. “Accurate,” “successful” and
“validated” occurrences refer to specific numerical checks or retained case-level
results, or appear inside limitations/negated claims. There is no claim of a
production model, general PINN superiority or validated Almaty emitter attribution.
Application word counts are exact under whitespace counting; the Common App
activity is 132 characters. No awards, publication venues or personal credentials
were invented.

## Typesetting, bibliography and visual inspection

- Paper: **14 A4 pages**, comprising 12 main pages and two appendix pages.
  Seven numbered equations, eight existing scientific figures, four tables and
  six references. All citations and entries match; existing reference metadata
  was preserved without new browsing or invented identifiers.
- Poster: **A1 landscape, approximately 841 × 594 mm**, one page. Existing Q
  comparison and mechanically selected failure images are unchanged.
- Slides: **10**, native editable text and benchmark table, original embedded
  scientific figures, ten embedded note sections and matching Markdown notes.
  Author metadata was corrected from the exporter's generic default before
  package validation. The PDF was exported with bundled LibreOffice.
- All 14 paper pages, the full poster and all ten slide-PDF pages were rendered
  and inspected. Artifact Tool previews were also inspected. No clipping,
  overlapping content, broken references or missing figures remained. Following
  the final math-typography correction, only paper page 4 changed in a pixel
  comparison; that page was re-inspected. Other paper pages and the poster were
  pixel-identical to their inspected renderings.
- PDF fonts are embedded. A PowerPoint application-level test was not performed;
  package validation, Artifact Tool import/render and LibreOffice PDF export were.

The PDF skill guided local typesetting and page-by-page render review. The
presentation skill guided editable slide construction and package/layout
validation. No decorative scientific figures were generated.

## Checks and fixes

The isolated report-only environment uses the existing reporting pins plus
ReportLab 4.4.9, pypdf 6.10.0 and Pillow 12.3.0. The scientific environment was not
modified. The full scientific test suite was deliberately not run because it
includes training. Its earlier 451-pass result remains historical only.

Delivery checks cover PDF openability/page counts/metadata, required numbers,
PowerPoint ZIP integrity, ten slides/notes, native table presence, unchanged
embedded figures, no external slide relationships, evidence hashes, exact
description lengths, bibliography completeness, no-overwrite guards, protected
paths and no scientific imports. Existing report/docs tests are retained.

During development of the delivery layer:

1. A bibliography parser initially misread the domain `[0,1]` as citations;
   the distinction is now explicit and tested, without changing references.
2. Visual inspection found an emphasis parser consuming the mathematical stars
   in `C*` and `T*`. The renderer was corrected to preserve attached stars,
   with a regression test. No scientific equation was revised.
3. An intermediate link-check run correctly failed while the audit and final
   check report had not yet been created. Those deliverables were completed;
   the link test was not relaxed.
4. The old author assertion now checks the user's confirmed name. This is a
   metadata expectation change, not a weakened scientific test.
5. Staging identified ASCII-heavy PDFs as text. Explicit binary attributes for
   the four new document packages prevent newline conversion and meaningless
   whitespace diffs, while leaving the document bytes untouched. A regression
   test checks those Git attributes. An extra blank line in the notes was removed.

Initial package installation needed network permission; installing into the
temporary report-only environment succeeded. Matplotlib and LibreOffice emitted
font-cache-location warnings in initial builds; writable reporting cache paths
were used and final rendered text was checked. No scientific warnings were
suppressed. The PPTX authoring runtime is environment-provided; its portability
limit is documented in the presentation README rather than promising a public
npm installation that has not been verified.

See [machine-readable checks](../presentation/delivery_checks.json) for exact
artifact sizes and hashes, [PDF provenance](../presentation/pdf_build_provenance.json)
for fonts/packages and [manual publishing actions](PUBLISHING_CHECKLIST.md).

## Final non-training verification

- Targeted delivery tests: **26 passed**.
- Combined report/document/delivery tests: **68 passed, 0 failed, 0 skipped,
  0 warnings** (27 reporting, 15 repository documentation, 26 delivery).
- Documentation checker: **54 public Markdown files, 293 links, 0 errors**.
  External URLs are not live availability checks.
- Evidence verification and separate report rebuild: passed, preserving J1's
  failed confirmation. All 41 evidence and 33 canonical artifact hashes match.
- Fresh paper/poster rebuild: passed; extracted text is identical to the
  delivered PDFs. Page geometry, metadata and sizes pass the delivery checker.
- PPTX finalizer: package integrity, ten-slide layout, required native table,
  font policy and Artifact Tool re-import all passed, with zero findings/warnings.
- All four binary deliverables open structurally; manual visual checks are
  described above. Bibliography completeness and all headline-number checks pass.
- `git diff --check`: passed. Scientific paths have zero diff against the
  starting polish commit. The separately documented strict archive-metadata
  exception remains unresolved by design, not silently repaired.

No GitHub-hosted CI run, push, tag, release or repository-settings change was
performed. The final commit is a local delivery commit only.
