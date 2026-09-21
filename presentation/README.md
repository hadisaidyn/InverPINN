# Presentation deliverables

Khadis Aidyn confirmed the public author name. The research conclusion remains
**J1 failed confirmation: 23/30 recoveries, required 27/30; classical 30/30**.

- [A1 landscape poster](InverPINN_Poster.pdf), with existing Q and retained-failure figures.
- [Editable ten-slide PowerPoint](InverPINN_Presentation.pptx).
- [Slide PDF](InverPINN_Presentation.pdf), exported with bundled LibreOffice.
- [Speaker notes](speaker_notes.md), also embedded in the PPTX; target 5–7 minutes.
- [Content source](content.json), including per-slide evidence references.
- [PDF build provenance](pdf_build_provenance.json).
- [Delivery checks](delivery_checks.json), including hashes and structural checks.

## Reproduction and editability

The [PDF build](../scripts/build_final_pdf.py) uses public Python dependencies
and reproduces the paper/poster without scientific computation. See the
[reproduction guide](../docs/reproducibility.md#formal-paper-and-poster).

The [slide builder](../scripts/build_delivery_slides.mjs) used the bundled
`@oai/artifact-tool` 2.8.59 authoring runtime and its presentation finalizer.
That runtime is environment-provided, not promised as a publicly installable
npm dependency. A clean clone can open/edit the delivered PPTX without it;
building the PPTX from JavaScript requires an installation providing these
environment variables:

```text
ARTIFACT_TOOL_ENTRY      entry module for @oai/artifact-tool
PRESENTATION_SKILL_DIR   installed presentation skill directory
RUNTIME_PYTHON          authoring runtime Python executable
RUNTIME_NODE_MODULES    authoring runtime Node dependency directory
RUNTIME_NODE            authoring runtime Node executable
```

With those variables configured, run using that Node executable:

```bash
node scripts/build_delivery_slides.mjs artifacts/slides_reproduction
```

The destination must be a new `artifacts/` subdirectory. The builder creates a
candidate, validates all ten slides and the native benchmark table, then exports
a separate final PPTX. It changes only the export's generic core metadata to the
confirmed author/title before validation. The scientific images remain embedded
byte-for-byte, with their aspect ratios preserved. Text, table cells and notes
are editable; existing scientific plots remain image evidence rather than being
redrawn. No model or solver is imported.

For delivery, the PDF was exported by bundled LibreOffice 26.8.0.0.alpha0 and
all ten pages were visually inspected. The PPTX was package-validated and
rendered; it was not tested in Microsoft PowerPoint itself. Rendering in another
office application may differ. DejaVu Sans is used consistently; install that
font if the editing application substitutes it.

No QR code, public URL, email, affiliation or license has been invented.
