# Manual publishing checklist

Public author name **Khadis Aidyn** was confirmed on 2026-09-21. It is already
applied to the citation, paper, poster and deck. No affiliation, ORCID, public
contact address or DOI has been supplied. These are not required fields.

Only outstanding manual choices/actions appear below; local artifact generation
and validation are complete in the [delivery inventory](FINAL_DELIVERABLES.md).

- [ ] Confirm first-person application/CV statements against actual contributions,
  including AI assistance, collaborators and personal technical work.
- [ ] Choose a public Git committer email before publishing; the existing identity
  uses an automatically derived local-machine address. No address was invented.
- [ ] License choice remains manual if reuse permission is desired.
- [ ] Create an empty GitHub repository named **InverPINN**.
- [ ] Add its actual remote URL after creation; no remote currently exists.
- [ ] Optionally rename the branch to `main`, or keep `scientific-revision`.
- [ ] Push only after approving publication and checking the intended files.
- [ ] Verify GitHub Actions completes successfully on the hosted runner.
- [ ] Set the description and topics below.
- [ ] Upload the existing workflow figure as a social preview if desired.
- [ ] Check GitHub citation rendering and manuscript/PDF links after the push.
- [ ] Optionally create a GitHub release named **InverPINN v1.0.0 - Research Freeze**
  using the [prepared notes](release_notes_v1.0.0.md). No tag or release exists yet.

## Repository metadata

Description: Physics-informed neural-network research on sparse pollution-source
inversion, inverse PDEs, and source-strength bias.

Topics: `physics-informed-neural-networks`, `pinn`, `inverse-problems`,
`scientific-machine-learning`, `pytorch`, `advection-diffusion`, `air-pollution`,
`source-inversion`, `pde`.

Social preview: [existing scientific workflow](../paper/generated/figures/01_schematic.png).
Keep the failed confirmation visible in release text: J1 23/30; required 27/30;
classical 30/30. A public research release is not a validated production model.
