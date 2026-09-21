# InverPINN speaker notes

Khadis Aidyn. Target talk: 5-7 minutes.

## Slide 1: InverPINN

InverPINN studies an inverse problem: using measured concentrations to estimate the location and strength of an unknown source. The experiments are synthetic, with known physics and known truth for evaluation. That lets us ask a precise question about the inverse method. The final answer was mixed: the PINN often localized sources well, but its source-strength bias persisted and it failed the required recovery rate. A classical inverse method performed better in this controlled setting.

**Key number:** Final J1 recovery: 23/30.

**Transition:** First, what makes source inference different from predicting concentration?

Sources: `paper/manuscript.md`, `paper/evidence/final/decision.json`.

## Slide 2: Sparse measurements, unknown source

A monitor records pollution after wind and diffusion have moved it. It does not directly measure the source. Reconstructing the concentration field and recovering the source are therefore different tasks. Here the unknowns are two location coordinates and a Gaussian peak intensity. Twenty fixed virtual sensors provide observations. Almaty motivates the application, but these experiments do not identify real emitters or reproduce the city's sensor network. Synthetic truth makes the errors measurable.

**Key number:** 20 fixed sensors.

**Transition:** The forward relationship is the advection-diffusion equation.

Sources: `paper/manuscript.md`.

## Slide 3: Known transport physics

The equation balances change in concentration with wind transport, diffusion and a source. The source is one stationary Gaussian. Wind, diffusion and source width are known, so inference concerns only location and strength. We prescribe zero concentration initially and on all four edges of a unit square. These absorbing boundaries simplify the problem. They are not a general atmospheric boundary model, and all units remain synthetic rather than representing real emissions.

**Key number:** u=0.35, v=-0.15, D=0.005, sigma=0.08.

**Transition:** A PINN combines sensor fitting with this physical relationship.

Sources: `paper/manuscript.md`, `paper/generated/tables/table1_physics.md`.

## Slide 4: Neural field and source inference

The PINN represents the concentration field with a neural network and uses automatic differentiation for the equation's derivatives. Its objective includes sensor error and the PDE residual. The revised field enforces positivity and the initial and boundary conditions by construction. However, source parameters do not directly enter the sensor loss when the field is held fixed. Observation information reaches source inference through coupling to the PDE residual. This distinction motivates the later bias diagnosis.

**Key number:** Both final neural methods use 12,000 updates.

**Transition:** Before interpreting inverse errors, the numerical reference needed checking.

Sources: `paper/manuscript.md`, `docs/source_field_coupling.md`.

## Slide 5: Reference quality and blind evaluation

Manufactured solutions and refinement checks preceded inverse experiments. The coarse reference was rejected for accuracy, and the final reference used a 641 by 641 grid. Thirty new final scenarios followed a frozen protocol. Fitting used sparse observations, while source truth and dense fields supported evaluation only. Three numerical-reference estimates exceeded the one-percent target and remained in the report. These are Richardson estimates, not certified error bounds. The required joint recovery rate was fixed in advance.

**Key number:** Final requirement: at least 27/30 joint recoveries.

**Transition:** Earlier tests had already exposed a gap between physical validity and recovery.

Sources: `paper/manuscript.md`, `paper/evidence/final/protocol.md`.

## Slide 6: Early failure and persistent Q bias

On an earlier, separate held-out set, the initial PINN recovered fourteen of thirty cases and predicted negative concentrations at many evaluation points. Hard constraints later removed those violations. They did not ensure correct source estimates. The revised PINN continued to underestimate source strength, including every case in the observability diagnostic set. These historical experiments are separate from the final thirty scenarios and are not pooled to improve a headline number.

**Key number:** Historical B0 median negativity: 35.424%.

**Transition:** Could weak sensor information alone explain the PINN's failures?

Sources: `paper/manuscript.md`, `paper/evidence/historical/initial_summary.json`.

## Slide 7: Classical inversion exposed the recovery gap

Classical inversion searches source parameters through a numerical forward solver. It recovered all thirty final cases using the same observations as the PINNs. Separate conditional-strength diagnostics also recovered Q accurately when location was known. Together these findings weaken insufficient information as a complete explanation. The displayed historical observability map describes local sensitivity across source locations. It is not a posterior map or proof of global uniqueness. The comparison remains specific to the tested source family and known physics.

**Key number:** Classical final recovery: 30/30.

**Transition:** J1 then tested an analytical intervention in the PINN's amplitude optimization.

Sources: `paper/manuscript.md`, `docs/observability_diagnosis.md`.

## Slide 8: J1 profiles source amplitude analytically

For a fixed neural field and source location, the PDE residual is linear in source amplitude. Its least-squares optimum therefore has a closed form. J1 uses differentiable variable projection, subject to the positive amplitude domain, rather than optimizing Q as an independent Adam parameter. This uses no source truth. But the optimum is conditional on the learned field and sampled residual. An imperfect field can still produce a biased optimum. Profiling improved the development results without eliminating that possibility.

**Key number:** Q is profiled each update, plus a fixed terminal profile.

**Transition:** The decisive question was how the frozen method performed on fresh cases.

Sources: `paper/manuscript.md`, `docs/source_field_coupling.md`.

## Slide 9: J1 failed final confirmation

J1 recovered twenty-three of thirty sources jointly, or 76.67 percent. The preregistered requirement was twenty-seven, so confirmation failed even though its median errors passed the other gates. Median localization was 0.005574, relative Q error 5.975 percent and concentration L2 7.423 percent. Classical inversion recovered all thirty with lower median errors. J1 still underestimated Q in twenty-eight cases. Exact nonnegativity and zero initial and boundary violations did not compensate for the failed recovery rate.

**Key number:** J1: 23/30. Required: 27/30. Classical: 30/30.

**Transition:** The conclusion depends on the retained failures, not only the medians.

Sources: `paper/generated/tables/table3_final_benchmark.md`, `paper/evidence/final/decision.json`.

## Slide 10: Physical validity did not guarantee recovery

This is the final case with the largest localization error, selected mechanically rather than for appearance. Its source-strength error was 62.183 percent and field L2 error was 69.724 percent. It remained in every aggregate. The evidence supports a formulation and optimization limitation in these PINNs, without uniquely isolating the cause. The project stopped after failed confirmation. Future noise, meteorology or real-data studies would require separate authorization and validation. The lesson is to distinguish physical consistency and attractive medians from dependable inverse recovery.

**Key number:** Worst displayed case: Q error 62.183%; L2 69.724%.

**Transition:** Questions.

Sources: `paper/manuscript.md`, `paper/evidence/manifest.json`.
