# Figure gallery

Eight existing, unchanged figures from the completed study. Final confirmation
failed: **J1 23/30, required 27/30; classical 30/30**. The
[manuscript](../paper/manuscript.md) supplies full captions; the
[canonical index](../paper/generated/README.md) also links vector PDFs.

## 1. Controlled workflow

![Gaussian source, transport, sensors and inverse methods](../paper/generated/figures/01_schematic.png)

The source-to-observation-to-inference workflow and controlled assumptions.
[Methods and study design](../paper/manuscript.md).

## 2. Numerical reference qualification

![Numerical convergence and reference resolution](../paper/generated/figures/02_numerical_convergence.png)

Refinement tests motivate the high-resolution reference; estimated errors are not certified bounds.
[Numerical validation](numerical_validation.md) · [Reference report](synthetic_reference.md).

## 3. True and inferred locations

![Source locations for the three frozen methods](../paper/generated/figures/03_source_locations.png)

All 30 final cases are shown for the paired methods, without removing failures.
[Final confirmatory results](J1_confirmatory_results.md).

## 4. Source-strength bias

![True versus predicted Gaussian source strength](../paper/generated/figures/04_source_strength.png)

J1 underestimated Q in 28/30 cases; analytical PDE profiling did not eliminate signed bias.
[Final confirmatory results](J1_confirmatory_results.md).

## 5. Error distributions

![Localization, strength and concentration error distributions](../paper/generated/figures/05_error_distributions.png)

Distributions expose failure tails that median errors alone hide.
[Final benchmark table](../paper/generated/tables/table3_final_benchmark.md).

## 6. Observability and sensor geometry

![Observability diagnostics with sensor geometry](../paper/generated/figures/06_observability.png)

Historical information diagnostics help interpret the recovery gap, not prove global uniqueness or causation.
[Observability diagnosis](observability_diagnosis.md).

## 7. Mechanically selected recovery

![Reference, predictions and errors for final case 003](../paper/generated/figures/07_success.png)

Case 003 is the median-localization case among the 23 successful J1 recoveries, not a manually chosen attractive example.
[Case-selection rules](../paper/supplement.md).

## 8. Mechanically selected failure

![Reference, predictions and errors for final case 030](../paper/generated/figures/08_failure.png)

Case 030 has the largest J1 localization error; its source and field errors remain in every aggregate.
[Failure discussion](J1_confirmatory_results.md).

[Back to documentation](README.md) · [Report-only reproduction](reproducibility.md).
