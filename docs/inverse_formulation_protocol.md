# Revision 7 development-only protocol (pre-fit)

This is model development, not independent validation. Revision 6 is preserved
at 199559f901bd21981f155d70550e3c82d06dde70. All historical results are consumed.
Their bytes are hashed for preservation, not read for candidate selection.

Before generation, freeze this document, the YAML protocol, derivation, source
plan and original code/config hashes. The new seed is 2009202607. Twenty-four
independently permuted/jittered Latin-hypercube draws cover x,y in [.25,.75]
and Q in [.8,1.4], one draw per marginal stratum. This deliberately includes
12 right-half locations and all y/Q strata, without selecting prior failure
coordinates or rejecting any draw. Marginal strata do not guarantee every
joint subregion. No outcome-dependent redraw is allowed. Reconstruct prior
100 scenario plans solely to exclude exact source/seed/ID overlap; additionally
exclude configured historical source values. A checksum inventory protects
every existing result file, not just the last benchmark.

Keep the complete controlled problem, sensor layout and observation schedule.
Generate 161/321/641-square refinements; always use 641-square reference.
Retain and flag all estimated 1% Richardson target misses. Estimates are not
certified bounds. Freeze the complete reference/input manifest before fitting.
Compute the previously defined 201-square scaled Jacobian/SVD diagnostics and
the median observed-sensor-RMS split before PINN performance exists. These are
descriptive information strata, never exclusions or training inputs.

## Fixed candidates and budgets

The separate source_field_coupling.md derivation precedes modified training.
J0 is byte-unchanged B_revised, 12000 updates, seed42. Read-only gradient
instrumentation must pass a bitwise checkpoint-equivalence test.
J1 uses 12000 joint field/location Adam updates with exact differentiable
Q projection on each 512-point PDE batch. J2 uses 60 cycles of 150 field-only
then 50 location-only updates, profiling Q in source blocks. Adam states persist
across cycles; architecture, initial field/location, lr, batches and weights
match J0. Both terminate with one Q-only profile on 8192 separately seeded
fitting points. No truth or evaluation points enter profiling.
J3 reuses J0 location and observation-only 201-square Q projection; report its
total runtime including its J0 prerequisite. Its concentration is numerical,
not the original neural field. Its native-grid centered-difference continuous
PDE diagnostic differs from neural autograd and is labeled separately; do not
claim strict PDE dominance from these nonidentical diagnostics.

All four methods run on all24 primary cases. The four index-selected IDs
001,009,017,024 also use seeds142,242, giving three total seeds including42.
These 8 extra fits per neural candidate are never substituted for primary
results. J3 reuses the corresponding J0 location. Total: 96 neural fits,
1,152,000 optimizer updates plus 64 terminal Q profiles, and32 hybrid profiles.
No multistart truth selection. No J4 is planned: J2 already implements the
suggested combined A/B/C structure. No additional location optimizer is
planned: J2 already uses the proposed field-frozen location-only Adam.
If results do not justify a distinct new intervention, stop with J0–J3.

## Selection fixed before aggregate performance

A pure PINN can advance only if *all* primary advancement criteria in YAML
hold, no failed fits occur, and Q underestimation is not universal (equality
tolerance1e-6). Median PDE must be at most J0 median plus
max(10% of J0 median,1e-4); also apply this rule to PDE90th percentile.
Require >=20% improvement in median absolute relative Q error and absolute
mean signed Q bias. Relative to J0, localization median/90th/worst, Q90th/worst,
L2 median/90th/worst, mean RMSE/MAE must each be <=1.10 times baseline;
recovery count cannot decrease. Require zero negative predictions and exact
IC/BC in all primary and additional-seed runs. Additional-seed recovery count
cannot fall below J0's count on the same 12 scenario-seed pairs; all fits must
finish. Report ranges, variance, runtime and every outlier. These constraints
avoid choosing solely on one median. If multiple candidates pass, rank by
recovery count, Q median, location median, then runtime (all directions clear).
No hybrid B_revised_2 is eligible here because its PDE diagnostic is not the
same continuous autograd test; a hybrid success can motivate a separately
specified later validation, not silently become a pure PINN freeze.

Promising targets are development advancement criteria, not generalization
claims: median location<=.02, Qrelative<=.10, L2<=.15; no universal low Q;
exact nonnegative IC/BC; no material PDE regression. Signed errors are
Qpred-Qtrue in physical peak-intensity units. Secondary recovery remains
location<=.08 AND Qrelative<=.20. Report distributions/all failed fits.
Mechanistic evidence must distinguish a possible structural pathway from a
uniquely established causal explanation. No new held-out, noise, sensor-count,
wind, multi-source or Almaty work is permitted.
