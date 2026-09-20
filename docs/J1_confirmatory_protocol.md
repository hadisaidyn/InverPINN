# Revision 8: final J1 confirmatory protocol (before generation)

J1 was generated during development but was not promoted under the previous
selection rule. Revision #8 evaluates it as a newly frozen hypothesis on
independent data. Revision7's 19.83% Q-error and19.44% signed-bias improvements
did not meet its20% gates. This history is not revised or reinterpreted as a pass.

The verified starting state is scientific-revision, clean, commit
33892895805d692c8c3960d8646c43d6a4f7536d. J1_confirmatory.yaml extracts the exact
executed J1 recipe from that commit, not a newly tuned candidate. All existing
scientific source and configuration files are SHA256-locked. Separate hashes
record J1 configuration, training, MLP, constrained field, profiling, residual,
and classical-inverse code. The fitting execution is separately archived
before its first fit. Any implementation change after that point stops execution
for review; a genuine invalidating bug must not be patched and silently resumed.

## Fixed method

J1 uses the unchanged three-hidden-layer64-wide tanh MLP, coordinate normalization,
and C=16x(1-x)y(1-y)(t/.8)softplus(f). x/y are sigmoid parameters initialized
at.5; Q starts at1.1 but is an analytically profiled buffer, never an Adam
parameter. Uniform512-point PDE batches give Q=max(dtype.tiny,sum(G R0)/sum(G²)),
R0=Ct+.35Cx-.15Cy-.005(Cxx+Cyy), with float64 accumulation and differentiable
projection. Adam lr.001 runs exactly12,000 field/location updates; no scheduler,
rescue or extension. The final Q-only profile uses8,192 fitting points with
seed44. Primary fitting seed42, batch generator43, data512, IC128, BC64/edge,
weights10/1/1/1, C*=.88,T*=.8, CPUfloat32 and one torch thread are unchanged.
The same read-only100-update gradient logger used in Revision7 is retained.
IC and BC losses are exactly zero under the hard transform.

B_revised is the unmodified frozen comparator, seed42 and12,000 updates.
Classical inversion is the unchanged Revision6 algorithm: independent201²
forward solver dt.00025, three separated starts selected by observation-only
profiled MSE from the fixed21×21 location grid [.25,.75]²; bounded[0,1] x/y
least squares with analytic nonnegative Q profiling, max_nfev60 and unchanged
tolerances/scaling/sensitivity. All starts are saved; selection uses sparse
MSE, never truth. Classical inversion is not part of J1.

## Untouched scenarios and reference

Before drawing sources, freeze this protocol and YAML. New seed2009202608
produces30 IID uniform draws in x,y∈[.25,.75],Q∈[.8,1.4], named
final_blind_v1_001…030. Exclude every prior development, heldout, fresh-blind,
diagnostic_v2 and development_v3 plan and historical configured source tuple.
Prior results are checksum-protected, not read for model selection. No redraws,
replacements, manual source selection or exclusion of inconvenient cases.

Use exactly one Gaussian, sigma.08,u.35,v-.15,D.005,zeroIC,zeroDirichletBC,
the identical20-sensor geometry and81 times0…0.8, zero added noise. References
always use641²,dt.00003125;161²/321² refinements estimate accuracy. Flag and
retain every estimated1% target miss. Richardson estimates are not certified
error bounds. A serious numerical failure stops for review, without replacement.
Freeze source/reference/input hashes and manifest before fitting. Truth is
separated from inputs and may not enter any fitting API or worker I/O. Terminal
attempt/checkpoint records precede evaluation, for successful and failed fits.

## Endpoints and decision, frozen before generation

ALL of the following are required; comparisons use unrounded values:

- At least27/30 joint recoveries: Euclidean localization<=.08 AND |Qhat-Q|/|Q|<=.20.
- Median localization<=.02; median absolute relative Q error<=.10.
- Median concentration relative L2<=.15.
- Absolute median signed relative Q error |median((Qhat-Q)/Q)|<=.10.
- Not all30 Q estimates are underestimates (equality absolute tolerance1e-6).
- Negative fraction=0, BC maximum violation=0, IC maximum violation=0.
- Median J1 PDE residual RMS<=1.10×median B_revised PDE residual RMS, using
  the exact same paired30 cases and independent evaluation points. This is a
  ratio of paired-sample medians, not a median of per-case ratios. Both methods'
  complete finite diagnostics are required; missing evaluations cannot pass.

All90 attempts must finish with evaluable finite outcomes for advancement;
failed attempts remain in raw tables and recovery denominators. A failed fit
is not retried. Unexpected implementation errors preserve partial artifacts
and stop for review, not automatic rescue.

Concentration metrics use all641² nodes at t=.1,.2,.4,.8. Physical diagnostics
use8,192 interior,4,096 initial and1,024/edge boundary points, seed2009202808,
independent of fitting. Report PDE RMSE/MAE, BC/IC RMSE/max, negative fraction
and minimum concentration. Classical field reconstruction uses its201² solution
bilinearly mapped to the reference grid, retaining its discretization mismatch.

Distributions: mean, median, sampleSD(ddof1), Q25,Q75,Q90,worst, finite/missing
counts. Recovery uses the fixed denominator30 and95% Wilson interval.
Signed Q and signed relative Q means/medians and under/over/equal counts are
explicit. No headline best-of-seeds or additional primary fitting seeds.

Paired J1-minus-B_revised differences cover localization,Qerror,L2,PDE and
runtime (also RMSE/MAE). Ties: |delta|<=1e-8+1e-6 max(|a|,|b|). Bootstrap
10,000 resamples of complete paired scenario indices, seed2009202818; percentile
95% intervals for mean and median differences, reporting missing-pair counts.
Do not interpret statistical significance as the advancement rule. J1-minus-
classical source/field gaps and J1/classical runtime ratios are reported.
Runtime includes all fitting starts and fixed logging/checkpoint overhead;
classical shared response-bank construction is reported separately, together
with amortized cost per30 cases. No claim of universal runtime superiority.

Before fitting, freeze the median observed-sensor-RMS split. Existing scaled
Jacobian/SVD diagnostics are computed only for descriptive failure analysis,
never passed to training. Rank/condition proxies are not posterior uncertainty.
Worst three J1 examples are chosen mechanically by localization descending,
scenario ID for ties; computational failures rank first and remain explicit.
Report source truth/estimate,Q,errors,signal,smallest scaled singular value,
and independent PDE residual. All failure associations are exploratory.

## Conditional robustness, only after all confirmation gates pass

If and only if all criteria pass, copy J1_confirmatory.yaml byte-for-byte to
InverPINN_final.yaml; controlled method name InverPINN-J1. Never retune it.
If any criterion fails, do not create that file, do not generate robustness
scenarios, and stop the pure-model-development loop. No Revision9 small tweak.

If passed, automatically generate20 new IID diagnostic scenarios, seed2009202838,
excluding all consumed and final-blind scenarios. Keep the same true physics
and641² reference. One seed42 fit per distinct condition; no selection/tuning.
Freeze their manifest before fitting. All perturbation studies use these same
underlying cases and reuse the identical noiseless20-sensor/correct-wind control.

R1: nested5/10/20 subsets of the existing geometry, deterministic farthest-point
coverage. First index maximizes distance from centroid; subsequent indices
maximize distance to the selected set; lowest original index breaks exact ties.
Freeze indices before outcomes. No noise; correct wind.

R2:20 sensors; z_l=z+l*s*Z, l∈{0,.05,.10,.20},
s=sqrt(mean(z²)) over the complete noiseless scenario readings. Z has IID N(0,1)
entries from a per-scenario child of seed2009202848, reused across levels.
No clipping or deletion of negative noisy readings, including initial readings;
the model still imposes the exact known zero IC. This conflict is part of the
declared measurement-noise model, not an invitation to change constraints.

R3:no measurement noise,20 sensors, truewind(.35,-.15). Inference wind is
unchanged, multiplied by.9 or1.1, or rotated by-10/+10 degrees using the usual
counterclockwise rotation matrix. Speed is preserved for rotations; D unchanged.
Same frozen model settings, newly fitted to each perturbed inference problem.

Report raw source/field errors, recovery and failure regions for all levels,
including the six requested robustness figures. Model settings never adapt.
Robustness describes operating limits, not another selection stage. No real
Almaty, AirKaz, ERA5 or other source family work is authorized in this revision.
