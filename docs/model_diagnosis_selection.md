# Development-only candidate-selection rule (before candidate results)

Revision 4 may nominate **at most one** B_revised. The old held-out set is
consumed. These ten development cases are model-selection data, not an unbiased
performance estimate, and no p-value or confidence interval will disguise
that distinction. All attempted candidates retain all ten cases and failures.

First exclude any candidate with a training/evaluation failure, any negative
concentration in the prescribed dense diagnostic grid, or worse mean or
maximum physical BC/IC RMS violation than B0. Also exclude candidates with a
worse mean independent, physical-unit PDE RMS than B0. A zero-loss training
claim cannot substitute for these separate diagnostics. Exact homogeneous
constraints are preferred to a similar numerical penalty, but do not guarantee
that the interior solution is accurate.

Next require **both mean and median** localization error, relative Q error,
and concentration relative L2 to be no worse than B0. These non-regression
criteria avoid selecting a localization improvement that destroys amplitude
or concentration accuracy. No post-hoc weighted score is permitted.

Among eligible candidates remove those Pareto-dominated on median localization,
median Q error and median concentration L2. If more than one remains, use the
following declared lexicographic priorities: lower median Q error, then lower
median concentration L2, then lower median localization error, then lower mean
training runtime. Report all tradeoffs, including means, physical maxima and
runtime, even when not used as final tie breakers. Initialization diagnostics
remain separate from independent optimization-seed validation.

If none is eligible, do **not** invent a successful revised configuration or
relax this rule. Report no nomination and no justification for a fresh test.
If a candidate is nominated, freeze its complete configuration; testing it on
fresh unseen cases belongs to a later milestone. Development success alone
does not validate inverse identifiability or a real-data causal interpretation.

For multistart, the source truth and dense field are forbidden inputs to
within-case selection. Select the smallest common independently sampled
sensor-plus-physics objective, using fixed diagnostic points and all observed
sensor readings, with lexicographic start-name tie breaking. Report the sum of
all start runtimes, not only the chosen run. Never select the truth-best start.
