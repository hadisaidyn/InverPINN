# Source–field coupling: derivation before Revision 7 fitting

## Objective and information pathways

Write `C_theta=16x(1-x)y(1-y)(t/T) softplus(f_theta)` and
`G=exp(-((x-xs)^2+(y-ys)^2)/(2 sigma^2))`. Q is peak source
intensity (concentration/time), not total emitted mass. Coordinates and time
use the unchanged synthetic physical units. The equation is
`C_t+u C_x+v C_y-D(C_xx+C_yy)=Q G`, with zero initial and Dirichlet data.

For sensor readings `z_j`, the actual objective is

`L = 10/C*^2 mean_j(C_theta(z_j.coordinates)-z_j.value)^2`
`  + T*^2/C*^2 mean_i(R0_theta(z_i)-Q G_i)^2`
`  + 1/C*^2 (mean IC^2 + mean BC^2)`,

where `C*=0.88`, `T*=0.8`, and
`R0=C_t+u C_x+v C_y-D(C_xx+C_yy)`. IC and BC are identically zero
under the hard transform. Batches are uniformly sampled, not importance
weighted. Preference weights and unit factors are separate.

At fixed theta, `partial L_data/partial (xs,ys,Q)=0`. The data gradient
changes theta; subsequent PDE gradients transmit that change to the source.
With `r=R0-QG`, `partial L_PDE/partial Q=-2 mean(G r)` and
`partial L_total/partial raw_Q=-2 w_PDE mean(G r) sigmoid(raw_Q)`
for J0's softplus parameter. Location derivatives are
`-2 w_PDE Q mean(r G (x-xs)/sigma^2)` (and the analogous y term),
then multiplied by the sigmoid-location Jacobian. Small Q also attenuates
location information in this partial gradient. These are partial derivatives;
the fitted field's dependence on Q is an optimization pathway, not an explicit
forward PDE solution inside the data loss.

Finite sparse observations do not uniquely constrain an arbitrary neural
field. Perturbations approximately zero at sensors can change derivatives at
collocation points. Thus smaller Q and a different field can jointly reduce
this finite penalty objective without solving the exact inverse problem.
This establishes a possible mechanism, not its empirical causal magnitude.
Profiling the PDE residual alone does not introduce direct sensor dependence.

## Exact weighted amplitude profile

For fixed theta and location, let positive weights `a_i` include quadrature
or sampling weights and any fixed PDE normalization. Minimize
`sum_i a_i (R0_i-Q G_i)^2 / sum_i a_i`. Define
`N=sum a_i G_i R0_i`, `H=sum a_i G_i^2`.
For `H>0`, the unrestricted minimizer is `N/H`. Under `Q>=epsilon`,
`Q_profile=max(epsilon,N/H)`. We use the dtype's smallest positive normal
epsilon, matching the numerical positive domain of J0, not a scientific
prior. An open domain Q>0 has only an infimum when N<=0; the numerical
epsilon explicitly resolves this boundary case. H=0/nonfinite inputs fail
loudly; no ridge or hidden amplitude prior is introduced. Accumulation is
float64, then returned in the input dtype. The actual uniform objective has
`a_i=1`; the common PDE weight cancels. If extra Q penalties are introduced,
this formula would no longer be the correct profile.

J1 uses differentiable variable projection of the current collocation batch.
At an interior optimum `partial L/partial Q=0`, so the envelope theorem
makes its gradient equal to a detached exact block optimum; we test this.
At the epsilon boundary the active branch has zero Q-profile derivative.
Q is a checkpointed buffer, never an Adam parameter. After the fixed budget,
Q alone is profiled on a predeclared independent *fitting* set of 8192 points,
not the independent evaluation points. This terminal operation is included
in J1/J2 budgets and cannot inspect observations' source truth.

J2 alternates 150 field-only updates (source fixed) and 50 location-only
updates (field fixed; Q reprofiled each step), for 60 cycles: 9000 field and
3000 location updates. Separate persistent Adam states have lr=0.001.
It is an equal-total-update comparison, not equal field-update or wall-time
comparison. During a source block, coordinate differentiation of the frozen
field remains enabled, but field parameters cannot receive optimizer updates.

J3, explicitly **PINN-location + forward-profiled-Q**, keeps J0's location,
solves the independent 201-square numerical PDE for unit Q, and profiles
`Q=max(epsilon,h^T observations/h^T h)`. It then reconstructs the numerical
field at that location and amplitude. This is hybrid inference, not a pure
PINN, and does not change location. A biased location can still bias Q.

## Planned empirical checks

All quantification uses only development_v3. Log field/source gradients and
data/PDE gradient cosine at updates 1, 100, 200, ...; no callback may mutate
parameters, gradients or RNG. Evaluate loss over all observed readings at
these checkpoints (no truth). Compare Q's distance to its eventual
observation-conditioned optimum with successive full-sensor loss changes.
This is a descriptive trajectory test, not proof of a unique causal mechanism.
Predetermined examples are IDs 001, 009, 017, 024. No prior scenario outcomes
are reopened for selection. Results will be reported separately after fitting.
