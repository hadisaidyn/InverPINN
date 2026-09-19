# Revision 4: dimensional audit and a consistent dimensionless objective

This derivation concerns only the existing synthetic unit-square problem. It
does not assign metres, seconds or micrograms to the Almaty study. The code's
synthetic length, time and concentration units must not be mistaken for those
real-world units.

## Unchanged governing equation

\[
C_t+uC_x+vC_y=D(C_{xx}+C_{yy})+S,\qquad
S=Q\exp[-((x-x_s)^2+(y-y_s)^2)/(2\sigma^2)].
\]

The residual is left minus right: **Ct + u Cx + v Cy - D Laplacian(C) - S**.
Advection carries pollution with the wind; diffusion smooths differences;
the nonnegative source adds pollution. Q is the Gaussian's *peak rate*, with
units concentration/time, not its integrated emission. On the whole plane the
integral is 2 pi sigma² Q; on the finite square Gaussian tails are truncated.

Initial concentration is zero. Concentration is zero on all four boundary
edges for all times. These are absorbing Dirichlet conditions, not zero flux.
For positive D and a nonnegative source the parabolic maximum principle implies
nonnegative concentration. Sparse sensor agreement alone does not imply this.

## Audit of B0

The MLP maps x,y,t to [-1,1] internally. Autograd differentiates with respect to
the original coordinates, so the input-normalization chain rule is included.
This is **not** nondimensionalization of the entire equation. C and the source
are not normalized. Data/IC/BC losses have units C²; the PDE loss has units
(C/time)². B0's numerical weights (10,1,1,1) implicitly select a time-unit
convention. Changing the time unit without changing those weights changes the
optimization problem. This does not establish an incorrect PDE sign or
incorrect derivative, and does not by itself prove the cause of Q bias.

## Characteristic scales fixed without scenario truth

Use L=1 synthetic length (domain side), T=0.8 synthetic time (observation
window), Q*=1.1 concentration/time (midpoint of the already specified prior
[0.8,1.4]), and C*=T Q*=0.88 concentration. C* is an upper-bound-style production
scale, not a fitted sensor standard deviation or a true concentration maximum.
The velocity scale is U*=L/T=1.25; diffusivity scale is L²/T=1.25; source scale
is C*/T=1.1. All scales are fixed across all ten development cases.

Define xi=x/L, eta=y/L, tau=t/T, c=C/C*, u_hat=uT/L, v_hat=vT/L,
D_hat=DT/L², q=QT/C*, sigma_hat=sigma/L. Applying the chain rule gives

\[
c_\tau+\hat u c_\xi+\hat v c_\eta
-\hat D(c_{\xi\xi}+c_{\eta\eta})-\hat S=0,
\quad \hat S=(T/C_*)S.
\]

Here u_hat=0.28, v_hat=-0.12, D_hat=0.004, sigma_hat=0.08. No advection,
diffusion or source sign changes. The physical residual r and dimensionless
residual r_hat obey **r_hat=(T/C*)r**. This equivalence will be tested with an
analytic function containing nonzero time, advection and curvature terms.

For a dimensionless objective use sensor/IC/BC errors divided by C* and PDE
residual divided by C*/T **before squaring**. Thus

\[
J=10 L_{data}/C_*^2+T^2 L_{PDE}/C_*^2
  +L_{IC}/C_*^2+L_{BC}/C_*^2.
\]

The numbers (10,1,1,1) remain dimensionless preference weights; scales and
weights are separate configuration fields. A physical-coordinate network with
this residual/error scaling is mathematically equivalent to solving the fully
nondimensional PDE, with autograd handling the chain rule. Input scaling alone
is not sufficient. Physical-unit residuals remain separately reported so
candidate diagnostics remain comparable. In this particular synthetic unit
system, the relative PDE preference changes by T²=0.64 versus B0; there is no
mathematical guarantee that this will improve optimization.

## Positive and exact-constraint formulations to investigate

On x,y in [0,1], t in [0,T], let

\[
E(x,y,t)=16\,x(1-x)y(1-y)t/T.
\]

E is dimensionless, nonnegative and no greater than one. It is exactly zero
on each boundary and at the initial time. A model C=E softplus(f) with one
synthetic concentration unit as output scale therefore satisfies the required
homogeneous IC and BC exactly and is nonnegative. The PDE must differentiate
the *complete product*, not only f. A positive-only comparison removes E but
uses the same softplus transform. No projection, clipping, hidden flux
condition or true-source envelope is involved.

This formulation can suppress parameter gradients near t=0 or an edge; a
large negative f can also saturate softplus. It changes the approximation
space, and its representational/optimization consequences require tests and
measured residuals. At t=0 in the open interior, Ct=16x(1-x)y(1-y)softplus(f)/T.
Matching a nonzero Gaussian there may require a very large f near an edge.
The Gaussian forcing is not exactly zero on the boundary, so smooth classical
compatibility at the initial-boundary corner is not guaranteed even for the
underlying PDE. We enforce boundary values, not an additional PDE equation on
the edges. Residual evaluation on the edges is labeled diagnostic only.

## Completed numerical scale audit

`scripts/supplement_model_diagnosis.py` evaluates all six terms independently
on the existing 8192 fixed spacetime diagnostic points for each of ten B0 and
ten B5 checkpoints. It never fits a model or changes the selection. Physical
and dimensionless values use the same predictions and points; every term is
multiplied by T/C*=.8/.88 for the latter. Signed arrays are retained in NPZ and
mean/RMS/MAE/maxima in CSV, rather than confusing the RMS of a sum with a sum
of RMSs. The physical RMS medians are:

| Term | B0 | B5 |
|---|---:|---:|
| C_t | .089171 | .098577 |
| u C_x | .094300 | .100409 |
| v C_y | .041338 | .047418 |
| D C_xx | .015443 | .016980 |
| D C_yy | .017986 | .020118 |
| S | .113198 | .124137 |
| r | .008215 | .005684 |

The six term medians span about 7.3-fold, not a gross orders-of-magnitude unit
disparity. Their physical units agree. This does not make B0's **objective**
unit-invariant or certify its conditioning: coordinate/parameter gradients
and unit-compensating weights are separate issues. The B1 scaling-only control
does not improve recovery, so no claim that nondimensionalization alone fixes
Q underestimation is justified. The statistics were added after model selection
and are labeled descriptive post-selection evidence in the diagnosis report.
