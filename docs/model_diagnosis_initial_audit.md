# B0 audit completed before candidate implementation/results

All ten original development fits were rerun, with identical final metrics to
revision 3 (zero difference in localization, relative Q, relative L2,
negativity, and PDE/BC/IC RMS). The observer-invariance test also compares
parameters and trajectories bit-for-bit. The original sparse-only fitting API
is preserved; the audit uses a separate adapter to the lower-level hook.

Observed B0 means: localization 0.04846350, relative Q 0.34239444, relative L2
0.37716990, negative fraction 0.37060962, PDE RMS 0.00730588, BC RMS 0.00252655,
IC RMS 0.00302838. Medians are 0.03645983, 0.32246124 and 0.37049153 for the
three reconstruction/parameter criteria respectively.

At all 110 sampled early checkpoints (updates 1..10 and 100 across ten cases),
the signed physical PDE dL/dQ was positive (mean 0.01728261, range
0.01137431..0.02768402). Gradient descent therefore initially pushes Q down.
Data/IC/BC have exactly zero *direct* source derivatives by construction.
Later Q can increase through the field/PDE coupling. At update 6000 the mean
physical dL/dQ is -0.00016522, not a persistent universal downward gradient.

Softplus Jacobians across sampled checkpoints range 0.140016..0.709266:
attenuation is real but there is no observed numerical saturation or
disconnected source graph. In the final 1000 updates the median weighted
network gradient norms are data 0.064557, PDE 0.005977, BC 0.008452, IC
0.006863. Data/PDE gradient cosines are negative at 47.86% of sampled points.
No single claim that the PDE always dominates (or always becomes irrelevant)
is justified. Gradient norms also depend on parameterization and are not
independent causal effect sizes.

With 512 uniform points, the expected count inside radius sigma=.08, for an
interior center, is 512*pi*.08²=10.294; inside 2 sigma it is 41.177. Audited
coverage is close to these values, with no empty sigma neighborhoods. The
residual maps may still show concentrated residual error; this does not alone
establish insufficient sampling. Therefore no adaptive collocation is added.

Hypotheses to test, not established causes: positive-source/field tradeoff,
weak physical constraints, parameterization attenuation, source initialization,
and incomplete optimization. A fixed ablation ladder, two extra Q starts, an
exponential-Q control, one longer Adam budget, and four off-center location
starts will each use all ten development scenarios. No outcome-based early
stopping or per-scenario truth-based initialization is allowed. L-BFGS is not
included: the minimal controlled convergence comparison doubles the identical
stochastic Adam trajectory; implementing a deterministic full-batch L-BFGS
stage would also change the sampling/objective and obscure that comparison.
