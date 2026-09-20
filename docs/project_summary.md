# InverPINN: final project summary

## Technical summary

InverPINN evaluates sparse inverse source recovery for transient 2D advection–diffusion with one Gaussian source. The work separates numerical error, observation informativeness, physical admissibility and inverse optimization. Manufactured solutions verify first-order upwind advection, second-order diffusion and first-order Euler stepping. A 641² reference supplies 20 fixed sensors and dense evaluation under known transport and zero added noise.

PyTorch PINNs represent concentration and infer source location/amplitude. Hard constraints eliminate negative concentrations and homogeneous IC/BC violations, but not amplitude bias. Conditional-Q oracle diagnostics and classical joint inversion demonstrate accurate numerical recovery under the tested observation design. J1 analytically minimizes PDE residual over Q, leaving field/location optimization to Adam.

The central result is unsuccessful confirmation: **J1 recovered 23/30 fresh cases, below the preregistered 27/30 requirement**. Median localization, relative Q error and concentration L2 were 0.005574, 5.975% and 7.423%; Q was underestimated in 28/30. Classical inversion recovered 30/30 with medians 0.001111, 1.218% and 1.507%. Exact physical constraints did not guarantee inverse correctness. Field/source inconsistency and weak-signal associations support a formulation/optimization explanation, without isolating an exclusive causal mechanism.

The contribution is a reproducible evaluation and diagnosis—not a generally superior PINN or verified emitter detector. Independent scenarios, frozen protocols, failure retention and artifact hashes protect that distinction. The final paper rebuild reads saved evidence only. The controlled research phase is complete.

## Simple summary

We asked whether an AI that knows the physics of pollution transport could work backward from sparse sensors to find an unknown pollution source. We created simulated pollution where the source's position and strength were exactly known, and checked that the simulator became more accurate on a finer grid.

The AI often found the location accurately but tended to underestimate how strong the source was. Making its concentrations nonnegative and enforcing the right starting and edge conditions fixed obvious physical mistakes, but not every wrong estimate.

We compared it with a different approach: run the pollution equation with possible sources and find the source that best matches the sensors. That classical method recovered all 30 sources in the final test. This showed that missing sensor information was not the whole explanation for the AI's failures. It does not prove that classical methods are always better.

A mathematical improvement helped the AI, but the untouched test was decisive. We required at least 27 successes out of 30. It achieved **23**, so we reported failure rather than lowering the target. The research shows where an appealing method remains unreliable and why checking failures matters.

Almaty's pollution motivated the question, but these were simplified simulations, not identified real emitters. We make no claim about illegal sources or responsibility for actual pollution.

See the [paper](../paper/manuscript.md), [reproduction guide](reproducibility.md) and [application descriptions](application_description.md).
