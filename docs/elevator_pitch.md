# InverPINN elevator pitches

Drafts for Khadis Aidyn. Timing assumes a conversational pace. Confirm first-person
claims against actual work and acknowledge implementation assistance as appropriate.

## 15-second version

I studied whether a physics-informed neural network could locate a pollution
source from sparse sensors. It often found the location, but underestimated
strength and failed blind confirmation. A classical solver did better.

## 30-second version

InverPINN works backward from sparse pollution measurements to estimate a source's
location and strength. I studied this using controlled synthetic data, PyTorch and
an advection-diffusion equation. The final PINN often localized the source well,
but recovered only 23 of 30 cases when 27 were required. A classical inverse solver
recovered all 30. The main finding was persistent source-strength bias, preserved
as a failed hypothesis rather than presented as a solved pollution problem.

## 60-second version

Pollution sensors measure concentration after wind and diffusion have moved it.
InverPINN asks whether a physics-informed neural network can work backward to
estimate the original source. The project used one simulated Gaussian source,
known transport and 20 fixed sensors, so the source truth was available for
evaluation. Numerical convergence checks preceded inverse experiments, and the
final methods and success criteria were frozen before new blind scenarios.

The PINN often found the location accurately, but underestimated source strength.
An analytical amplitude calculation improved it without solving reliability:
the final model recovered 23 of 30 cases, below the required 27. Classical
inversion recovered all 30 from identical observations. This weakens insufficient
sensor information as a complete explanation in the tested setting. The project
retained every failure and stopped model development. It does not identify real
Almaty emitters or establish performance with real sensor noise.
