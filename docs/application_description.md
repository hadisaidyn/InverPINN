# InverPINN: honest application descriptions

Word counts use whitespace-separated words; hyphenated expressions count as one. These are project descriptions, not assertions of publication, awards, affiliation or sole authorship. Personal contributions should be stated only if accurate.

## 50-word description

InverPINN investigated whether physics-informed neural networks could recover pollution sources from sparse synthetic sensors. Using PyTorch, differential equations, numerical validation, and blind testing, the project diagnosed persistent source-strength bias. Its final model recovered 23 of 30 cases, missing the predefined threshold, while classical inversion recovered all 30 under controlled assumptions.

## 100-word description

InverPINN explored an inverse problem: recovering an unknown pollution source from sparse concentration measurements. The project combined PyTorch automatic differentiation, a validated advection-diffusion simulator, reproducible experiments, and independent blind testing. Physical constraints eliminated negative concentrations, but source strength remained systematically underestimated. Analytical amplitude profiling improved recovery without solving the reliability problem. On 30 fresh synthetic cases, the final PINN recovered 23, below the preregistered requirement of 27; classical inversion recovered all 30. The contribution is an evidence-based diagnosis of a scientific machine-learning limitation, not a claim of real emitter identification. Failed results were preserved rather than tuned away after evaluation.

## 150-word description

InverPINN studied whether physics-informed neural networks can infer the location and strength of a pollution source from sparse sensor measurements. Motivated by air-quality questions in Almaty, the research began with controlled synthetic data, known transport physics, and exactly specified source parameters. It used PyTorch automatic differentiation, finite-difference simulation, mathematical source profiling, automated tests, and reproducible configurations. Numerical convergence checks preceded inverse experiments, and separate development and blind scenarios prevented test outcomes from guiding model selection. Exact physical constraints removed negative concentrations but did not guarantee correct source recovery. The final method recovered 23 of 30 fresh scenarios, below the preregistered threshold of 27, while classical inversion recovered 30. Persistent source-strength underestimation revealed a gap between physical consistency and reliable inference. The project preserved failures, compared alternative explanations, and reported limitations openly. Its contribution is rigorous evaluation and scientific integrity, not verified real-world pollution-source discovery or universal superiority over classical methods.

## Activities-list version

Studied pollution-source inversion with PyTorch/PDEs; validated simulations and blind tests; reported PINN bias and a failed reliability threshold.

## Resume bullet

- Developed and evaluated a reproducible PyTorch/PDE source-inversion study; validated numerical convergence, diagnosed source-strength bias, and documented 23/30 blind recoveries versus 30/30 for classical inversion, preserving the failed 27/30 acceptance criterion.

Use “developed” only to the extent it describes the applicant's actual role; otherwise name the verified contribution.

## Interview explanation

“My question was whether a neural network constrained by pollution physics could work backward from a few sensors to find a source. The surprising result was that it often found the location but underestimated the source's strength. I checked the simulator, enforced physical conditions, and compared the network with a classical inverse solver. That solver recovered every source in the final controlled test, so insufficient sensor information was not the whole explanation. A mathematical improvement helped the network, but it still missed our predefined threshold: 23 successes instead of 27 out of 30. Reporting that failure was central to the project. It taught me to distinguish an attractive model from evidence that it is dependable. We did not identify actual emitters in Almaty.”

Adapt first-person statements to the actual work and acknowledge collaborators/tools appropriately. [Technical evidence](project_summary.md) accompanies these versions.
