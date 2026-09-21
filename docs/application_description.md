# InverPINN: honest application descriptions

Word counts use whitespace-separated words; hyphenated expressions count as one. These are project descriptions, not assertions of publication, awards, affiliation or sole authorship. Personal contributions should be stated only if accurate.

Public author: **Khadis Aidyn**, confirmed on 2026-09-21. First-person answers are
drafts for personal review before submission. The project used AI-assisted coding
and documentation; do not claim sole implementation or derivations that you did
not personally perform or verify.

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

## Interview: Tell me about your research project

“My question was whether a neural network constrained by pollution physics could work backward from a few sensors to find a source. The surprising result was that it often found the location but underestimated the source's strength. I checked the simulator, enforced physical conditions, and compared the network with a classical inverse solver. That solver recovered every source in the final controlled test, so insufficient sensor information was not the whole explanation. A mathematical improvement helped the network, but it still missed our predefined threshold: 23 successes instead of 27 out of 30. Reporting that failure was central to the project. It taught me to distinguish an attractive model from evidence that it is dependable. We did not identify actual emitters in Almaty.”

Adapt first-person statements to the actual work and acknowledge collaborators/tools appropriately. [Technical evidence](project_summary.md) accompanies these versions.

## Common App activity (150-character maximum)

Studied pollution-source inversion with PyTorch/PDEs; tested blind recovery and reported PINN bias: 23/30 cases vs. classical 30/30.

## Longer internal activity description

InverPINN investigated inverse advection-diffusion with PyTorch, automatic
differentiation, numerical validation and independent synthetic benchmarks.
It compared neural and classical source recovery, diagnosed persistent amplitude
bias and retained a failed final confirmation: J1 recovered 23/30 when 27/30 were
required, versus classical recovery of 30/30. This is a project-level description;
identify the applicant's actual implementation and analysis contributions separately.

## 250-word description

InverPINN explored whether a physics-informed neural network could infer the location and strength of a pollution source from sparse concentration observations. The setting was controlled: one Gaussian source, known wind and diffusion, twenty fixed sensors, and no added measurement noise. Source parameters were known for evaluation, while concentration references came from a numerically checked finite-difference solver.

The project separated mathematical correctness, physical admissibility, and inverse reliability. PyTorch automatic differentiation supplied equation derivatives. Analytical functions and grid refinement checked derivatives and simulation quality. Hard constraints eliminated negative concentrations and enforced zero initial and boundary conditions. Separate development and blind scenario sets prevented evaluated cases from becoming fresh validation data.

Despite those controls, the PINN underestimated source strength. Observability diagnostics and classical inversion tested whether the observations were insufficient. Classical inversion recovered all thirty final sources using identical observations. An analytical variable-projection method, J1, also removed source amplitude from the neural optimizer parameters.

J1 improved recovery but failed final confirmation: twenty-three of thirty cases recovered jointly, below the preregistered requirement of twenty-seven. Its median localization error was 0.005574, relative strength error 5.975%, and concentration L2 error 7.423%. Twenty-eight strength estimates remained below truth.

The scientific contribution is a documented recovery gap and a preserved failed hypothesis. Physical consistency and median errors did not establish dependable source inference. All failures remained in the report, and development stopped after confirmation failed. The conclusions concern this synthetic setting, not real Almaty emitters, noisy sensors, uncertain meteorology, or universal superiority of one inverse method over another.

## Interview: What did you learn from a result that did not work?

I learned why a success criterion must be fixed before seeing the final result.
J1's median errors looked promising, and it enforced physical constraints exactly.
But it recovered only 23 of 30 cases when 27 were required. Changing the threshold
or trying new settings on those same cases would have changed the scientific
question. The appropriate response was to preserve the failure, compare plausible
explanations and describe what the evidence actually supported. A failed hypothesis
can be informative without becoming a successful model.

## Interview: What was technically hardest?

The difficult technical distinction was between numerical-reference error,
observation information and inverse-method failure. A stable simulator can still
be inaccurate, and a physically admissible neural field can still imply a wrong
source. The project checked derivatives and refinement, then used conditional-Q
diagnostics and a classical inverse solver on identical observations. Those
comparisons helped isolate the recovery gap. I would explain the particular
checks I personally performed rather than equating the repository's capabilities
with my own unaided work.

## Interview: What did you personally do?

I defined the research question and the staged validation requirements, including
synthetic ground truth, frozen benchmarks and explicit reporting of failed results.
The project used AI-assisted implementation and documentation. My account should
distinguish code I wrote, code produced with assistance, derivations I checked and
analyses I personally interpreted. I would name those specific contributions
rather than claim sole implementation of the entire repository.

Before using this answer, replace broad contribution descriptions with concrete,
truthful examples you can explain. No personal technical contribution beyond the
documented project direction has been independently verified in this delivery.
