# InverPINN Engineering Rules

InverPINN is an incremental research project for solving the inverse two-dimensional advection--diffusion problem with physics-informed neural networks (PINNs). Its eventual objective is estimating unknown PM2.5 sources from sparse Almaty observations and ERA5 meteorology.

## Non-negotiable rules

1. Build the project incrementally; do not implement the full system in one stage.
2. Every stage must add automated tests appropriate to its scope.
3. Begin with synthetic data whose ground truth is exactly known. Do not use real data until the synthetic pipeline is validated.
4. Use PyTorch for all neural-network and tensor computations.
5. Keep physics equations and residual definitions separate from neural-network code.
6. Make experiments reproducible: use explicit random seeds and version-controlled configuration files.
7. Write clear docstrings that explain the mathematics, including assumptions, variables, units, and boundary/initial conditions where applicable.
8. Never silently change a failing test to make it pass. State and investigate the mathematical or modelling reason for failure before changing either implementation or expectations.
9. Prefer small, direct implementations over unnecessary abstractions.
10. Every experiment must save its plots, metrics, checkpoints, and the exact resolved configuration that produced them.

## Initial scope

This repository starts as a scaffold only. A PINN, PDE solver, synthetic-data generator, and real-data ingestion are intentionally out of scope until a later, separately tested stage.
