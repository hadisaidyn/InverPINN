# Project website copy

Copy only; no website has been created or published.

## Hero headline

InverPINN: when physical consistency is not enough for source recovery

## Subheadline

Khadis Aidyn's controlled study of source-strength bias in physics-informed
neural networks for sparse pollution-source inversion.

## Problem

Sensors observe pollution after transport. Recovering where it originated and
how strong the source was requires solving an inverse problem, not merely
interpolating concentrations between sensors.

## Method

A PyTorch PINN represents concentration and combines sparse observations with
the advection-diffusion equation. Numerical reference checks, separate development
and blind scenarios, exact physical constraints and classical inversion test its
reliability. J1 analytically profiles source amplitude from its PDE residual.

## Main result

**J1 failed final confirmation: 23/30 recoveries (76.67%), below the preregistered
27/30 requirement (90%). Classical inversion recovered 30/30.**

J1 median localization error was 0.005574, relative Q error 5.975% and concentration
L2 error 7.423%. Classical medians were 0.001111, 1.218% and 1.507%. J1 underestimated
Q in 28 cases and overestimated it in two, despite exact positivity, IC and BC.

## Why the result matters

Better median errors and physical admissibility did not establish reliable inverse
recovery. Classical recovery on identical observations weakens a purely
insufficient-information explanation under these controlled assumptions.

## Limitations

One Gaussian, known wind and diffusion, one sensor layout and zero added noise.
No validated real Almaty emitter attribution, noise robustness, uncertain-weather
robustness or general ranking of all PINN and classical methods follows.

## Paper

[Formal manuscript PDF](../paper/InverPINN_Paper.pdf) and
[report-only reproduction](reproducibility.md).

## GitHub

The repository is prepared locally. Add its verified public URL only after
publication; no repository address is currently available. No public contact
information was supplied, so no contact block is included.
