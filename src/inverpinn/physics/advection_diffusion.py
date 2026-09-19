r"""Mathematical terms for transient two-dimensional advection--diffusion.

This module contains physics only: it receives concentration derivatives and
physical parameters, then evaluates the governing equation.  It deliberately
does not create a neural network or compute derivatives from coordinates.

The equation is

.. math::

    \frac{\partial C}{\partial t}
    + u\frac{\partial C}{\partial x}
    + v\frac{\partial C}{\partial y}
    = D\left(
        \frac{\partial^2 C}{\partial x^2}
        + \frac{\partial^2 C}{\partial y^2}
      \right) + S(x, y, t).

Here ``C(x, y, t)`` is pollutant concentration, ``u`` and ``v`` are wind
speeds in the x and y directions, ``D`` is the diffusion coefficient, and
``S`` is the pollution source rate.

For a concrete SI-unit convention, if concentration has units ``[C]``, then
``x`` and ``y`` have units metres ``[m]`` and ``t`` has seconds ``[s]``:

* ``u`` and ``v`` have units ``[m / s]``;
* ``D`` has units ``[m^2 / s]``;
* ``S`` has units ``[C / s]``.

Consequently every term in the equation has units ``[C / s]``.  No boundary
or initial condition is imposed here; those belong to a later, tested stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


ScalarOrTensor = float | torch.Tensor


@dataclass(frozen=True)
class AdvectionDiffusionTerms:
    """The individual rates that make up the advection--diffusion equation.

    Each field has units of concentration per unit time.  Keeping the terms
    visible makes it easy to inspect which physical effect contributes to a
    residual without mixing this physics with neural-network code.
    """

    concentration_time_rate: torch.Tensor
    x_advection_rate: torch.Tensor
    y_advection_rate: torch.Tensor
    diffusion_rate: torch.Tensor
    source_rate: torch.Tensor


def advection_diffusion_terms(
    concentration_t: torch.Tensor,
    concentration_x: torch.Tensor,
    concentration_y: torch.Tensor,
    concentration_xx: torch.Tensor,
    concentration_yy: torch.Tensor,
    wind_x: ScalarOrTensor,
    wind_y: ScalarOrTensor,
    diffusion: ScalarOrTensor,
    source: ScalarOrTensor,
) -> AdvectionDiffusionTerms:
    r"""Evaluate every term in the two-dimensional transient PDE.

    Think of a small square of air. ``concentration_time_rate`` says how fast
    its pollution amount is changing. Wind moves pollution sideways, which is
    represented by the two advection rates. Diffusion is the natural spreading
    from crowded places toward less crowded places. The source rate adds new
    pollution, such as from a chimney.

    Args:
        concentration_t: :math:`\partial C / \partial t`, with units ``[C/s]``.
        concentration_x: :math:`\partial C / \partial x`, with units ``[C/m]``.
        concentration_y: :math:`\partial C / \partial y`, with units ``[C/m]``.
        concentration_xx: :math:`\partial^2 C / \partial x^2`, units ``[C/m^2]``.
        concentration_yy: :math:`\partial^2 C / \partial y^2`, units ``[C/m^2]``.
        wind_x: x-direction wind speed :math:`u`, units ``[m/s]``.
        wind_y: y-direction wind speed :math:`v`, units ``[m/s]``.
        diffusion: diffusion coefficient :math:`D`, units ``[m^2/s]``.
        source: source rate :math:`S(x, y, t)`, units ``[C/s]``.

    Returns:
        The five rates used in the governing equation. Inputs follow PyTorch
        broadcasting rules, so scalar physical parameters may be shared across
        a batch of derivative values.
    """
    # A wind speed multiplied by a spatial slope gives a concentration change
    # per second: [m/s] * [C/m] = [C/s].
    x_advection_rate = wind_x * concentration_x
    y_advection_rate = wind_y * concentration_y

    # The two curvatures form the Laplacian. Diffusion multiplies that shape:
    # [m^2/s] * [C/m^2] = [C/s].
    diffusion_rate = diffusion * (concentration_xx + concentration_yy)

    # ``source`` can be a number or a tensor; adding zero makes a numeric
    # scalar use the same tensor dtype/device as the other PDE terms.
    source_rate = concentration_t * 0 + source

    return AdvectionDiffusionTerms(
        concentration_time_rate=concentration_t,
        x_advection_rate=x_advection_rate,
        y_advection_rate=y_advection_rate,
        diffusion_rate=diffusion_rate,
        source_rate=source_rate,
    )


def advection_diffusion_residual(
    concentration_t: torch.Tensor,
    concentration_x: torch.Tensor,
    concentration_y: torch.Tensor,
    concentration_xx: torch.Tensor,
    concentration_yy: torch.Tensor,
    wind_x: ScalarOrTensor,
    wind_y: ScalarOrTensor,
    diffusion: ScalarOrTensor,
    source: ScalarOrTensor,
) -> torch.Tensor:
    r"""Return the residual of the transient 2D advection--diffusion equation.

    The residual is the left-hand side minus the right-hand side:

    .. math::

        r = C_t + uC_x + vC_y - D(C_{xx} + C_{yy}) - S.

    A physically exact solution has ``r = 0``.  In a future PINN stage, an
    optimizer can reduce this value while a neural network supplies the
    derivatives.  This function itself is only the calculator for the physics.
    Every returned value has units ``[C/s]``.
    """
    terms = advection_diffusion_terms(
        concentration_t=concentration_t,
        concentration_x=concentration_x,
        concentration_y=concentration_y,
        concentration_xx=concentration_xx,
        concentration_yy=concentration_yy,
        wind_x=wind_x,
        wind_y=wind_y,
        diffusion=diffusion,
        source=source,
    )

    return (
        terms.concentration_time_rate
        + terms.x_advection_rate
        + terms.y_advection_rate
        - terms.diffusion_rate
        - terms.source_rate
    )
