"""Spatial emission rates, independent of transport and neural networks."""

import math

import torch


def gaussian_source(
    x: torch.Tensor,
    y: torch.Tensor,
    x_s: float,
    y_s: float,
    Q: float,
    sigma: float,
) -> torch.Tensor:
    r"""Evaluate S(x,y) = Q exp(-((x-x_s)^2 + (y-y_s)^2)/(2 sigma^2)).

    Imagine emissions drawn as a smooth hill. Its center is (x_s, y_s),
    its height is Q, and sigma controls its width. One sigma away from the
    center, the height is Q * exp(-1/2). Coordinates and sigma must use the
    same length unit, so the exponent has no units. Q and S have units of
    concentration per time, as required by the advection--diffusion equation.

    Q is the peak emission intensity, NOT the total emission rate. Over the
    entire plane the integral is 2*pi*Q*sigma**2; a finite domain truncates it.
    This is a time-independent source field, not the concentration resulting
    from transport. It imposes no initial or boundary conditions.

    x and y must be floating-point PyTorch tensors with compatible shapes,
    devices, and units. Broadcasting supports points, batches, and grids.
    Parameters are finite scalars, sigma > 0 and Q >= 0. Tensor operations
    preserve gradients with respect to the input coordinates.
    """
    if not all(math.isfinite(value) for value in (x_s, y_s, Q, sigma)):
        raise ValueError("Source parameters must be finite.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if Q < 0:
        raise ValueError("Q must be nonnegative for a pollution source.")

    # Squared distance measures how far a point is from the emission center.
    distance_squared = (x - x_s) ** 2 + (y - y_s) ** 2
    # Farther points have a more negative exponent and therefore less emission.
    return Q * torch.exp(-distance_squared / (2 * sigma**2))
