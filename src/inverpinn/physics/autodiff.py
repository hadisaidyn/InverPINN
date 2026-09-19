"""Coordinate derivatives for pointwise concentration functions using autograd."""

import torch

from inverpinn.physics.advection_diffusion import advection_diffusion_residual


def concentration_derivatives(C: torch.Tensor, coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
    r"""Calculate C_t, C_x, C_y, C_xx, C_yy as columns of shape [N,1].

    coordinates is the exact floating-point [N,3] tensor of physical [x,y,t]
    used to compute C, and must have requires_grad=True BEFORE evaluation.
    C must have shape [N] or [N,1] and stay attached to that computation graph.
    Do not detach C, reconstruct coordinates, or evaluate under no_grad().
    For an explicitly constant field use C=coordinates[:, :1]*0 + constant;
    a disconnected output is rejected because it could indicate a broken graph.

    Each output row must depend ONLY on the corresponding coordinate row.
    The vector-Jacobian product with ones then gives each point's derivatives.
    Ordinary pointwise MLPs satisfy this assumption; cross-sample attention,
    batch reductions, or training-mode batch normalization do not. Independence
    cannot be inferred from these tensors and is the caller's responsibility.

    create_graph=True retains the graph for second derivatives and later
    differentiation with respect to network parameters. Affine functions have
    exactly zero curvature; explicit zero graph connections handle that case.
    Use smooth activations (e.g. tanh); ReLU has zero second derivative almost
    everywhere and is unsuitable for representing a smooth diffusion residual.

    A derivative is the slope of concentration in a direction. Differentiating
    a spatial slope again measures curvature, which drives diffusion. With
    physical coordinates, units are C/time, C/length, and C/length². If scaling
    occurs inside the network, autograd includes its chain-rule factors.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] == 0:
        raise ValueError("coordinates must have shape [N,3], N > 0, ordered [x,y,t].")
    if not coordinates.is_floating_point() or not coordinates.requires_grad:
        raise ValueError("coordinates must be floating point with requires_grad=True before evaluation.")
    if C.shape not in ((len(coordinates),), (len(coordinates), 1)):
        raise ValueError("C must have shape [N] or [N,1].")
    if not C.requires_grad:
        raise ValueError("C is detached; evaluate the field with autograd enabled.")
    first = torch.autograd.grad(C, coordinates, torch.ones_like(C), create_graph=True, allow_unused=True)[0]
    if first is None:
        raise ValueError("C is not connected to the supplied coordinates.")
    # Zero-valued connections preserve differentiation for constant slopes.
    first = first + coordinates * 0
    second_x = torch.autograd.grad(first[:, 0], coordinates, torch.ones_like(first[:, 0]), create_graph=True)[0]
    second_y = torch.autograd.grad(first[:, 1], coordinates, torch.ones_like(first[:, 1]), create_graph=True)[0]
    return {"C_t": first[:, 2:3], "C_x": first[:, 0:1], "C_y": first[:, 1:2],
            "C_xx": second_x[:, 0:1] + coordinates[:, :1]*0,
            "C_yy": second_y[:, 1:2] + coordinates[:, :1]*0}


def autograd_pde_residual(C: torch.Tensor, coordinates: torch.Tensor, *, u, v, D, S) -> torch.Tensor:
    r"""Return r=C_t+u*C_x+v*C_y-D*(C_xx+C_yy)-S with shape [N,1].

    This is left minus right for C_t+u*C_x+v*C_y=D*(C_xx+C_yy)+S.
    Parameters may be scalars or per-point tensors [N] / [N,1]; columns
    prevent accidental [N,N] broadcasting. Tensor parameter gradients remain
    attached. S is the evaluated source rate, not a callable. D is spatially
    constant for this PDE; variable D would require a different diffusion law.
    No training, loss reduction, or boundary/initial conditions are applied.
    """
    derivatives = concentration_derivatives(C, coordinates)

    def column(value):
        if not isinstance(value, torch.Tensor):
            return value
        if value.ndim == 0:
            return value
        if value.shape in ((len(coordinates),), (len(coordinates), 1)):
            return value.reshape(-1, 1)
        raise ValueError("PDE parameters must be scalars or tensors of shape [N] or [N,1].")

    return advection_diffusion_residual(
        derivatives["C_t"], derivatives["C_x"], derivatives["C_y"],
        derivatives["C_xx"], derivatives["C_yy"], column(u), column(v), column(D), column(S))
