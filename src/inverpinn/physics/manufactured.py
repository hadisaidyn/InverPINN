"""Exact smooth PDE solutions for numerical validation, not emission models.

On [0,1]^2, write C=a(t)P(x,y). Then the forcing is
S=a'(t)P+a(t)[u P_x+v P_y-D(P_xx+P_yy)]. Both supported profiles vanish on
the boundary and a(0)=0. All calculations use consistent synthetic units.
See docs/numerical_validation.md for the derivative-by-derivative derivation.
"""

import math

import torch


def spatial_terms(x, y, *, profile, u, v, D):
    """Return P and u*P_x+v*P_y-D*laplacian(P), evaluated analytically.

    sine: P=sin(pi*x)sin(pi*y), laplacian(P)=-2*pi^2*P.
    polynomial: P=x(1-x)y(1-y), laplacian(P)=-2[x(1-x)+y(1-y)].
    The latter's centered second differences are exact, not approximate.
    These functions support autograd for independent derivative checks.
    """
    if profile == "sine":
        p = torch.sin(math.pi*x) * torch.sin(math.pi*y)
        px = math.pi * torch.cos(math.pi*x) * torch.sin(math.pi*y)
        py = math.pi * torch.sin(math.pi*x) * torch.cos(math.pi*y)
        laplacian = -2*math.pi**2*p
    elif profile == "polynomial":
        p = x*(1-x)*y*(1-y)
        px, py = (1-2*x)*y*(1-y), x*(1-x)*(1-2*y)
        laplacian = -2*(x*(1-x)+y*(1-y))
    else:
        raise ValueError(f"Unknown manufactured profile: {profile}")
    return p, u*px + v*py - D*laplacian


def time_factors(t, *, profile):
    """Return a(t), a'(t): t,1 for sine; exp(t)-1,exp(t) for polynomial.

    Linear time removes the exact solution's Euler time-truncation term
    during spatial refinement. Exponential time exposes Euler's first order
    during temporal refinement. expm1 avoids cancellation near t=0.
    """
    if profile == "sine":
        return t, 1.0
    if profile == "polynomial":
        return math.expm1(t), math.exp(t)
    raise ValueError(f"Unknown manufactured profile: {profile}")


def error_norms(numerical, reference, *, dx, dy):
    """Return integral L2, Linf and relative L2, using trapezoidal weights.

    Integral L2 has concentration*length units in 2D, Linf concentration
    units; relative L2 is dimensionless. A zero reference has no relative
    error denominator, so return None rather than a fabricated zero.
    """
    if numerical.shape != reference.shape or numerical.ndim != 2:
        raise ValueError("Error norms require equal-shaped 2D fields.")
    if not torch.isfinite(numerical).all() or not torch.isfinite(reference).all():
        raise ValueError("Error norms require finite fields.")
    weights = torch.ones_like(reference)
    weights[[0, -1], :] *= 0.5
    weights[:, [0, -1]] *= 0.5
    difference = numerical-reference
    l2 = float(torch.sqrt(dx*dy*torch.sum(weights*difference**2)))
    denominator = float(torch.sqrt(dx*dy*torch.sum(weights*reference**2)))
    return {"L2_error": l2, "Linf_error": float(difference.abs().max()),
            "relative_L2_error": l2/denominator if denominator else None}
