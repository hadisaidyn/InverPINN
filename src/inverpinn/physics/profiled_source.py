"""Truth-independent amplitude elimination for a Gaussian PDE source."""

import torch

from inverpinn.physics.diagnostic_source import DiagnosticGaussianSource


def profile_Q(R0, G, weights=None):
    r"""Minimize sum w(R0-QG)^2 for Q>=dtype.tiny, with no amplitude prior.

    R0 has concentration/time units, G is dimensionless and Q is a peak
    intensity, not integrated mass. Uniform collocation gives w=1. Fixed
    global loss factors cancel. A zero source column makes Q unidentifiable
    and is rejected rather than regularized. Float64 sums reduce cancellation;
    the returned scalar retains gradients to R0/G and uses R0's dtype.
    The tiny floor represents the infimum of the open Q>0 domain numerically.
    """
    if R0.shape != G.shape or not R0.numel() or not R0.is_floating_point() or R0.dtype != G.dtype:
        raise ValueError("R0 and G must be nonempty matching floating tensors.")
    w = torch.ones_like(G) if weights is None else weights
    if w.shape != G.shape or not torch.isfinite(w).all() or (w < 0).any() or not (w > 0).any():
        raise ValueError("Weights must match, be finite/nonnegative and not all zero.")
    if not torch.isfinite(R0).all() or not torch.isfinite(G).all() or (G < 0).any():
        raise FloatingPointError("Nonfinite residual or invalid Gaussian column.")
    r, g, w = R0.double(), G.double(), w.double()
    H = (w*g*g).sum()
    if not torch.isfinite(H) or H <= 0:
        raise ValueError("Zero/nonfinite Gaussian energy: amplitude unidentified.")
    q = ((w*g*r).sum()/H).clamp_min(torch.finfo(R0.dtype).tiny).to(R0.dtype)
    if not torch.isfinite(q):
        raise FloatingPointError("Nonfinite amplitude profile.")
    return q


class ProfiledGaussianSource(DiagnosticGaussianSource):
    """Sigmoid x/y parameters, fixed sigma, and a non-optimizer Q buffer."""

    def __init__(self, x_s, y_s, Q, sigma):
        super().__init__(x_s, y_s, Q, sigma)
        del self.raw_Q
        self.register_buffer("profiled_Q", torch.tensor(Q))

    @property
    def Q(self):
        return self.profiled_Q

    def unit(self, x, y):
        return torch.exp(-((x-self.x_s)**2+(y-self.y_s)**2)/(2*self.sigma**2))

    def set_Q(self, value):
        """Persist a computed profile, not a truth value or optimizer step."""
        if not torch.isfinite(value).all() or value.numel()!=1 or value <= 0:
            raise ValueError("Profiled Q must be positive finite scalar.")
        with torch.no_grad():
            self.profiled_Q.copy_(value.detach())
