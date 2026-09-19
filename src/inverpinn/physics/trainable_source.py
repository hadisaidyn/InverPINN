"""Constrained parameters for a single inverse Gaussian emission source."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TrainableGaussianSource(nn.Module):
    r"""S(x,y)=Q*exp(-((x-x_s)^2+(y-y_s)^2)/(2*sigma²)).

    x_s=sigmoid(raw_x), y_s=sigmoid(raw_y) remain inside [0,1]. In exact
    arithmetic endpoints are approached asymptotically; floating-point sigmoid
    may saturate at an endpoint. Q=softplus(raw_Q)+epsilon stays strictly
    positive even when softplus underflows. epsilon is the dtype's smallest
    positive normal value, not a meaningful physical lower bound. No clipping
    or post-optimizer projection is performed. All three raw values are
    nn.Parameters; sigma is a known fixed buffer in length units.

    Initial coordinates must be strictly interior to have finite logits; the
    initial Q must exceed epsilon. Q denotes peak concentration/time, not
    integrated emissions. Raw parameters and sigma are checkpointed together.
    """

    def __init__(self, x_s: float, y_s: float, Q: float, sigma: float):
        super().__init__()
        if not all(math.isfinite(v) for v in (x_s, y_s, Q, sigma)):
            raise ValueError("Source initialization must be finite.")
        if not (0 < x_s < 1 and 0 < y_s < 1 and sigma > 0 and Q > torch.finfo(torch.float32).tiny):
            raise ValueError("Require 0<x_s,y_s<1, sigma>0 and Q above float32 tiny.")
        self.raw_x = nn.Parameter(torch.tensor(math.log(x_s)-math.log1p(-x_s)))
        self.raw_y = nn.Parameter(torch.tensor(math.log(y_s)-math.log1p(-y_s)))
        q = Q-torch.finfo(torch.float32).tiny
        # Stable inverse softplus: avoids exp(Q), which overflows for large Q.
        self.raw_Q = nn.Parameter(torch.tensor(q+math.log(-math.expm1(-q))))
        self.register_buffer("sigma", torch.tensor(sigma))

    @property
    def x_s(self):
        return torch.sigmoid(self.raw_x)

    @property
    def y_s(self):
        return torch.sigmoid(self.raw_y)

    @property
    def Q(self):
        return F.softplus(self.raw_Q)+torch.finfo(self.raw_Q.dtype).tiny

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate source intensity while retaining gradients to all parameters."""
        return self.Q*torch.exp(-((x-self.x_s)**2+(y-self.y_s)**2)/(2*self.sigma**2))

    def estimates(self) -> dict[str, float]:
        """Detached physical values for logging only, never for residual evaluation."""
        return {"x_s": self.x_s.detach().item(), "y_s": self.y_s.detach().item(), "Q": self.Q.detach().item()}


class TrainableGaussianMixture(nn.Module):
    """Sum K constrained Gaussian sources, initially supporting experiments at K=2.

    Source labels are arbitrary: permuting components leaves S unchanged.
    K and widths are fixed hypotheses, while each location and strength trains.
    Distinct initial locations avoid identical gradients from exact symmetry;
    no repulsion or artificial ordering constraint is imposed. Coincident sources
    can be unidentifiable, even though the combined field is well defined.
    """

    def __init__(self, sources: list[dict]):
        super().__init__()
        if not sources:
            raise ValueError("At least one source specification is required.")
        self.sources = nn.ModuleList([TrainableGaussianSource(**s) for s in sources])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return sum(source(x, y) for source in self.sources)

    def estimates(self):
        return {"sources": [source.estimates() for source in self.sources]}
