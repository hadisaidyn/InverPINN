"""Float64 analytical checks, limiting cases, and parameter-gradient tests."""

import pytest
import torch

from inverpinn.physics.autodiff import concentration_derivatives, autograd_pde_residual
from inverpinn.models.mlp import ConcentrationMLP


def points():
    return torch.tensor([[0.2, -0.3, 0.4], [0.7, 0.5, -0.2], [-0.6, 0.8, 0.9]],
                        dtype=torch.float64, requires_grad=True)


def check(actual, expected):
    for key, value in expected.items():
        assert actual[key].shape == (3, 1)
        torch.testing.assert_close(actual[key][:, 0], value, rtol=1e-12, atol=1e-12)


def test_polynomial_with_mixed_terms():
    p = points()
    x, y, t = p.unbind(1)
    C = x**3 + 2*y**3 + 3*t**2 + x*y*t
    check(concentration_derivatives(C, p), dict(C_x=3*x*x+y*t, C_y=6*y*y+x*t,
          C_t=6*t+x*y, C_xx=6*x, C_yy=12*y))


def test_trigonometric_exponential():
    p = points()
    x, y, t = p.unbind(1)
    C = torch.exp(-t)*torch.sin(2*x)*torch.cos(3*y)
    check(concentration_derivatives(C[:, None], p), dict(C_t=-C,
          C_x=2*torch.exp(-t)*torch.cos(2*x)*torch.cos(3*y),
          C_y=-3*torch.exp(-t)*torch.sin(2*x)*torch.sin(3*y), C_xx=-4*C, C_yy=-9*C))


@pytest.mark.parametrize("constant", [False, True])
def test_affine_and_constant_fields(constant):
    p = points()
    C = p[:, 0]*0+7 if constant else 2*p[:, 0]-3*p[:, 1]+4*p[:, 2]+7
    zero = torch.zeros(3, dtype=torch.float64)
    check(concentration_derivatives(C, p), dict(C_t=zero if constant else zero+4,
          C_x=zero if constant else zero+2, C_y=zero if constant else zero-3,
          C_xx=zero, C_yy=zero))


def test_manufactured_residual_and_source_sign():
    p = points()
    x, y, t = p.unbind(1)
    C = x*x+3*y*y+2*t
    source = 2+0.4*2*x-0.2*6*y-0.1*8
    r = autograd_pde_residual(C, p, u=0.4, v=-0.2, D=0.1, S=source)
    torch.testing.assert_close(r, torch.zeros_like(r), atol=1e-12, rtol=0)
    r = autograd_pde_residual(C, p, u=0.4, v=-0.2, D=0.1, S=source+1)
    torch.testing.assert_close(r, -torch.ones_like(r), atol=1e-12, rtol=0)


def test_residual_parameter_gradient_exact():
    p = points()
    a = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    D = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    C = a*(p[:, 0]**2+p[:, 1]**2+p[:, 2])
    r = autograd_pde_residual(C, p, u=0., v=0., D=D, S=0.)
    da, dD = torch.autograd.grad(r.sum(), (a, D))
    torch.testing.assert_close(da, 3*(1-4*D))
    torch.testing.assert_close(dD, -12*a)


def test_input_scaling_chain_rule():
    p = points()
    z = 2*(p-torch.tensor([0., 0., -1.])) / torch.tensor([2., 4., 5.])-1
    C = z[:, 0]**2+z[:, 1]**2+z[:, 2]
    ones = torch.ones(3, dtype=torch.float64)
    check(concentration_derivatives(C, p), dict(C_t=ones*0.4, C_x=2*z[:, 0],
          C_y=z[:, 1], C_xx=ones*2, C_yy=ones*0.5))


def test_real_mlp_supports_residual_backpropagation():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = ConcentrationMLP([0, 0, 0], [1, 1, 1], width=8, depth=2).double()
    p = points()
    r = autograd_pde_residual(model(p), p, u=0.3, v=-0.2, D=0.1, S=1.)
    r.square().mean().backward()
    # The output bias disappears under differentiation; hidden weights must not.
    assert model.network[0].weight.grad is not None
    assert torch.isfinite(model.network[0].weight.grad).all()
    assert model.network[0].weight.grad.abs().sum() > 0


def test_broken_graphs_are_rejected():
    p = points()
    with pytest.raises(ValueError, match="detached"):
        concentration_derivatives((p[:, 0]**2).detach(), p)
    with pytest.raises(ValueError, match="not connected"):
        concentration_derivatives(torch.ones(3, requires_grad=True), p)
    with pytest.raises(ValueError, match="requires_grad"):
        concentration_derivatives(p[:, 0], p.detach())
