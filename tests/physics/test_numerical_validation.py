"""Independent analytic checks of PDE signs, accuracy, and storage equivalence."""

import math

import pytest
import torch

from inverpinn.physics.finite_difference import solve_advection_diffusion, stability_number
from inverpinn.physics.manufactured import error_norms, spatial_terms, time_factors


def mesh(n):
    axis = torch.linspace(0, 1, n, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return x, y


def manufactured_run(n, dt, steps, profile="sine", u=0.35, v=-0.15, D=0.005):
    x, y = mesh(n)
    p, transport = spatial_terms(x, y, profile=profile, u=u, v=v, D=D)

    def forcing(_x, _y, t):
        a, ap = time_factors(t, profile=profile)
        return ap*p+a*transport

    numerical = solve_advection_diffusion(nx=n, ny=n, dt=dt, steps=steps, u=u, v=v, D=D,
        forcing=forcing, return_history=False)
    exact = time_factors(dt*steps, profile=profile)[0]*p
    return error_norms(numerical, exact, dx=1/(n-1), dy=1/(n-1)), numerical


@pytest.mark.parametrize("profile", ["sine", "polynomial"])
@pytest.mark.parametrize("u,v", [(0.35, -0.15), (-0.35, 0.15)])
def test_manufactured_forcing_matches_independent_exact_derivatives(profile, u, v):
    """Compute C directly, then autograd all derivatives independently of S."""
    x = torch.tensor([0.13, 0.31, 0.63, 0.87], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.21, 0.79, 0.48, 0.33], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.11, 0.29, 0.51, 0.73], dtype=torch.float64, requires_grad=True)
    if profile == "sine":
        c = t*torch.sin(math.pi*x)*torch.sin(math.pi*y)
    else:
        c = torch.expm1(t)*x*(1-x)*y*(1-y)
    ct, cx, cy = torch.autograd.grad(c.sum(), (t, x, y), create_graph=True)
    cxx = torch.autograd.grad(cx.sum(), x, retain_graph=True)[0]
    cyy = torch.autograd.grad(cy.sum(), y)[0]
    p, transport = spatial_terms(x, y, profile=profile, u=u, v=v, D=0.005)
    a, ap = (t, 1) if profile == "sine" else (torch.expm1(t), torch.exp(t))
    supplied = ap*p+a*transport
    torch.testing.assert_close(supplied, ct+u*cx+v*cy-0.005*(cxx+cyy), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("u,v", [(0.35, -0.15), (-0.35, 0.15)])
def test_mixed_transport_is_accurate_and_first_order_under_refinement(u, v):
    errors = [manufactured_run(n, 0.004/factor**2, 50*factor**2, u=u, v=v)[0]
              for n, factor in [(21, 1), (41, 2), (81, 4)]]
    assert errors[-1]["relative_L2_error"] < 0.01
    for coarse, fine in zip(errors, errors[1:]):
        assert fine["Linf_error"] < coarse["Linf_error"]
        order = math.log2(coarse["L2_error"]/fine["L2_error"])
        assert 0.8 < order < 1.2


def test_diffusion_spatial_convergence_is_second_order():
    errors = [manufactured_run(n, 0.004/factor**2, 50*factor**2, u=0, v=0)[0]
              for n, factor in [(21, 1), (41, 2), (81, 4)]]
    for coarse, fine in zip(errors, errors[1:]):
        assert 1.8 < math.log2(coarse["L2_error"]/fine["L2_error"]) < 2.2


def test_polynomial_spatial_diffusion_is_exact():
    x, y = mesh(41)
    p, a = spatial_terms(x, y, profile="polynomial", u=0, v=0, D=0.005)
    lap = (p[1:-1, :-2]+p[1:-1, 2:]+p[:-2, 1:-1]+p[2:, 1:-1]-4*p[1:-1, 1:-1])*40**2
    torch.testing.assert_close(-0.005*lap, a[1:-1, 1:-1], atol=1e-14, rtol=1e-11)


def test_temporal_error_decreases_at_first_order_without_spatial_error_floor():
    errors = [manufactured_run(41, 0.008/factor, 50*factor, profile="polynomial", u=0, v=0)[0]
              for factor in [1, 2, 4, 8]]
    for coarse, fine in zip(errors, errors[1:]):
        assert 0.9 < math.log2(coarse["L2_error"]/fine["L2_error"]) < 1.1
        assert fine["Linf_error"] < coarse["Linf_error"]


@pytest.mark.parametrize("steps", [0, 7])
def test_final_only_matches_full_history_and_preserves_initial_and_boundaries(steps):
    options = dict(nx=13, ny=17, dt=0.002, steps=steps, u=-0.35, v=0.15, D=0.005,
        sources=[dict(x_s=0.3, y_s=0.7, Q=1, sigma=0.08)], forcing=lambda x, y, t: -t*torch.ones_like(x))
    history = solve_advection_diffusion(**options)
    final = solve_advection_diffusion(**options, return_history=False)
    torch.testing.assert_close(final, history[-1], atol=0, rtol=0)
    assert torch.count_nonzero(history[0]) == 0
    assert torch.count_nonzero(history[:, [0, -1], :]) == 0
    assert torch.count_nonzero(history[:, :, [0, -1]]) == 0


def test_forcing_is_additive_and_uses_old_euler_time():
    options = dict(nx=11, ny=11, dt=0.05, steps=5, u=0, v=0, D=0)
    source = dict(x_s=0.3, y_s=0.7, Q=1, sigma=0.08)
    gaussian = solve_advection_diffusion(**options, sources=[source])
    combined = solve_advection_diffusion(**options, sources=[source], forcing=lambda x, y, t: torch.full_like(x, t))
    # Left endpoint sum dt * (0 + dt + ... + (n-1)dt), not right endpoint.
    n = torch.arange(6, dtype=torch.float64)
    expected = 0.05**2*n*(n-1)/2
    torch.testing.assert_close((combined-gaussian)[:, 1:-1, 1:-1], expected[:, None, None].expand(6, 9, 9))


@pytest.mark.parametrize("bad", [lambda x, y, t: x[:1], lambda x, y, t: x*float("nan")])
def test_bad_forcing_fails_loudly(bad):
    with pytest.raises(ValueError, match="finite.*tensor"):
        solve_advection_diffusion(nx=11, ny=11, dt=0.01, steps=1, u=0, v=0, D=0, forcing=bad)


def test_unstable_forcing_run_is_rejected_before_callback():
    def forbidden(x, y, t):
        pytest.fail("Unstable run must fail before evaluating forcing")
    with pytest.raises(ValueError, match="Unstable timestep"):
        solve_advection_diffusion(nx=81, ny=81, dt=0.02, steps=1, u=0.35, v=-0.15,
                                  D=0.005, forcing=forbidden, return_history=False)
    assert stability_number(dx=1/80, dy=1/80, dt=0.002, u=0.35, v=-0.15, D=0.005) == pytest.approx(0.336)


def test_error_norms_are_integrals_not_grid_dependent_vector_norms():
    for n in [11, 41]:
        reference = torch.ones((n, n), dtype=torch.float64)
        norms = error_norms(3*reference, reference, dx=1/(n-1), dy=1/(n-1))
        assert norms == pytest.approx(dict(L2_error=2, Linf_error=2, relative_L2_error=2))
    assert error_norms(reference, reference*0, dx=1/40, dy=1/40)["relative_L2_error"] is None


@pytest.mark.parametrize("profile", ["sine", "polynomial"])
def test_manufactured_initial_and_boundary_values(profile):
    x, y = mesh(41)
    p, _ = spatial_terms(x, y, profile=profile, u=0.35, v=-0.15, D=0.005)
    assert torch.count_nonzero(time_factors(0, profile=profile)[0]*p) == 0
    for boundary in [p[0], p[-1], p[:, 0], p[:, -1]]:
        torch.testing.assert_close(boundary, torch.zeros_like(boundary), atol=1e-15, rtol=0)
