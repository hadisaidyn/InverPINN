"""Exact limiting cases and transport properties for the forward solver."""
import pytest
import torch
from inverpinn.physics.finite_difference import solve_advection_diffusion, stability_number
from inverpinn.physics.sources import gaussian_source

SOURCE = dict(x_s=0.5, y_s=0.5, Q=1.0, sigma=0.08)


def run(**kwargs):
    """Use a small grid and stable baseline timestep."""
    options = dict(nx=21, ny=21, dt=0.005, steps=20, u=0.0, v=0.0, D=0.0)
    return solve_advection_diffusion(**(options | kwargs))


def test_no_emissions_stays_zero():
    assert torch.count_nonzero(run(u=0.3, v=-0.2, D=0.01)) == 0


def test_no_transport_exact_solution_and_boundaries():
    """With no transport, the interior solution is exactly C=t*S."""
    history = run(ny=17, sources=[SOURCE])
    assert history.shape == (21, 17, 21)
    y, x = torch.meshgrid(torch.linspace(0, 1, 17, dtype=torch.float64),
                          torch.linspace(0, 1, 21, dtype=torch.float64), indexing="ij")
    source = gaussian_source(x, y, **SOURCE)
    t = torch.arange(21, dtype=torch.float64) * 0.005
    torch.testing.assert_close(history[:, 1:-1, 1:-1], t[:, None, None] * source[1:-1, 1:-1])
    assert torch.count_nonzero(history[0]) == 0
    assert torch.count_nonzero(history[:, [0, -1], :]) == 0
    assert torch.count_nonzero(history[:, :, [0, -1]]) == 0


def test_diffusion_spreads_symmetrically():
    diffused = run(D=0.01, sources=[SOURCE])
    local = run(sources=[SOURCE])
    torch.testing.assert_close(diffused, diffused.flip(1))
    torch.testing.assert_close(diffused, diffused.flip(2))
    assert diffused[-1, 10, 10] < local[-1, 10, 10]
    assert diffused[-1, 10, 14] > local[-1, 10, 14]
    assert diffused.min() >= 0


@pytest.mark.parametrize("u,v", [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)])
def test_wind_moves_centroid_downstream(u, v):
    history = run(u=u, v=v, sources=[SOURCE])
    axis = torch.linspace(0, 1, 21, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coordinate = x if u else y
    centroid = (history[-1] * coordinate).sum() / history[-1].sum()
    assert (centroid - 0.5) * (u + v) > 0


def test_multiple_sources_superpose():
    a, b = SOURCE | {"x_s": 0.3}, SOURCE | {"y_s": 0.7, "Q": 2}
    opts = dict(u=-0.2, v=0.1, D=0.01)
    torch.testing.assert_close(run(sources=[a, b], **opts),
                               run(sources=[a], **opts) + run(sources=[b], **opts))


def test_combined_stability_bound():
    """Separate advection and diffusion bounds are insufficient when combined."""
    assert stability_number(dx=0.1, dy=0.1, dt=0.1, u=0.5, v=0, D=0.015) == pytest.approx(1.1)
    with pytest.raises(ValueError, match="Unstable timestep"):
        run(nx=11, ny=11, dt=0.1, u=0.5, D=0.015)
    assert run(nx=11, ny=11, dt=0.1, u=1, sources=[SOURCE]).min() >= 0


@pytest.mark.parametrize("opts", [dict(D=-1), dict(dt=0), dict(nx=2), dict(steps=-1), dict(u=float("nan"))])
def test_invalid_parameters(opts):
    with pytest.raises(ValueError):
        run(**opts)
