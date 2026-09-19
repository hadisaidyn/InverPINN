"""Analytic interpolation, noise, and input-contract checks for sparse sensors."""

import pytest
import torch

from inverpinn.data.sensors import sample_sensors


def field():
    """A bilinear spatial field on a rectangular nonuniform grid is exact."""
    x = torch.tensor([0., 0.2, 0.6, 1.], dtype=torch.float64)
    y = torch.tensor([0., 0.7, 1.], dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    t = torch.arange(4, dtype=torch.float64)
    return 2*xx + 3*yy + xx*yy + t[:, None, None], x, y


def test_manual_bilinear_interpolation_and_boundaries():
    c, x, y = field()
    positions = torch.tensor([[0., 0.], [1., 1.], [0.2, 0.7], [0.37, 0.43]], dtype=torch.float64)
    returned, readings = sample_sensors(c, x, y, positions=positions)
    px, py = positions.T
    expected = 2*px + 3*py + px*py + torch.arange(4)[:, None]
    torch.testing.assert_close(readings, expected)
    torch.testing.assert_close(returned, positions)
    assert readings.shape == (4, 4)


def test_seed_reproducibility_and_rng_isolation():
    c, x, y = field()
    state = torch.random.get_rng_state().clone()
    p, m = sample_sensors(c, x, y, n_sensors=10, noise_std=0.1, seed=17)
    p2, m2 = sample_sensors(c, x, y, n_sensors=10, noise_std=0.1, seed=17)
    assert torch.equal(p, p2) and torch.equal(m, m2)
    assert torch.equal(state, torch.random.get_rng_state())
    p3, _ = sample_sensors(c, x, y, n_sensors=10, seed=18)
    assert not torch.equal(p, p3)
    assert torch.all((p >= 0) & (p <= 1))
    _, manual = sample_sensors(c, x, y, positions=p, noise_std=0.1, seed=17)
    torch.testing.assert_close(m, manual)


def test_gaussian_noise_statistics_and_no_clipping():
    c = torch.zeros((1000, 2, 2), dtype=torch.float64)
    axis = torch.tensor([0., 1.], dtype=torch.float64)
    _, noisy = sample_sensors(c, axis, axis, n_sensors=100, noise_std=0.2, seed=7)
    assert abs(noisy.mean().item()) < 0.003
    assert abs(noisy.std().item() - 0.2) < 0.003
    assert noisy.min() < 0 < noisy.max()


@pytest.mark.parametrize("kwargs", [dict(), dict(n_sensors=0), dict(n_sensors=2, positions=[[0, 0]]),
    dict(positions=[[1.1, 0.2]]), dict(positions=[[float("nan"), 0.]]),
    dict(positions=[]), dict(n_sensors=2, noise_std=-1)])
def test_invalid_sampling_requests(kwargs):
    with pytest.raises(ValueError):
        sample_sensors(*field(), **kwargs)


def test_bad_axes_and_shape():
    c, x, y = field()
    with pytest.raises(ValueError):
        sample_sensors(c, x.flip(0), y, n_sensors=2)
    with pytest.raises(ValueError):
        sample_sensors(c, x[:2], y, n_sensors=2)
