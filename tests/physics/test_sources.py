"""Exact Gaussian properties tested with deterministic coordinates."""

import pytest
import torch

from inverpinn.physics.sources import gaussian_source


def test_maximum_at_source_center() -> None:
    """The exponential is at most one, with equality only at the center."""
    axis = torch.linspace(0, 1, 101, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    intensity = gaussian_source(x, y, 0.30, 0.70, 1.0, 0.08)
    assert intensity.argmax().item() == 70 * 101 + 30
    assert intensity[70, 30].item() == pytest.approx(1.0)


@pytest.mark.parametrize("direction", [(1, 0), (0, 1), (-1, 0), (0, -1)])
def test_intensity_decreases_with_distance(direction: tuple[int, int]) -> None:
    """A radial Gaussian decays strictly in every outward direction for Q > 0."""
    distance = torch.tensor([0.0, 0.08, 0.16, 0.24], dtype=torch.float64)
    intensity = gaussian_source(
        0.30 + direction[0] * distance,
        0.70 + direction[1] * distance,
        0.30, 0.70, 1.0, 0.08,
    )
    assert torch.all(intensity[:-1] > intensity[1:])
    torch.testing.assert_close(intensity, torch.exp(-0.5 * torch.arange(4).double() ** 2))


def test_strength_scales_intensity_linearly() -> None:
    """Tripling peak emissions triples the field at every sampled location."""
    x = torch.tensor([0.30, 0.38, 0.46], dtype=torch.float64)
    y = torch.tensor([0.70, 0.75, 0.80], dtype=torch.float64)
    base = gaussian_source(x, y, 0.30, 0.70, 1.0, 0.08)
    torch.testing.assert_close(gaussian_source(x, y, 0.30, 0.70, 3.0, 0.08), 3 * base)
    torch.testing.assert_close(gaussian_source(x, y, 0.30, 0.70, 0.0, 0.08), torch.zeros_like(base))


@pytest.mark.parametrize("sigma", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_width_is_rejected(sigma: float) -> None:
    """Width must be finite and positive for the Gaussian to be defined."""
    with pytest.raises(ValueError):
        gaussian_source(torch.tensor(0.3), torch.tensor(0.7), 0.3, 0.7, 1.0, sigma)
