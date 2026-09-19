"""Tests for the physics-only 2D transient advection--diffusion equation."""

import torch

from inverpinn.physics.advection_diffusion import advection_diffusion_residual


def _derivatives() -> dict[str, torch.Tensor]:
    """Return fixed derivatives in a hypothetical concentration unit system."""
    return {
        "concentration_t": torch.tensor([2.0, -1.0]),
        "concentration_x": torch.tensor([3.0, 4.0]),
        "concentration_y": torch.tensor([-2.0, 5.0]),
        "concentration_xx": torch.tensor([7.0, -3.0]),
        "concentration_yy": torch.tensor([11.0, 2.0]),
    }


def test_zero_wind_removes_both_advection_terms() -> None:
    """Still air leaves only time change, diffusion, and source in the residual."""
    derivatives = _derivatives()

    residual = advection_diffusion_residual(
        **derivatives,
        wind_x=0.0,
        wind_y=0.0,
        diffusion=0.5,
        source=torch.tensor([1.0, 6.0]),
    )

    expected = derivatives["concentration_t"] - 0.5 * (
        derivatives["concentration_xx"] + derivatives["concentration_yy"]
    ) - torch.tensor([1.0, 6.0])
    torch.testing.assert_close(residual, expected)


def test_zero_diffusion_removes_laplacian_term() -> None:
    """Without diffusion, wind transport and sources remain in the residual."""
    derivatives = _derivatives()

    residual = advection_diffusion_residual(
        **derivatives,
        wind_x=2.0,
        wind_y=-0.5,
        diffusion=0.0,
        source=3.0,
    )

    expected = (
        derivatives["concentration_t"]
        + 2.0 * derivatives["concentration_x"]
        - 0.5 * derivatives["concentration_y"]
        - 3.0
    )
    torch.testing.assert_close(residual, expected)


def test_zero_source_removes_source_term() -> None:
    """With no emissions, the source contributes exactly zero to the residual."""
    derivatives = _derivatives()

    residual = advection_diffusion_residual(
        **derivatives,
        wind_x=1.5,
        wind_y=-2.0,
        diffusion=0.25,
        source=0.0,
    )

    expected = (
        derivatives["concentration_t"]
        + 1.5 * derivatives["concentration_x"]
        - 2.0 * derivatives["concentration_y"]
        - 0.25 * (derivatives["concentration_xx"] + derivatives["concentration_yy"])
    )
    torch.testing.assert_close(residual, expected)


def test_residual_is_invariant_to_a_change_of_length_unit() -> None:
    """Changing metres to kilometres preserves the physical rate ``[C / s]``.

    A velocity is 1,000 times smaller in km/s, a first spatial derivative is
    1,000 times larger in C/km, a diffusion coefficient is 1,000,000 times
    smaller in km^2/s, and a second derivative is 1,000,000 times larger in
    C/km^2. Each product therefore remains the same concentration rate.
    """
    derivatives_in_metres = _derivatives()
    residual_in_metres = advection_diffusion_residual(
        **derivatives_in_metres,
        wind_x=800.0,
        wind_y=-300.0,
        diffusion=50_000.0,
        source=7.0,
    )

    metres_per_kilometre = 1_000.0
    derivatives_in_kilometres = {
        "concentration_t": derivatives_in_metres["concentration_t"],
        "concentration_x": derivatives_in_metres["concentration_x"] * metres_per_kilometre,
        "concentration_y": derivatives_in_metres["concentration_y"] * metres_per_kilometre,
        "concentration_xx": derivatives_in_metres["concentration_xx"] * metres_per_kilometre**2,
        "concentration_yy": derivatives_in_metres["concentration_yy"] * metres_per_kilometre**2,
    }
    residual_in_kilometres = advection_diffusion_residual(
        **derivatives_in_kilometres,
        wind_x=800.0 / metres_per_kilometre,
        wind_y=-300.0 / metres_per_kilometre,
        diffusion=50_000.0 / metres_per_kilometre**2,
        source=7.0,
    )

    torch.testing.assert_close(residual_in_kilometres, residual_in_metres)
