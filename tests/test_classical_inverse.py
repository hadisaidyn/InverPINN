"""Recovery with exactly the same discrete PDE and observation operator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from inverpinn.data.sensors import sample_sensors
from inverpinn.physics.finite_difference import solve_advection_diffusion
from inverpinn.physics.inverse_source import fit_source


SOLVER = dict(nx=17, ny=17, dt=0.01, steps=30, u=0.2, v=-0.1, D=0.01)
SOURCE = dict(x_s=0.35, y_s=0.65, Q=1.2, sigma=0.12)


def synthetic(noise_std=0.):
    field = solve_advection_diffusion(**SOLVER, sources=[SOURCE])
    axis = torch.linspace(0, 1, 17, dtype=torch.float64)
    positions, observations = sample_sensors(field, axis, axis, n_sensors=20,
                                             noise_std=noise_std, seed=7)
    return field.numpy(), positions.numpy(), observations.numpy()


@pytest.mark.parametrize("noise_std", [0., 0.0005])
def test_recovers_source_and_reduces_sensor_mismatch(noise_std):
    truth, positions, observations = synthetic(noise_std)
    result, field, readings, history = fit_source(positions, observations,
        solver=SOLVER, sigma=SOURCE["sigma"], initial=[0.5, 0.5, 0.7])
    assert result.success
    tolerance = 1e-6 if noise_std == 0 else 0.02
    np.testing.assert_allclose(result.x, [0.35, 0.65, 1.2], atol=tolerance, rtol=0)
    assert np.mean((readings-observations)**2) < history[0]["mse"]
    assert np.linalg.norm(field-truth)/np.linalg.norm(truth) < (1e-6 if noise_std == 0 else 0.02)
    assert all(0 <= r["x_s"] <= 1 and 0 <= r["y_s"] <= 1 and r["Q"] > 0 for r in history)


def test_budget_exhaustion_is_not_reported_as_success():
    _, positions, observations = synthetic()
    result, _, _, _ = fit_source(positions, observations, solver=SOLVER,
        sigma=0.12, initial=[0.5, 0.5, 0.7], max_nfev=1)
    assert not result.success
    assert result.status == 0


@pytest.mark.parametrize("change,match", [
    ({"sigma": 0.}, "sigma"),
    ({"initial": [0.5, 0.5, -1.]}, "initial"),
    ({"solver": SOLVER | {"sources": [SOURCE]}}, "no sources"),
    ({"solver": SOLVER | {"dt": 1.}}, "Unstable"),
])
def test_invalid_inputs_and_cfl_fail(change, match):
    _, positions, observations = synthetic()
    kwargs = dict(solver=SOLVER, sigma=0.12, initial=[0.5, 0.5, 0.7]) | change
    with pytest.raises(ValueError, match=match):
        fit_source(positions, observations, **kwargs)


def test_nonfinite_observations_fail():
    _, positions, observations = synthetic()
    observations[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite observations"):
        fit_source(positions, observations, solver=SOLVER, sigma=0.12, initial=[0.5, 0.5, 0.7])


def test_runner_artifacts_and_no_truth_leakage(tmp_path):
    spec = importlib.util.spec_from_file_location("classical_runner",
        Path(__file__).resolve().parents[1] / "scripts/fit_classical_inverse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth, positions, observations = synthetic()
    estimates = []
    for i in range(2):
        dataset = tmp_path / f"dataset{i}.npz"
        output = tmp_path / f"run{i}"
        np.savez(dataset, C=truth+i, x=np.linspace(0,1,17), y=np.linspace(0,1,17),
            t=np.arange(31)*0.01, u=0.2, v=-0.1, D=0.01,
            sensor_positions_20=positions, measurements_20=observations,
            source_positions=[[0.35+i*0.1, 0.65]], source_Q=[1.2+i])
        config = dict(dataset=str(dataset), output_dir=str(output), n_sensors=20,
                      seed=7, sigma=0.12, initial=[0.5,0.5,0.7], max_nfev=100, tolerance=1e-9)
        metrics = module.run(config)
        estimates.append(metrics["source_estimate"])
        for name in ("config.yaml", "metrics.json", "checkpoint.npz", "predictions.npz",
                     "optimization_history.csv", "classical_inverse.png", "report.md"):
            assert (output / name).is_file()
        with np.load(output / "checkpoint.npz") as saved:
            np.testing.assert_array_equal(saved["observations"], observations)
            np.testing.assert_array_equal(saved["positions"], positions)
        with pytest.raises(FileExistsError):
            module.run(config)
    assert estimates[0] == estimates[1]
