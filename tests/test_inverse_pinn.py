"""Constraints, gradient flow, joint updates, logging and truth isolation."""

import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from inverpinn.physics.sources import gaussian_source
from inverpinn.physics.trainable_source import TrainableGaussianSource
from inverpinn.physics.autodiff import autograd_pde_residual
from inverpinn.training.forward_pinn import train_forward_pinn

ROOT = Path(__file__).resolve().parents[1]


def test_initialization_matches_gaussian():
    source = TrainableGaussianSource(0.3, 0.7, 1., 0.08)
    assert source.estimates() == pytest.approx(dict(x_s=0.3, y_s=0.7, Q=1.))
    x, y = torch.tensor([0.2, 0.3]), torch.tensor([0.6, 0.7])
    torch.testing.assert_close(source(x, y), gaussian_source(x, y, 0.3, 0.7, 1., 0.08))
    assert len(list(source.parameters())) == 3
    assert not source.sigma.requires_grad


@pytest.mark.parametrize("raw", [-1000., 1000.])
def test_extreme_parameters_stay_constrained(raw):
    source = TrainableGaussianSource(0.5, 0.5, 1., 0.1)
    with torch.no_grad():
        for p in source.parameters():
            p.fill_(raw)
    estimates = source.estimates()
    assert 0 <= estimates["x_s"] <= 1
    assert 0 <= estimates["y_s"] <= 1
    assert 0 < estimates["Q"] < float("inf")


def test_exact_source_parameter_gradients_through_residual():
    source = TrainableGaussianSource(0.4, 0.6, 0.8, 0.2).double()
    points = torch.tensor([[0.5, 0.7, 0.2]], dtype=torch.float64, requires_grad=True)
    s = source(points[:, :1], points[:, 1:2])
    residual = autograd_pde_residual(points[:, :1]*0, points, u=0., v=0., D=0., S=s)
    gradients = torch.autograd.grad(residual.sum(), tuple(source.parameters()))
    expected_x = -s*(0.5-source.x_s)/source.sigma**2*source.x_s*(1-source.x_s)
    expected_y = -s*(0.7-source.y_s)/source.sigma**2*source.y_s*(1-source.y_s)
    expected_q = -s/source.Q*torch.sigmoid(source.raw_Q)
    for actual, expected in zip(gradients, (expected_x, expected_y, expected_q)):
        torch.testing.assert_close(actual, expected.squeeze())


def tiny_config():
    config = yaml.safe_load((ROOT / "configs/inverse_pinn.yaml").read_text())
    config.update(steps=3, width=8, depth=2, data_batch=4, collocation_batch=16, initial_batch=4, boundary_per_edge=2)
    return config


def test_joint_optimization_and_logging(caplog):
    config = tiny_config()
    source = TrainableGaussianSource(**config["source_initial"])
    initial = {key: p.detach().clone() for key, p in source.named_parameters()}
    with caplog.at_level(logging.INFO):
        model, optimizer, history = train_forward_pinn(torch.tensor([[0.3, 0.7]]), torch.tensor([0., 0.1]),
            torch.tensor([[0.], [0.05]]), dict(u=0.1, v=-0.1, D=0.01), config,
            logging.getLogger("test.inverse"), source_model=source)
    for key, p in source.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad)
        assert not torch.equal(initial[key], p)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in list(model.parameters())+list(source.parameters())}
    assert [r["epoch"] for r in history] == [1, 2, 3]
    for name, estimate in source.estimates().items():
        assert history[-1][f"source/{name}"] == estimate
    assert "Source epoch=3" in caplog.text


def test_cli_truth_does_not_change_fit(tmp_path):
    checkpoints = []
    for run in (0, 1):
        path = tmp_path / f"data{run}.npz"
        axis = np.linspace(0, 1, 3)
        np.savez(path, C=np.full((2,3,3), 1.+run), x=axis, y=axis, t=np.array([0., 0.1]),
                 sensor_positions_1=np.array([[0.3, 0.7]]), measurements_1=np.array([[0.], [0.05]]),
                 u=0.1, v=0., D=0.01, source_positions=np.array([[0.3+0.2*run, 0.7]]), source_Q=np.array([1.+run]))
        config = tiny_config()
        output = tmp_path / f"run{run}"
        config.update(dataset=str(path), output_dir=str(output), n_sensors=1)
        config_path = tmp_path / f"config{run}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        subprocess.run([sys.executable, str(ROOT / "scripts/train_inverse_pinn.py"), "--config", str(config_path)],
                       check=True, capture_output=True, text=True)
        checkpoints.append(torch.load(output / "model.pt", weights_only=True))
        assert len((output / "training_history.jsonl").read_text().splitlines()) == 3
        estimates = json.loads((output / "source_estimate.json").read_text())
        restored = TrainableGaussianSource(**config["source_initial"])
        restored.load_state_dict(checkpoints[-1]["source_state_dict"])
        assert restored.estimates() == estimates
    for state in ("model_state_dict", "source_state_dict"):
        for key in checkpoints[0][state]:
            assert torch.equal(checkpoints[0][state][key], checkpoints[1][state][key])
