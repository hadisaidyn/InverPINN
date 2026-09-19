"""Check regression, reproducibility, and separation of training from truth."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from inverpinn.models.mlp import ConcentrationMLP

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline_script", ROOT / "scripts/train_data_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_model_shape_and_checkpoint():
    model = ConcentrationMLP([0, 0, 0], [1, 1, 2], width=8, depth=2)
    points = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 1.0]])
    assert model(points).shape == (2, 1)
    restored = ConcentrationMLP([0, 0, 0], [1, 1, 1], width=8, depth=2)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(points), model(points))


def test_sensor_only_regression_is_reproducible():
    positions = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    times = torch.linspace(0, 1, 8)
    readings = 0.1 + 0.2*positions[:, 0] + 0.3*positions[:, 1] + 0.4*times[:, None]
    config = dict(seed=3, epochs=250, batch_size=32, learning_rate=0.01, width=16, depth=2)
    a, scale, history = baseline.train_observations(positions, times, readings, [0, 0, 0], [1, 1, 1], config)
    b, other_scale, other_history = baseline.train_observations(positions, times, readings, [0, 0, 0], [1, 1, 1], config)
    assert history[-1] < history[0] * 0.05
    assert scale == other_scale and history == other_history
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


def test_relative_l2():
    truth = np.array([3., 4.])
    assert baseline.relative_l2(truth * 1.1, truth) == pytest.approx(0.1)
    assert baseline.relative_l2(truth, np.zeros(2)) is None


def test_cli_truth_cannot_affect_training(tmp_path):
    config = yaml.safe_load((ROOT / "configs/data_baseline.yaml").read_text())
    config.update(n_sensors=2, epochs=3, width=8, depth=1, batch_size=4)
    axis = np.linspace(0, 1, 3)
    states = []
    for i in range(2):
        dataset = tmp_path / f"data{i}.npz"
        output = tmp_path / f"run{i}"
        np.savez(dataset, x=axis, y=axis, t=axis, C=np.full((3, 3, 3), i+1.),
                 sensor_positions_2=np.array([[0.2, 0.3], [0.7, 0.8]]), measurements_2=np.full((3, 2), 0.1))
        config.update(dataset=str(dataset), output_dir=str(output))
        path = tmp_path / f"config{i}.yaml"
        path.write_text(yaml.safe_dump(config))
        subprocess.run([sys.executable, str(ROOT / "scripts/train_data_baseline.py"), "--config", str(path)],
                       check=True, capture_output=True, text=True)
        states.append(torch.load(output / "model.pt", weights_only=True)["model_state_dict"])
        with np.load(output / "predictions.npz") as saved:
            metrics = json.loads((output / "metrics.json").read_text())
            assert metrics["relative_l2"] == pytest.approx(baseline.relative_l2(saved["C_predicted"], np.full((3,3,3), i+1.)))
        for name in ("training_curve.png", "predicted_concentration.png"):
            with Image.open(output / name) as image:
                image.verify()
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key])
