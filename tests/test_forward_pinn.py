"""Forward PINN integration: known physics, finite checks and artifacts."""

import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from inverpinn.training.forward_pinn import train_forward_pinn, require_finite

ROOT = Path(__file__).resolve().parents[1]


def small_config():
    config = yaml.safe_load((ROOT / "configs/pinn.yaml").read_text())
    config.update(steps=3, width=8, depth=2, data_batch=4, collocation_batch=8, initial_batch=4, boundary_per_edge=2)
    return config


def test_reproducible_forward_training():
    known = dict(u=0.1, v=-0.1, D=0.01, sources=[dict(x_s=0.3, y_s=0.7, Q=1., sigma=0.08)])
    p, t = torch.tensor([[0.3, 0.7]]), torch.tensor([0., 0.1])
    observations = torch.tensor([[0.], [0.05]])
    args = (p, t, observations, known, small_config(), logging.getLogger("test.pinn"))
    a, _, ha = train_forward_pinn(*args)
    b, _, hb = train_forward_pinn(*args)
    assert ha == hb
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key])
    assert set(k for k in ha[0] if k.startswith("loss/")) == {"loss/data", "loss/pde", "loss/initial", "loss/boundary", "loss/total"}
    assert not any("source" in key for key in a.state_dict())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_fails_loudly(value):
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        require_finite(torch.tensor([value]), "test gradient")
    with pytest.raises(FloatingPointError, match="observations"):
        train_forward_pinn(torch.tensor([[0.3, 0.7]]), torch.tensor([0., 0.1]),
            torch.tensor([[0.], [value]]), {}, small_config(), logging.getLogger("test.pinn"))


def test_forward_pinn_command_and_metrics(tmp_path):
    config = small_config()
    axis = np.linspace(0, 1, 4)
    dataset = tmp_path / "dataset.npz"
    truth = np.ones((3, 4, 4))
    np.savez(dataset, C=truth, x=axis, y=axis, t=np.array([0., 0.1, 0.2]),
        u=0.1, v=0., D=0.01, source_positions=np.array([[0.3, 0.7]]), source_Q=np.array([1.]),
        source_sigma=np.array([0.08]), sensor_positions_1=np.array([[0.3, 0.7]]), measurements_1=np.array([[0.], [0.05], [0.1]]))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}))
    output = tmp_path / "run"
    config.update(dataset=str(dataset), metadata=str(metadata), output_dir=str(output), n_sensors=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    subprocess.run([sys.executable, str(ROOT / "scripts/train_forward_pinn.py"), "--config", str(path)],
                   check=True, capture_output=True, text=True)
    for name in ("ground_truth", "pinn_prediction", "absolute_error", "training_losses"):
        with Image.open(output / f"{name}.png") as image:
            image.verify()
    metrics = json.loads((output / "metrics.json").read_text())["space_time"]
    with np.load(output / "predictions.npz") as arrays:
        error = arrays["C_predicted"]-truth
        assert metrics["mae"] == pytest.approx(np.abs(error).mean())
        assert metrics["rmse"] == pytest.approx(np.sqrt((error**2).mean()))
        assert metrics["relative_l2"] == pytest.approx(np.linalg.norm(error)/np.linalg.norm(truth))
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    assert checkpoint["step"] == 3
    assert len((output / "training_history.jsonl").read_text().splitlines()) == 3
