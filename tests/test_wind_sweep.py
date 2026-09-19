"""Controlled wind design, paired statistics and unchanged nuisance variables."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import yaml

from inverpinn.data.sensors import sample_sensors
from inverpinn.evaluation.sensor_sweep import ERROR_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("run_wind_sweep")


def configuration():
    return yaml.safe_load((ROOT / "configs/wind_sweep.yaml").read_text())


def test_wind_design(monkeypatch):
    module = runner(monkeypatch)
    conditions = module.wind_conditions(configuration())
    assert len(conditions) == 6
    assert sum(c["is_reference"] for c in conditions) == 1
    assert sum(c["magnitude_comparison"] for c in conditions) == 3
    assert sum(c["direction_comparison"] for c in conditions) == 4
    assert {(c["u"], c["v"]) for c in conditions} == {(0.,0.), (0.2,0.), (0.4,0.), (0.,0.4), (-0.4,0.), (0.,-0.4)}
    assert conditions[0]["wind_direction_deg"] is None


def test_paired_changes(monkeypatch):
    module = runner(monkeypatch)
    config = configuration()
    conditions = module.wind_conditions(config)
    rows = []
    for condition in conditions:
        for seed in config["seeds"]:
            error = seed/100 + (0 if condition["is_reference"] else 0.05)
            rows.append(dict(condition=condition["condition"], seed=seed, n_sensors=20, status="success",
                             **{key: error for key in ERROR_COLUMNS}))
    raw = pd.DataFrame(rows)
    summary = module.summarize_wind(raw, conditions, config["seeds"])
    np.testing.assert_allclose(summary.loc[~summary.is_reference, "paired_localization_delta_mean"], 0.05)
    np.testing.assert_allclose(summary.paired_localization_delta_std, 0, atol=1e-15)
    with pytest.raises(ValueError):
        module.summarize_wind(raw.iloc[:-1], conditions, config["seeds"])


def test_controlled_data_and_tidy_outputs(tmp_path, monkeypatch):
    module = runner(monkeypatch)
    sweep = importlib.import_module("run_sensor_sweep")
    forward = yaml.safe_load((ROOT / "configs/forward.yaml").read_text())
    forward["solver"].update(nx=7, ny=7, steps=3)
    forward_path = tmp_path / "forward.yaml"
    forward_path.write_text(yaml.safe_dump(forward))
    config = configuration() | dict(forward_config=str(forward_path), inverse_config=str(ROOT / "configs/inverse_pinn.yaml"), output_dir=str(tmp_path / "out"))

    def fake_train(command, **kwargs):
        run = yaml.safe_load(Path(command[-1]).read_text())
        output = Path(run["output_dir"])
        output.mkdir()
        with np.load(run["dataset"]) as dataset:
            estimate = dict(x_s=0.3+float(dataset["u"])*0.01, y_s=0.7+float(dataset["v"])*0.01, Q=1.)
        (output / "metrics.json").write_text(json.dumps(dict(source_estimate=estimate, relative_l2=0.1)))

    monkeypatch.setattr(sweep.subprocess, "run", fake_train)
    summary = module.run_wind_sweep(config)
    output = Path(config["output_dir"])
    raw = pd.read_csv(output / "raw_results.csv")
    assert len(raw) == 30 and raw.status.eq("success").all()
    assert summary.n_runs.tolist() == [5]*6
    assert "localization_delta_vs_reference" in raw
    saved = []
    for condition in ("calm", "speed_0.4_angle_90"):
        with np.load(output / condition / "datasets/seed_0.npz") as data:
            arrays = {key: data[key].copy() for key in data.files}
        _, clean = sample_sensors(torch.from_numpy(arrays["C"]), torch.from_numpy(arrays["x"]),
            torch.from_numpy(arrays["y"]), positions=torch.from_numpy(arrays["sensor_positions_20"]))
        saved.append((arrays, arrays["measurements_20"]-clean.numpy()))
    a, b = saved[0][0], saved[1][0]
    for key in ("sensor_positions_20", "source_positions", "source_Q", "source_sigma", "D", "x", "y", "t", "noise_std"):
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_allclose(saved[0][1], saved[1][1], rtol=0, atol=1e-16)
    assert not np.array_equal(a["C"], b["C"])
    with Image.open(output / "localization_error_vs_wind.png") as image:
        image.verify()


def test_common_unstable_timestep_rejected(tmp_path, monkeypatch):
    module = runner(monkeypatch)
    forward = yaml.safe_load((ROOT / "configs/forward.yaml").read_text())
    forward["solver"]["dt"] = 1.
    path = tmp_path / "forward.yaml"
    path.write_text(yaml.safe_dump(forward))
    config = configuration() | dict(forward_config=str(path), inverse_config=str(ROOT / "configs/inverse_pinn.yaml"), output_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="Unstable wind"):
        module.run_wind_sweep(config)
    assert not (tmp_path / "out").exists()
