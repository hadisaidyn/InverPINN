"""Relative-noise calibration, paired experiments and error-bar statistics."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import yaml

from inverpinn.data.sensors import relative_noise_std, sample_sensors
from inverpinn.evaluation.sensor_sweep import ERROR_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def test_noise_percentage_definition():
    clean = torch.tensor([0., 2.], dtype=torch.float64)
    assert relative_noise_std(clean, 20) == pytest.approx(0.2*np.sqrt(2))
    assert relative_noise_std(clean, 0) == 0
    with pytest.raises(ValueError):
        relative_noise_std(torch.zeros(2), 5)
    with pytest.raises(ValueError):
        relative_noise_std(clean, -1)


def test_paired_locations_and_noise_scaling():
    field = torch.ones((200,2,2), dtype=torch.float64)
    axis = torch.tensor([0.,1.], dtype=torch.float64)
    p, clean = sample_sensors(field, axis, axis, n_sensors=20, seed=3)
    p2, low = sample_sensors(field, axis, axis, n_sensors=20, seed=3, noise_std=relative_noise_std(clean, 2))
    p3, high = sample_sensors(field, axis, axis, n_sensors=20, seed=3, noise_std=relative_noise_std(clean, 20))
    assert torch.equal(p, p2) and torch.equal(p, p3)
    assert torch.equal(clean, torch.ones_like(clean))
    torch.testing.assert_close(high-clean, 10*(low-clean), atol=1e-14, rtol=1e-12)


def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("run_noise_sweep")


def test_noise_statistics_are_sample_std(monkeypatch):
    module = runner(monkeypatch)
    raw = pd.DataFrame([dict(noise_percent=p, n_sensors=20, seed=s, status="success",
                            **{key: float(s+1) for key in ERROR_COLUMNS})
                       for p in [0,2,5,10,20] for s in range(5)])
    summary = module.summarize_noise(raw, [0,2,5,10,20], list(range(5)))
    np.testing.assert_allclose(summary.localization_error_mean, 3)
    np.testing.assert_allclose(summary.localization_error_std, np.sqrt(2.5))
    with pytest.raises(ValueError):
        module.summarize_noise(raw.iloc[:-1], [0,2,5,10,20], list(range(5)))


def test_noise_runner_artifacts(tmp_path, monkeypatch):
    module = runner(monkeypatch)

    def fake_run(config):
        out = Path(config["output_dir"])
        out.mkdir()
        pd.DataFrame([dict(noise_percent=config["noise_percent"], n_sensors=20, seed=s, status="success",
                           **{key: (s+1)/100 for key in ERROR_COLUMNS}) for s in config["seeds"]]).to_csv(out / "raw_results.csv", index=False)

    monkeypatch.setattr(module, "run_sweep", fake_run)
    config = yaml.safe_load((ROOT / "configs/noise_sweep.yaml").read_text())
    config["output_dir"] = str(tmp_path / "out")
    result = module.run_noise_sweep(config)
    assert len(result) == 5
    raw = pd.read_csv(tmp_path / "out/raw_results.csv")
    assert len(raw) == 25 and raw.noise_percent.nunique() == 5
    with Image.open(tmp_path / "out/localization_error_vs_noise.png") as image:
        image.verify()
    assert (tmp_path / "out/report.md").is_file()


def test_percent_mode_produces_calibrated_datasets(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    sweep = importlib.import_module("run_sensor_sweep")
    axis = np.array([0.,1.])
    dataset = tmp_path / "truth.npz"
    np.savez(dataset, C=np.full((2,2,2), 2.), x=axis, y=axis, t=axis, u=0., v=0., D=0.01,
             source_positions=np.array([[0.3,0.7]]), source_Q=np.array([1.]), source_sigma=np.array([0.08]))

    def fake_train(command, **kwargs):
        cfg = yaml.safe_load(Path(command[-1]).read_text())
        out = Path(cfg["output_dir"])
        out.mkdir()
        (out / "metrics.json").write_text(json.dumps(dict(source_estimate=dict(x_s=0.3,y_s=0.7,Q=1.), relative_l2=0.1)))

    monkeypatch.setattr(sweep.subprocess, "run", fake_train)
    output = tmp_path / "out"
    sweep.run_sweep(dict(sensor_counts=[20], seeds=list(range(5)), workers=1, noise_percent=10,
        reference_dataset=str(dataset), inverse_config=str(ROOT / "configs/inverse_pinn.yaml"), output_dir=str(output)))
    raw = pd.read_csv(output / "raw_results.csv")
    np.testing.assert_allclose(raw.noise_std, 0.2)
    with np.load(output / "datasets/seed_0.npz") as saved:
        assert saved["noise_percent"] == 10
        assert saved["clean_sensor_rms"] == pytest.approx(2)
        assert saved["noise_std"] == pytest.approx(0.2)
