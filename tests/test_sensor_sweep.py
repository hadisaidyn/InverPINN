"""Sensor sweep completeness, exact sample statistics and experiment dispatch."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from inverpinn.evaluation.sensor_sweep import ERROR_COLUMNS, summarize_results

ROOT = Path(__file__).resolve().parents[1]


def records():
    return pd.DataFrame([dict(n_sensors=n, seed=seed, status="success",
        **{key: float(seed+1) for key in ERROR_COLUMNS}) for n in [5,10,20,50] for seed in range(5)])


def test_exact_mean_and_sample_std():
    result = summarize_results(records(), [5,10,20,50], list(range(5)))
    assert result.n_runs.tolist() == [5]*4
    for key in ERROR_COLUMNS:
        np.testing.assert_allclose(result[f"{key}_mean"], 3)
        np.testing.assert_allclose(result[f"{key}_std"], np.sqrt(2.5))


@pytest.mark.parametrize("defect", ["missing", "duplicate", "failed", "nan"])
def test_incomplete_or_invalid_results_cannot_be_summarized(defect):
    raw = records()
    if defect == "missing":
        raw = raw.iloc[:-1]
    elif defect == "duplicate":
        raw.loc[0, "seed"] = 1
    elif defect == "failed":
        raw.loc[0, "status"] = "failed"
    else:
        raw.loc[0, "localization_error"] = float("nan")
    with pytest.raises(ValueError):
        summarize_results(raw, [5,10,20,50], list(range(5)))


def load_runner():
    spec = importlib.util.spec_from_file_location("sweep_script", ROOT / "scripts/run_sensor_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_config(tmp_path):
    reference = tmp_path / "reference.npz"
    axis = np.linspace(0,1,3)
    np.savez(reference, C=np.ones((2,3,3)), x=axis, y=axis, t=np.array([0.,0.1]),
             u=0.1, v=0., D=0.01, source_positions=np.array([[0.3,0.7]]), source_Q=np.array([1.]), source_sigma=np.array([0.08]))
    base = yaml.safe_load((ROOT / "configs/inverse_pinn.yaml").read_text())
    base.update(steps=2, width=8, depth=1, data_batch=4, collocation_batch=8, initial_batch=4, boundary_per_edge=2)
    base_path = tmp_path / "inverse.yaml"
    base_path.write_text(yaml.safe_dump(base))
    return dict(reference_dataset=str(reference), inverse_config=str(base_path), sensor_counts=[5,10,20,50],
                seeds=[0,1,2,3,4], workers=2, noise_std=0.005, output_dir=str(tmp_path / "sweep"))


def test_twenty_run_dispatch_and_saved_artifacts(tmp_path, monkeypatch):
    runner = load_runner()
    config = setup_config(tmp_path)

    def fake_train(command, **kwargs):
        # Exercise orchestration with deterministic mock results, not expensive training.
        run = yaml.safe_load(Path(command[-1]).read_text())
        directory = Path(run["output_dir"])
        directory.mkdir()
        (directory / "metrics.json").write_text(json.dumps(dict(
            source_estimate=dict(x_s=0.3, y_s=0.7, Q=0.9), relative_l2=0.2)))

    monkeypatch.setattr(runner.subprocess, "run", fake_train)
    summary = runner.run_sweep(config)
    out = Path(config["output_dir"])
    raw = pd.read_csv(out / "raw_results.csv")
    assert len(raw) == 20 and raw.status.eq("success").all()
    assert summary.n_runs.tolist() == [5]*4
    np.testing.assert_allclose(summary.relative_strength_error_mean, 0.1)
    np.testing.assert_allclose(summary.relative_strength_error_std, 0, atol=1e-15)
    with np.load(out / "datasets/seed_0.npz") as a, np.load(out / "datasets/seed_1.npz") as b:
        assert not np.array_equal(a["sensor_positions_5"], b["sensor_positions_5"])
        assert not np.array_equal(a["measurements_5"], b["measurements_5"])
        np.testing.assert_array_equal(a["sensor_positions_5"], a["sensor_positions_50"][:5])
    with Image.open(out / "error_vs_sensor_count.png") as image:
        image.verify()
    assert (out / "report.md").is_file() and (out / "summary.csv").is_file()


def test_failed_run_is_preserved(tmp_path, monkeypatch):
    runner = load_runner()
    config = setup_config(tmp_path)
    config["sensor_counts"] = [5]

    def fail(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="failed runs"):
        runner.run_sweep(config)
    out = Path(config["output_dir"])
    raw = pd.read_csv(out / "raw_results.csv")
    assert len(raw) == 5 and raw.status.eq("failed").all()
    assert not (out / "summary.csv").exists()


def test_real_training_smoke(tmp_path):
    runner = load_runner()
    config = setup_config(tmp_path)
    config["sensor_counts"] = [5]
    result = runner.run_sweep(config)
    assert result.n_runs.tolist() == [5]
    raw = pd.read_csv(Path(config["output_dir"]) / "raw_results.csv")
    assert raw.status.eq("success").all()
    for directory in raw.run_dir:
        assert (Path(directory) / "model.pt").is_file()
