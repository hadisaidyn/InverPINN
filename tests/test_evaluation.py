"""Exact metric values, undefined denominators, and persisted reports."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from inverpinn.evaluation.metrics import (source_localization_error,
    relative_source_strength_error, concentration_metrics)


def test_exact_metrics():
    assert source_localization_error([0.3, 0.4], [0., 0.]) == pytest.approx(0.5)
    assert relative_source_strength_error(0.8, 1.) == pytest.approx(0.2)
    assert relative_source_strength_error(1.2, 1.) == pytest.approx(0.2)
    result = concentration_metrics(np.array([4., 6.]), np.array([3., 4.]))
    assert result == pytest.approx(dict(relative_l2=np.sqrt(5)/5, mae=1.5, rmse=np.sqrt(2.5)))


def test_perfect_predictions_and_zero_reference():
    assert source_localization_error([0.3, 0.7], [0.3, 0.7]) == 0
    assert relative_source_strength_error(1., 1.) == 0
    assert concentration_metrics([1., 2.], [1., 2.]) == dict(relative_l2=0., mae=0., rmse=0.)
    assert relative_source_strength_error(0., 0.) is None
    assert concentration_metrics([1.], [0.]) == dict(relative_l2=None, mae=1., rmse=1.)


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")]])
def test_invalid_arrays(bad):
    with pytest.raises(ValueError):
        concentration_metrics(bad, bad)


def test_invalid_shapes_and_strength():
    with pytest.raises(ValueError):
        concentration_metrics(np.ones((3, 1)), np.ones(3))
    with pytest.raises(ValueError):
        source_localization_error([[0, 0]], [0, 0])
    with pytest.raises(ValueError):
        relative_source_strength_error(-1, 1)


def test_evaluation_report_and_coordinate_validation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("evaluation_script", root / "scripts/evaluate_inverse_pinn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "run"
    run.mkdir()
    dataset = tmp_path / "truth.npz"
    axis = np.array([0., 1.])
    truth = np.ones((2, 2, 2))
    np.savez(dataset, C=truth, x=axis, y=axis, t=axis,
             source_positions=np.array([[0.3, 0.4]]), source_Q=np.array([1.]))
    np.savez(run / "predictions.npz", C_predicted=truth*2, x=axis, y=axis, t=axis)
    (run / "config.yaml").write_text(yaml.safe_dump(dict(dataset=str(dataset),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(), steps=1)))
    (run / "source_estimate.json").write_text(json.dumps(dict(x_s=0., y_s=0., Q=0.8)))
    output = tmp_path / "evaluation"
    report = module.evaluate_run(run, output)
    assert report["source_localization_error"] == pytest.approx(0.5)
    assert report["space_time"] == dict(relative_l2=1., mae=1., rmse=1.)
    assert json.loads((output / "metrics.json").read_text()) == report
    assert "20.000%" in (output / "report.md").read_text()
    with Image.open(output / "source_localization.png") as image:
        image.verify()
    np.savez(run / "predictions.npz", C_predicted=truth, x=axis+1, y=axis, t=axis)
    with pytest.raises(ValueError, match="coordinates do not match"):
        module.evaluate_run(run, tmp_path / "bad")
