"""Interpolation limits and independence from unavailable spatial truth."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from inverpinn.models.interpolation import idw_interpolate, rbf_interpolate, ordinary_kriging

POSITIONS = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])


@pytest.mark.parametrize("method", [idw_interpolate, rbf_interpolate, ordinary_kriging])
def test_exact_sensor_values_and_constant_fields(method):
    values = np.array([[1.,2.,3.,4.],[5.,4.,3.,2.]])
    np.testing.assert_allclose(method(POSITIONS,values,POSITIONS), values, atol=1e-12)
    queries = np.array([[0.2,0.3],[0.7,0.6]])
    np.testing.assert_allclose(method(POSITIONS,np.full((2,4),3.),queries),3.,atol=1e-12)
    assert method(POSITIONS,values,queries).shape == (2,2)


def test_idw_midpoint():
    np.testing.assert_allclose(idw_interpolate([[0.,0.],[1.,0.]], [[2.,4.]], [[0.5,0.]]), [[3.]])


def test_rbf_reproduces_affine_fields():
    values = 2*POSITIONS[:,0]+3*POSITIONS[:,1]+1
    queries = np.array([[0.3,0.4],[-0.1,0.7]])
    np.testing.assert_allclose(rbf_interpolate(POSITIONS,values[None],queries)[0], 2*queries[:,0]+3*queries[:,1]+1, atol=1e-12)


@pytest.mark.parametrize("method", [idw_interpolate, rbf_interpolate, ordinary_kriging])
def test_duplicate_sensors_fail(method):
    with pytest.raises(ValueError, match="Duplicate"):
        method([[0.,0.],[0.,0.]], [[1.,2.]], [[0.5,0.5]])


def test_baseline_predictions_do_not_depend_on_truth(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("baseline_eval", root / "scripts/evaluate_spatial_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictions = []
    for i in range(2):
        dataset = tmp_path / f"data{i}.npz"
        axis = np.linspace(0,1,3)
        values = np.array([[1.,2.,3.,4.],[2.,3.,4.,5.]])
        np.savez(dataset, C=np.full((2,3,3),1.+i), x=axis,y=axis,t=np.array([0.,1.]),sensor_positions_4=POSITIONS,measurements_4=values)
        config = dict(dataset=str(dataset),output_dir=str(tmp_path / f"out{i}"),n_sensors=4,idw_power=2.,rbf_smoothing=0.,kriging_range=0.2,kriging_nugget=0.)
        metrics = module.evaluate(config)
        assert len(metrics) == 6
        with np.load(tmp_path / f"out{i}/predictions.npz") as saved:
            predictions.append({key:saved[key].copy() for key in ("IDW","RBF","ordinary_kriging")})
    for key in predictions[0]:
        np.testing.assert_array_equal(predictions[0][key],predictions[1][key])
