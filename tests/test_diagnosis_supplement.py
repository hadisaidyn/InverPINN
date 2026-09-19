"""Analytical and safety checks for post-selection, read-only diagnostics."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from inverpinn.evaluation.diagnosis_supplement import (
    inferred_raw_parameters, pde_terms, region_masks, scaled_gradient_rows, source_motion, statistics)
from inverpinn.physics.autodiff import autograd_pde_residual
from inverpinn.physics.nondimensional import CharacteristicScales


def test_all_six_terms_match_analytical_polynomial_and_residual():
    # C=x²+3y²+2xt+4t² includes both curvatures, time and signed winds.
    points = torch.tensor([[.2, .3, .4], [.8, .7, .6]], dtype=torch.float64, requires_grad=True)
    x, y, t = points.split(1, dim=1)
    C = x**2+3*y**2+2*x*t+4*t**2
    source = x*0+.7
    terms = pde_terms(C, points, u=.4, v=-.2, D=.03, S=source)
    expected = dict(C_t=2*x+8*t, u_C_x=.4*(2*x+2*t), v_C_y=-1.2*y,
                    D_C_xx=x*0+.06, D_C_yy=x*0+.18, S=source)
    for key, value in expected.items():
        torch.testing.assert_close(terms[key], value, rtol=1e-13, atol=1e-13)
    exact_r = sum(expected[key] for key in ("C_t", "u_C_x", "v_C_y"))-expected["D_C_xx"]-expected["D_C_yy"]-source
    torch.testing.assert_close(terms["residual"], exact_r)
    torch.testing.assert_close(terms["residual"], autograd_pde_residual(C, points, u=.4, v=-.2, D=.03, S=source))
    # Every dimensionless term obtains the SAME T/C* multiplier.
    scales = CharacteristicScales(length=2., time=3., concentration=4.)
    torch.testing.assert_close(scales.residual(terms["residual"]), .75*exact_r)


def test_pde_terms_reject_source_broadcast_and_nonfinite():
    points = torch.tensor([[.2, .3, .4]], requires_grad=True)
    C = points[:, :1]**2
    with pytest.raises(ValueError, match="column"):
        pde_terms(C, points, u=0., v=0., D=.1, S=torch.zeros(1))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        pde_terms(C, points, u=0., v=0., D=.1, S=torch.full((1, 1), float("nan")))


@pytest.mark.parametrize("u,D", [(float("nan"), .1), (0., -.1), (0., float("inf"))])
def test_pde_terms_reject_invalid_transport(u, D):
    points = torch.tensor([[.2, .3, .4]], requires_grad=True)
    with pytest.raises(ValueError, match="transport"):
        pde_terms(points[:, :1], points, u=u, v=0., D=D, S=torch.zeros(1, 1))


def test_region_masks_exclude_edges_and_initial_time_but_allow_overlap():
    points = [[0., .5, .2], [.05, .5, .2], [.5, .5, 0.], [.5, .5, .2], [.9, .9, .2]]
    masks = region_masks(points, [.05, .5], source_radius=.16, boundary_width=.08)
    assert masks["global_interior"].tolist() == [False, True, False, True, True]
    assert masks["boundary_band"].tolist() == [False, True, False, False, False]
    assert masks["near_estimate"].tolist() == [False, True, False, False, False]
    np.testing.assert_array_equal(masks["near_estimate"] | masks["away_from_estimate"], masks["global_interior"])
    assert (masks["near_estimate"] & masks["boundary_band"]).sum() == 1


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_statistics_do_not_hide_empty_or_nonfinite(values):
    with pytest.raises(ValueError, match="finite"):
        statistics(values)


def test_statistics_keep_sign_and_exact_zero():
    result = statistics([-3., 4.])
    assert result == dict(count=2, mean=.5, rmse=np.sqrt(12.5), mae=3.5, maxabs=4.)
    assert statistics([0., 0.])["rmse"] == 0


def history():
    return pd.DataFrame({"epoch":[1, 2, 3, 4], "source/x_s":[.1, .5, .1, .1],
                         "source/y_s":[.2, .2, .2, .3], "source/Q":[1., 3., 1., .5]})


def test_motion_uses_postupdate_endpoints_and_does_not_hide_excursion():
    result = source_motion(history(), 1, 4, 2.)
    assert result["location_displacement"] == pytest.approx(.1)
    assert result["max_location_excursion"] == pytest.approx(.4)
    assert result["Q_change"] == -.5
    assert result["Q_change_over_prior_scale"] == -.25
    assert result["max_Q_excursion"] == 2.
    assert result["last_step_Q_change"] == -.5


@pytest.mark.parametrize("epochs", [[0, 1, 2, 3], [1, 2, 2, 4], [1, 2, 4, 5]])
def test_motion_rejects_off_by_one_duplicate_and_missing_updates(epochs):
    frame = history()
    frame["epoch"] = epochs
    with pytest.raises(ValueError, match="consecutive"):
        source_motion(frame, 1, 4, 1.)


def test_inferred_logits_reproduce_logged_physical_values_not_claim_exact_raw():
    frame = pd.DataFrame(dict(x_s=[.2, .8], y_s=[.3, .9], Q=[.15, 1.1]))
    values = inferred_raw_parameters(frame)
    np.testing.assert_allclose(torch.sigmoid(torch.from_numpy(values["raw_x_inferred"])), frame.x_s, atol=1e-14)
    np.testing.assert_allclose(torch.sigmoid(torch.from_numpy(values["raw_y_inferred"])), frame.y_s, atol=1e-14)
    np.testing.assert_allclose(torch.nn.functional.softplus(torch.from_numpy(values["raw_Q_inferred"])), frame.Q, atol=1e-14)
    with pytest.raises(ValueError, match="saturated"):
        inferred_raw_parameters(pd.DataFrame(dict(x_s=[0.], y_s=[.5], Q=[.5])))


def test_scaled_source_gradients_apply_parameter_chain_rule_and_zero_is_missing():
    frame = pd.DataFrame(dict(x_s=[.5, .5], y_s=[.5, .5], Q=[1., 1.],
        x_s_raw_gradient=[.3, .3], y_s_raw_gradient=[.4, .4],
        x_s_physical_gradient=[3., 3.], y_s_physical_gradient=[4., 4.], Q_physical_gradient=[-2., 0.]))
    result = scaled_gradient_rows(frame, CharacteristicScales(length=2., time=4., concentration=12.))
    assert result.location_scaled_norm.tolist() == [10., 10.]
    assert result.Q_scaled_gradient.tolist() == [-6., 0.]
    assert result.location_to_Q_scaled_ratio.iloc[0] == pytest.approx(10/6)
    assert np.isnan(result.location_to_Q_scaled_ratio.iloc[1])
    assert "raw_x_inferred" not in frame  # diagnostics must not mutate input


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("supplement_runner", scripts/"supplement_model_diagnosis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_manifest_rejects_mutation_and_path_escape(runner, tmp_path):
    root = tmp_path/"report"
    root.mkdir()
    path = root/"record.json"
    path.write_text("original")
    manifest = {"record.json":runner.sha256(path)}
    runner.verify_manifest(root, manifest)
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        runner.verify_manifest(root, manifest)
    with pytest.raises(ValueError, match="escaped"):
        runner.verify_manifest(root, {"../outside.json":"unused"})


@pytest.mark.parametrize("existing", ["diagnostic_supplement", "diagnosis_summary.json"])
def test_supplement_refuses_existing_outputs_before_model_or_truth_io(runner, monkeypatch, tmp_path, existing):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    # Process audit hook has separate strict tests; avoid installing one for
    # this filesystem-only preflight unit test.
    monkeypatch.setattr(runner, "install_development_read_guard", lambda _: None)
    root = tmp_path/"results"
    root.mkdir()
    (root/existing).write_text("preserve me")
    (tmp_path/"config.yaml").write_text(yaml.safe_dump(dict(output_dir="results", benchmark_root="unused")))
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.supplement(dict(diagnosis_config="config.yaml", output_subdir="diagnostic_supplement"))
    assert (root/existing).read_text() == "preserve me"
