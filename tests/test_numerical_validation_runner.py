"""Reproducible validation artifacts and overwrite protection on small grids."""

import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml


def runner():
    path = Path(__file__).parents[1]/"scripts/validate_numerics.py"
    spec = importlib.util.spec_from_file_location("validate_numerics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_end_to_end_and_no_overwrite(tmp_path):
    config = yaml.safe_load(Path("configs/numerical_validation.yaml").read_text())
    synthetic = yaml.safe_load(Path("configs/dataset.yaml").read_text())
    synthetic["solver"].update(nx=21, ny=21, dt=0.004, steps=25)
    synthetic_path = tmp_path/"synthetic.yaml"
    synthetic_path.write_text(yaml.safe_dump(synthetic))
    original_bytes = synthetic_path.read_bytes()
    config.update(synthetic_config=str(synthetic_path))
    config["spatial"].update(grids=[11, 21, 41], final_time=0.1, dt_over_h_squared=1.6)
    config["temporal"].update(grid=21, final_time=0.1, initial_dt=0.008)
    config["gaussian"].update(grids=[11, 21, 41])
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    mod = runner()
    output = mod.validate(path, tmp_path/"validation")
    required = ["numerical_convergence.csv", "spatial_convergence.png", "temporal_convergence.png",
        "spatial_convergence.pdf", "temporal_convergence.pdf", "validation_summary.json", "config.yaml",
        "final_fields.npz", "environment.json", "provenance.json", "source_snapshot.zip"]
    assert set(p.name for p in output.iterdir()) == set(required)
    frame = pd.read_csv(output/"numerical_convergence.csv")
    assert frame["CFL_number"].max() <= 1
    assert set(frame["study_type"]) == {"spatial_advection_diffusion", "spatial_diffusion", "spatial_dt_half",
        "temporal", "gaussian_grid_difference", "gaussian_finest", "gaussian_time_difference"}
    assert frame["final_time"].to_numpy() == pytest.approx(0.1)
    assert frame["runtime"].min() >= 0
    summary = json.loads((output/"validation_summary.json").read_text())
    assert len(summary["spatial_orders"]) == 2
    assert len(summary["temporal_orders"]) == 3
    assert summary["target_is_project_requirement"] is False
    resolved = yaml.safe_load((output/"config.yaml").read_text())
    assert resolved["synthetic_configuration"] == synthetic
    assert synthetic_path.read_bytes() == original_bytes
    provenance = json.loads((output/"provenance.json").read_text())
    assert "git_commit" in provenance
    with zipfile.ZipFile(output/"source_snapshot.zip") as archive:
        assert "src/inverpinn/physics/finite_difference.py" in archive.namelist()
    with np.load(output/"final_fields.npz") as fields:
        assert np.isfinite(fields["gaussian_41"]).all()
        assert fields["gaussian_41"].shape == (41, 41)
    saved = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        mod.validate(path, output)
    assert saved == {p.name: p.read_bytes() for p in output.iterdir()}
    # Timings/provenance are not deterministic quantities; numeric fields are.
    second = mod.validate(path, tmp_path/"second")
    with np.load(output/"final_fields.npz") as a, np.load(second/"final_fields.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_timestep_hits_requested_end_without_exceeding_requested_step():
    dt, steps = runner().timestep(0.1, 0.03)
    assert dt <= 0.03
    assert dt*steps == pytest.approx(0.1)


def test_non_nested_comparison_refuses_interpolation():
    import torch
    with pytest.raises(ValueError, match="nested"):
        runner().restrict(torch.zeros((12, 12)), 7)
