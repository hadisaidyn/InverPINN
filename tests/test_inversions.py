"""Analytical ladders test temperature-gradient signs, units, gaps and event timing."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from inverpinn.data.inversions import profile_layers, diagnose_inversions, synthetic_profiles


def profile(t, h=(0, 100, 200), bounds=(0, 1000)):
    return profile_layers(t, h, height_bounds_m=bounds)


def diagnose(ds=None, **kwargs):
    options = dict(start="2020-01-01T00:00:00Z", end="2020-01-01T04:00:00Z",
                   cadence_seconds=3600, height_bounds_m=[0, 1000])
    return diagnose_inversions(synthetic_profiles() if ds is None else ds, **(options | kwargs))


def test_linear_profile_exact_strength_depth_gradient():
    layers, qc = profile([270, 271, 272])
    assert qc["status"] == "detected"
    assert layers[0] == dict(layer_index=0, base_m_agl=0., top_m_agl=200., depth_m=200.,
        strength_K=2., gradient_K_per_m=.01, max_level_spacing_m=100.,
        base_type="surface_based", top_unresolved=True)


@pytest.mark.parametrize("t", [[272,271,270], [270,270,270]])
def test_cooling_and_isothermal_are_not_inversions(t):
    layers, qc = profile(t)
    assert layers == []
    assert qc["status"] == "not_detected_in_sampled_layers"


def test_stable_potential_temperature_is_not_temperature_inversion():
    # Environmental cooling of 6 K/km is less than the dry adiabatic cooling
    # rate (~9.8 K/km), but actual temperature still falls: no thermal inversion.
    layers, _ = profile([280, 279.4, 278.8])
    assert not layers


def test_no_arbitrary_strength_cutoff_or_offset_dependence():
    for offset in (0, 10):
        layers, _ = profile(np.array([270., 270.000001, 270.000002])+offset)
        assert len(layers) == 1
        assert layers[0]["strength_K"] == pytest.approx(.000002)


def test_missing_internal_level_never_bridged():
    layers, qc = profile([270, np.nan, 280])
    assert not layers and qc["status"] == "insufficient_data"
    assert qc["invalid_levels"] == 1
    layers, qc = profile([270, np.nan, 280,281], [0, 100,200,300])
    assert len(layers) == 1 and layers[0]["base_type"] == "unresolved_base"
    assert layers[0]["base_m_agl"] == 200
    assert qc["partial_profile"]


def test_multiple_layers_and_resolved_elevated_base():
    layers, _ = profile([270,272,271,273,270], [0,100,200,300,400])
    assert len(layers) == 2
    assert [r["base_type"] for r in layers] == ["surface_based", "elevated"]
    assert not any(r["top_unresolved"] for r in layers)
    layers, _ = profile([270,271,271,272], [0,100,200,300])
    assert len(layers) == 2  # Do not silently merge an isothermal gap.


def test_two_metre_or_lower_profile_limit_not_claimed_surface_based():
    layers, _ = profile([270,271,272], [2,100,200])
    assert layers[0]["base_type"] == "unresolved_base"
    layers, _ = profile([270,271,272], bounds=(100,200))
    assert layers[0]["base_m_agl"] == 100
    assert layers[0]["base_type"] == "unresolved_base"


def test_below_ground_and_out_of_domain_excluded():
    layers, qc = profile([260,270,269,280], [-100,0,100,2000])
    assert not layers and qc["below_ground_levels"] == 1
    assert qc["valid_pairs"] == 1


@pytest.mark.parametrize("t,h,bounds", [
    ([270,271],[0,0],[0,1000]), ([270,271],[100,0],[0,1000]),
    ([0,271],[0,100],[0,1000]), ([270,np.inf],[0,100],[0,1000]),
    ([270,271],[0,np.inf],[0,1000]), ([270],[0],[0,1000]),
    ([270,271],[0,100],[100,0]), ([270,271],[0,100],[-1,100])])
def test_malformed_profiles_raise(t, h, bounds):
    with pytest.raises(ValueError):
        profile(t,h,bounds)


def test_known_events_missingness_and_local_timezone():
    layers, coverage, events = diagnose()
    assert len(layers) == 4 and len(events) == 2
    assert list(coverage.status) == ["detected", "detected", "insufficient_data", "not_detected_in_sampled_layers", "detected"]
    assert list(events.sample_count) == [2,1]
    assert list(events.sample_span_hours) == [1.,0.]
    assert events.iloc[0].following_status == "insufficient_data"
    assert events.iloc[0].max_strength_K == 4.
    assert events.iloc[0].start_local == "2020-01-01T06:00:00+06:00"


def test_missing_timestamp_not_skipped_and_sites_separate():
    ds = synthetic_profiles().isel(time=[0,1,3,4])
    assert list(diagnose(ds)[1].status)[2] == "insufficient_data"
    other = ds.assign_coords(site=["other"])
    import xarray as xr
    joined = xr.concat([ds,other], dim="site")
    assert len(diagnose(joined)[2]) == 4


def test_timestamp_resolution_does_not_change_events():
    ds = synthetic_profiles()
    expected = diagnose(ds)[2]
    for resolution in ("us", "ns"):
        shifted = ds.assign_coords(time=pd.DatetimeIndex(ds.time.values).as_unit(resolution))
        pd.testing.assert_frame_equal(diagnose(shifted)[2], expected)


def test_empty_events_distinguish_isothermal_and_no_coverage():
    ds = synthetic_profiles()
    ds.air_temperature.values[:] = 270.
    layers, coverage, events = diagnose(ds)
    assert layers.empty and events.empty and "start_utc" in events.columns
    assert set(coverage.status) == {"not_detected_in_sampled_layers"}
    ds.air_temperature.values[:] = np.nan
    assert set(diagnose(ds)[1].status) == {"insufficient_data"}


def test_irregular_height_spacing_and_missing_height():
    h = np.array([0, 20, 370.])
    layers, _ = profile(270+.003*h, h)
    assert layers[0]["gradient_K_per_m"] == pytest.approx(.003)
    assert layers[0]["max_level_spacing_m"] == 350.
    layers, qc = profile([270,275,280], [0,np.nan,370])
    assert not layers and qc["status"] == "insufficient_data"


@pytest.mark.parametrize("problem", ["surface", "potential", "units", "time_basis", "order", "duplicates", "cadence", "kind", "height"])
def test_dataset_contract_fails_loudly(problem):
    ds = synthetic_profiles()
    if problem == "surface": ds = ds.drop_vars("air_temperature")
    if problem == "potential": ds.air_temperature.attrs["standard_name"] = "air_potential_temperature"
    if problem == "units": ds.air_temperature.attrs["units"] = "degC"
    if problem == "time_basis": ds.attrs["time_basis"] = "local"
    if problem == "order": ds = ds.isel(time=[1,0,2,3,4])
    if problem == "duplicates": ds = ds.isel(time=[0,0,2,3,4])
    if problem == "cadence": ds = ds.assign_coords(time=ds.time+np.timedelta64(1,"m"))
    if problem == "kind": ds.attrs["data_kind"] = "unknown"
    if problem == "height": ds.height_agl.attrs.pop("positive")
    with pytest.raises(ValueError): diagnose(ds)


def runner():
    spec = importlib.util.spec_from_file_location("inversion_runner", Path(__file__).parents[1]/"scripts/diagnose_inversions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_end_to_end_and_repeatability(tmp_path):
    config = yaml.safe_load(Path("configs/inversions_demo.yaml").read_text())
    outputs = []
    for name in ("a", "b"):
        target = tmp_path/name
        result = runner().run(config | dict(output_dir=str(target)))
        assert result["candidate_events"] == 2
        assert {"config.yaml", "inversion_events.csv", "inversion_layers.csv", "profile_coverage.csv",
                "profiles_checkpoint.nc", "provenance.json", "summary.json", "report.md", "inversion_diagnostic.png"} <= {p.name for p in target.iterdir()}
        outputs.append((target/"inversion_events.csv").read_bytes())
        assert set(pd.read_csv(target/"inversion_events.csv").data_kind) == {"synthetic_test"}
    assert outputs[0] == outputs[1]


def test_default_real_run_blocks_without_misleading_zero_table(tmp_path):
    config = yaml.safe_load(Path("configs/inversions.yaml").read_text())
    target = tmp_path/"blocked"
    with pytest.raises(ValueError, match="Study dates"):
        runner().run(config | dict(output_dir=str(target)))
    assert "not_assessed" in (target/"summary.json").read_text()
    assert not (target/"inversion_events.csv").exists()


def test_real_missing_profiles_and_synthetic_mislabel_rejected(tmp_path):
    config = yaml.safe_load(Path("configs/inversions_demo.yaml").read_text()) | dict(
        data_kind="real", profiles_file=str(tmp_path/"input.nc"), usage_permission_confirmed=True,
        source_description="A test of rejection", output_dir=str(tmp_path/"missing"))
    with pytest.raises(FileNotFoundError, match="Vertical-profile"):
        runner().run(config)
    synthetic_profiles().to_netcdf(config["profiles_file"])
    with pytest.raises(ValueError, match="Synthetic profiles"):
        runner().run(config | dict(output_dir=str(tmp_path/"mislabel")))
