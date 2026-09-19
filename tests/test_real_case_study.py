"""Synthetic-only checks; no tests pretend to establish an observed Almaty episode."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml

from inverpinn.data.coordinates import to_projected
from inverpinn.data.episode import split_sensors,select_winter_episode,episode_inputs
from inverpinn.physics.case_study import scaled_transport_residual
from inverpinn.training.case_study import train_case_study


def runner():
    spec=importlib.util.spec_from_file_location("real_case",Path(__file__).resolve().parents[1]/"scripts/run_real_case_study.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    """Four winter hours, five sensors, known regional wind and increasing PM."""
    lon=np.tile(np.array([76.7,76.8,76.9,77.,77.1]),4)
    lat=np.full(len(lon),43.25)
    x,y=to_projected(lon,lat)
    hour=np.repeat(np.arange(4),5)
    obs=xr.Dataset(dict(sensor_id=("observation",np.tile([f"s{i}" for i in range(5)],4)),
        pm25=("observation",40.+hour*2,{"units":"ug m-3"}),ready_for_model=("observation",np.ones(20,dtype=np.int8)),
        x=("observation",x),y=("observation",y),longitude=("observation",lon),latitude=("observation",lat)),
        coords=dict(observation=np.arange(20),time=("observation",np.datetime64("2020-01-01T00:30:00")+hour*np.timedelta64(1,"h"))),
        attrs=dict(crs="EPSG:32643",time_basis="UTC",purpose="retrospective_reanalysis",pm25_timestamp_meaning="instantaneous"))
    obs.time.attrs["timezone"]="UTC"
    obs.x.attrs["units"]=obs.y.attrs["units"]="m"
    dims=("time","latitude","longitude")
    met=xr.Dataset(dict(u_x=(dims,np.ones((4,3,3)),{"units":"m s-1"}),v_y=(dims,np.zeros((4,3,3)),{"units":"m s-1"}),
        t2m=(dims,np.full((4,3,3),270.),{"units":"K"}),t2m_valid=(dims,np.ones((4,3,3),dtype=np.int8)),
        blh_valid=(dims,np.zeros((4,3,3),dtype=np.int8)),era5_time_present=("time",np.ones(4,dtype=np.int8))),
        coords=dict(time=np.datetime64("2020-01-01T00:00:00")+np.arange(4)*np.timedelta64(1,"h"),latitude=[43.,43.25,43.5],longitude=[76.5,77.,77.5]),
        attrs=dict(crs="EPSG:32643",time_basis="UTC",purpose="retrospective_reanalysis",bbox_west_south_east_north=[76.5,43.,77.5,43.5]))
    met.time.attrs["timezone"]="UTC"
    return obs,met


def configuration(tmp_path):
    root=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((root/"configs/real_case_study.yaml").read_text())
    config.update(data_kind="synthetic_test",observations=str(tmp_path/"modeling.nc"),meteorology=str(tmp_path/"meteorology.nc"),
        provenance=str(tmp_path/"provenance.json"),output_dir=str(tmp_path/"result"),training_seeds=[0,1],map_size=9)
    config["episode"].update(duration_hours=2,minimum_sensors_per_hour=3)
    config["training"].update(steps=3,width=8,depth=2,data_batch=8,collocation_batch=16,initial_batch=8,boundary_per_edge=2)
    return config


def write_fixture(tmp_path):
    obs,met=fixture()
    obs.to_netcdf(tmp_path/"modeling.nc")
    met.to_netcdf(tmp_path/"meteorology.nc")
    manifest=dict(source_label="synthetic fixture",usage_permission="Generated in tests",outputs_sha256={
        name:runner().sha256(tmp_path/name) for name in ("modeling.nc","meteorology.nc")})
    (tmp_path/"provenance.json").write_text(json.dumps(manifest))


def test_scaled_residual_matches_analytical_chain_rule():
    p=torch.tensor([[0.1,0.2,0.3],[0.5,0.4,0.8]],dtype=torch.float64,requires_grad=True)
    c=p[:,:1]**2+p[:,1:2]**2+p[:,2:3]
    wind=torch.tensor([[2.,3.],[4.,-1.]],dtype=torch.float64)
    L,T,D=1000.,3600.,50.
    source=1+2*wind[:,:1]*T/L*p[:,:1]+2*wind[:,1:2]*T/L*p[:,1:2]-4*D*T/L**2
    residual=scaled_transport_residual(c,p,wind_m_s=wind,diffusion_m2_s=D,source_scaled=source,length_m=L,time_scale_seconds=T)
    torch.testing.assert_close(residual,torch.zeros_like(residual),atol=1e-12,rtol=0)


def test_episode_selection_ignores_heldout_pm_and_explains_winter():
    obs,met=fixture()
    train,held=split_sensors(obs,seed=42,holdout_fraction=0.2)
    selected,table=select_winter_episode(obs,met,train,duration_hours=2,minimum_sensors_per_hour=3)
    assert selected["start_utc"]=="2020-01-01T02:00:00+00:00"
    assert len(table)==3
    changed=obs.copy(deep=True)
    changed.pm25.values[np.isin(changed.sensor_id,held)]=1e9
    other,_=select_winter_episode(changed,met,train,duration_hours=2,minimum_sensors_per_hour=3)
    assert selected==other
    assert split_sensors(changed,seed=42,holdout_fraction=0.2)==(train,held)


def test_missing_met_hour_not_bridged_and_no_summer_selection():
    obs,met=fixture()
    met.era5_time_present.values[1]=0
    met.era5_time_present.values[3]=0
    with pytest.raises(ValueError,match="No complete winter"):
        select_winter_episode(obs,met,["s0","s1","s2"],duration_hours=2,minimum_sensors_per_hour=3)
    obs,met=fixture()
    obs=obs.assign_coords(time=obs.time+np.timedelta64(182,"D"))
    met=met.assign_coords(time=met.time+np.timedelta64(182,"D"))
    with pytest.raises(ValueError,match="No complete winter"):
        select_winter_episode(obs,met,["s0","s1","s2"],duration_hours=2,minimum_sensors_per_hour=3)


def test_ties_choose_earliest_and_background_is_train_only():
    obs,met=fixture()
    obs.pm25.values[:]=20.
    obs.pm25.values[np.asarray(obs.sensor_id=="s4")]=1000.
    train,held=["s0","s1","s2","s3"],["s4"]
    selected,_=select_winter_episode(obs,met,train,duration_hours=2,minimum_sensors_per_hour=3)
    assert selected["start_utc"]=="2020-01-01T00:00:00+00:00"
    frame,winds,domain,diagnostics=episode_inputs(obs,met,selected,train,held)
    assert diagnostics["initial_boundary_background_ug_m3"]==20.
    assert domain["length_m"]>1000
    np.testing.assert_allclose(winds,[[1,0],[1,0]])
    with pytest.raises(ValueError,match="training rows only"):
        train_case_study(frame,winds,domain,20.,{},seed=0)


def test_input_guards_and_hash_validation(tmp_path):
    module=runner()
    config=configuration(tmp_path)
    with pytest.raises(FileNotFoundError,match="No episode or real result"):
        module.load_inputs(config)
    write_fixture(tmp_path)
    real=config | dict(data_kind="real",permission_confirmed=False)
    with pytest.raises(ValueError,match="permission"):
        module.load_inputs(real)
    with pytest.raises(ValueError,match="Synthetic"):
        module.load_inputs(real | dict(permission_confirmed=True))
    (tmp_path/"modeling.nc").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="hash mismatch"):
        module.load_inputs(config)


def test_end_to_end_artifacts_and_reproducibility(tmp_path):
    write_fixture(tmp_path)
    config=configuration(tmp_path)
    module=runner()
    metrics=module.run(config)
    output=Path(config["output_dir"])
    assert len(metrics)==4
    for name in ("source_intensity.nc","source_intensity.png","report.md","config.yaml","episode.json", "episode_candidates.csv",
                 "sensor_predictions.csv","estimated_components.csv","seed_0/model.pt","seed_1/training_losses.png"):
        assert (output/name).exists()
    report=(output/"report.md").read_text()
    for marker in ("## Measured facts","## Model estimates","## Interpretation","No real measured facts","not a calibrated source probability"):
        assert marker in report
    with xr.open_dataset(output/"source_intensity.nc") as data:
        assert data.attrs["data_kind"]=="synthetic_test"
        assert data.source_mean.attrs["units"]=="ug m-3 s-1"
        maps=data.source_intensity.values.copy()
        assert np.isfinite(maps).all() and (maps>=0).all()
    # Changing withheld targets must not change fitting, priors, episode, or maps.
    with xr.open_dataset(tmp_path/"modeling.nc") as ds:
        obs=ds.load()
    resolved=yaml.safe_load((output/"config.yaml").read_text())
    obs.pm25.values[np.isin(obs.sensor_id.values,resolved["heldout_sensors"])]=1e5
    obs.to_netcdf(tmp_path/"changed.nc")
    manifest=json.loads((tmp_path/"provenance.json").read_text())
    manifest["outputs_sha256"]["modeling.nc"]=module.sha256(tmp_path/"changed.nc")
    (tmp_path/"changed_provenance.json").write_text(json.dumps(manifest))
    other=config | dict(output_dir=str(tmp_path/"other"),observations=str(tmp_path/"changed.nc"),provenance=str(tmp_path/"changed_provenance.json"))
    module.run(other)
    with xr.open_dataset(tmp_path/"other/source_intensity.nc") as data:
        np.testing.assert_array_equal(maps,data.source_intensity.values)
    with pytest.raises(FileExistsError):
        module.run(config)
