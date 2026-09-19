"""Known synthetic fields verify projection, ERA5 normalization and causal joins."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod
import pytest
import xarray as xr

from inverpinn.data.coordinates import to_projected, to_wgs84, project_wind
from inverpinn.data.era5 import ingest_era5, utc_times, cache_raw
from inverpinn.data.modeling import merge_observations


BBOX = [76.6,43.1,77.3,43.4]
START, END = "2020-01-01T00:00:00Z", "2020-01-01T03:00:00Z"


def raw_dataset(blh=True, hours=(0,1,2,3)):
    lon, lat = np.array([76.5,77.5]), np.array([43.5,43.0])
    hh, yy, xx = np.meshgrid(hours, lat, lon, indexing="ij")
    dims = ("valid_time", "latitude", "longitude")
    fields = dict(u10=(dims, 2*(xx-76)+3*(yy-43)+hh, {"units":"m s**-1"}),
        v10=(dims, np.full(xx.shape,2.), {"units":"m s-1"}),
        t2m=(dims, 280+(xx-76)+hh, {"units":"K"}))
    if blh:
        fields["blh"] = (dims, np.full(xx.shape,500.), {"units":"m"})
    return xr.Dataset(fields, coords=dict(valid_time=np.datetime64("2020-01-01T00:00:00")+np.array(hours)*np.timedelta64(1,"h"),
                                          latitude=lat,longitude=lon))


def ingest(tmp_path, ds=None):
    path = tmp_path / "input.nc"
    (raw_dataset() if ds is None else ds).to_netcdf(path)
    return ingest_era5([path],bbox=BBOX,start=START,end=END,raw_cache_dir=tmp_path / "raw")


def sensors(times=("2020-01-01T00:30:00Z",)):
    return pd.DataFrame(dict(sensor_id=["s1"]*len(times),timestamp=times,
                             longitude=[77.]*len(times),latitude=[43.25]*len(times),pm25=[12.]*len(times)))


def test_projection_round_trip_and_axis_order():
    lon, lat = np.meshgrid(np.linspace(76.5,77.3,9), np.linspace(43.,43.5,7))
    x,y = to_projected(lon,lat)
    back_lon,back_lat = to_wgs84(x,y)
    np.testing.assert_allclose(back_lon,lon,rtol=0,atol=1e-9)
    np.testing.assert_allclose(back_lat,lat,rtol=0,atol=1e-9)
    cx,cy = to_projected(75.,0.)
    assert cx == pytest.approx(500000.,abs=1e-6)
    assert cy == pytest.approx(0.,abs=1e-6)
    with pytest.raises(ValueError,match="zone 43N"):
        to_projected(43.25,76.9)  # Swapped axes must not silently project.
    with pytest.raises(ValueError):
        to_wgs84(np.nan,100.)


def test_wind_jacobian_matches_geodesic_displacement():
    u,v = 5.,-3.
    ux,vy = project_wind(76.9,43.25,u,v)
    dt=0.01
    lon,lat,_ = Geod(ellps="WGS84").fwd(76.9,43.25,np.degrees(np.arctan2(u,v)),np.hypot(u,v)*dt)
    x0,y0 = to_projected(76.9,43.25)
    x1,y1 = to_projected(lon,lat)
    np.testing.assert_allclose([ux,vy],[(x1-x0)/dt,(y1-y0)/dt],atol=2e-6,rtol=0)
    east_x,east_y = project_wind(76.9,43.25,1.,0.)
    assert east_y > 0  # Grid north differs from true north east of central meridian.


def test_timezones_require_explicit_basis_and_honor_historical_offsets():
    times=utc_times(["2020-01-01T06:00:00","2025-01-01T05:00:00"],"Asia/Almaty")
    assert list(times.hour) == [0,0]
    with pytest.raises(ValueError,match="explicit timezone"):
        utc_times(["2020-01-01T00:00:00"])
    with pytest.raises(ValueError,match="full ISO date"):
        utc_times(["22-Mar"],"Asia/Almaty")
    with pytest.raises(Exception):
        utc_times(["2024-02-29T23:30:00"],"Asia/Almaty")  # Repeated local hour.


def test_normalization_units_projection_and_raw_cache(tmp_path):
    met,provenance = ingest(tmp_path)
    assert np.all(np.diff(met.latitude)>0)
    assert met.u10.dims == ("time","latitude","longitude")
    assert met.t2m.attrs["units"] == "K"
    assert met.attrs["crs"] == "EPSG:32643"
    assert met.x.dims == ("latitude","longitude")
    original = (tmp_path/"input.nc").read_bytes()
    assert Path(provenance[0]["cached"]).read_bytes() == original
    assert provenance[0]["sha256"] == hashlib.sha256(original).hexdigest()
    assert cache_raw(tmp_path/"input.nc",tmp_path/"raw") == provenance[0]


def test_normalizes_negative_wrapped_longitude(tmp_path):
    ds=raw_dataset().assign_coords(longitude=[-283.5,-282.5])
    met,_=ingest(tmp_path,ds)
    np.testing.assert_allclose(met.longitude,[76.5,77.5])


@pytest.mark.parametrize("change,match", [
    ("units","units"), ("expver","dimensions"), ("missing_variable","u10"), ("outside","bbox")])
def test_invalid_inputs_fail(tmp_path,change,match):
    ds=raw_dataset()
    if change == "units": ds.t2m.attrs["units"]="C"
    if change == "expver": ds=ds.expand_dims(expver=[1,5])
    if change == "missing_variable": ds=ds.drop_vars("u10")
    if change == "outside": ds=ds.assign_coords(longitude=[75.,76.])
    with pytest.raises(ValueError,match=match):
        ingest(tmp_path,ds)


def test_duplicate_era5_chunks_rejected(tmp_path):
    path=tmp_path/"input.nc"
    raw_dataset().to_netcdf(path)
    with pytest.raises(ValueError,match="Overlapping"):
        ingest_era5([path,path],bbox=BBOX,start=START,end=END,raw_cache_dir=tmp_path/"raw")


def test_affine_interpolation_and_past_only_time(tmp_path):
    met,_=ingest(tmp_path)
    out,qc=merge_observations(sensors(),met)
    assert out.u10.item() == pytest.approx(2.75)
    assert out.t2m.item() == pytest.approx(281.)
    assert out.meteorology_age_seconds.item() == 1800
    assert out.meteorology_time.values[0] == np.datetime64("2020-01-01T00:00:00")
    assert out.ready_for_model.item() == 1
    assert out.longitude.item() == 77.
    assert qc["spatially_interpolated_rows"] == 1
    future_changed=met.copy(deep=True)
    for name in ("u10","v10","t2m","blh"):
        future_changed[name].values[1:]=1e6
    other,_=merge_observations(sensors(),future_changed)
    xr.testing.assert_identical(out,other)


def test_missing_hour_not_bridged_and_optional_blh(tmp_path):
    met,_=ingest(tmp_path,raw_dataset(blh=False,hours=(0,2,3)))
    assert met.era5_time_present.values.tolist()==[1,0,1,1]
    out,qc=merge_observations(sensors(("2020-01-01T00:30:00Z","2020-01-01T01:30:00Z")),met)
    assert out.ready_for_model.values.tolist()==[1,0]
    assert np.isnan(out.u10.values[1])
    assert out.qc_missing_era5_hour.values.tolist()==[0,1]
    assert out.qc_blh_missing.values.tolist()==[1,1]
    required,_=merge_observations(sensors(),met,require_blh=True)
    assert required.ready_for_model.item()==0


def test_missing_positive_weight_corner_not_filled(tmp_path):
    ds=raw_dataset()
    ds.u10.values[0,0,0]=np.nan
    met,_=ingest(tmp_path,ds)
    out,_=merge_observations(sensors(),met)
    assert np.isnan(out.u10.item())
    assert out.qc_u10_missing.item()==1
    # An exact node uses only that node, even if unused neighbors are missing.
    met.attrs["bbox_west_south_east_north"]=[76.5,43.,77.5,43.5]
    obs=sensors().assign(longitude=77.5,latitude=43.)
    node,_=merge_observations(obs,met)
    assert np.isfinite(node.u10.item())


def test_qc_keeps_bad_rows_and_rejects_duplicates(tmp_path):
    met,_=ingest(tmp_path)
    obs=pd.concat([sensors().assign(sensor_id=str(i)) for i in range(4)],ignore_index=True)
    obs.loc[0,"pm25"]=-1
    obs.loc[1,"longitude"]=np.nan
    obs.loc[2,"latitude"]=44.
    obs.loc[3,"pm25"]=np.nan
    out,qc=merge_observations(obs,met)
    assert qc["rows"]==4 and qc["not_ready"]==4
    assert out.pm25_observed.values[0]==-1
    assert np.isnan(out.pm25.values[0])
    assert out.qc_coordinates_invalid.values[1]==1
    assert out.qc_outside_bbox.values[2]==1
    with pytest.raises(ValueError,match="Duplicate"):
        merge_observations(pd.concat([sensors(),sensors()]),met)


def test_stale_no_past_and_real_time_rejected(tmp_path):
    met,_=ingest(tmp_path)
    out,_=merge_observations(sensors(),met,max_age_seconds=60)
    assert out.qc_meteorology_stale.item()==1 and np.isnan(out.u10.item())
    before,_=merge_observations(sensors(("2019-12-31T23:30:00Z",)),met)
    assert before.qc_no_past_meteorology.item()==1
    with pytest.raises(ValueError,match="real-time"):
        merge_observations(sensors(),met,purpose="operational_forecast")
    with pytest.raises(ValueError,match="instantaneous"):
        merge_observations(sensors(),met,timestamp_meaning="daily_average")


def test_physical_invalid_met_values_are_flagged_not_repaired(tmp_path):
    ds=raw_dataset()
    ds.t2m.values[0,0,0]=-1
    ds.blh.values[0,1,1]=-10
    met,_=ingest(tmp_path,ds)
    assert np.min(met.t2m.values)==-1
    out,_=merge_observations(sensors(),met)
    assert out.qc_t2m_missing.item()==1
    assert out.qc_blh_missing.item()==1
    assert out.ready_for_model.item()==0


def test_non_hourly_era5_timestamps_rejected(tmp_path):
    ds=raw_dataset()
    ds=ds.assign_coords(valid_time=ds.valid_time+np.timedelta64(5,"m"))
    with pytest.raises(ValueError,match="exact hours"):
        ingest(tmp_path,ds)


def test_masked_netcdf_values_stay_missing(tmp_path):
    ds=raw_dataset()
    ds.u10.values[0,0,0]=np.nan
    ds.u10.encoding["_FillValue"]=-9999.
    met,_=ingest(tmp_path,ds)
    assert int((met.u10_valid==0).sum())==1
    assert int(np.isnan(met.u10).sum())==1


def test_future_pm_values_do_not_change_earlier_features(tmp_path):
    met,_=ingest(tmp_path)
    frame=sensors(("2020-01-01T00:30:00Z","2020-01-01T02:30:00Z"))
    a,_=merge_observations(frame,met)
    frame.loc[1,"pm25"]=1000000.
    b,_=merge_observations(frame,met)
    xr.testing.assert_identical(a.isel(observation=0),b.isel(observation=0))


def test_raw_cache_tampering_fails(tmp_path):
    path=tmp_path/"input.nc"
    raw_dataset().to_netcdf(path)
    record=cache_raw(path,tmp_path/"raw")
    Path(record["cached"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError,match="corruption"):
        cache_raw(path,tmp_path/"raw")


def test_end_to_end_netcdf_and_reproducibility(tmp_path):
    spec=importlib.util.spec_from_file_location("prepare_modeling",Path(__file__).resolve().parents[1]/"scripts/prepare_modeling_dataset.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    met_path=tmp_path/"era5.nc"
    raw_dataset(blh=False).to_netcdf(met_path)
    obs_path=tmp_path/"sensors.csv"
    sensors().to_csv(obs_path,index=False)
    base=dict(era5_files=[str(met_path)],sensor_csv=str(obs_path),bbox=BBOX,start=START,end=END,
        raw_cache_dir=str(tmp_path/"cache"),sensor_timezone=None,max_age_seconds=3600.,require_blh=False,
        purpose="retrospective_reanalysis",pm25_units="ug m-3",timestamp_meaning="instantaneous",
        source_label="synthetic test",usage_permission="synthetic generated in test")
    outputs=[]
    for i in range(2):
        config=base | dict(output_dir=str(tmp_path/f"out{i}"))
        qc=module.run(config)
        assert qc["ready_for_model"]==1
        with xr.open_dataset(tmp_path/f"out{i}"/"modeling.nc") as saved:
            outputs.append(saved.load())
        for name in ("config.yaml","meteorology.nc","provenance.json","qc_summary.json","qc_positions.png","qc_report.md"):
            assert (tmp_path/f"out{i}"/name).is_file()
        with pytest.raises(FileExistsError): module.run(config)
    xr.testing.assert_identical(*outputs)
    assert outputs[0].time.attrs["timezone"]=="UTC"
    with pytest.raises(ValueError,match="separate"):
        module.run(base | dict(output_dir=str(tmp_path/"cache"/"processed")))
