"""Auditable winter episode selection without held-out PM concentrations."""

import numpy as np
import pandas as pd

from inverpinn.data.coordinates import to_projected


def split_sensors(observations, *, seed, holdout_fraction):
    """Split sorted valid station IDs, not individual readings, using a fixed seed."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one.")
    usable = observations.ready_for_model.values == 1
    ids = np.unique(observations.sensor_id.values[usable].astype(str))
    if len(ids) < 4:
        raise ValueError("Need at least four QC-valid sensors for training/held-out separation.")
    order = np.random.default_rng(seed).permutation(ids)
    count = max(1, int(np.floor(len(ids)*holdout_fraction)))
    if len(ids)-count < 3:
        raise ValueError("Need at least three training sensors.")
    return sorted(order[count:].tolist()), sorted(order[:count].tolist())


def select_winter_episode(observations, meteorology, training_ids, *, duration_hours,
                          minimum_sensors_per_hour, local_timezone="Asia/Almaty"):
    """Choose highest mean of hourly spatial medians among complete winter windows.

    First average each training sensor within a UTC hour; then take the median
    across sensors so heavily sampled stations receive no extra weight. Require
    every hour of each UTC-aligned candidate to have enough training sensors
    and complete ERA5 winds/temperature over the configured geographic box.
    Candidate hour starts must be in Dec/Jan/Feb in the configured IANA zone.
    Ties choose the earliest start. End is exclusive. This is a retrospective
    episode-selection score, NOT a health threshold or a test-set metric.
    Held-out sensor PM never participates. Save all candidate scores for audit.
    """
    if type(duration_hours) is not int or duration_hours < 2:
        raise ValueError("duration_hours must be an integer >=2.")
    if type(minimum_sensors_per_hour) is not int or minimum_sensors_per_hour < 2:
        raise ValueError("minimum_sensors_per_hour must be >=2.")
    keep = (observations.ready_for_model.values == 1) & np.isin(observations.sensor_id.values.astype(str), training_ids)
    times = pd.DatetimeIndex(observations.time.values).tz_localize("UTC")
    frame = pd.DataFrame(dict(hour=times[keep].floor("h"), sensor=observations.sensor_id.values[keep].astype(str),
                              pm25=observations.pm25.values[keep]))
    if frame.empty or not np.isfinite(frame.pm25).all() or (frame.pm25 < 0).any():
        raise ValueError("No usable nonnegative training measurements.")
    per_sensor = frame.groupby(["hour","sensor"], sort=True).pm25.mean()
    hourly = per_sensor.groupby("hour").agg(["median","count"])
    west,south,east,north = meteorology.attrs["bbox_west_south_east_north"]
    area = meteorology.sel(longitude=slice(west,east),latitude=slice(south,north))
    if min(area.sizes["longitude"],area.sizes["latitude"]) < 1:
        raise ValueError("Study box must contain ERA5 grid nodes for regional wind approximation.")
    met_times = pd.DatetimeIndex(area.time.values).tz_localize("UTC")
    good = area.era5_time_present.values.astype(bool)
    for key in ("u_x","v_y","t2m"):
        good &= np.isfinite(area[key].values).all(axis=(1,2))
    good &= area.t2m_valid.values.astype(bool).all(axis=(1,2))
    good_hours = set(met_times[good])
    candidates=[]
    for start in hourly.index:
        hours=pd.date_range(start,periods=duration_hours,freq="h")
        winter=all(month in (12,1,2) for month in hours.tz_convert(local_timezone).month)
        complete=hours.isin(hourly.index).all()
        if not winter or not complete:
            continue
        subset=hourly.loc[hours]
        if (subset["count"] < minimum_sensors_per_hour).any() or not all(h in good_hours for h in hours):
            continue
        candidates.append(dict(start_utc=start.isoformat(),end_utc=(start+pd.Timedelta(hours=duration_hours)).isoformat(),
            score_mean_hourly_median=float(subset["median"].mean()), minimum_training_sensors=int(subset["count"].min())))
    if not candidates:
        raise ValueError("No complete winter episode passes the predeclared PM/meteorology coverage requirements.")
    table=pd.DataFrame(candidates).sort_values(["score_mean_hourly_median","start_utc"],ascending=[False,True]).reset_index(drop=True)
    chosen=table.iloc[0].to_dict()
    chosen.update(selection="maximum mean hourly training-sensor median; ties earliest UTC start",
                  winter_definition=f"December/January/February, {local_timezone}", candidate_count=len(table))
    return chosen,table


def episode_inputs(observations, meteorology, episode, training_ids, heldout_ids):
    """Prepare irregular rows and hourly regional winds; no temporal gap filling.

    Square projected domain encloses the configured geographic study box.
    ERA5 projected winds are reduced to latitude-area-weighted regional means,
    excluding interpolation halo nodes. This is an explicit spatial-uniformity
    approximation, not fine-scale meteorology downscaling. Temperature/BLH are
    diagnostics only in this first 2-D transport model.
    """
    start,end=pd.Timestamp(episode["start_utc"]),pd.Timestamp(episode["end_utc"])
    times=pd.DatetimeIndex(observations.time.values).tz_localize("UTC")
    keep=(observations.ready_for_model.values==1)&(times>=start)&(times<end)
    selected=observations.isel(observation=np.flatnonzero(keep))
    frame=pd.DataFrame({name:selected[name].values for name in ("sensor_id","x","y","longitude","latitude","pm25")})
    frame["time_seconds"]=(times[keep]-start).total_seconds()
    frame["split"]=np.where(frame.sensor_id.astype(str).isin(training_ids),"train","heldout")
    if not set(frame.sensor_id.astype(str)).issubset(set(training_ids)|set(heldout_ids)):
        raise ValueError("Unexpected sensor outside predeclared split.")
    if not (frame.split=="train").any() or not (frame.split=="heldout").any():
        raise ValueError("Selected episode must contain training and held-out observations.")
    west,south,east,north=meteorology.attrs["bbox_west_south_east_north"]
    xx,yy=to_projected([west,west,east,east],[south,north,south,north])
    length=float(max(np.ptp(xx),np.ptp(yy)))
    x0,y0=float((np.min(xx)+np.max(xx)-length)/2),float((np.min(yy)+np.max(yy)-length)/2)
    if not ((frame.x>=x0)&(frame.x<=x0+length)&(frame.y>=y0)&(frame.y<=y0+length)).all():
        raise ValueError("Sensor lies outside projected study domain.")
    hours=pd.date_range(start,end,freq="h",inclusive="left").tz_localize(None)
    area=meteorology.sel(time=hours,longitude=slice(west,east),latitude=slice(south,north))
    weights=np.cos(np.deg2rad(area.latitude.values))[:,None]*np.ones((1,area.sizes["longitude"]))
    winds=np.column_stack([np.sum(area[key].values*weights,axis=(1,2))/weights.sum() for key in ("u_x","v_y")])
    if not np.isfinite(winds).all():
        raise ValueError("Missing hourly regional wind; no imputation is permitted.")
    first=frame[(frame.split=="train")&(frame.time_seconds<3600)].groupby("sensor_id").pm25.mean()
    background=float(first.median())
    diagnostics=dict(temperature_min_K=float(area.t2m.min()),temperature_max_K=float(area.t2m.max()),
        blh_missing_cells=int((area.blh_valid.values==0).sum()),
        initial_boundary_background_ug_m3=background,background_method="median training-sensor mean in first episode hour")
    return frame,winds,dict(x0=x0,y0=y0,length_m=length,duration_seconds=(end-start).total_seconds()),diagnostics
