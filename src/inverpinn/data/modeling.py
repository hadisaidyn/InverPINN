"""Pair sensor rows with past-valid ERA5; retain every row and explicit QC flags."""

import numpy as np
import pandas as pd
import xarray as xr

from inverpinn.data.coordinates import PROJECTED_CRS, to_projected, project_wind
from inverpinn.data.era5 import VARIABLES, utc_times, validate_bbox


def merge_observations(observations, meteorology, *, sensor_timezone=None,
                       max_age_seconds=3600., require_blh=False,
                       purpose="retrospective_reanalysis", pm25_units="ug m-3",
                       timestamp_meaning="instantaneous"):
    r"""Build one observation-indexed xarray Dataset and a JSON-ready QC summary.

    Input columns: sensor_id, timestamp, longitude, latitude, pm25. Concentration
    must be ug/m³, coordinates WGS84 degrees. Explicitly instantaneous sensor
    timestamps only: daily or centered averages need a different observation
    operator and are rejected rather than mislabeled as point samples.

    At each observation t, choose max(t_ERA5 <= t), bounded by max_age_seconds.
    A reindexed missing hour is NOT skipped in favor of an older good value.
    No nearest-time match, forward fill across gaps, or temporal interpolation.
    Spatial bilinear weights are applied on the native lon/lat grid at this
    single hour. Every positive-weight corner must be valid for each variable;
    zero-weight missing corners have no effect. No extrapolation or gap fill.
    East/north winds are then transformed at the sensor to projected velocity.

    Original lon/lat, timestamp strings and PM values are retained. Missing or
    negative PM, missing meteorology and invalid coordinates are flagged; rows
    are never silently dropped. ready_for_model gates required fields; optional
    missing blh does not block unless require_blh=True. This is retrospective:
    ERA5 assimilation/release timing cannot support strict real-time forecasts.
    No PM values from other rows, fitted normalization or future rows enter any
    feature. Later temporal/spatial train/test splits remain the caller's job.
    """
    if purpose != "retrospective_reanalysis":
        raise ValueError("ERA5 cannot guarantee real-time availability; use vintage forecasts for operational prediction.")
    if pm25_units != "ug m-3" or timestamp_meaning != "instantaneous":
        raise ValueError("Require explicit ug m-3 PM and instantaneous timestamps; convert/resolve averages first.")
    if not np.isfinite(max_age_seconds) or max_age_seconds < 0:
        raise ValueError("max_age_seconds must be finite and nonnegative.")
    if not isinstance(require_blh, bool):
        raise ValueError("require_blh must be boolean.")
    required = {"sensor_id", "timestamp", "longitude", "latitude", "pm25"}
    if not required.issubset(observations) or len(observations) == 0:
        raise ValueError(f"Require nonempty sensor rows with columns {sorted(required)}.")
    frame = observations.copy().reset_index(drop=True)
    if frame.sensor_id.isna().any() or frame.sensor_id.astype(str).str.strip().eq("").any():
        raise ValueError("sensor_id must be nonempty.")
    frame["sensor_id"] = frame.sensor_id.astype(str)
    times = utc_times(frame.timestamp, sensor_timezone)
    frame["time_utc"] = times
    if frame.duplicated(["sensor_id", "time_utc"]).any():
        raise ValueError("Duplicate sensor/time after UTC conversion; resolve explicitly.")
    for name in ("longitude", "latitude", "pm25"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    count = len(frame)
    lon, lat, pm = (frame[name].to_numpy(dtype=float) for name in ("longitude", "latitude", "pm25"))
    west, south, east, north = validate_bbox(meteorology.attrs["bbox_west_south_east_north"])
    begin, end = utc_times([meteorology.attrs["requested_start_utc"], meteorology.attrs["requested_end_utc"]])
    coords_valid = np.isfinite(lon) & np.isfinite(lat) & (lon >= 72) & (lon <= 78) & (lat >= 0) & (lat <= 84)
    inside = coords_valid & (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    flags = dict(qc_pm25_missing=~np.isfinite(pm), qc_pm25_negative=np.isfinite(pm) & (pm < 0),
                 qc_coordinates_invalid=~coords_valid, qc_outside_bbox=coords_valid & ~inside,
                 qc_outside_interval=np.asarray((times < begin) | (times > end)))
    x, y = np.full(count, np.nan), np.full(count, np.nan)
    if coords_valid.any():
        x[coords_valid], y[coords_valid] = to_projected(lon[coords_valid], lat[coords_valid])
    clock = times.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    met_clock = np.asarray(meteorology.time.values, dtype="datetime64[ns]")
    if len(met_clock) == 0 or np.any(np.diff(met_clock) <= np.timedelta64(0, "ns")):
        raise ValueError("Meteorology times must strictly increase.")
    ids = np.searchsorted(met_clock, clock, side="right")-1
    has_past = ids >= 0
    safe = np.maximum(ids, 0)
    met_time = np.full(count, np.datetime64("NaT", "ns"))
    met_time[has_past] = met_clock[safe[has_past]]
    age = (clock-met_time)/np.timedelta64(1, "s")
    flags["qc_no_past_meteorology"] = ~has_past
    flags["qc_meteorology_stale"] = has_past & (age > max_age_seconds)
    flags["qc_missing_era5_hour"] = has_past & (meteorology.era5_time_present.values[safe] == 0)
    usable = inside & ~flags["qc_outside_interval"] & has_past & ~flags["qc_meteorology_stale"] & ~flags["qc_missing_era5_hour"]
    values = {key: np.full(count, np.nan) for key in VARIABLES}
    interpolated = np.zeros(count, dtype=np.int8)
    gx, gy = meteorology.longitude.values, meteorology.latitude.values
    # No extrapolation even if the caller supplies a malformed processed grid.
    usable &= (lon >= gx[0]) & (lon <= gx[-1]) & (lat >= gy[0]) & (lat <= gy[-1])
    for row in np.flatnonzero(usable):
        ix = np.clip(np.searchsorted(gx, lon[row], side="right")-1, 0, len(gx)-2)
        iy = np.clip(np.searchsorted(gy, lat[row], side="right")-1, 0, len(gy)-2)
        a, b = (lon[row]-gx[ix])/(gx[ix+1]-gx[ix]), (lat[row]-gy[iy])/(gy[iy+1]-gy[iy])
        weights = np.array([(1-a)*(1-b), a*(1-b), (1-a)*b, a*b])
        yy, xx = np.array([iy,iy,iy+1,iy+1]), np.array([ix,ix+1,ix,ix+1])
        positive = weights > 0
        interpolated[row] = np.count_nonzero(positive) > 1
        for key in VARIABLES:
            corners = meteorology[key].values[safe[row], yy, xx]
            valid = meteorology[f"{key}_valid"].values[safe[row], yy, xx].astype(bool)
            if valid[positive].all() and np.isfinite(corners[positive]).all():
                values[key][row] = np.dot(weights[positive], corners[positive])
    for key in VARIABLES:
        flags[f"qc_{key}_missing"] = ~np.isfinite(values[key])
    held = usable & (age > 0)
    ux, vy = np.full(count, np.nan), np.full(count, np.nan)
    if coords_valid.any():
        ux[coords_valid], vy[coords_valid] = project_wind(lon[coords_valid], lat[coords_valid],
            values["u10"][coords_valid], values["v10"][coords_valid])
    flags["qc_projected_wind_missing"] = ~np.isfinite(ux) | ~np.isfinite(vy)
    ready = ~np.logical_or.reduce([value for key,value in flags.items() if require_blh or key != "qc_blh_missing"])
    out = xr.Dataset(coords={"observation": np.arange(count)})
    for key, value in dict(sensor_id=frame.sensor_id.to_numpy(str),
        timestamp_original=frame.timestamp.astype(str).to_numpy(str), time=clock,
        longitude=lon, latitude=lat, x=x, y=y, pm25_observed=pm,
        pm25=np.where(np.isfinite(pm) & (pm >= 0), pm, np.nan),
        meteorology_time=met_time, meteorology_age_seconds=age,
        meteorology_held=held.astype(np.int8),
        spatially_interpolated=interpolated, u_x=ux, v_y=vy,
        ready_for_model=ready.astype(np.int8), **values).items():
        out[key] = ("observation", value)
    for name, mask in flags.items():
        out[name] = ("observation", mask.astype(np.int8), {"flag_values": [0,1], "flag_meanings": "clear flagged"})
    units = VARIABLES | dict(pm25="ug m-3", pm25_observed="ug m-3", longitude="degrees_east",
        latitude="degrees_north", x="m", y="m", u_x="m s-1", v_y="m s-1", meteorology_age_seconds="s")
    for name, unit in units.items():
        out[name].attrs["units"] = unit
    for name in ("time", "meteorology_time"):
        out[name].attrs["timezone"] = "UTC"
    out.u10.attrs["vector_basis"] = "true eastward"
    out.v10.attrs["vector_basis"] = "true northward"
    out.u_x.attrs["vector_basis"] = "EPSG:32643 easting coordinate velocity"
    out.v_y.attrs["vector_basis"] = "EPSG:32643 northing coordinate velocity"
    out = out.set_coords(["time", "longitude", "latitude", "x", "y"])
    out["crs"] = xr.DataArray(0, attrs=PROJECTED_CRS.to_cf())
    out.attrs = dict(crs="EPSG:32643", geographic_crs="EPSG:4326", time_basis="UTC",
        temporal_method="latest valid-time hour <= observation; missing hour not skipped",
        spatial_method="bilinear in native longitude/latitude, strict valid positive-weight corners",
        missing_policy="retain rows and NaNs with explicit QC; no imputation",
        purpose=purpose, pm25_timestamp_meaning=timestamp_meaning,
        leakage_limit="retrospective reanalysis; not a real-time availability guarantee",
        naive_sensor_timezone=str(sensor_timezone), max_age_seconds=float(max_age_seconds), require_blh=int(require_blh))
    qc = dict(rows=count, ready_for_model=int(ready.sum()), not_ready=int((~ready).sum()),
        flag_counts={key:int(mask.sum()) for key,mask in flags.items()},
        spatially_interpolated_rows=int(interpolated.sum()), held_meteorology_rows=int(held.sum()),
        missing_era5_hours=int((meteorology.era5_time_present.values == 0).sum()),
        era5_invalid_cells={key:int((meteorology[f"{key}_valid"].values == 0).sum()) for key in VARIABLES},
        notes="Counts overlap; invalid rows are retained. Optional BLH policy is in configuration.")
    return out, qc
