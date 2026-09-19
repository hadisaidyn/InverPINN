"""Strict local ERA5 NetCDF ingestion; no retrieval, temporal fill or gap repair."""

import hashlib
from datetime import datetime
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from inverpinn.data.coordinates import PROJECTED_CRS, to_projected, project_wind

VARIABLES = {"u10": "m s-1", "v10": "m s-1", "t2m": "K", "blh": "m"}


def utc_times(values, naive_timezone=None):
    """Normalize full timestamps to timezone-aware UTC, with no timezone guess.

    Naive sensor timestamps require an explicit IANA timezone (e.g.
    Asia/Almaty), so historical offset changes are honored. Ambiguous and
    nonexistent local times raise. Text dates must start YYYY-MM-DD; yearless
    AirKaz archive labels are intentionally unsupported. Aware offsets prevail.
    """
    result = []
    for value in values:
        if isinstance(value, str):
            if not re.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]|$)", value):
                raise ValueError("Timestamp text must include a full ISO date YYYY-MM-DD.")
        elif not isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
            raise ValueError("Timestamps must be ISO text or datetime values, not numeric epochs.")
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError("Missing timestamp cannot be aligned safely.")
        if timestamp.tzinfo is None:
            if naive_timezone is None:
                raise ValueError("Naive timestamps require an explicit timezone.")
            timestamp = timestamp.tz_localize(naive_timezone, ambiguous="raise", nonexistent="raise")
        result.append(timestamp.tz_convert("UTC"))
    return pd.DatetimeIndex(result)


def cache_raw(path, cache_dir):
    """Copy an input byte-for-byte into a content-addressed raw cache.

    Never overwrite an existing different file. Returns provenance including
    SHA-256; a repeated import verifies and reuses the cached copy.
    """
    source = Path(path).resolve()
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}{source.suffix.lower()}"
    if destination.exists():
        with destination.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Raw cache corruption: {destination}")
    else:
        shutil.copyfile(source, destination)
    return dict(original=str(source), cached=str(destination), sha256=digest)


def validate_bbox(bbox):
    """Validate [west,south,east,north] WGS84 degrees within UTM zone 43N."""
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        raise ValueError("bbox must be finite [west,south,east,north].")
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise ValueError("bbox bounds must increase.")
    to_projected([west, east], [south, north])
    return west, south, east, north


def ingest_era5(paths, *, bbox, start, end, raw_cache_dir):
    """Return (processed Dataset, provenance) for local hourly ERA5 NetCDF.

    Required variables: u10/v10 (m/s), t2m (K); optional blh (m). Accept
    time or valid_time, descending latitude, and 0..360 longitudes. Require
    deterministic hourly single-level data on one regular geographic grid;
    expver/member/forecast-step dimensions are rejected, never averaged.
    Files are temporal chunks with identical grids and disjoint timestamps.

    Keep a bracketing spatial halo around bbox for bilinear sampling. Reindex
    to every hour from floor(start) through floor(end), inclusive; the left
    support hour is explicit for non-hourly sensor times. Missing hours/cells
    remain NaN with validity flags. No temporal interpolation or extrapolation.
    ERA5 CF time is UTC. Output adds 2-D projected x/y and projected velocities
    while preserving latitude/longitude and original east/north components.
    """
    west, south, east, north = validate_bbox(bbox)
    begin, finish = utc_times([start, end])
    if finish < begin:
        raise ValueError("end must not precede start.")
    if not paths:
        raise ValueError("At least one ERA5 NetCDF input is required.")
    chunks, provenance = [], []
    for path in paths:
        if Path(path).suffix.lower() not in (".nc", ".nc4", ".netcdf"):
            raise ValueError("Only decoded ERA5 NetCDF is supported; convert GRIB explicitly first.")
        record = cache_raw(path, raw_cache_dir)
        provenance.append(record)
        with xr.open_dataset(record["cached"]) as raw:
            # Select only the requested region/time before loading large arrays.
            ds = raw
            for axis, lo, hi in (("longitude", west, east), ("latitude", south, north)):
                if axis not in ds.coords or ds[axis].dims != (axis,):
                    raise ValueError(f"Require one-dimensional ERA5 coordinate {axis}.")
                values = np.asarray(ds[axis].values, float)
                if axis == "longitude":
                    values = ((values+180) % 360)-180
                order = np.argsort(values)
                sorted_values = values[order]
                if len(values) < 2 or not np.isfinite(values).all() or lo < sorted_values[0] or hi > sorted_values[-1]:
                    raise ValueError("ERA5 grid must cover bbox with bracketing cells.")
                left = max(0, np.searchsorted(sorted_values, lo, side="right")-1)
                right = min(len(values)-1, np.searchsorted(sorted_values, hi, side="left"))
                ds = ds.isel({axis: order[left:right+1]})
            time_name = "valid_time" if "valid_time" in ds.dims else "time"
            if time_name not in ds.coords or ds[time_name].dims != (time_name,) or ds[time_name].dtype.kind != "M":
                raise ValueError("Require decoded one-dimensional standard-calendar ERA5 time.")
            time_values = ds[time_name].values
            original_times = pd.DatetimeIndex(time_values)
            if original_times.hasnans or original_times.has_duplicates or not original_times.equals(original_times.floor("h")):
                raise ValueError("ERA5 times must be unique, nonmissing exact hours.")
            selected = (time_values >= begin.floor("h").tz_localize(None).to_datetime64()) & (time_values <= finish.floor("h").tz_localize(None).to_datetime64())
            ds = ds.isel({time_name: np.flatnonzero(selected)})
            ds = ds[[key for key in ("u10","10u","v10","10v","t2m","2t","blh") if key in ds]].load()
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        for axis in ("time", "latitude", "longitude"):
            if axis not in ds.coords or ds[axis].dims != (axis,):
                raise ValueError(f"Require one-dimensional ERA5 coordinate {axis}.")
        if ds.time.dtype.kind != "M":
            raise ValueError("Require decoded standard-calendar datetime64 ERA5 time.")
        if ds.time.attrs.get("timezone", "UTC") != "UTC":
            raise ValueError("ERA5 timestamps must be UTC.")
        stamps = pd.DatetimeIndex(ds.time.values)
        if stamps.hasnans or stamps.has_duplicates or not stamps.equals(stamps.floor("h")):
            raise ValueError("ERA5 times must be unique, nonmissing exact hours.")
        lon = np.asarray(ds.longitude.values, float)
        lat = np.asarray(ds.latitude.values, float)
        if not np.isfinite(lon).all() or not np.isfinite(lat).all() or np.any(np.abs(lat) > 90):
            raise ValueError("Invalid ERA5 geographic coordinates.")
        ds = ds.assign_coords(longitude=((lon+180) % 360)-180).sortby(["latitude", "longitude", "time"])
        for axis in ("longitude", "latitude"):
            values = ds[axis].values
            if len(values) < 2 or np.any(np.diff(values) <= 0) or not np.allclose(np.diff(values), np.diff(values)[0]):
                raise ValueError("ERA5 requires unique regular latitude/longitude axes of length >=2.")
        for axis, lo, hi in (("longitude", west, east), ("latitude", south, north)):
            values = ds[axis].values
            if lo < values[0] or hi > values[-1]:
                raise ValueError("ERA5 grid must cover bbox with bracketing cells; no spatial extrapolation.")
            left = max(0, np.searchsorted(values, lo, side="right")-1)
            right = min(len(values)-1, np.searchsorted(values, hi, side="left"))
            ds = ds.isel({axis: slice(left, right+1)})
        data = {}
        for name, unit in VARIABLES.items():
            aliases = {"u10": ["u10", "10u"], "v10": ["v10", "10v"], "t2m": ["t2m", "2t"], "blh": ["blh"]}[name]
            found = [key for key in aliases if key in ds]
            if not found and name == "blh":
                continue
            if len(found) != 1:
                raise ValueError(f"Require exactly one ERA5 variable for {name}.")
            variable = ds[found[0]]
            if set(variable.dims) != {"time", "latitude", "longitude"}:
                raise ValueError(f"{name}: unsupported dimensions; explicitly resolve expver/member/step first.")
            allowed_units = {"m s-1", "m s**-1", "m/s"} if name in ("u10", "v10") else {unit}
            if variable.attrs.get("units") not in allowed_units:
                raise ValueError(f"{name}: missing/unsupported units {variable.attrs.get('units')!r}.")
            data[name] = variable.transpose("time", "latitude", "longitude").astype(float)
        clean = xr.Dataset(data)
        if "blh" not in clean:
            clean["blh"] = xr.full_like(clean.t2m, np.nan)
        chunks.append(clean)
    for chunk in chunks[1:]:
        for axis in ("latitude", "longitude"):
            if not np.array_equal(chunk[axis], chunks[0][axis]):
                raise ValueError("ERA5 chunks must share the same grid.")
    ds = xr.concat(chunks, dim="time", join="exact").sortby("time")
    if pd.DatetimeIndex(ds.time.values).has_duplicates:
        raise ValueError("Overlapping ERA5 timestamps: resolve explicitly, not last-file-wins.")
    ds["era5_time_present"] = xr.DataArray(np.ones(ds.sizes["time"], dtype=np.int8), dims="time", coords={"time": ds.time})
    hours = pd.date_range(begin.floor("h"), finish.floor("h"), freq="h").tz_localize(None)
    ds = ds.reindex(time=hours)
    ds["era5_time_present"] = ds.era5_time_present.fillna(0).astype(np.int8)
    for name, unit in VARIABLES.items():
        valid = np.isfinite(ds[name])
        if name in ("t2m", "blh"):
            valid = valid & (ds[name] > 0 if name == "t2m" else ds[name] >= 0)
        ds[f"{name}_valid"] = valid.astype(np.int8)
        ds[name].attrs = dict(units=unit, grid_mapping="crs")
    lon, lat = np.meshgrid(ds.longitude.values, ds.latitude.values)
    x, y = to_projected(lon, lat)
    ds["x"] = (("latitude", "longitude"), x, {"units": "m", "standard_name": "projection_x_coordinate"})
    ds["y"] = (("latitude", "longitude"), y, {"units": "m", "standard_name": "projection_y_coordinate"})
    u, v = project_wind(lon, lat, ds.u10.where(ds.u10_valid).values, ds.v10.where(ds.v10_valid).values)
    ds["u_x"] = (("time", "latitude", "longitude"), u, {"units": "m s-1", "grid_mapping": "crs"})
    ds["v_y"] = (("time", "latitude", "longitude"), v, {"units": "m s-1", "grid_mapping": "crs"})
    ds["crs"] = xr.DataArray(0, attrs=PROJECTED_CRS.to_cf())
    ds.longitude.attrs = dict(units="degrees_east", standard_name="longitude")
    ds.latitude.attrs = dict(units="degrees_north", standard_name="latitude")
    ds.time.attrs = dict(timezone="UTC", standard_name="time")
    ds.u10.attrs["vector_basis"] = "true eastward"
    ds.v10.attrs["vector_basis"] = "true northward"
    ds = ds.set_coords(["x", "y"])
    ds.attrs = dict(crs="EPSG:32643", geographic_crs="EPSG:4326", time_basis="UTC",
        bbox_west_south_east_north=list(bbox), requested_start_utc=begin.isoformat(), requested_end_utc=finish.isoformat(),
        spatial_grid="native latitude/longitude with 2D UTM coordinates, not a regular UTM raster",
        gap_policy="No fill; explicit hourly gaps and validity flags", purpose="retrospective_reanalysis")
    return ds, provenance
