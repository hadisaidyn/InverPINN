"""Resolved thermal inversions from actual-temperature profiles, without tuning.

The NOAA/NWS definition is dT/dh > 0. See docs/thermal_inversions.md for
scientific references, sampling limitations and the event convention. This is
not a pollution model or a probability estimate. All reported times are UTC.
"""

import numpy as np
import pandas as pd
import xarray as xr

from inverpinn.data.era5 import utc_times


LAYER_COLUMNS = ["site", "time_utc", "longitude", "latitude", "layer_index",
    "base_m_agl", "top_m_agl", "depth_m", "strength_K", "gradient_K_per_m",
    "max_level_spacing_m", "base_type", "top_unresolved"]
COVERAGE_COLUMNS = ["site", "time_utc", "longitude", "latitude", "status",
    "valid_pairs", "eligible_levels", "invalid_levels", "below_ground_levels",
    "partial_profile", "layer_count", "strongest_layer_K"]
EVENT_COLUMNS = ["site", "longitude", "latitude", "start_utc", "last_detected_utc",
    "start_local", "last_detected_local", "sample_count", "sample_span_hours",
    "max_strength_K", "max_depth_m", "min_base_m_agl", "max_level_spacing_m",
    "partial_profile_samples", "preceding_status", "following_status"]


def profile_layers(temperature, height_agl, *, height_bounds_m):
    """Return layers and coverage for one already vertically ordered profile.

    Imagine climbing a ladder with a thermometer. If it gets warmer as you
    climb, that section is inverted. We compare neighbouring rungs only. A
    missing reading cannot tell us what happened across that rung.

    T is actual air temperature in K; h is metres above ground. A layer has
    strength T_top-T_base (K), depth h_top-h_base (m), and mean gradient in K/m.
    The test is strictly positive, with no arbitrary noise/severity cutoff.
    """
    t, h = np.asarray(temperature, dtype=float), np.asarray(height_agl, dtype=float)
    bounds = np.asarray(height_bounds_m, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 <= bounds[0] < bounds[1]:
        raise ValueError("height_bounds_m must be explicit finite 0 <= lower < upper.")
    if t.ndim != 1 or t.shape != h.shape or t.size < 2:
        raise ValueError("Temperature and height need matching 1D profiles with >=2 levels.")
    if np.isinf(t).any() or np.isinf(h).any() or np.any(t[np.isfinite(t)] <= 0):
        raise ValueError("Reject infinities and nonpositive Kelvin temperatures; use NaN for missing.")
    if np.any(np.diff(h[np.isfinite(h)]) <= 0):
        raise ValueError("Finite heights must be strictly increasing; do not silently reorder levels.")
    valid = np.isfinite(t) & np.isfinite(h) & (h >= bounds[0]) & (h <= bounds[1])
    pairs = valid[:-1] & valid[1:]
    positive = pairs & (np.diff(t) > 0)
    layers = []
    j = 0
    while j < len(positive):
        if not positive[j]:
            j += 1
            continue
        base = j
        while j < len(positive) and positive[j]:
            j += 1
        top = j
        depth, strength = h[top]-h[base], t[top]-t[base]
        if h[base] == 0:
            base_type = "surface_based"
        elif base > 0 and pairs[base-1]:
            base_type = "elevated"
        else:
            base_type = "unresolved_base"
        layers.append(dict(layer_index=len(layers), base_m_agl=float(h[base]),
            top_m_agl=float(h[top]), depth_m=float(depth), strength_K=float(strength),
            gradient_K_per_m=float(strength/depth),
            max_level_spacing_m=float(np.diff(h[base:top+1]).max()),
            base_type=base_type, top_unresolved=bool(top == len(pairs) or not pairs[top])))
    # Missing heights could belong inside the study domain, so count conservatively.
    invalid = ~np.isfinite(h) | ((h >= bounds[0]) & (h <= bounds[1]) & ~np.isfinite(t))
    qc = dict(status="detected" if layers else
              "not_detected_in_sampled_layers" if pairs.any() else "insufficient_data",
        valid_pairs=int(pairs.sum()), eligible_levels=int(valid.sum()),
        invalid_levels=int(invalid.sum()), below_ground_levels=int(np.sum(h < 0)),
        partial_profile=bool(invalid.any()), layer_count=len(layers),
        strongest_layer_K=max((row["strength_K"] for row in layers), default=np.nan))
    return layers, qc


def diagnose_inversions(dataset, *, start, end, cadence_seconds, height_bounds_m,
                        local_timezone="Asia/Almaty"):
    """Diagnose each column separately and group consecutive detected samples.

    Input uses the documented NetCDF contract (time, site, level). The requested
    interval is inclusive and must lie on an explicit sampling schedule. Missing
    times become missing profiles, never interpolation. Event span is last minus
    first observation, not an inferred physical duration. No spatial averaging
    of temperatures is done before detection.
    """
    ds = dataset
    for name in ("air_temperature", "height_agl"):
        if name not in ds or ds[name].dims != ("time", "site", "level"):
            raise ValueError(f"{name}(time,site,level) required; surface-only meteorology is insufficient.")
    if ds.air_temperature.attrs.get("units") != "K" or ds.height_agl.attrs.get("units") != "m":
        raise ValueError("Profile units must be K and m AGL.")
    if ds.air_temperature.attrs.get("standard_name") != "air_temperature":
        raise ValueError("Actual air_temperature required, not potential/skin temperature.")
    if ds.height_agl.attrs.get("positive") != "up" or ds.attrs.get("time_basis") != "UTC":
        raise ValueError("Explicit positive=up height_agl and time_basis=UTC required.")
    if ds.attrs.get("data_kind") not in ("real", "synthetic_test") or not ds.attrs.get("source_description"):
        raise ValueError("data_kind and source_description required.")
    if ds.sizes["site"] < 1 or ds.sizes["level"] < 2:
        raise ValueError("At least one site and two levels required.")
    sites = list(map(str, ds.site.values))
    if len(set(sites)) != len(sites) or any(not s.strip() for s in sites):
        raise ValueError("Nonempty unique site identifiers required.")
    for name, bounds in (("longitude", (-180, 180)), ("latitude", (-90, 90))):
        if name not in ds or ds[name].dims != ("site",):
            raise ValueError(f"{name}(site) required.")
        values = ds[name].values
        if not np.isfinite(values).all() or np.any((values < bounds[0]) | (values > bounds[1])):
            raise ValueError(f"Invalid WGS84 {name}.")
    period = utc_times([start, end])
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0 or period[1] < period[0]:
        raise ValueError("Positive cadence and ordered UTC study period required.")
    step = pd.Timedelta(seconds=cadence_seconds)
    if step.value <= 0 or (period[1]-period[0]).value % step.value:
        raise ValueError("Study endpoints must align with cadence.")
    times = pd.DatetimeIndex(ds.time.values)
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("Unique increasing finite profile timestamps required.")
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    # DatetimeIndex.asi8 uses its stored resolution (possibly microseconds),
    # whereas Timestamp.value and Timedelta.value are nanoseconds.
    times = times.as_unit("ns")
    selected = (times >= period[0]) & (times <= period[1])
    if np.any((times[selected].asi8-period[0].value) % step.value):
        raise ValueError("Profile timestamps do not match configured cadence; no silent resampling.")
    schedule = pd.date_range(period[0], period[1], freq=step)
    schedule.tz_convert(local_timezone)  # Validate even when no events exist.
    aligned = ds.assign_coords(time=times.tz_localize(None)).reindex(time=schedule.tz_localize(None))
    layers, coverage = [], []
    for si, site in enumerate(sites):
        where = dict(site=site, longitude=float(ds.longitude.values[si]), latitude=float(ds.latitude.values[si]))
        for ti, instant in enumerate(schedule):
            found, qc = profile_layers(aligned.air_temperature.values[ti, si],
                aligned.height_agl.values[ti, si], height_bounds_m=height_bounds_m)
            stamp = dict(time_utc=instant.isoformat())
            layers.extend(where | stamp | row for row in found)
            coverage.append(where | stamp | qc)
    layer_table = pd.DataFrame(layers, columns=LAYER_COLUMNS)
    coverage_table = pd.DataFrame(coverage, columns=COVERAGE_COLUMNS)
    events = []
    for site in sites:
        records = coverage_table.loc[coverage_table.site == site].to_dict("records")
        i = 0
        while i < len(records):
            if records[i]["status"] != "detected":
                i += 1
                continue
            first = i
            while i < len(records) and records[i]["status"] == "detected":
                i += 1
            a, b = records[first], records[i-1]
            start_t, last_t = pd.Timestamp(a["time_utc"]), pd.Timestamp(b["time_utc"])
            subset = layer_table[(layer_table.site == site) &
                layer_table.time_utc.between(a["time_utc"], b["time_utc"])]
            events.append(dict(site=site, longitude=a["longitude"], latitude=a["latitude"],
                start_utc=a["time_utc"], last_detected_utc=b["time_utc"],
                start_local=start_t.tz_convert(local_timezone).isoformat(),
                last_detected_local=last_t.tz_convert(local_timezone).isoformat(),
                sample_count=i-first, sample_span_hours=(last_t-start_t).total_seconds()/3600,
                max_strength_K=float(subset.strength_K.max()), max_depth_m=float(subset.depth_m.max()),
                min_base_m_agl=float(subset.base_m_agl.min()),
                max_level_spacing_m=float(subset.max_level_spacing_m.max()),
                partial_profile_samples=sum(r["partial_profile"] for r in records[first:i]),
                preceding_status=records[first-1]["status"] if first else "study_boundary",
                following_status=records[i]["status"] if i < len(records) else "study_boundary"))
    return layer_table, coverage_table, pd.DataFrame(events, columns=EVENT_COLUMNS)


def synthetic_profiles():
    """Exact five-hour method check: two events separated by a missing hour.

    Artificial values are not meteorological observations of Almaty. There are
    surface and elevated layers, an isothermal profile and a missing profile.
    """
    temperature = np.array([[270, 272, 274, 273, 275], [270, 271, 272, 271, 270],
                            [np.nan]*5, [275]*5, [270, 271, 272, 271, 270]], dtype=float)[:, None, :]
    heights = np.broadcast_to([0., 100., 200., 500., 1000.], temperature.shape).copy()
    return xr.Dataset(dict(
        air_temperature=(("time", "site", "level"), temperature,
                         dict(units="K", standard_name="air_temperature")),
        height_agl=(("time", "site", "level"), heights, dict(units="m", positive="up")),
        longitude=("site", [76.9]), latitude=("site", [43.25])),
        coords=dict(time=pd.date_range("2020-01-01", periods=5, freq="h"),
                    site=["synthetic_column"], level=np.arange(5)),
        attrs=dict(time_basis="UTC", data_kind="synthetic_test",
                   source_description="Manufactured temperature profiles; not observations."))
