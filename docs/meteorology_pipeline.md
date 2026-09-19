# ERA5 and PM2.5 modeling dataset

This stage is an offline, deterministic data-preparation pipeline. It does not
download ERA5, scrape AirKaz, train a network or automatically resolve the
AirKaz permissions/date issues described in [the access report](airkaz_data_access.md).

## Run

From the repository root, with requirements installed:

```bash
# Fully synthetic, reproducible demonstration, no external data required:
python scripts/run_meteorology_demo.py

# For authorized local inputs, edit configs/almaty.yaml first:
python scripts/prepare_modeling_dataset.py --config configs/almaty.yaml
```

The demo creates a small ERA5-shaped NetCDF and PM CSV using seed 42. These
are schema/interpolation fixtures, **not real atmospheric measurements or
forward-PDE ground truth**. One missing meteorology hour, one negative PM value
and one missing longitude intentionally demonstrate QC. All nine observations
remain; four are model-ready and five flagged. Use a new output directory for
another run; existing processed results are never overwritten.

## Obtaining compatible ERA5 inputs

Use the official [ERA5 hourly single-level catalogue and download form](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download).
On 19 September 2026 it offered Reanalysis and NetCDF4 (Experimental), with
account/login and licence acceptance before submitting a request. Select the
two 10 m wind components, 2 m temperature, and boundary-layer height when
available. Request hourly data and enough geographic coverage to bracket the
study boundary. The repository does not supply credentials or invent endpoints.

The supported input is decoded NetCDF (`.nc`, `.nc4`, `.netcdf`), not ZIP/GRIB.
Convert other formats explicitly, retaining originals and conversion provenance.
Acceptable variable names are `u10`/`10u`, `v10`/`10v`, `t2m`/`2t`, and `blh`.
Only one alias may be present per variable. Coordinates must be one-dimensional
`latitude`, `longitude`, and `time` or `valid_time`. Data arrays must contain
exactly these dimensions. Ensemble members, expver streams, pressure levels,
forecast reference-time/step grids and conflicting overlaps require an explicit
upstream selection, not an undocumented mean or first-entry selection.

Multiple files can represent disjoint time chunks on the same spatial grid;
each chunk must contain both winds and temperature. BLH can be absent in any
chunk. This version does not combine separate variable-only files. The selected
area/time is sliced before data arrays are loaded into memory. Input chunks
must use a regular latitude/longitude grid; descending latitude is reordered and
longitude is normalized to `[-180,180)`. A bracketing spatial halo is retained.

The [ECMWF ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)
describes hourly validity times, instantaneous parameter conventions and the
horizontal reference. Do not pass monthly means or accumulated fields into this
instantaneous-variable pipeline merely because their arrays look similar.

## Coordinates and vector components

- Input sensor coordinates: WGS84 geographic longitude/latitude in degrees,
  EPSG:4326. Longitude comes first, always.
- Modeling coordinates: WGS84 / UTM zone 43N, EPSG:32643, easting/northing in
  metres. The EPSG definition in pyproj's PROJ database covers 72–78°E,
  0–84°N, including Almaty. Central meridian 75°E, central scale 0.9996,
  false easting 500000 m. Coordinates are absolute, not normalized to `[0,1]`.
- Bounding boxes use `[west,south,east,north]`. The example `[76.5,43.0,77.3,43.5]`
  is configurable, not an administrative city boundary. This differs from
  CDS request ordering; never reuse this list as a CDS request without mapping.
- `to_projected(longitude,latitude)` and `to_wgs84(x,y)` provide both directions.
  Finite, in-zone inputs are required. Altitudes are not transformed. pyproj
  uses `always_xy=True` to prevent geographic axis-order surprises, as described
  in its [Transformer documentation](https://pyproj4.github.io/pyproj/stable/api/transformer.html).
- ERA5 is stored on its native geographic grid with auxiliary 2-D projected
  x/y coordinates. It is **not** silently resampled onto a uniform UTM raster.
  The merged dataset has projected coordinates at each sensor row and retains
  its original longitude/latitude.
- `u10` and `v10` retain true eastward/northward components. `u_x` and `v_y`
  are projected coordinate velocities: `[u_x,v_y]ᵀ = J [u10,v10]ᵀ`, where J
  is the local projection Jacobian. Symmetric one-metre WGS84 geodesic steps
  approximate J, incorporating both convergence and map scale. Tests compare
  these velocities to independently projected short geodesic displacements.

The existing unit-square synthetic PDE solver is not automatically compatible
with these dimensional coordinates or spatially varying meteorology. Integrating
them into a physical model requires consistent length/time scaling and a
separate decision about variable-wind advection/diffusion operators. No such
scientific-model change is hidden in this ingestion stage.

## Units and supported input sensor schema

| Field | Meaning | Stored unit |
|---|---|---|
| `u10`, `v10` | Wind 10 m above the surface, true east/north basis | m s-1 |
| `u_x`, `v_y` | Projected-coordinate wind velocities | m s-1 |
| `t2m` | Temperature 2 m above the surface | K |
| `blh` | Boundary-layer height | m |
| `pm25`, `pm25_observed` | PM2.5 concentration | ug m-3 |
| `longitude`, `latitude` | Original geographic coordinates | degrees_east, degrees_north |
| `x`, `y` | Projected coordinates | m |
| `meteorology_age_seconds` | Observation time minus candidate meteorology time | s |

ERA5 unit attributes are checked, not inferred from magnitudes. Wind accepts
`m s-1`, `m s**-1` or `m/s`; temperature requires `K`, BLH requires `m`.
Celsius files must be explicitly converted upstream with documented provenance.
BLH absence becomes NaN plus a flag, never an assumed height. Invalid Kelvin
values `<=0`, BLH `<0`, and nonfinite fields are flagged. Raw source values are
preserved in the processed meteorology; only valid values enter interpolation.
No arbitrary high/low wind or concentration threshold is imposed.

Sensor CSV has one row per observation and these required columns:

```csv
sensor_id,timestamp,longitude,latitude,pm25
sensor_01,2020-01-01T00:30:00Z,76.90,43.25,18.2
```

Sensor identifiers are read as strings. The configuration must specify
`pm25_units: ug m-3` and `timestamp_meaning: instantaneous`. Blank concentration
or coordinates become flagged missing values; malformed numeric strings fail.
Original concentration is retained in `pm25_observed`. Negative PM stays in that
field but has NaN in modeling `pm25` and is not model-ready; it is not clipped
to zero. Coordinates are retained per observation, including possible relocation.

## Time basis and future-information controls

ERA5 decoded CF `time`/`valid_time` is UTC. The requested start/end must have
explicit offsets. Sensor ISO timestamps with offsets are converted to UTC;
naive timestamps require a provider-confirmed IANA timezone in configuration.
`Asia/Almaty` uses historical timezone rules, **not a hard-coded current offset**.
Ambiguous/nonexistent local times, missing timestamps, yearless dates and
duplicate sensor/time pairs after conversion raise errors. They are not guessed
or averaged. Daily/centered averages are rejected because they need an interval
observation operator, not an instantaneous label. Original timestamp strings
are preserved. NetCDF datetime64 values are timezone-naive storage with explicit
UTC attributes and CF time encoding.

For a requested interval, hourly meteorology is reindexed from `floor(start)`
through `floor(end)`, inclusive. The left support hour is deliberately retained
for off-hour observations. Absent hours stay NaN and `era5_time_present=0`.
For each sensor row at t, the candidate is the latest grid hour **no later than
t**, with an explicit maximum age. Exact hour matches use that hour; off-hour
matches use a piecewise-constant prior hour and are flagged `meteorology_held`.
If that hour is absent or too old, no older good value is substituted.
`meteorology_time`/age describe the candidate even when QC prevents use.

The same hour is spatially interpolated to each sensor using bilinear weights
on longitude/latitude. Every positive-weight corner must be valid. Missing
zero-weight corners do not contaminate exact-grid-node matches. No temporal
linear interpolation, nearest-future matching, spatial extrapolation, temporal
backfill, mean imputation, smoothing, or full-dataset normalization occurs.
Tests change future meteorology and PM rows and require unchanged earlier
features. Train/validation/test splitting is deliberately not performed here;
later fitted preprocessing must use training data only.

**Important limit:** these controls prevent future *rows/valid times* entering
the merge. They cannot make ERA5 causal in a real-time availability sense.
ECMWF uses assimilation windows and publishes preliminary reanalysis after a
delay; later revisions may replace it. This implementation therefore accepts
only `purpose: retrospective_reanalysis` and rejects operational forecasting.
An operational study needs forecast vintages/issue times and verified availability
metadata, not an assumed publication lag. See the official
[ERA5 catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
and [assimilation/update documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation).

## Output, caching and QC

Raw files are copied byte-for-byte to `raw_cache_dir/era5/` and
`raw_cache_dir/pm25/`, named by SHA-256 plus extension. Reuse verifies the cache
hash; corruption raises. Original files are never modified. Processed output
must be a separate directory tree. Generated data remains ignored by Git.

Output directory contents:

- `modeling.nc`: the one clean, observation-indexed modeling dataset. All rows
  retained; explicit `ready_for_model` mask, QC flags, source timestamps,
  projected/geographic coordinates, fields and unit metadata.
- `meteorology.nc`: normalized regional meteorology, validity flags, hourly
  presence flags, and CRS metadata, including its spatial interpolation halo.
- `qc_summary.json` and `qc_report.md`: counts for invalid/missing PM,
  coordinates, out-of-box/interval rows, missing/stale meteorology, optional BLH,
  and missing/invalid source cells. Flag counts overlap; they are not additive.
- `qc_positions.png`: projected sensor positions colored by row readiness.
  Rows without projectable coordinates remain in QC but cannot appear on a map.
- `config.yaml`, `provenance.json`: resolved input paths, raw and output hashes,
  configuration/seed, code hashes, package/PROJ/timezone-provider metadata,
  source and user-supplied usage permission. A permission string is a record,
  not proof of a licence. No model is trained; these files plus `modeling.nc`
  form the data-preparation checkpoint.
- `failure.txt` if a run fails. Partial output is not a successful dataset.

All required fields must be valid for `ready_for_model=1`. Missing BLH is
non-blocking only when `require_blh=false`. There is no implicit row deletion:
downstream users must explicitly choose the readiness policy and account for
excluded rows. A dataset with zero ready rows is reported as such, not repaired.

Verification uses synthetic fixtures, not licensed AirKaz data. The installed
NetCDF4/NumPy 2.5 stack emitted binary-compatibility/deprecation warnings during
tests; tested reads, masked values and write/read round trips succeeded. Warnings
are not suppressed. Freeze and validate a compatible dependency environment
before a large real-data research run.
