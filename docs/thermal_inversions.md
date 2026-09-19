# Thermal inversion diagnostic: definition and evidence

Research and method decision recorded 2026-09-19, before implementation.

## Scientific basis

1. [NOAA/NWS glossary, temperature inversion](https://forecast.weather.gov/glossary.php?word=inversion)
   defines an inversion as air temperature increasing with height. A surface-based
   layer starts at the surface; an elevated layer starts above it.
2. [Fang et al. (2026), Atmospheric Chemistry and Physics 26, 4089–4104,
   doi:10.5194/acp-26-4089-2026](https://acp.copernicus.org/articles/26/4089/2026/),
   section 2.2, detects positive temperature gradients in radiosonde profiles and
   reports layer strength and thickness. That study additionally uses 100 m and
   0.5 K filters, spline processing, and a 2000 m base-height restriction. These
   are study-specific operational choices, not the physical definition. We do
   **not** transfer these filters, smoothing, or its near-surface classification
   to Almaty without instrument/resolution validation. Its China pollution
   associations are not evidence about a particular Almaty episode.
3. [ECMWF/Copernicus ERA5 pressure-level dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)
   provides hourly upper-air fields on 37 pressure levels and a 0.25 degree grid.
   It is an assimilating reanalysis, not direct local measurements. Both the
   finite vertical sampling and smoothed terrain limit shallow-inversion detection.

All three pages were read through gstack browse. The AMS glossary was blocked;
it is not used as supporting evidence. No atmospheric data were downloaded.

## Chosen definition

For adjacent supplied levels with valid **actual air temperature** T (K) and
height above ground h (m), compute g = (T[j+1]-T[j])/(h[j+1]-h[j]).
A resolved inversion segment has g > 0. Exactly isothermal segments are not
inversions. Join consecutive positive segments into one layer; never bridge a
missing, isothermal, or decreasing segment. Report base, top, thickness,
temperature rise, mean gradient, and largest level spacing.

This zero sign boundary follows the physical definition; no tuned minimum
strength, depth, duration, wind, PM2.5, or boundary-layer-height threshold is used.
The output is a **resolved inversion candidate**, not a probability or a
measurement-error-certified detection. Tiny positive differences remain in the
table and need interpretation against documented instrument/model uncertainty.
Potential temperature increasing with height indicates static stability, which
is not the same condition as actual temperature increasing with height.

Study height bounds are an explicit input defining the investigated domain,
not a severity threshold. Bounds must be justified for the study and are not
silently selected here. An inversion above these bounds is not assessed.

Only a layer beginning at an explicitly supplied 0 m reference is labelled
surface_based. A layer starting after a valid nonpositive segment is elevated.
A layer starting at the lowest available level or after missing data has
unresolved_base: it might extend toward the surface. A 2 m air measurement must
keep its 2 m height; it is not a ground/skin temperature. Layer tops without a
following valid nonpositive segment are flagged unresolved as well.

## Input contract and preparation

The diagnostic takes processed NetCDF with dimensions `time, site, level`:

- `air_temperature(time,site,level)`, units `K`, standard_name `air_temperature`;
- `height_agl(time,site,level)`, units `m`, positive up, same levels;
- `longitude(site)` and `latitude(site)` in WGS84 degrees, site identifiers;
- ascending unique timestamps with dataset `time_basis: UTC`;
- dataset `data_kind: real` or `synthetic_test`, and `source_description`.

Every profile's finite heights must ascend strictly. NaNs remain missing;
infinities and nonpositive Kelvin temperatures are rejected. Keep missing
internal levels as NaNs, rather than deleting them and creating artificial
adjacency. Below-ground levels are excluded and counted. No temperature, time,
or height interpolation is performed. Preserve raw inputs outside processed
outputs. The runner records the input and code hashes and resolved configuration.

For ERA5, obtain co-located temperature and vertical height information, not
just t2m/u10/v10/blh. Preparation must document terrain, vertical datum and any
geopotential-to-height conversion; pressure is not height. Exclude below-ground
or extrapolated pressure levels using surface pressure and terrain before export,
retaining invalid levels as NaN. Higher-resolution model-level profiles or
quality-controlled radiosondes may resolve structures missed by pressure levels.
This stage consumes prepared profiles; it does not invent a download API or
implement undocumented GRIB/model-level height reconstruction.

## Events, coverage and temporal limits

At each site and scheduled timestamp, record `detected`,
`not_detected_in_sampled_layers`, or `insufficient_data`. The latter includes
missing timestamps and profiles with no valid adjacent in-domain pair.
Partial profiles can support detection but cannot rule out other layers.
`not_detected_in_sampled_layers` never means an inversion is physically absent
between, below or above sampled levels. Coverage is reported separately.

An event is a consecutive run of scheduled samples containing at least one
candidate at the same site. This is an **any-layer episode**, not tracking the
same air mass or vertical layer. No gaps are bridged. First/last sampled times,
sample count and their elapsed span are reported. A single observation has
zero elapsed span, not zero physical duration. Exact onset, end and continuous
persistence between samples are unknown. Neighbouring statuses are saved so
missing coverage is distinguishable from a negative sampled diagnostic.
UTC is the storage basis; event tables also show Asia/Almaty local timestamps
using date-aware timezone conversion, not a fixed UTC offset.

## Study-period availability

No verified real Almaty vertical profiles or confirmed study dates are present
in this repository. Existing meteorology is single-level synthetic method-check
data; the 2020 dates in `configs/almaty.yaml` are placeholders. Consequently, a
real inversion-event table cannot yet be calculated. The default configuration
must fail with an explicit availability report, not emit an empty table that
could be mistaken for zero events. Synthetic event tables are method checks only.

This diagnostic does not prove pollution causation or identify emitters. Use it
as retrospective meteorological context; event grouping uses later samples and
must not be used as a real-time prediction feature without redesign.
