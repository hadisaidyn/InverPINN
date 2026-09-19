# Exploratory Almaty winter case study

## Current status

The pipeline is implemented and exercised on synthetic fixtures. **No verified
real Almaty dataset is currently present in this workspace. No observed winter
episode has been selected, no real-data fit has been performed, and no real
emitter/source map is claimed.** The existing ingestion demo is explicitly
synthetic and cannot be passed off as real by this runner.

Supply the previous preprocessing stage's `modeling.nc`, `meteorology.nc` and
`provenance.json` from authorized real inputs. Confirm AirKaz usage rights and
resolve its date/timezone and coordinate issues first; see
[AirKaz access research](airkaz_data_access.md) and
[the ingestion contract](meteorology_pipeline.md).

```bash
python scripts/run_real_case_study.py --config configs/real_case_study.yaml
```

Edit the input paths and permission declaration before running. The default
configuration intentionally remains blocked. `permission_confirmed: true` is a
user attestation, not independent legal verification. The runner checks input
hashes against the paired preprocessing manifest and rejects known synthetic,
placeholder or unverified descriptions in real-data mode. No download, scraping,
external publication or contact with providers occurs.

## Predeclared winter-episode selection

1. Split QC-valid sensor IDs using a fixed seed (42), withholding 20% of station
   IDs, at least one. Require at least three training stations. This is a
   within-episode spatial holdout, not a forecasting split.
2. Among training stations only, average each station's readings within a UTC
   hour, then take the median across stations. This avoids giving higher-rate
   stations more weight merely because they report more often.
3. Enumerate UTC-hour-aligned 24-hour windows. Every hour must fall in December,
   January or February under `Asia/Almaty` historical timezone rules and have
   at least three reporting training stations plus complete regional ERA5 wind
   and temperature coverage. No gaps are filled.
4. Select the largest mean of those hourly medians; ties choose the earliest
   start. End time is exclusive. Save every admissible window and its score in
   `episode_candidates.csv`, and the chosen window/rule in `episode.json`.

This defines a high-pollution window relative to the supplied data, **not a
health-standard exceedance or a independently confirmed pollution event**.
Selection is retrospective and biased toward higher reported concentrations.
It does not establish how representative the episode is of Almaty winters.
Held-out PM values never select the episode or its background prior, train the
network, choose checkpoints or tune hyperparameters. Tests change withheld
values and require bitwise-identical source maps.

## Physical model and priors

The square projected domain encloses the supplied study bounding box. Retain
EPSG:32643 easting/northing metres and original longitude/latitude. Let L be
the square side length, T the configured time scale, and C* the configured
concentration scale. The network uses:

`xi=(x-x0)/L`, `eta=(y-y0)/L`, `tau=(t-t0)/T`, `c=C/C*`.

The physics residual is

`c_tau + (u*T/L)c_xi + (v*T/L)c_eta - (D*T/L²)(c_xixi+c_etaeta) - s`.

The estimated dimensional forcing is `S=s*C*/T`, measured in **µg m⁻³ s⁻¹**.
These chain-rule factors are tested against analytical derivatives. The source
is a sum of K positive, stationary Gaussians, with bounded trainable locations
and positive trainable amplitudes. The initial configuration assumes K=2 and
5 km widths. These are basis-function hypotheses, **not evidence of two actual
emitters**. Estimated components must not be labeled as known facilities.

ERA5 projected winds are reduced to latitude-area-weighted regional means at
each hour, using native grid nodes inside the study box and excluding the
interpolation halo. They are piecewise constant in time and uniform in space.
No missing hour is bridged. This is an explicit approximation to avoid claiming
unresolved neighborhood-scale winds. A fixed effective diffusivity of 100 m²/s
is an assumption in the default configuration, not an ERA5 measurement.

The first training hour's median of station means defines a constant background
for both the initial field and all four boundaries. These soft constraints are
empirical priors; they are not observed spatial initial/boundary conditions.
All weights, scales, diffusion, Gaussian widths, initialization, training budget
and seeds are saved. Temperature and BLH are diagnostic context only, not inputs
to an invented diffusivity or vertical mixing formula in this initial model.

The PDE omits topography, spatial wind variation, chemistry, deposition,
vertical mixing, time-varying emissions and external pollutant inflow beyond
the simple boundary prior. It also approximates the projection locally as a
planar metric. Positive inferred forcing can compensate for these omissions,
sensor calibration error or background error. It cannot be converted to a
validated mass emission flux without additional physical information.

## Outputs and language policy

The generated report has separate sections:

| Category | Included content | Claims explicitly excluded |
|---|---|---|
| Measured facts | Reported PM readings, their coordinates/times, coverage, QC and input provenance | Ground-truth source locations, independent calibration verification or legal status |
| Model estimates | Conditional effective-source intensity and seed spread; sensor reconstruction metrics | Calibrated source likelihood/probability, mass emission rates, dense-field or localization accuracy |
| Interpretation | Possible explanations, confounders and follow-up validation needs | Attribution to a facility/owner, illegal or undocumented activity, regulatory violations |

ERA5 is reanalysis, not a direct local meteorological measurement, and is
described in the assumptions/model context rather than as measured PM facts.
Known synthetic runs are labeled **SYNTHETIC METHOD CHECK — NOT REAL OBSERVATIONS**.

Saved products include:

- `source_intensity.nc`: per-seed source maps, their mean and sample SD, physical
  units, projected/geographic coordinates, episode dates and interpretation limits.
- `source_intensity.png`: effective-intensity map and optimizer-seed SD map,
  with training/held-out sensor markers. No addresses or facility labels.
- `estimated_components.csv`: explicitly estimated Gaussian component parameters,
  not an inventory of observed sources.
- `sensor_predictions.csv`, `sensor_metrics.csv`: per-run train/held-out MAE,
  RMSE and relative L2 against sensor readings. Negative concentration predictions
  are counted rather than clipped. There is no dense observed concentration
  reference, so no dense reconstruction or source-localization error is reported.
- Per-seed checkpoints, optimizer states, loss curves and epoch-by-epoch logs,
  including source estimates. Every seed uses the final fixed-budget checkpoint;
  held-out errors do not select a run or trigger retraining.
- Selected observations, hourly regional winds, episode candidates, observation
  summary, exact resolved configuration, input provenance and code hashes.

## Uncertainties and responsible interpretation

Three optimizer seeds expose some fitting instability, not calibrated uncertainty.
Source components are permutation-symmetric; the aggregate source maps are
summarized instead of averaging arbitrary component labels. Narrow seed spread
does not imply a uniquely identified source or adequate physical model.

Before treating a real map as evidence even for scientific source hypotheses,
evaluate sensitivity to diffusion, boundary/background assumptions, K/widths,
sensor subsets, episode choice and meteorological representation. Independent
field measurements and emissions records are required for attribution. These
follow-up studies are not silently performed or claimed by this first pipeline.

Held-out-station performance measures spatial reconstruction within a selected
episode, not causal inference, future forecasting or source identification.
Reanalysis contains assimilation/revision information unavailable in real time.
Both omitted physics and errors in the data may produce apparent hotspots.
No map from this pipeline establishes illegal, undocumented or otherwise
noncompliant emissions. Report the assumptions beside every published map.
