# Historical AirKaz PM2.5 access

Checked 19 September 2026. Status: an official historical workbook is publicly
downloadable, but research/republication permission and temporal metadata still
need clarification. No scraper, undocumented API client, or real-data ingestion
pipeline has been written. The downloaded workbook was inspected read-only in a
temporary directory, not added to training data or redistributed in this repo.
The inspected workbook's SHA-256 was
`87118944b46a7e6538c9d2fe75ab9b584ffdfb86e18453345f70c60abb98b939`.

## Verified access routes

| Source / route | What is verified | Mechanism | Reuse status |
|---|---|---|---|
| [AirKaz homepage](https://airkaz.org/) and [Almaty page](https://airkaz.org/almaty.php) | Both link an Almaty daily PM2.5 archive described as March 2017–September 2020 | Follow the visible Excel download link | Site footer requires author consent for use of site materials; no open-data license was found on these pages |
| [Official XLSX file](https://airkaz.org/xls/daily_almaty_6oct20.xlsx) | File downloaded successfully, contains station metadata and daily station-value table | Ordinary browser download, no account/token needed in this check | Availability is not permission to republish or bulk reuse; obtain written scope of permission |
| [AirKaz Almaty live map](https://airkaz.org/almaty.php) | Public station popup and short hour-selector interface | View through the official website | Not a verified long-term historical download mechanism or documented public API |
| Owner-approved export / research agreement | Appropriate permission-seeking next step, **not a confirmed offered service** | Ask the AirKaz author/operator for an authorized export and terms | Availability, price, contact address, dates and export schema remain unverified |

The official site's footer credits Pavel and states in Russian that use of site
materials without the author's consent infringes copyright. This is a report
of the published notice, not a legal determination. No permissive license or
exception for academic work was verified. The hosting-support credit to GoHost
does not establish that the host can grant data rights. No email address or API
endpoint is inferred from domain names or page internals.

## Fields and coordinates in the official workbook

Inspected the downloaded file's sheet names, header rows and bounded metadata
ranges without editing it. Source: [published workbook](https://airkaz.org/xls/daily_almaty_6oct20.xlsx).

| Sheet | Observed fields / structure | Interpretation and limits |
|---|---|---|
| `sensors` | `id`, `name`, `city`, `lat`, `lng` (A1:E1) | Station identifier/name/city and numeric geographic coordinates. A formal CRS, accuracy, installation height, relocation history and period of validity are not supplied by these headers |
| `data` | `Row Labels`, numeric station-ID columns, `Grand Total`, trailing blank column | Wide table of daily values; join station columns to `sensors.id`. `Grand Total` is an aggregate, **not a sensor**. Blank entries occur and must not become zero |
| `coil_comp` | `date`, `pm2.5, мкг`, `ТЭЦ2, тонн`, `ТЭЦ3, тонн` | Ancillary comparison table, not a replacement for per-station observations; definitions of the plant quantities require clarification |
| `march_comp` | Day/month row labels and year columns 2017–2020 | Ancillary cross-year comparison, not a raw timestamped sensor export |

The publisher labels the archive as daily PM2.5 concentration; the live map
explicitly labels PM2.5 in µg/m³. The main data sheet's bare station-ID headers
do not themselves declare units, aggregation completeness rules, calibration,
quality flags or whether values were corrected. Confirm that the archive uses
the same concentration units before modeling.

Coordinate QA is necessary. For example, `sensors` includes station 38737023
(`new10`), city `Алматы`, with `(lat,lng)=(49.628,73.016)`, far from the Almaty
cluster near `(43.2,76.9)`. Do not silently move, relabel or drop it. Ask the
provider whether this reflects a city-label error, relocation or another issue.
Do not assume current map positions describe historical deployment positions.

The `sensors` worksheet advertises an extent exceeding one million rows even
though the first 100 rows contain only 49 nonempty rows including the header.
That extent is not a sensor count. A bounded read-only inspection was used after
a general workbook importer spent excessive time on the large declared range.

## Timestamps and historical coverage

- Homepage description: daily averages, March 2017–September 2020.
- In the actual `data` sheet, A2 is the **text** `22-Mar`; subsequent labels
  repeat day/month across year transitions (e.g. `31-Dec`, `1-Jan`). The year
  is not encoded in these main-table row labels. Reconstructing years from row
  order and the advertised date range would be an inference, not verified
  timestamp metadata.
- The last inspected labels extend from `30-Sep` through `7-Oct`, despite the
  homepage's September cutoff and the file name containing `6oct20`. Neither
  filename nor row ordering should be treated as authoritative full dates
  without provider confirmation.
- Daily aggregation timezone, day start/end convention, minimum valid samples,
  handling of missing hours and original sampling cadence were not verified.
  Do not assign UTC, a fixed offset, or today's Almaty offset retrospectively.
- The live map popup inspected on 19 September showed station 108, a name,
  AQI, current PM2.5, daily-average PM2.5, a temperature field that was unavailable,
  and an update timestamp such as `2026-09-19 17:14:16`. No timezone/UTC offset
  was stated in that popup. The hour selector does not establish arbitrary
  historical date access or a stable export interface.

These observations come from the [workbook](https://airkaz.org/xls/daily_almaty_6oct20.xlsx)
and [live Almaty page](https://airkaz.org/almaty.php). Daily averages alone are
too coarse to treat as instantaneous observations for InverPINN's transient PDE
without explicitly modeling temporal averaging and meteorological aggregation.

## Other leads and limits of verification

AirKaz links [DOI 10.4209/aaqr.2019.09.0464](https://doi.org/10.4209/aaqr.2019.09.0464)
as a 2020 Almaty air-quality study. The publisher page encountered a security
verification interstitial during this check. Its data-availability statement,
supplementary files and licensing were **not verified**. A paper's open-access
license would not automatically license a third-party measurement database.

[OpenAQ Explorer](https://explore.openaq.org/) was also checked as a possible
legitimate aggregator, but returned an uncaught client exception. AirKaz-specific
provenance, station IDs, historical coverage and permission were not established.
Do not substitute unrelated Kazakhstan stations or claim that OpenAQ provides
AirKaz history without verifying provider attribution and dataset-specific terms.

Google, DuckDuckGo and Brave search attempts encountered bot challenges; no
challenge was bypassed. Bing returned irrelevant results. Accordingly, this is
not an exhaustive claim that no other archive exists. No current documented
AirKaz API, post-2020 historical bulk export, authenticated research portal or
permissively licensed third-party mirror was verified in this session. Internal
website requests would not, by themselves, constitute an authorized public API.

## Permission and metadata request before ingestion

Request from a verified AirKaz operator/author contact:

1. Permission for research use, local storage, derived models/figures, publication
   and redistribution (including whether raw data can accompany a paper).
2. Requested station IDs and date range; available cadence, format, costs and
   approved download/export mechanism or documented API, if one exists.
3. UTC timestamps or explicit timezone/day-aggregation rules, unambiguous full
   dates and explanation of the archive's October rows.
4. Station coordinates/CRS and deployment/relocation dates, sensor models,
   calibration/corrections, units, missing-value codes and quality flags.
5. Required attribution, retention/deletion rules, access limits and whether
   use with third-party meteorological data is permitted.

No request has been sent on the user's behalf. Until these questions are
resolved, continue synthetic experiments and keep AirKaz ingestion disabled.
