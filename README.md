# ELG 5163 Machine Vision: Pre-fire vegetation indices and burn prediction

This project compares how well three pre-fire vegetation indices predict areas that later burned, using Copernicus Sentinel-2 L2A imagery and EFFIS burnt-area perimeters as labels. The study region is mainland Spain and Portugal (plus the Balearic Islands).

## Project goal

Evaluate the predictive value of:

- NDVI
- Temporal NDVI (pre-fire NDVI compared with the same season in the 3 previous years)
- DFI (Dead Fuel Index)
- optional positive control: post-fire dNDVI

Using a traditional, non-ML Bayesian histogram approach.

## Main workflow

`main.py` runs these stages in order. `selftest` only runs when requested (`--stages selftest`). When `fetch` is included, `main.py` first requests a CDSE token and stops immediately if the credentials are missing or rejected.

| Stage | Code | What it does |
|---|---|---|
| `selftest` | `tests/selftest_synthetic.py` | Opt-in. Checks on synthetic data that the comparison method ranks an informative index (DFI) above a weak one (NDVI). |
| `select` | `firedetect/effis.py` | Downloads EFFIS burnt-area perimeters for the region (if not cached), keeps Spanish and Portuguese fires, and picks a random sample of fires to study. |
| `fetch` | `firedetect/sentinel2.py`, `firedetect/cdse.py` | Downloads one pre-fire Sentinel-2 L2A scene per selected fire, plus the same window in each of the 3 previous years for Temporal NDVI. |
| `build` | `firedetect/dataset.py`, `raster.py`, `indices.py` | Computes the indices and builds a burned vs unburned pixel dataset with cloud masking and fire-aware sampling. |
| `compare` | `firedetect/evaluate.py`, `bayes.py`, `plots.py` | Fits class-conditional histograms and compares indices with leave-one-fire-out validation. |

## Indices

All three are computed per pixel from the pre-fire scene in `firedetect/indices.py`:

```text
NDVI          = (B08 - B04) / (B08 + B04)
Temporal NDVI = NDVI - mean NDVI of the same pixel and date window in the previous 3 years
DFI           = 100 * (1 - B12/B11) * (B04/B08)
```

where `B04` = red, `B08` = NIR, `B11` = SWIR1 and `B12` = SWIR2.

Temporal NDVI is negative where vegetation is less green than its usual state for that time of year, independent of how green that land cover normally is. In each prior year, pixels that are cloudy or otherwise not clear (SCL not 4 or 5) are left out of that year's average. Pixels that are not clear in any prior year are left out of the dataset entirely, so all three indices are compared on exactly the same pixels. `--baseline-years` sets how many prior years are averaged (`0` drops Temporal NDVI from the comparison).

## Data sources

| Data | Provider | How it is obtained | Needed? |
|---|---|---|---|
| Burnt-area perimeters (labels) | [EFFIS](https://forest-fire.emergency.copernicus.eu/applications/data-and-services) (European Forest Fire Information System, Copernicus Emergency Management Service / JRC) | Downloaded automatically by the `select` stage | Yes |
| Sentinel-2 L2A imagery (index inputs and prior-year baselines) | [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu), Sentinel Hub Process API | Downloaded automatically by the `fetch` stage | Yes |
| Land cover mask | [ESA WorldCover 10 m](https://esa-worldcover.org) | Manual download (see below) | Optional |

### EFFIS burnt-area perimeters

- **Source:** the EFFIS WFS service at `https://maps.effis.emergency.copernicus.eu/effis`, yearly layers `ms:modis.ba.poly.<year>` (currently 2016–2025).
- **Content:** one polygon per fire with attributes including `id`, `FIREDATE`, `COUNTRY`, `PROVINCE`, `AREA_HA` and land-cover percentages.
- **Coverage:** fires of roughly 30 ha and larger, mapped from MODIS imagery. The pipeline additionally skips fires under 50 ha (`--min-area-ha`).
- **Access:** public, no account needed. Use is subject to the [EFFIS data license](https://forest-fire.emergency.copernicus.eu/about-effis/data-license); credit EFFIS / Copernicus EMS in reports.
- **Region filter:** a lon/lat box around Iberia (`-9.6, 35.9, 4.4, 43.9`) is applied while downloading, then the `COUNTRY` attribute keeps only `ES` and `PT` (the box also touches southern France, Andorra and Morocco).
- **Size:** about 175 MB on disk for 2018–2025 (yearly cache + merged file + clipped region file). About 2 minutes to download.

Alternatively, pass your own perimeter file with `--effis-shp` (any format GeoPandas reads, any extent).

### Sentinel-2 L2A imagery

- **Source:** Sentinel Hub Process API on CDSE (`https://sh.dataspace.copernicus.eu/api/v1/process`), collection `sentinel-2-l2a`.
- **Bands:** B03, B04 (red), B08 (NIR), B11 (SWIR1), B12 (SWIR2), SCL (scene classification, for cloud masking) and dataMask, as 16-bit integer GeoTIFF.
- **Scene:** the fire's bounding box plus a 500 m margin (`--margin-m`), in the fire's UTM zone, at 10 m resolution. Very large fires are coarsened so no side exceeds 2500 px.
- **Time window:** 60 to 10 days before the fire date (`--window-days-before`, `--window-days-after`). Sentinel Hub builds a least-cloudy mosaic from that window (`mosaickingOrder=leastCC`).
- **Prior-year baselines:** the same grid and window shifted back 1, 2 and 3 years (`--baseline-years`), saved to `data/baseline/`. Because the grid is identical, each baseline lines up pixel for pixel with the pre-fire scene.
- **Size:** about 0.5–10 MB per scene depending on fire size (88 MB at most, for fires over ~25 km across). The default 20-fire sample is about 110 MB per year of imagery, so about 440 MB with 3 baseline years.
- **Access:** free CDSE account and OAuth client (see [Credentials](#credentials)). The free tier includes 10,000 processing units (PU) per month. A scene costs `(width x height / 512²) x (6 bands / 3)` PU (dataMask is free, 16-bit output, one mosaic), so 0.3–2 PU for a typical fire and about 42 PU at the 2500 px cap. The default 20-fire run costs about **60 PU for the pre-fire scenes plus 180 PU for 3 baseline years**. Only successful requests are charged, and already-downloaded scenes are skipped on re-runs (0 PU). The EFFIS download does not use CDSE quota.
- **License:** Copernicus Sentinel data are free and open; credit "Contains modified Copernicus Sentinel data [year]".

### ESA WorldCover (optional)

Restricts sampling to tree cover, shrubland and grassland (WorldCover classes 10, 20, 30). Download the 10 m tiles covering Iberia from [esa-worldcover.org](https://esa-worldcover.org) (CC BY 4.0), merge them into one GeoTIFF (or a VRT), and pass it with `--worldcover-raster data/worldcover.tif`. It is reprojected onto each Sentinel-2 scene automatically.

## Where things are stored

Everything under `data/` and `outputs/` is generated and git-ignored; it is recreated by `python main.py`.

| Path | Contents | Created by |
|---|---|---|
| `config/authentication.env` | Your CDSE credentials (git-ignored, never commit) | You, from the template |
| `data/effis_cache/` | Raw EFFIS GeoJSON, one file per year and region box | `select` |
| `data/effis_iberia_<start>-<end>.gpkg` | Merged EFFIS perimeters for the region and years | `select` |
| `data/fires_region.gpkg` | Standardised perimeters (`fire_id`, `fire_date`, `area_ha`) of all fires in the region, used to exclude neighbouring fires from unburned rings | `select` |
| `data/selected_fires.csv` | The fires chosen for this run | `select` |
| `data/sentinel2/<fire_id>__<start>_<end>.tif` | One pre-fire Sentinel-2 scene per fire | `fetch` |
| `data/baseline/<fire_id>__<start>_<end>.tif` | The same scene in each prior year, for Temporal NDVI | `fetch` |
| `data/dataset.csv` | Sampled pixels: `fire_id`, `fire_label`, `burned`, `ndvi`, `temporal_ndvi`, `dfi`, `source_image` | `build` |
| `outputs/` | Comparison table and plots (see [Output files](#output-files)) | `compare` |
| `outputs/logs/run_<date>_<time>.txt` | Full console output of each `main.py` run, including errors | every run |

## Requirements

- **Python 3.10 or newer** (developed with 3.14) and the packages in `requirements.txt`: numpy, pandas, scipy, scikit-learn, matplotlib, rasterio, geopandas, shapely, pyproj, requests.
- **Internet access** for the `select` (EFFIS) and `fetch` (CDSE) stages. `build`, `compare` and `selftest` run offline.
- **A free CDSE account with an OAuth client** for each person who runs `fetch`.
- **Disk space:** about 650 MB for the default run (EFFIS ~175 MB, imagery ~440 MB for 20 fires with 3 baseline years), more for more fires or a WorldCover mosaic.
- **CDSE quota:** about 240 of the 10,000 free monthly processing units for the default 20-fire run.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Credentials

Each user needs their own CDSE OAuth client:

1. Create a free account at https://dataspace.copernicus.eu.
2. Open the Sentinel Hub dashboard at https://shapps.dataspace.copernicus.eu/dashboard/, go to **User settings -> OAuth clients** and create a client. Copy the secret immediately; it is shown only once.
3. Copy the template and fill in your values:

```bash
copy config\authentication.env.example config\authentication.env
```

`config/authentication.env` is git-ignored, so your credentials stay local. `main.py` loads it automatically; variables already set in your shell (`$env:CDSE_CLIENT_ID`, `$env:CDSE_CLIENT_SECRET`) take precedence.

## Running the workflow

```bash
python main.py
```

Useful variations:

```bash
python main.py --stages select                   # preview which fires are chosen (no CDSE needed)
python main.py --stages fetch                    # download scenes for the current selection
python main.py --stages build compare            # re-run the analysis on downloaded scenes (offline)
python main.py --stages selftest                 # synthetic check of the comparison method
python main.py --max-fires 40 --country ES       # more fires, Spain only
python main.py --start-date 2022-01-01 --end-date 2023-12-31
python main.py --fire-id 214381 278330           # specific EFFIS fire IDs
python main.py --baseline-years 1                # cheaper baseline (1 prior year)
python main.py --worldcover-raster data/worldcover.tif
python main.py --help                            # all options
```

`fetch`, `build` and `compare` work on the fires listed in `data/selected_fires.csv`, so options that change the selection (`--fire-id`, `--max-fires`, dates) need the `select` stage in the same run.

Fire selection is a random sample (fixed seed, so it is reproducible) of fires from `--start-date` (default 2018-01-01) that are at least `--min-area-ha` (default 50 ha). Other regions: `--region ontario`, or `--region-file` with a boundary polygon.

## How each stage works

### fetch

- Uses the CDSE OAuth2 client-credentials flow; one token is reused and renewed shortly before it expires.
- Requests Sentinel-2 L2A from the Sentinel Hub Process API with `mosaickingOrder=leastCC`, as GeoTIFF in UTM with a 2500 px per side cap.
- Retries on rate limits and server errors, and skips scenes that already exist.
- `--continue-on-fetch-error` keeps going when a single fire fails.

### build

- Reads each pre-fire GeoTIFF and matches it to its fire (and its baselines) by the fire ID in the filename.
- Computes NDVI, Temporal NDVI and DFI.
- Applies cloud masking using SCL (keeps vegetation and bare soil, classes 4 and 5) and dataMask.
- Rasterizes the fire perimeter in the scene's UTM grid.
- Takes burned pixels from the fire area eroded by 20 m (`--erode-m`).
- Takes unburned pixels from a 250 m ring around the perimeter (`--buffer-m`), excluding other fires.
- Samples up to 5000 pixels per class per fire (`--max-samples-per-class`) and writes one CSV.

### compare

- Splits each index into 30 bins and counts burned and unburned pixels per bin with Laplace smoothing (`alpha=1`).
- Applies Bayes' rule to get P(burned | bin), the posterior curve.
- Evaluates with leave-one-fire-out cross-validation: each fire is scored by a model built from all other fires.
- Reports ROC-AUC, PR-AUC, lift in the top 10% and Brier score, and draws the plots below.

### selftest

Creates synthetic data where DFI is informative and NDVI is weak, then checks that the comparison ranks DFI above NDVI. Also runnable on its own with `python -m tests.selftest_synthetic`.

## Output files

- `data/dataset.csv`
- `outputs/index_comparison.csv`
- `outputs/hist_ndvi.png`, `outputs/hist_temporal_ndvi.png`, `outputs/hist_dfi.png`
- `outputs/posterior_ndvi.png`, `outputs/posterior_temporal_ndvi.png`, `outputs/posterior_dfi.png`
- `outputs/roc_curve.png`
- `outputs/pr_curve.png`
- `outputs/calibration.png`

## Important implementation notes

- The project avoids ML for the main result; the model is a traditional Bayesian histogram classifier (counting and Bayes' rule). scikit-learn is only used to compute evaluation metrics.
- Class imbalance is handled by reporting ranked metrics such as ROC-AUC, PR-AUC, and lift at top 10% instead of accuracy.
- Burned pixels are sampled inside each perimeter after erosion, and unburned pixels are sampled in a ring outside the perimeter while excluding other fire areas.

## File structure

```text
.
├── main.py                          # command line; runs the stages in order
├── firedetect/
│   ├── config.py                    # regions, band order, endpoints, constants
│   ├── utils.py                     # shared helpers (column detection, safe names, run log)
│   ├── effis.py                     # select: download EFFIS perimeters, pick fires
│   ├── cdse.py                      # credentials, OAuth token, Sentinel Hub requests
│   ├── sentinel2.py                 # fetch: pre-fire and prior-year scenes
│   ├── raster.py                    # read GeoTIFF, cloud mask, rasterize, WorldCover
│   ├── indices.py                   # NDVI, Temporal NDVI, DFI
│   ├── dataset.py                   # build: burned / unburned sampling -> dataset.csv
│   ├── bayes.py                     # histogram classifier
│   ├── evaluate.py                  # compare: leave-one-fire-out and metrics
│   └── plots.py                     # figures
├── tests/
│   └── selftest_synthetic.py
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   ├── authentication.env.example   # template, committed
│   └── authentication.env           # your credentials, git-ignored
├── data/                            # generated, git-ignored
│   ├── effis_cache/
│   ├── effis_iberia_2018-latest.gpkg
│   ├── fires_region.gpkg
│   ├── selected_fires.csv
│   ├── sentinel2/
│   ├── baseline/
│   └── dataset.csv
├── outputs/                         # generated, git-ignored
└── .venv/                           # git-ignored
```

## Reference

This project follows the workflow described in the ELG 5163 assignment brief, using Sentinel-2 L2A imagery and burnt-area perimeters as labels for a pre-fire burn prediction comparison.

Data credits: EFFIS / Copernicus Emergency Management Service (burnt areas); Copernicus Sentinel-2 data via the Copernicus Data Space Ecosystem; ESA WorldCover (if used).
