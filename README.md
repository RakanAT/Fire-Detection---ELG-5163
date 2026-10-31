# ELG 5163 Machine Vision: Pre-fire vegetation indices and burn prediction

This project compares how well three pre-fire vegetation indices predict areas that later burned, using Copernicus Sentinel-2 L2A imagery and EFFIS burnt-area perimeters as labels. The study region is mainland Spain and Portugal (plus the Balearic Islands).

## Project goal

Evaluate the predictive value of:

- NDVI
- temporal NDVI anomaly
- DFI (Dead Fuel Index)
- optional positive control: post-fire dNDVI

Using a traditional, non-ML Bayesian histogram approach.

## Main workflow

`main.py` runs these stages in order:

| Stage | Script | What it does |
|---|---|---|
| `selftest` | `selftest_synthetic.py` | Checks on synthetic data that the Stage 3 method ranks an informative index (DFI) above a weak one (NDVI). |
| `select` | `main.py` | Downloads EFFIS burnt-area perimeters for the region (if not cached), keeps Spanish and Portuguese fires, and picks a random sample of fires to study. |
| `fetch` | `01_fetch_sentinel2.py` | Downloads one pre-fire Sentinel-2 L2A scene per selected fire from CDSE. |
| `build` | `02_build_dataset.py` | Builds a burned vs unburned pixel dataset with cloud masking and fire-aware sampling. |
| `compare` | `03_bayes_compare.py` | Fits class-conditional histograms and compares indices with leave-one-fire-out validation. |

## Data sources

| Data | Provider | How it is obtained | Needed? |
|---|---|---|---|
| Burnt-area perimeters (labels) | [EFFIS](https://forest-fire.emergency.copernicus.eu/applications/data-and-services) (European Forest Fire Information System, Copernicus Emergency Management Service / JRC) | Downloaded automatically by the `select` stage | Yes |
| Sentinel-2 L2A imagery (index inputs) | [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu), Sentinel Hub Process API | Downloaded automatically by the `fetch` stage | Yes |
| Land cover mask | [ESA WorldCover 10 m](https://esa-worldcover.org) | Manual download (see below) | Optional |
| Prior-year NDVI baseline | Sentinel-2 L2A via CDSE | Manual (see below) | Optional, but required for a meaningful anomaly index |

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
- **Size:** about 0.5–10 MB per fire depending on fire size (88 MB at most, for fires over ~25 km across). The default 20-fire sample is about 110 MB, most of it from one very large fire.
- **Access:** free CDSE account and OAuth client (see [Credentials](#credentials)). The free tier includes 10,000 processing units (PU) per month. A scene costs `(width x height / 512²) x (6 bands / 3)` PU (dataMask is free, 16-bit output, one mosaic), so 0.3–2 PU for a typical fire and about 42 PU at the 2500 px cap. The default 20-fire run costs about **60 PU**. Only successful requests are charged, and already-downloaded scenes are skipped on re-runs (0 PU). The EFFIS download does not use CDSE quota.
- **License:** Copernicus Sentinel data are free and open; credit "Contains modified Copernicus Sentinel data [year]".

### ESA WorldCover (optional)

Restricts sampling to tree cover, shrubland and grassland (WorldCover classes 10, 20, 30). Download the 10 m tiles covering Iberia from [esa-worldcover.org](https://esa-worldcover.org) (CC BY 4.0), merge them into one GeoTIFF (or a VRT), and pass it with `--worldcover-raster data/worldcover.tif`. It is reprojected onto each Sentinel-2 scene automatically.

### Prior-year NDVI baseline (optional)

The anomaly index is `NDVI - baseline NDVI`. **Without `--baseline-dir` the baseline is zero, so the anomaly is identical to NDVI.** A baseline raster must be a 7-band GeoTIFF in the same format and on the same grid as the fire's scene. Running `01_fetch_sentinel2.py` for the same fire with `--fire-date` set one or more years earlier and `--output-dir data/baseline_ndvi` produces that grid. Note that `02_build_dataset.py` currently matches baselines to scenes by raster size only, so this is not yet automated for many fires.

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
| `data/cache/` | Request cache directory | `fetch` |
| `data/dataset.csv` | Sampled pixels: `fire_id`, `fire_label`, `burned`, `ndvi`, `anomaly`, `dfi`, `source_image` | `build` |
| `outputs/` | Comparison table and plots (see [Output files](#output-files)) | `compare` |

## Requirements

- **Python 3.10 or newer** (developed with 3.14) and the packages in `requirements.txt`: numpy, pandas, scipy, scikit-learn, matplotlib, rasterio, geopandas, shapely, pyproj, requests.
- **Internet access** for the `select` (EFFIS) and `fetch` (CDSE) stages. `build`, `compare` and `selftest` run offline.
- **A free CDSE account with an OAuth client** for each person who runs `fetch`.
- **Disk space:** about 300 MB for the default run (EFFIS ~175 MB, imagery ~110 MB for 20 fires), more for more fires or a WorldCover mosaic.
- **CDSE quota:** about 60 of the 10,000 free monthly processing units for the default 20-fire run.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Credentials

Each team member needs their own CDSE OAuth client:

1. Create a free account at https://dataspace.copernicus.eu.
2. Open the Sentinel Hub dashboard at https://shapps.dataspace.copernicus.eu/dashboard/, go to **User settings -> OAuth clients** and create a client. Copy the secret immediately; it is shown only once.
3. Copy the template and fill in your values:

```bash
copy config\authentication.env.example config\authentication.env
```

`config/authentication.env` is git-ignored, so your credentials stay local. `main.py` loads it automatically; variables already set in your shell (`$env:CDSE_CLIENT_ID`, `$env:CDSE_CLIENT_SECRET`) take precedence.

## Running the whole workflow

```bash
python main.py
```

Useful variations:

```bash
python main.py --stages select                   # preview which fires are chosen (no CDSE needed)
python main.py --stages build compare            # re-run the analysis on downloaded scenes
python main.py --max-fires 40 --country ES       # more fires, Spain only
python main.py --start-date 2022-01-01 --end-date 2023-12-31
python main.py --fire-id 214381 278330           # specific EFFIS fire IDs
python main.py --worldcover-raster data/worldcover.tif --baseline-dir data/baseline_ndvi
python main.py --help                            # all options
```

Fire selection is a random sample (fixed seed, so it is reproducible) of fires from `--start-date` (default 2018-01-01) that are at least `--min-area-ha` (default 50 ha). Other regions: `--region ontario`, or `--region-file` with a boundary polygon.

The individual stage scripts can still be run on their own, as below.

## Stage 1: fetch pre-fire Sentinel-2 data

```bash
python 01_fetch_sentinel2.py \
  --effis-shp data/fires_region.gpkg \
  --fire-id 214381 \
  --window-days-before 60 \
  --window-days-after 10 \
  --output-dir data/sentinel2
```

Notes:

- The script uses the CDSE OAuth2 client-credentials flow; credentials come from the environment (`main.py` loads them from `config/authentication.env`).
- The fire date is read from the perimeter file; pass `--fire-date YYYY-MM-DD` if it has no date column.
- It requests Sentinel Hub Process API imagery for Sentinel-2 L2A with `mosaickingOrder=leastCC`.
- It downloads GeoTIFF output in UTM projection with a `2500 px per side` cap.
- It includes retry and rate-limit handling, and skips scenes that already exist.

## Stage 2: build the training dataset

```bash
python 02_build_dataset.py \
  --image-dir data/sentinel2 \
  --effis-shp data/fires_region.gpkg \
  --worldcover-raster data/worldcover.tif \
  --baseline-dir data/baseline_ndvi \
  --output-csv data/dataset.csv \
  --buffer-m 250
```

This stage:

- reads the downloaded GeoTIFFs and matches each one to its fire by filename
- computes NDVI, anomaly, and DFI
- applies cloud masking using SCL (keeps vegetation and bare soil, classes 4 and 5) and dataMask
- rasterizes the fire perimeter in the scene's UTM grid
- builds burned pixels from the fire area eroded by 20 m (`--erode-m`)
- builds unburned pixels as a 250 m ring around the perimeter while excluding other fires
- writes one CSV with the target label and index values

## Stage 3: Bayesian model comparison

```bash
python 03_bayes_compare.py \
  --dataset-csv data/dataset.csv \
  --output-dir outputs \
  --index-names ndvi anomaly dfi
```

This step:

- fits class-conditional histograms with Laplace smoothing (`alpha=1`)
- computes Bayes posterior probability in each index bin
- evaluates via leave-one-fire-out cross-validation
- outputs the comparison table and plots for:
  - histogram comparisons
  - posterior curves
  - ROC
  - precision-recall
  - calibration

## Synthetic self-test

```bash
python selftest_synthetic.py
```

This synthetic test creates fake data where DFI is informative and NDVI is weak, then checks that the Stage 3 method ranks DFI above NDVI.

## Output files

Expected outputs include:

- `data/dataset.csv`
- `outputs/index_comparison.csv`
- `outputs/hist_ndvi.png`
- `outputs/hist_anomaly.png`
- `outputs/hist_dfi.png`
- `outputs/posterior_ndvi.png`
- `outputs/posterior_anomaly.png`
- `outputs/posterior_dfi.png`
- `outputs/roc_curve.png`
- `outputs/pr_curve.png`
- `outputs/calibration.png`

## Important implementation notes

- The project avoids ML for the main result; the main model is traditional Bayesian histogram comparison.
- Class imbalance is handled by reporting ranked metrics such as ROC-AUC, PR-AUC, and lift at top 10% instead of accuracy.
- Burned pixels are sampled inside each perimeter after erosion, and unburned pixels are sampled in a ring outside the perimeter while excluding other fire areas.
- The DFI formula used in the code follows the request:

```text
DFI = 100 * (1 - B12/B11) * (B4/B8)
```

where `B4` = red, `B8` = NIR, `B11` = SWIR1, and `B12` = SWIR2.

## File structure

```text
.
├── main.py                          # runs the whole workflow
├── 01_fetch_sentinel2.py
├── 02_build_dataset.py
├── 03_bayes_compare.py
├── selftest_synthetic.py
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
│   └── dataset.csv
├── outputs/                         # generated, git-ignored
└── .venv/                           # git-ignored
```

## Reference

This project follows the workflow described in the ELG 5163 assignment brief, using Sentinel-2 L2A imagery and burnt-area perimeters as labels for a pre-fire burn prediction comparison.

Data credits: EFFIS / Copernicus Emergency Management Service (burnt areas); Copernicus Sentinel-2 data via the Copernicus Data Space Ecosystem; ESA WorldCover (if used).
