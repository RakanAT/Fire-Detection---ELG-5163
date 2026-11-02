# ELG 5163 Machine Vision: Pre-fire vegetation indices and burn prediction

This project tests whether pre-fire vegetation indices from Copernicus Sentinel-2 L2A imagery can indicate which areas are more likely to burn, using EFFIS burnt-area perimeters as labels. The study region is mainland Spain and Portugal (plus the Balearic Islands): 100 randomly selected fires from 2018–2025, of which 97 have usable cloud-free imagery. The sample is frozen in `config/selected_fires.csv`, so everyone who runs the pipeline analyses the same fires (see [Reproducibility](#reproducibility)).

The classifier is a traditional, non-ML Bayesian histogram method (counting pixels per index value and applying Bayes' rule), evaluated with leave-one-fire-out validation: every fire is scored by a model built only from the other fires.

## Findings

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Do pre-fire indices tell which areas within a fire's surroundings will burn? | **No.** Single indices score ROC-AUC 0.48–0.53, all nine combined 0.56, against land 0–250 m and 1–3 km from the fire alike. Neighbourhood dead-fuel shares and dry clumps do no better. | `index_comparison.csv`, `near_vs_far.csv`, `roc_curve.png`, `hist_*` / `posterior_*`, fire maps |
| 2 | Does the landscape do better? | **A little.** Slope, valley/ridge position and distance to firebreaks together reach 0.61; adding the indices to them gives 0.61. Fires reach ridges and stop in valleys and near fields, roads, villages and water. | `index_comparison.csv` (landscape rows), `landscape_comparison.csv` |
| 3 | Do pre-fire indices predict how severely an area burns? | **Yes, moderately.** NDVI 0.67: greener, denser vegetation before the fire burned more severely. | `severity_comparison.csv`, `severity_roc_curve.png` |
| 4 | Do clumps of dry vegetation mark fire-prone areas? | **Not testable in this design.** Clumps usually straddle the perimeter (36% of burned and of unburned pixels sit in one), so a comparison against the fire's own surroundings cannot credit them. Testing them as a hazard indicator needs burned areas compared with random vegetation across the region. | clump rows in `index_comparison.csv` and `severity_comparison.csv` |

Checks on the method and labels:
- **Positive control:** post-fire dNDVI separates burned from unburned pixels with ROC-AUC 0.85 (median 0.92 per fire), so the method works when a signal exists and the EFFIS perimeters are mostly right.
- **Already-burned "pre-fire" scenes:** two fires whose EFFIS date was later than the real fire were found and excluded (`config/excluded_fires.csv`); the build stage checks every fire for this.
- **Weak post-fire scars:** fires whose own dNDVI separates poorly are marked `weak_post_fire_scar` in `per_fire.csv` (doubtful perimeter, or a light burn that regreened quickly).

What the scores mean: ROC-AUC 0.5 is chance, 1.0 is perfect. "Lift top 10%" is how much more often the 10% of pixels the model ranks riskiest burned than average (1.0 = no better than random).

## Workflow

`main.py` runs these stages in order. `selftest` only runs when requested (`--stages selftest`). When `fetch` is included, `main.py` first requests a CDSE token and stops immediately if the credentials are missing or rejected.

| Stage | Code | What it does |
|---|---|---|
| `selftest` | `tests/selftest_synthetic.py` | Opt-in. Checks on synthetic data that the comparison ranks an informative index (DFI) above a weak one (NDVI). |
| `select` | `firedetect/effis.py` | Downloads EFFIS burnt-area perimeters for the region (if not cached), keeps Spanish and Portuguese fires, and takes the frozen sample from `config/selected_fires.csv` (or draws a new one with `--new-sample`). |
| `fetch` | `firedetect/sentinel2.py`, `cdse.py`, `worldcover.py`, `terrain.py` | Per fire: pre-fire, prior-year and post-fire Sentinel-2 scenes, land-cover and elevation patches, for the 10 m main scenes and the wider 20 m context scenes. |
| `build` | `firedetect/dataset.py`, `indices.py`, `landscape.py`, `raster.py` | Computes every index and landscape factor per pixel; samples burned and unburned pixels into the main and context datasets; checks each fire for an already-burned "pre-fire" scene. |
| `compare` | `firedetect/evaluate.py`, `bayes.py`, `plots.py` | Leave-one-fire-out comparison; writes the tables and plots listed under [Output files](#output-files). |
| `maps` | `firedetect/maps.py` | One map per fire of the pixels each index flagged as likely to burn against the pixels that burned. |

## Indices

### Sentinel-2 bands used

Every scene is downloaded as Level-2A surface reflectance (atmospherically corrected), as integer digital numbers (reflectance × 10,000). The indices are ratios, so this scaling cancels out.

| Band | Name | Centre wavelength | Native resolution | Used for |
|---|---|---|---|---|
| B03 | Green | 560 nm | 10 m | downloaded but not used by the indices |
| B04 | Red | 665 nm | 10 m | NDVI, DFI; absorbed by chlorophyll, so dark over healthy vegetation |
| B08 | Near infrared (NIR) | 842 nm | 10 m | NDVI, DFI, NDMI, NBR; strongly reflected by healthy leaf structure |
| B11 | Shortwave infrared 1 (SWIR1) | 1610 nm | 20 m | DFI, NDMI; absorbed by water in leaves and soil |
| B12 | Shortwave infrared 2 (SWIR2) | 2190 nm | 20 m | DFI, NBR; absorbed by water, bright over dry and burnt material |
| SCL | Scene classification | – | 20 m | cloud, shadow, water and snow masking (keeps classes 4 vegetation, 5 bare soil) |
| dataMask | Data mask | – | – | marks pixels with no data |

B11, B12 and SCL are resampled to the 10 m grid by Sentinel Hub (bilinear for the bands).

### Pre-fire indices compared

Computed per pixel in `firedetect/indices.py` and `firedetect/landscape.py`.

| Indicator | Bands | Formula | What it measures |
|---|---|---|---|
| **NDVI** | B04, B08 | `(B08 − B04) / (B08 + B04)` | Greenness: amount of healthy, photosynthesising vegetation (−1 to 1; dense vegetation ~0.6–0.9) |
| **Temporal NDVI** | B04, B08 (pre-fire and 3 prior years) | `NDVI − mean NDVI of the same pixel and window in the previous 3 years` | Greenness compared with the pixel's usual state for that time of year; negative = browner than usual |
| **DFI** (Dead Fuel Index) | B04, B08, B11, B12 | `100 × (1 − B12/B11) × (B04/B08)` | Dead, dry fuel: the SWIR term falls as material dries out, the red/NIR term rises as green leaves disappear |
| **NDMI** (Normalised Difference Moisture Index) | B08, B11 | `(B08 − B11) / (B08 + B11)` | Vegetation water content; lower = drier fuel |
| **NBR** (Normalised Burn Ratio) | B08, B12 | `(B08 − B12) / (B08 + B12)` | Healthy vs dry or burnt material; high for healthy vegetation, low for dry fuel and burn scars |
| **Dead fuel within 100 m / 300 m** | B08, B11 (via NDMI) | share of vegetation pixels with `NDMI < 0` in a 210 m × 210 m / 610 m × 610 m window | Fuel continuity: how much dry vegetation surrounds a pixel |
| **Dry clump size** | B08, B11 (via NDMI) | dry pixels → morphological opening with a 20 m disk (removes isolated pixels and thin lines) → 8-connected components → area (ha) of the clump the pixel is in, 0 outside clumps | Concentrated rather than scattered dry fuel |
| **Distance to large dry clump** | B08, B11 (via NDMI) | distance transform to the nearest clump of at least 5 ha | How close a pixel is to a substantial body of dry fuel |
| **Combined** (naive Bayes) | all of the above | `P(burned │ all) ∝ P(burned) × Π P(index value │ burned)` | The nine pre-fire indices together |

How they are calculated in practice:
- Each normalised difference (NDVI, NDMI, NBR) is `(a − b) / (a + b)`, set to 0 where `a + b` is 0. In DFI, SWIR1 is clipped to a tiny positive value to avoid dividing by zero, and `B04/B08` is 0 where NIR is 0.
- Temporal NDVI: in each prior year, pixels that are not clear (SCL not 4 or 5) are left out of that year's average. Pixels with no clear prior year are left out of the dataset, so every index is compared on exactly the same pixels.
- A vegetation pixel is **dry** when `NDMI < 0`: it reflects more SWIR1 than NIR, i.e. low leaf water. About 40% of sampled vegetation pixels are dry by this rule (21% of trees, 63% of shrubland, 54% of grassland).
- The dead-fuel shares are a moving-window (box) filter: dry vegetation pixels in the window divided by cloud-free vegetation pixels in it, so clouds, non-vegetation and the image edge are ignored rather than counted as "not dry". Window sizes are set in metres, so they cover the same ground on the 10 m and 20 m scenes.

### Landscape factors (reference for finding 2)

Not vegetation indices: they show what the landscape alone explains, as single factors (`landscape_comparison.csv`), as a **landscape** naive Bayes model, and combined with the indices (**combined indices + landscape**).

| Factor | Source | How |
|---|---|---|
| **Slope** | Copernicus DEM (30 m) | degrees from horizontal, from the elevation gradient |
| **Valley vs ridge (TPI)** | Copernicus DEM | elevation minus the mean elevation within 300 m: positive on ridges, negative in valleys |
| **Distance to firebreak** | ESA WorldCover | distance transform to the nearest cropland, built-up, bare, snow, water, wetland, mangrove or moss pixel |
| **Land cover** | ESA WorldCover | tree / shrub / grass, treated as categories (models only) |

Weather (temperature, humidity, wind, the Fire Weather Index, lightning) is not used: available datasets have 9–25 km cells, so a fire and its surroundings share the same values and weather cannot separate them in this design. It would matter in a regional comparison against random vegetation (see finding 4).

### Post-fire measures

| Measure | Bands | Formula | Used for |
|---|---|---|---|
| **dNDVI** | B04, B08 (pre- and post-fire) | `pre-fire NDVI − post-fire NDVI` | Positive control (not a prediction): checks the labels and the method |
| **RBR** (Relativised Burn Ratio) | B08, B12 (pre- and post-fire) | `dNBR / (pre-fire NBR + 1.001)`, with `dNBR = pre-fire NBR − post-fire NBR` | Burn severity label for finding 3 (Parks et al. 2014) |

The post-fire scene covers 5 to 45 days after the fire date; there, SCL classes 2 (dark area, often charcoal), 4, 5 and 7 are accepted, and cloudy pixels are left out for these measures only.

**Severity analysis:** within burned pixels, each pre-fire model is tested on whether it predicts the more severely burned half (RBR above its median) vs the less severely burned half. RBR is used instead of dNBR because dNBR contains the pre-fire NBR itself, so pre-fire NBR, NDVI and NDMI would predict it partly by construction (with dNBR their ROC-AUC was about 0.08 higher).

## How the comparison works

**Sampling** (`build`): burned pixels come from the fire perimeter eroded by 20 m (`--erode-m`), so mixed edge pixels are avoided. Unburned pixels come from zones outside the perimeter:

| Dataset | Scenes | Unburned zones | Used for |
|---|---|---|---|
| `data/dataset.csv` | 10 m, perimeter + 500 m | 0–250 m from the perimeter | findings 1–4, maps |
| `data/context/dataset.csv` | 20 m, perimeter + 3 km | 0–250 m (`near`) and 1–3 km (`far`) | `near_vs_far.csv` |

Only cloud-free tree, shrub and grassland pixels (WorldCover classes 10, 20, 30) are used. Other fires within 30 days of a fire are kept out of its unburned zones, as they may be part of the same event; land burned in earlier or later years counts as unburned by this fire. Up to 5000 pixels are sampled per fire for the burned area and for each zone (`--max-samples-per-class`).

**Classifier** (`compare`):
- Each index is split into 30 equal-width bins (between its 0.5th and 99.5th percentiles; the outer bins are open-ended). Burned and unburned pixels are counted per bin, plus 1 (Laplace smoothing), and Bayes' rule gives P(burned | bin).
- **Naive Bayes** combines several features by multiplying their likelihoods, `P(burned | all) ∝ P(burned) × Π P(value | burned)`, as if they were independent. With one feature it is exactly the single-index classifier. Land cover is treated as categories. The indices are correlated (NDVI, NDMI and NBR all use NIR), so combined probabilities are overconfident; the ranking (ROC-AUC) is affected much less.
- **Leave-one-fire-out:** every fire is scored by a model built from all other fires. Each fire's per-bin counts are computed once and the model for a held-out fire uses the total counts minus that fire's counts, which is identical to refitting without it. Bin edges are set once from all values (no labels involved).

**What P(burned) means:** "is a pixel with this value more like the burned area or the unburned comparison zone?", not a real-world probability of burning: sampling is roughly balanced and the comparison zone is chosen by design.

## Reproducibility

`data/` and `outputs/` are not committed, so each person who runs the pipeline downloads the data again. Two things keep their results the same as everyone else's:

- **Frozen fire sample:** `config/selected_fires.csv` (committed) lists the 100 fires (ID, date, area). `select` uses it instead of drawing a new sample, so later EFFIS updates (new fires, a new year's layer) cannot change which fires are analysed. A fire that EFFIS later removes is reported and skipped. To draw a different sample, run `python main.py --new-sample` (with any of `--max-fires`, `--start-date`, `--end-date`, `--min-area-ha`, `--country`); the new sample is saved to `config/selected_fires.csv`, to be committed.
- **Fixed period:** `--end-date` defaults to 2025-12-31, so the perimeters used to find neighbouring fires cover the same years.

Given the same fires, results are deterministic: pixel sampling uses fixed seeds, the classifier involves no randomness, and the least-cloudy Sentinel-2 mosaic of a date window, WorldCover and the DEM do not change. Small differences are only possible if Copernicus reprocesses archived imagery or EFFIS revises a perimeter.

A full run from scratch costs about 1,250 CDSE processing units of the runner's own account and roughly two hours of downloading. To only look at the results, share `outputs/` (about 50 MB); sharing `data/` (about 2.3 GB) lets others re-run `build`, `compare` and `maps` without downloading.

## Excluded fires and data checks

`config/excluded_fires.csv` (committed) lists fires that are left out, each with a reason. `fetch` does not download them and `build` skips them; when a new sample is drawn, `select` takes the next fire of its fixed random order in their place (so excluding a fire does not reshuffle the rest of the sample). The frozen sample keeps the 100 fires as drawn, so fire 214920, excluded after the sample was drawn, is listed there but skipped.

The `build` stage flags fires whose "pre-fire" scene already shows the burn scar, which happens when the EFFIS fire date is later than the real fire: compared with its unburned ring, NDVI inside the perimeter fell more than 0.2 below its prior-year mean **and** the SWIR ratio B12/B11 is more than 0.2 higher (charred ground). Flagged fires are only reported (a warning in the log and `suspect_pre_burned` in `per_fire.csv`); after confirming one on its map, add it to the exclude list. `weak_post_fire_scar` marks fires whose own dNDVI separates burned from unburned with ROC-AUC below 0.75.

## Output files

| File | Finding | What it shows |
|---|---|---|
| `outputs/index_comparison.csv`, `roc_curve.png` | 1, 2 | ROC-AUC and lift of every pre-fire index, their combination, the landscape models and the dNDVI control (burned vs 0–250 m ring) |
| `outputs/near_vs_far.csv` | 1, 2 | The same against unburned vegetation 0–250 m and 1–3 km away (20 m context scenes) |
| `outputs/hist_<index>.png`, `posterior_<index>.png` | 1 | Burned vs unburned distribution of each pre-fire index, and P(burned) per value with the share of pixels per bin (why curves that look informative still give ROC-AUC ≈ 0.5) |
| `outputs/landscape_comparison.csv` | 2 | Each landscape factor on its own, with its median for burned and unburned pixels (the direction) |
| `outputs/severity_comparison.csv`, `severity_roc_curve.png` | 3 | Predicting the more vs less severely burned half (RBR) within burned pixels |
| `outputs/fire_maps/<fire_id>.png` | 1, 4 | Per fire: pre- and post-fire false-colour images with the EFFIS perimeter, then hits, false alarms and misses of each index over shaded relief |
| `outputs/per_fire.csv` | checks | Per fire: date, area, land cover, held-out ROC-AUC per model, `suspect_pre_burned`, `weak_post_fire_scar` |
| `outputs/logs/run_<date>_<time>.txt` | – | Full console output of each run |

False-colour images show SWIR2, NIR and red as red, green and blue: healthy vegetation green, dry vegetation and bare soil pink, burn scars dark red. Each image is stretched to its own brightness range, so colours compare areas within a map, not between fires.

## Data sources

| Data | Provider | How it is obtained |
|---|---|---|
| Burnt-area perimeters (labels) | [EFFIS](https://forest-fire.emergency.copernicus.eu/applications/data-and-services) (European Forest Fire Information System, Copernicus Emergency Management Service / JRC) | `select` stage |
| Sentinel-2 L2A imagery (pre-fire, prior years, post-fire) | [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu), Sentinel Hub Process API | `fetch` stage |
| Land cover | [ESA WorldCover 10 m](https://esa-worldcover.org) | `fetch` stage (small patch per fire) |
| Elevation | [Copernicus DEM GLO-30](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) | `fetch` stage (small patch per fire) |

### EFFIS burnt-area perimeters

- **Source:** the EFFIS WFS service at `https://maps.effis.emergency.copernicus.eu/effis`, yearly layers `ms:modis.ba.poly.<year>` (currently 2016–2025).
- **Content:** one polygon per fire with attributes including `id`, `FIREDATE`, `COUNTRY`, `PROVINCE`, `AREA_HA` and land-cover percentages.
- **How they are drawn:** EFFIS Rapid Damage Assessment maps burnt areas semi-automatically from post-fire satellite imagery (MODIS 250 m, and Sentinel-2 for recent years) around active-fire detections, then reviews them by eye. `FIREDATE` is the date of first detection or mapping, not necessarily the ignition date. Unburned islands inside a perimeter are generally not cut out.
- **Coverage:** fires of roughly 30 ha and larger. The pipeline additionally skips fires under 50 ha (`--min-area-ha`).
- **Access:** public, no account needed. Use is subject to the [EFFIS data license](https://forest-fire.emergency.copernicus.eu/about-effis/data-license); credit EFFIS / Copernicus EMS in reports.
- **Region filter:** a lon/lat box around Iberia (`-9.6, 35.9, 4.4, 43.9`) is applied while downloading, then the `COUNTRY` attribute keeps only `ES` and `PT`.

Alternatively, pass your own perimeter file with `--effis-shp` (any format GeoPandas reads, any extent).

### Sentinel-2 L2A imagery

- **Source:** Sentinel Hub Process API on CDSE (`https://sh.dataspace.copernicus.eu/api/v1/process`), collection `sentinel-2-l2a`, bands B03, B04, B08, B11, B12, SCL and dataMask as 16-bit integer GeoTIFF.
- **Two scene sets per fire** (`config.SceneSet`), each on one grid in the fire's UTM zone so all its rasters line up pixel for pixel: main (perimeter + 500 m, 10 m, `data/`) and context (perimeter + 3 km, 20 m, `data/context/`). Very large fires are coarsened so no side exceeds 2500 px.
- **Time windows:** pre-fire 60 to 10 days before the fire date (`--window-days-before`, `--window-days-after`); the same window 1, 2 and 3 years earlier for Temporal NDVI (`--baseline-years`); post-fire 5 to 45 days after the fire (main set only). Sentinel Hub builds a least-cloudy mosaic from each window (`mosaickingOrder=leastCC`).
- **Cost:** free CDSE account and OAuth client (see [Credentials](#credentials)). The free tier includes 10,000 processing units (PU) per month. A scene costs `(width x height / 512²) x (6 bands / 3)` PU. The default 100 fires cost about **1,250 PU** in total, most of it from the few largest fires. `python main.py --estimate-only` prints what is still needed without downloading; already-downloaded scenes are skipped (0 PU).
- **License:** Copernicus Sentinel data are free and open; credit "Contains modified Copernicus Sentinel data [year]".

### ESA WorldCover land cover

- **Source:** ESA WorldCover 10 m 2021 v200, public Cloud-Optimised GeoTIFF tiles at `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/` (3 x 3 degree tiles).
- **How:** `fetch` reads only the window around each pre-fire scene, resamples it (nearest neighbour) onto the scene's grid and caches it in `<set>/worldcover/`. No account and no CDSE quota.
- **License:** CC BY 4.0; credit "ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium".

### Copernicus DEM (elevation)

- **Source:** Copernicus DEM GLO-30, public Cloud-Optimised GeoTIFF tiles at `https://copernicus-dem-30m.s3.amazonaws.com/` (1 x 1 degree tiles, 30 m). Not available through Sentinel Hub on the free CDSE tier.
- **How:** reading these tiles is slow (many HTTP round trips), so `fetch` downloads the native 30 m window covering each fire's largest scene once into `data/dem_raw/`, 8 fires in parallel, then resamples it (bilinear) onto each scene set's grid into `<set>/dem/`.
- **License:** Copernicus DEM, © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA.

## Setup

- **Python 3.10 or newer** (developed with 3.14) and the packages in `requirements.txt`.
- **Internet access** for `select` and `fetch`; `build`, `compare`, `maps` and `selftest` run offline once `fetch` has run.
- **Disk space:** about 2–3 GB for the default 100-fire run.

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

## Running

```bash
python main.py                                   # everything: select, fetch, build, compare, maps
python main.py --estimate-only                   # CDSE processing units still needed, nothing downloaded
python main.py --stages build compare maps       # re-run the analysis on downloaded data (offline, ~5 min + maps)
python main.py --stages compare                  # tables and plots only (~1 min)
python main.py --stages maps --map-fires 244088  # redraw the map of one fire
python main.py --stages selftest                 # synthetic check of the comparison method
python main.py --new-sample --max-fires 40 --country ES   # draw a different sample: fewer fires, Spain only
python main.py --help                            # all options
```

`fetch` and `build` work on the fires listed in `data/selected_fires.csv`, which `select` writes, so options that change the selection (`--new-sample`, `--fire-id`, the exclude list) need the `select` stage in the same run. A new sample takes the first `--max-fires` (default 100) fires, in a fixed random order (seed 42), from those between `--start-date` and `--end-date` (default 2018-01-01 to 2025-12-31) that are at least `--min-area-ha` (default 50 ha) and not excluded.

## File structure

```text
.
├── main.py                          # command line; runs the stages in order
├── firedetect/
│   ├── config.py                    # scene sets, zones, indices and models, thresholds, endpoints
│   ├── utils.py                     # shared helpers (column detection, safe names, exclude list, run log)
│   ├── effis.py                     # select: download EFFIS perimeters, pick fires
│   ├── cdse.py                      # credentials, OAuth token, Sentinel Hub requests
│   ├── sentinel2.py                 # fetch: pre-fire, prior-year and post-fire scenes, cost estimate
│   ├── worldcover.py                # fetch: land-cover patch per scene
│   ├── terrain.py                   # fetch: DEM patches; slope, hillshade
│   ├── raster.py                    # read GeoTIFF, cloud mask, rasterize, remote tiles onto a grid
│   ├── indices.py                   # NDVI, Temporal NDVI, DFI, NDMI, NBR, neighbourhood shares
│   ├── landscape.py                 # dry clumps, distance to firebreak, valley/ridge position
│   ├── dataset.py                   # build: per-scene layers, burned / unburned sampling
│   ├── bayes.py                     # histogram classifier, naive Bayes combination
│   ├── evaluate.py                  # compare: leave-one-fire-out and the result tables
│   ├── plots.py                     # histograms, posteriors, ROC curves
│   └── maps.py                      # maps: flagged vs burned pixels per fire
├── tests/
│   └── selftest_synthetic.py
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   ├── authentication.env.example   # template, committed
│   ├── authentication.env           # your credentials, git-ignored
│   ├── selected_fires.csv           # the frozen 100-fire sample, committed
│   └── excluded_fires.csv           # fires left out, with reasons, committed
├── data/                            # generated, git-ignored
│   ├── effis_cache/, effis_iberia_2018-2025.gpkg, fires_region.gpkg, selected_fires.csv
│   ├── sentinel2/, baseline/, postfire/, worldcover/, dem/, dem_raw/
│   ├── dataset.csv, fire_summary.csv
│   └── context/                     # sentinel2/, baseline/, worldcover/, dem/, dataset.csv
├── outputs/                         # generated, git-ignored
│   └── fire_maps/
└── .venv/                           # git-ignored
```

## Reference

This project follows the workflow described in the ELG 5163 assignment brief, using Sentinel-2 L2A imagery and burnt-area perimeters as labels for a pre-fire burn prediction comparison.

Data credits: EFFIS / Copernicus Emergency Management Service (burnt areas); Copernicus Sentinel-2 data via the Copernicus Data Space Ecosystem; ESA WorldCover 2021 (land cover); Copernicus DEM GLO-30 (elevation).
