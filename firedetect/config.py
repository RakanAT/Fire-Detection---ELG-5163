"""Project-wide constants: paths, scene sets, indices, study regions, band layout and service endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SceneSet:
    """A set of co-registered rasters per fire, all on one grid: the fire's bounding box plus margin_m, in UTM.

    MAIN is the 10 m set the main comparison uses (unburned ring up to 250 m). CONTEXT is a wider 20 m set
    used to compare the burned area with unburned vegetation 1-3 km away.
    """

    root: Path
    margin_m: float
    min_resolution_m: float

    @property
    def prefire_dir(self) -> Path:
        return self.root / "sentinel2"

    @property
    def baseline_dir(self) -> Path:
        return self.root / "baseline"

    @property
    def postfire_dir(self) -> Path:
        return self.root / "postfire"

    @property
    def worldcover_dir(self) -> Path:
        return self.root / "worldcover"

    @property
    def dem_dir(self) -> Path:
        return self.root / "dem"


MAIN_SCENES = SceneSet(ROOT / "data", margin_m=500.0, min_resolution_m=10.0)
CONTEXT_SCENES = SceneSet(ROOT / "data" / "context", margin_m=3000.0, min_resolution_m=20.0)

# Zones sampled from each scene set: (name, min distance, max distance) from the perimeter in metres.
# Burned pixels are always taken inside the eroded perimeter.
MAIN_ZONES = [("near", 0.0, 250.0)]
CONTEXT_ZONES = [("near", 0.0, 250.0), ("far", 1000.0, 3000.0)]

# Indices and models compared (see README "Indices"). Each single index is a histogram classifier; the named
# combinations multiply their features' likelihoods (naive Bayes).
PRE_FIRE_INDICES = ["ndvi", "temporal_ndvi", "dfi", "ndmi", "nbr", "dead_fuel_100m", "dead_fuel_300m",
                    "dry_patch_ha", "dist_large_dry_patch_m"]
# Landscape factors known before a fire. Not vegetation indices: a reference for what the landscape alone explains.
LANDSCAPE_FACTORS = ["slope", "tpi_300m", "dist_nonfuel_m"]
CONTROL_INDEX = "dndvi"  # post-fire, so not a prediction; checks that the burned-area labels are right
CATEGORICAL_FEATURES = ["land_cover"]  # one histogram bin per class instead of value ranges
COMBINED_MODELS = {
    "combined": PRE_FIRE_INDICES,
    "landscape": LANDSCAPE_FACTORS + ["land_cover"],
    "combined_landscape": PRE_FIRE_INDICES + LANDSCAPE_FACTORS + ["land_cover"],
}
INDEX_LABELS = {
    "ndvi": "NDVI", "temporal_ndvi": "Temporal NDVI", "dfi": "DFI", "ndmi": "NDMI", "nbr": "NBR",
    "dead_fuel_100m": "Dead fuel within 100 m", "dead_fuel_300m": "Dead fuel within 300 m",
    "dry_patch_ha": "Dry clump size", "dist_large_dry_patch_m": "Distance to large dry clump",
    "slope": "Slope", "tpi_300m": "Valley vs ridge (TPI)", "dist_nonfuel_m": "Distance to firebreak", "land_cover": "Land cover",
    "dndvi": "dNDVI (post-fire control)",
    "combined": "Combined indices (naive Bayes)",
    "landscape": "Landscape only (naive Bayes)",
    "combined_landscape": "Combined indices + landscape (naive Bayes)",
}

# Dry vegetation: NDMI < 0, i.e. more SWIR1 than NIR reflectance (low leaf water); ~40% of sampled vegetation pixels.
DEAD_FUEL_NDMI = 0.0
NEIGHBOURHOOD_RADII_M = {"dead_fuel_100m": 100.0, "dead_fuel_300m": 300.0}
# Dry clumps: dry vegetation after a morphological opening that removes isolated pixels and thin lines,
# grouped into 8-connected patches.
DRY_PATCH_OPENING_M = 20.0
LARGE_DRY_PATCH_HA = 5.0
# WorldCover classes that carry little fuel and act as firebreaks: cropland, built-up, bare, snow, water,
# wetland, mangrove, moss.
NON_FUEL_CLASSES = (40, 50, 60, 70, 80, 90, 95, 100)
TPI_RADIUS_M = 300.0  # topographic position: elevation minus the mean within this radius (+ ridge, - valley)
# Other fires within this many days of a fire may be part of the same event: excluded from its unburned zones.
# Land burned in earlier or later years counts as unburned by this fire.
CONCURRENT_FIRE_DAYS = 30

# Burn severity. dNBR (pre-fire NBR minus post-fire NBR) contains the pre-fire NBR itself, so pre-fire indices
# would predict it partly by construction. RBR = dNBR / (pre-fire NBR + 1.001) (Parks et al. 2014) removes most of
# that dependence; the severity analysis compares the more severely burned half of burned pixels (RBR above its
# median) with the less severely burned half.
RBR_OFFSET = 1.001

POST_FIRE_DAYS = (5, 45)  # post-fire window: from 5 to 45 days after the fire date
# Post-fire pixels used for dNDVI / dNBR: SCL dark area (often charcoal), vegetation, bare soil, unclassified.
POST_FIRE_SCL_CLASSES = (2, 4, 5, 7)
WEAK_SCAR_AUC = 0.75  # a fire whose own dNDVI separates burned from unburned worse than this has a doubtful perimeter

# Pre-burn check (dataset.pre_burn_check): a "pre-fire" scene that already shows the burn scar.
PRE_BURN_NDVI_DROP = 0.2  # NDVI fell this much more inside the perimeter than in the ring (vs prior years)
PRE_BURN_SWIR_EXCESS = 0.2  # B12/B11 this much higher inside the perimeter than in the ring

VALID_SCL_CLASSES = (4, 5)  # Sentinel-2 scene classification: vegetation, bare soil
WORLDCOVER_VEGETATION = (10, 20, 30)  # tree cover, shrubland, grassland
LAND_COVER_NAMES = {10: "tree", 20: "shrub", 30: "grass"}

# Approximate lon/lat bounding boxes (W, S, E, N), plus country codes used to drop the parts of
# neighbouring countries a box also covers (needs a country attribute, e.g. EFFIS COUNTRY).
# Pass --region-file with a boundary polygon for an exact clip instead.
REGIONS = {
    # Mainland Spain + Portugal + Balearics; the box also touches southern France, Andorra and Morocco.
    "iberia": {"bbox": (-9.6, 35.9, 4.4, 43.9), "countries": ["ES", "PT", "ESP", "PRT", "SPAIN", "PORTUGAL"]},
    "ontario": {"bbox": (-95.16, 41.68, -74.34, 56.86), "countries": None},
}

# Exact (case-insensitive) attribute names, in order of preference. Substring matching is avoided
# because e.g. "fire" also matches FIREDATE and "date" also matches LASTUPDATE.
ID_CANDIDATES = ("fire_id", "id", "fireid", "nfireid", "fire_number", "objectid")
DATE_CANDIDATES = ("fire_date", "firedate", "afsdate", "start_date", "startdate", "date")
COUNTRY_CANDIDATES = ("country", "iso2", "iso", "country_code", "cntr_code")

# EFFIS WFS: one burnt-area polygon layer per year (ms:modis.ba.poly.<year>), MODIS-mapped fires >= ~30 ha.
EFFIS_WFS = "https://maps.effis.emergency.copernicus.eu/effis"

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SENTINELHUB_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 6

# Band order of every downloaded GeoTIFF (see sentinel2.EVALSCRIPT).
BANDS = ("B03", "B04", "B08", "B11", "B12", "SCL", "dataMask")
MAX_SIDE_PX = 2500  # very large fires are coarsened so no side exceeds this
MAX_CLOUD_COVER = 100.0  # tile-level filter; leastCC mosaicking and SCL masking handle clouds
