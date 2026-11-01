"""Project-wide constants: paths, study regions, band layout and service endpoints."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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
NATIVE_RESOLUTION_M = 10.0
MAX_SIDE_PX = 2500  # very large fires are coarsened so no side exceeds this
MAX_CLOUD_COVER = 100.0  # tile-level filter; leastCC mosaicking and SCL masking handle clouds

INDEX_LABELS = {"ndvi": "NDVI", "temporal_ndvi": "Temporal NDVI", "dfi": "DFI"}

VALID_SCL_CLASSES = (4, 5)  # Sentinel-2 scene classification: vegetation, bare soil
WORLDCOVER_VEGETATION = (10, 20, 30)  # tree cover, shrubland, grassland
