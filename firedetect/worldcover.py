"""ESA WorldCover 10 m land cover (2021, v200), read on demand from ESA's public bucket.

The tiles are 3 x 3 degree Cloud-Optimised GeoTIFFs in EPSG:4326, so only the small window around each
scene is downloaded. The patch is resampled (nearest neighbour) onto the scene's grid and cached as
<scene set>/worldcover/<scene name>.tif. No account or CDSE quota is needed.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling

from .raster import cached_grid_layer, tiles_onto_grid

LOGGER = logging.getLogger(__name__)

TILE_URL = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
TILE_DEG = 3


def tile_urls(west: float, south: float, east: float, north: float) -> list[str]:
    """URLs of the 3-degree tiles covering a lon/lat box; tiles are named by their lower-left corner, e.g. N42W009."""
    urls = []
    for lat in range(math.floor(south / TILE_DEG) * TILE_DEG, math.ceil(north / TILE_DEG) * TILE_DEG, TILE_DEG):
        for lon in range(math.floor(west / TILE_DEG) * TILE_DEG, math.ceil(east / TILE_DEG) * TILE_DEG, TILE_DEG):
            tile = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            urls.append(TILE_URL.format(tile=tile))
    return urls


def land_cover(transform, crs, shape_hw, cache_path: Path) -> np.ndarray:
    """WorldCover classes on the given grid (0 where there is no tile, e.g. open sea)."""
    return cached_grid_layer(cache_path, transform, crs, shape_hw,
                             lambda: tiles_onto_grid(tile_urls, transform, crs, shape_hw, np.uint8, 0, Resampling.nearest))


def fetch_for_scene(scene_path: Path, cache_dir: Path) -> Path:
    """Download and cache the WorldCover patch matching a downloaded Sentinel-2 scene."""
    cache_path = cache_dir / scene_path.name
    if cache_path.exists():
        LOGGER.info("WorldCover already cached, skipping: %s", cache_path)
        return cache_path
    with rasterio.open(scene_path) as src:
        land_cover(src.transform, src.crs, (src.height, src.width), cache_path)
    LOGGER.info("Saved WorldCover patch to %s", cache_path)
    return cache_path
