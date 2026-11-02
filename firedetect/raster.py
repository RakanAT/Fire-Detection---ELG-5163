"""Raster operations: read GeoTIFFs, cloud masking, rasterizing perimeters, land-cover masking, and reading
public lon/lat tile sets (WorldCover, DEM) onto a scene's grid."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

from .config import VALID_SCL_CLASSES, WORLDCOVER_VEGETATION

LOGGER = logging.getLogger(__name__)
PAD_DEG = 0.002  # read slightly beyond the scene so edge pixels have source data
# GDAL settings for reading remote Cloud-Optimised GeoTIFFs with few, merged HTTP range requests.
REMOTE_COG_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MULTIRANGE": "YES",
                  "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES", "VSI_CACHE": "TRUE", "GDAL_HTTP_VERSION": "2",
                  "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif", "GDAL_INGESTED_BYTES_AT_OPEN": "32768"}


def read_raster(path: Path):
    """Return (bands array [band, row, col], affine transform, CRS)."""
    with rasterio.open(path) as src:
        return src.read(), src.transform, src.crs


def clear_sky_mask(scl: np.ndarray, data_mask: np.ndarray, classes=VALID_SCL_CLASSES) -> np.ndarray:
    """Keep pixels with data whose scene class is in classes (default vegetation or bare soil; drops cloud, shadow, water, snow)."""
    return (data_mask == 1) & np.isin(scl, classes)


def _read_window(src, west: float, south: float, east: float, north: float):
    """Read the part of an open lon/lat tile inside a box; returns (array, transform) or None if they don't overlap."""
    inv = ~src.transform
    col0, row0 = inv * (west, north)
    col1, row1 = inv * (east, south)
    col0, row0 = max(0, math.floor(col0)), max(0, math.floor(row0))
    col1, row1 = min(src.width, math.ceil(col1)), min(src.height, math.ceil(row1))
    if col1 <= col0 or row1 <= row0:
        return None
    window = Window(col0, row0, col1 - col0, row1 - row0)
    return src.read(1, window=window), src.window_transform(window)


def tiles_onto_grid(tile_urls, transform, crs, shape_hw, dtype, nodata, resampling: Resampling) -> np.ndarray:
    """Mosaic the parts of remote EPSG:4326 tiles (Cloud-Optimised GeoTIFFs) covering a grid, resampled onto it.

    tile_urls(west, south, east, north) lists the URLs of the tiles covering a lon/lat box; missing tiles
    (e.g. open sea) are skipped and leave nodata.
    """
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *array_bounds(*shape_hw, transform))
    west, south, east, north = west - PAD_DEG, south - PAD_DEG, east + PAD_DEG, north + PAD_DEG
    result = np.full(shape_hw, nodata, dtype=dtype)
    with rasterio.Env(**REMOTE_COG_ENV):
        for url in tile_urls(west, south, east, north):
            try:
                with rasterio.open(url) as src:
                    patch = _read_window(src, west, south, east, north)
            except RasterioIOError:
                LOGGER.info("No tile at %s (outside coverage); skipping.", url)
                continue
            if patch is None:
                continue
            data, patch_transform = patch
            resampled = np.full(shape_hw, nodata, dtype=dtype)
            reproject(source=data.astype(dtype), destination=resampled, src_transform=patch_transform, src_crs="EPSG:4326",
                      dst_transform=transform, dst_crs=crs, src_nodata=nodata, dst_nodata=nodata, resampling=resampling)
            result = np.where(resampled != nodata, resampled, result)
    return result


def reproject_file_onto_grid(path: Path, transform, crs, shape_hw, dtype, nodata, resampling: Resampling) -> np.ndarray:
    """Resample band 1 of a local raster onto a grid."""
    result = np.full(shape_hw, nodata, dtype=dtype)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=result, src_nodata=nodata, dst_transform=transform,
                  dst_crs=crs, dst_nodata=nodata, resampling=resampling)
    return result


def cached_grid_layer(cache_path: Path, transform, crs, shape_hw, make_layer) -> np.ndarray:
    """Read a single-band layer from cache_path, or compute it with make_layer() and cache it as a GeoTIFF."""
    if cache_path.exists():
        with rasterio.open(cache_path) as src:
            return src.read(1)
    layer = make_layer()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": shape_hw[0], "width": shape_hw[1], "count": 1, "dtype": layer.dtype.name,
               "crs": crs, "transform": transform, "compress": "deflate"}
    # Write then rename, so an interrupted run never leaves a partial file that a re-run would treat as cached.
    partial = cache_path.with_suffix(".part.tif")
    with rasterio.open(partial, "w", **profile) as dst:
        dst.write(layer, 1)
    partial.replace(cache_path)
    return layer


def rasterize_geom(geom, transform, shape_hw) -> np.ndarray:
    """Boolean mask of the pixels whose centre falls inside geom."""
    return rasterize([(geom, 1)], out_shape=shape_hw, transform=transform, fill=0, dtype="uint8").astype(bool)


def vegetation_mask(land_cover: np.ndarray) -> np.ndarray:
    """Keep tree cover, shrubland and grassland; drops cropland, built-up, bare ground and water."""
    return np.isin(land_cover, WORLDCOVER_VEGETATION)
