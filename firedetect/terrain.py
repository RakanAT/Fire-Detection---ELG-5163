"""Terrain from the Copernicus DEM GLO-30 (30 m), read from its public bucket.

The tiles are 1 x 1 degree Cloud-Optimised GeoTIFFs in EPSG:4326. Reading them is slow (many HTTP round trips),
so per fire the native 30 m window covering its largest scene is downloaded once into data/dem_raw/<fire id>.tif,
several fires in parallel. Each scene set's patch is then resampled locally (bilinear) from that window onto
the scene's grid and cached as <scene set>/dem/<scene name>.tif. No account or CDSE quota is needed.

From the elevation: slope (degrees from horizontal) and, for map backgrounds, shaded relief. The valley/ridge
position is in landscape.py.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import Resampling, transform_bounds

from .raster import PAD_DEG, cached_grid_layer, reproject_file_onto_grid, tiles_onto_grid

LOGGER = logging.getLogger(__name__)

TILE_URL = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/"
            "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif")
NODATA = -32767.0
NATIVE_DEG = 1 / 3600  # 1 arc-second
PARALLEL_DOWNLOADS = 8


def tile_urls(west: float, south: float, east: float, north: float) -> list[str]:
    """URLs of the 1-degree tiles covering a lon/lat box; tiles are named by their lower-left corner, e.g. N43 W006."""
    return [TILE_URL.format(lat=f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}", lon=f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}")
            for lat in range(math.floor(south), math.ceil(north))
            for lon in range(math.floor(west), math.ceil(east))]


def _download_raw(scene_path: Path, raw_path: Path) -> Path:
    """Native-resolution DEM window (EPSG:4326) covering a scene, cached at raw_path."""
    with rasterio.open(scene_path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *array_bounds(src.height, src.width, src.transform))
    west, south = west - 2 * PAD_DEG, south - 2 * PAD_DEG
    width = math.ceil((east + 2 * PAD_DEG - west) / NATIVE_DEG)
    height = math.ceil((north + 2 * PAD_DEG - south) / NATIVE_DEG)
    transform = from_origin(west, south + height * NATIVE_DEG, NATIVE_DEG, NATIVE_DEG)
    cached_grid_layer(raw_path, transform, "EPSG:4326", (height, width),
                      lambda: tiles_onto_grid(tile_urls, transform, "EPSG:4326", (height, width), np.float32, NODATA, Resampling.nearest))
    return raw_path


def elevation(transform, crs, shape_hw, cache_path: Path, raw_path: Path | None = None) -> np.ndarray:
    """Elevation in metres on the given grid (NaN where there is no DEM, e.g. open sea).

    Read from cache_path, else resampled from the fire's raw window at raw_path."""
    dem = cached_grid_layer(cache_path, transform, crs, shape_hw,
                            lambda: reproject_file_onto_grid(raw_path, transform, crs, shape_hw, np.float32, NODATA, Resampling.bilinear))
    return np.where(dem == NODATA, np.nan, dem).astype(np.float32)


def slope_aspect(dem: np.ndarray, pixel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (degrees clockwise from north, the direction the slope faces)."""
    d_row, d_col = np.gradient(dem, pixel_size_m)  # rows increase southwards, columns eastwards
    slope = np.degrees(np.arctan(np.hypot(d_row, d_col)))
    # Downslope direction is -gradient: east component -d_col, north component +d_row.
    aspect = (np.degrees(np.arctan2(-d_col, d_row)) + 360.0) % 360.0
    return slope.astype(np.float32), aspect.astype(np.float32)


def hillshade(dem: np.ndarray, pixel_size_m: float, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Shaded relief (0..1) lit from the north-west, for map backgrounds."""
    slope, aspect = slope_aspect(np.nan_to_num(dem, nan=float(np.nanmean(dem)) if np.isfinite(dem).any() else 0.0), pixel_size_m)
    zenith, slope_r, azimuth_r, aspect_r = map(np.radians, (90.0 - altitude, slope, azimuth, aspect))
    shade = np.cos(zenith) * np.cos(slope_r) + np.sin(zenith) * np.sin(slope_r) * np.cos(azimuth_r - aspect_r)
    return np.clip(shade, 0, 1)


def _width_m(scene_path: Path) -> float:
    with rasterio.open(scene_path) as src:
        return src.bounds.right - src.bounds.left


def _patches_for_fire(fire_key: str, scenes: list[tuple[Path, Path]], raw_dir: Path) -> list[Path]:
    """Raw window from the largest scene, then one patch per (pre-fire scene, DEM directory)."""
    outputs = [dem_dir / scene.name for scene, dem_dir in scenes]
    if all(p.exists() for p in outputs):
        return outputs
    raw_path = _download_raw(max((scene for scene, _ in scenes), key=_width_m), raw_dir / f"{fire_key}.tif")
    for (scene, _), out in zip(scenes, outputs):
        with rasterio.open(scene) as src:
            elevation(src.transform, src.crs, (src.height, src.width), out, raw_path)
    return outputs


def fetch_patches(jobs: dict[str, list[tuple[Path, Path]]], raw_dir: Path, workers: int = PARALLEL_DOWNLOADS) -> list[str]:
    """DEM patches for many fires in parallel. jobs maps fire id -> [(pre-fire scene, DEM directory), ...].

    Returns the fire ids that failed.
    """
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_patches_for_fire, key, scenes, raw_dir): key for key, scenes in jobs.items()}
        for done, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                future.result()
                print(f"DEM [{done}/{len(jobs)}] fire {key} OK", flush=True)
            except Exception as exc:
                LOGGER.error("DEM failed for fire %s: %s", key, exc)
                failed.append(key)
    return failed
