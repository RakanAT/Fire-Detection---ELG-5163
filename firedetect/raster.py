"""Raster operations: read GeoTIFFs, cloud masking, rasterizing perimeters and land-cover masking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

from .config import VALID_SCL_CLASSES, WORLDCOVER_VEGETATION


def read_raster(path: Path):
    """Return (bands array [band, row, col], affine transform, CRS)."""
    with rasterio.open(path) as src:
        return src.read(), src.transform, src.crs


def clear_sky_mask(scl: np.ndarray, data_mask: np.ndarray) -> np.ndarray:
    """Keep pixels with data whose scene class is vegetation or bare soil (drops cloud, shadow, water, snow)."""
    return (data_mask == 1) & np.isin(scl, VALID_SCL_CLASSES)


def rasterize_geom(geom, transform, shape_hw) -> np.ndarray:
    """Boolean mask of the pixels whose centre falls inside geom."""
    return rasterize([(geom, 1)], out_shape=shape_hw, transform=transform, fill=0, dtype="uint8").astype(bool)


def worldcover_mask(worldcover_path: str | None, transform, crs, shape_hw) -> np.ndarray:
    """Boolean mask of tree cover, shrubland and grassland; all True when no WorldCover raster is given."""
    if worldcover_path is None:
        return np.ones(shape_hw, dtype=bool)
    # Resample WorldCover onto the Sentinel-2 grid so any extent/projection works (e.g. a regional mosaic).
    worldcover = np.zeros(shape_hw, dtype=np.uint8)
    with rasterio.open(worldcover_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=worldcover,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return np.isin(worldcover, WORLDCOVER_VEGETATION)
