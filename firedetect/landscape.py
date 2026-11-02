"""Spatial (neighbourhood and landscape) features, computed per pixel on a scene's grid.

    dry_patch_ha            area of the clump of dry vegetation the pixel belongs to (0 outside clumps):
                            dry pixels (NDMI < 0) -> morphological opening (drops isolated pixels and thin lines)
                            -> 8-connected component labelling -> area per component
    dist_large_dry_patch_m  distance to the nearest clump of at least LARGE_DRY_PATCH_HA
    dist_nonfuel_m          distance to the nearest firebreak-like land cover (cropland, built-up, bare, water, ...)
    tpi_300m                topographic position: elevation minus the mean elevation within 300 m (+ ridge, - valley)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_opening, distance_transform_edt, label, uniform_filter

from .config import DRY_PATCH_OPENING_M, LARGE_DRY_PATCH_HA, NON_FUEL_CLASSES, TPI_RADIUS_M


def _disk(radius_px: int) -> np.ndarray:
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return x * x + y * y <= radius_px * radius_px


def _distance_to(mask: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Distance (m) from each pixel to the nearest True pixel; the image diagonal where there is none."""
    if not mask.any():
        return np.full(mask.shape, np.hypot(*mask.shape) * pixel_size_m, dtype=np.float32)
    return (distance_transform_edt(~mask) * pixel_size_m).astype(np.float32)


def dry_patches(dry: np.ndarray, vegetation: np.ndarray, pixel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """(clump area in ha per pixel, distance in m to the nearest large clump) for dry vegetation pixels."""
    radius_px = max(1, round(DRY_PATCH_OPENING_M / pixel_size_m))
    clumps = binary_opening(dry & vegetation, structure=_disk(radius_px))
    labels, _ = label(clumps, structure=np.ones((3, 3), dtype=bool))
    area_ha = np.bincount(labels.ravel()) * pixel_size_m ** 2 / 1e4
    area_ha[0] = 0.0  # background
    patch_ha = area_ha[labels].astype(np.float32)
    return patch_ha, _distance_to(patch_ha >= LARGE_DRY_PATCH_HA, pixel_size_m)


def distance_to_nonfuel(land_cover: np.ndarray, pixel_size_m: float) -> np.ndarray:
    return _distance_to(np.isin(land_cover, NON_FUEL_CLASSES), pixel_size_m)


def topographic_position(elevation: np.ndarray, pixel_size_m: float) -> np.ndarray:
    filled = np.where(np.isfinite(elevation), elevation, np.nanmean(elevation))
    size = 2 * round(TPI_RADIUS_M / pixel_size_m) + 1
    return np.where(np.isfinite(elevation), filled - uniform_filter(filled, size=size, mode="nearest"), np.nan).astype(np.float32)
