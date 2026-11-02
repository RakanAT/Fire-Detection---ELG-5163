"""Vegetation indices computed per pixel from Sentinel-2 L2A reflectance bands.

    NDVI          = (NIR - Red) / (NIR + Red)                     B08, B04   greenness
    Temporal NDVI = NDVI - mean NDVI of the same window in prior years       greenness vs usual
    DFI           = 100 * (1 - SWIR2 / SWIR1) * (Red / NIR)       B12, B11, B04, B08   dead fuel
    NDMI          = (NIR - SWIR1) / (NIR + SWIR1)                 B08, B11   vegetation water content
    NBR           = (NIR - SWIR2) / (NIR + SWIR2)                 B08, B12   fuel / burn ratio
    Dead fuel within r = share of vegetation pixels within a (2r x 2r) window with NDMI < 0   fuel continuity
Post-fire, dNDVI and dNBR (pre-fire minus post-fire) measure the burn scar, and RBR its severity.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), 0 where a + b is 0."""
    denom = a + b
    return np.divide(a - b, denom, out=np.zeros_like(a, dtype=np.float32), where=denom != 0)


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return normalized_difference(nir, red)


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Normalised Difference Moisture Index; lower values mean drier vegetation."""
    return normalized_difference(nir, swir1)


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalised Burn Ratio; high for healthy vegetation, low for dry fuel and burn scars."""
    return normalized_difference(nir, swir2)


def neighbourhood_fraction(flag: np.ndarray, counted: np.ndarray, radius_px: float) -> np.ndarray:
    """Share of counted pixels that are flagged, in a square window of side 2 * radius_px + 1 around each pixel.

    A moving-average (box) filter of flag * counted divided by one of counted, so pixels that are not counted
    (cloud, non-vegetation, outside the image) are ignored rather than treated as not flagged. NaN where the
    window holds no counted pixel.
    """
    size = 2 * int(round(radius_px)) + 1
    counted = counted.astype(np.float32)
    flagged = uniform_filter((flag & (counted > 0)).astype(np.float32), size=size, mode="constant")
    total = uniform_filter(counted, size=size, mode="constant")
    return np.divide(flagged, total, out=np.full(flag.shape, np.nan, dtype=np.float32), where=total > 1e-6)


def mean_ndvi(ndvi_layers: list[np.ndarray], clear_layers: list[np.ndarray]) -> np.ndarray:
    """Per-pixel mean NDVI over the layers where that pixel is clear; NaN where it is never clear."""
    ndvi_stack = np.stack(ndvi_layers)
    clear_stack = np.stack(clear_layers)
    total = np.where(clear_stack, ndvi_stack, 0.0).sum(axis=0)
    count = clear_stack.sum(axis=0)
    return np.divide(total, count, out=np.full(total.shape, np.nan, dtype=np.float32), where=count > 0)


def temporal_ndvi(current_ndvi: np.ndarray, baseline_ndvi: np.ndarray) -> np.ndarray:
    """Negative where vegetation is less green than its usual state for the time of year."""
    return current_ndvi - baseline_ndvi


def dfi(red: np.ndarray, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Dead Fuel Index; higher values indicate more dry, dead vegetation."""
    swir_ratio = np.divide(swir2, np.clip(swir1, 1e-6, None), out=np.zeros_like(swir1, dtype=np.float32))
    red_nir = np.divide(red, nir, out=np.zeros_like(red, dtype=np.float32), where=nir != 0)
    return 100.0 * (1.0 - swir_ratio) * red_nir
