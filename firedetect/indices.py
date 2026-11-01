"""Vegetation indices computed per pixel from Sentinel-2 L2A reflectance bands.

    NDVI          = (NIR - Red) / (NIR + Red)                     B08, B04
    Temporal NDVI = NDVI - mean NDVI of the same window in prior years
    DFI           = 100 * (1 - SWIR2 / SWIR1) * (Red / NIR)       B12, B11, B04, B08
"""

from __future__ import annotations

import numpy as np


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red
    return np.divide(nir - red, denom, out=np.zeros_like(red, dtype=np.float32), where=denom != 0)


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
