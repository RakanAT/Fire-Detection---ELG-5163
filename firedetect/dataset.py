"""Build the burned vs unburned pixel dataset from the downloaded pre-fire scenes.

Per scene: compute the indices (Temporal NDVI from the fire's prior-year scenes in the baseline directory),
mask clouds (and optionally non-vegetation), rasterize the fire perimeter
in the scene's UTM grid, then sample
    burned pixels   - inside the perimeter eroded by erode_m (avoids mixed edge pixels)
    unburned pixels - in a buffer_m ring outside the perimeter, excluding any other fire
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import array_bounds
from shapely.geometry import box

from . import indices
from .config import BANDS, ID_CANDIDATES
from .raster import clear_sky_mask, rasterize_geom, read_raster, worldcover_mask
from .utils import find_column, safe_name

LOGGER = logging.getLogger(__name__)


def _baseline_ndvi(baseline_dir: Path, fire_key: str, transform, shape_hw) -> np.ndarray | None:
    """Mean prior-year NDVI on the scene's grid (NaN where no prior year is clear), or None if there are no baselines."""
    ndvi_layers, clear_layers = [], []
    for tif in sorted(baseline_dir.glob(f"{fire_key}__*.tif")):
        data, base_transform, _ = read_raster(tif)
        if data.shape[1:] != shape_hw or base_transform != transform:
            LOGGER.warning("Baseline %s is not on the same grid as the pre-fire scene; ignoring it.", tif.name)
            continue
        b04, b08, scl, data_mask = (data[i].astype(np.float32) for i in (1, 2, 5, 6))
        ndvi_layers.append(indices.ndvi(b04, b08))
        clear_layers.append(clear_sky_mask(scl, data_mask))
    if not ndvi_layers:
        return None
    return indices.mean_ndvi(ndvi_layers, clear_layers)


def _sample(mask: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    picks = np.random.default_rng(seed).choice(len(rows), size=min(len(rows), n), replace=False)
    return rows[picks], cols[picks]


def build_dataset(image_dir: str, fires_file: str, output_csv: str, baseline_dir: str | None = None, buffer_m: float = 250.0,
                  erode_m: float = 20.0, max_samples_per_class: int = 5000, worldcover_raster: str | None = None) -> Path:
    """Without baseline_dir the temporal_ndvi column is left empty; otherwise fires without baselines are skipped."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    perimeters = gpd.read_file(fires_file)
    if perimeters.empty:
        raise ValueError(f"{fires_file} is empty.")
    if perimeters.crs is None:
        raise ValueError(f"{fires_file} has no CRS; cannot align perimeters with the imagery.")
    id_col = find_column(perimeters.columns, ID_CANDIDATES)
    if id_col is None:
        raise ValueError(f"Could not locate a fire-id column (tried {ID_CANDIDATES}) in {fires_file}.")
    perimeters = perimeters[perimeters.geometry.notna() & ~perimeters.geometry.is_empty].copy()
    perimeters["_key"] = perimeters[id_col].map(safe_name)

    tables = []
    for fire_index, tif_path in enumerate(sorted(image_dir.glob("**/*.tif"))):
        try:
            data, transform, crs = read_raster(tif_path)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable raster %s: %s", tif_path, exc)
            continue
        if data.shape[0] < len(BANDS):
            LOGGER.warning("Raster %s does not contain bands %s; skipping.", tif_path, ", ".join(BANDS))
            continue

        # Filenames are "<fire id>__<start>_<end>.tif" (see sentinel2.scene_path).
        fire_key = tif_path.stem.split("__")[0]
        if not (perimeters["_key"] == fire_key).any():
            LOGGER.warning("No perimeter with id %r for %s; skipping.", fire_key, tif_path.name)
            continue

        b03, b04, b08, b11, b12, scl, data_mask = (band.astype(np.float32) for band in data[:7])
        shape_hw = b04.shape
        ndvi = indices.ndvi(b04, b08)
        dfi = indices.dfi(b04, b08, b11, b12)
        valid = clear_sky_mask(scl, data_mask) & worldcover_mask(worldcover_raster, transform, crs, shape_hw)

        if baseline_dir:
            baseline = _baseline_ndvi(Path(baseline_dir), fire_key, transform, shape_hw)
            if baseline is None:
                LOGGER.warning("No baseline scenes for %s in %s; skipping (run the fetch stage).", tif_path.name, baseline_dir)
                continue
            temporal_ndvi = indices.temporal_ndvi(ndvi, baseline)
            # Every index is sampled on the same pixels, so only pixels with a clear prior year are kept.
            valid &= np.isfinite(baseline)
        else:
            temporal_ndvi = np.full(shape_hw, np.nan, dtype=np.float32)

        # Only perimeters overlapping this image matter; reproject them to the image's UTM grid
        # so buffer/erosion distances are in metres.
        footprint = gpd.GeoSeries([box(*array_bounds(*shape_hw, transform))], crs=crs).to_crs(perimeters.crs).iloc[0]
        nearby = perimeters[perimeters.intersects(footprint)].to_crs(crs)
        is_target = nearby["_key"] == fire_key
        if not is_target.any():
            LOGGER.warning("Perimeter %r does not overlap %s; skipping.", fire_key, tif_path.name)
            continue
        fire_geom = nearby[is_target].union_all()
        other_fires = nearby[~is_target]

        fire_mask = rasterize_geom(fire_geom, transform, shape_hw)
        core = fire_geom.buffer(-erode_m)
        burned_mask = rasterize_geom(core, transform, shape_hw) if not core.is_empty else fire_mask
        ring_mask = rasterize_geom(fire_geom.buffer(buffer_m), transform, shape_hw) & ~fire_mask
        if not other_fires.empty:
            ring_mask &= ~rasterize_geom(other_fires.union_all(), transform, shape_hw)

        burned_mask &= valid
        ring_mask &= valid
        if not burned_mask.any() or not ring_mask.any():
            LOGGER.warning("No cloud-free burned/unburned pixels for %s; skipping.", tif_path.name)
            continue

        for burned, mask, seed in ((1, burned_mask, 42 + fire_index), (0, ring_mask, 43 + fire_index)):
            rows, cols = _sample(mask, max_samples_per_class, seed)
            tables.append(pd.DataFrame({
                "fire_id": fire_index,
                "fire_label": fire_key,
                "burned": burned,
                "ndvi": ndvi[rows, cols].astype(np.float64),
                "temporal_ndvi": temporal_ndvi[rows, cols].astype(np.float64),
                "dfi": dfi[rows, cols].astype(np.float64),
                "source_image": tif_path.name,
            }))
        LOGGER.info("%s: %d burned, %d unburned samples", tif_path.name, len(tables[-2]), len(tables[-1]))

    if not tables:
        raise ValueError("No valid burned/unburned pixels were produced. Check the imagery, masks, and fire polygons.")

    df = pd.concat(tables, ignore_index=True)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Dataset written to %s with %d rows.", output_path, len(df))
    return output_path
