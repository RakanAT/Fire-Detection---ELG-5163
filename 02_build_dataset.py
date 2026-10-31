#!/usr/bin/env python3
"""Build a per-pixel pre-fire dataset from Sentinel-2 L2A GeoTIFFs.

The script loads one or more GeoTIFFs, applies a Sentinel-2 SCL + dataMask cloud
mask, computes NDVI, DFI, and a pre-fire anomaly, rasterizes the burnt perimeters,
and creates a balanced burned-vs-unburned CSV sample ready for the Bayes
comparison step.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

LOGGER = logging.getLogger("build_dataset")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a raster dataset for burned-area modeling.")
    parser.add_argument("--image-dir", required=True, help="Directory containing the Sentinel-2 GeoTIFFs.")
    parser.add_argument("--effis-shp", required=True, help="EFFIS burnt-area perimeter shapefile.")
    parser.add_argument("--worldcover-raster", help="Optional WorldCover raster with classes 10, 20, 30 kept.")
    parser.add_argument("--baseline-dir", help="Optional directory of prior-year NDVI rasters for anomaly estimation.")
    parser.add_argument("--output-csv", default="data/dataset.csv", help="Output CSV path.")
    parser.add_argument("--max-samples-per-class", type=int, default=5000, help="Cap the number of sampled pixels per class per fire.")
    parser.add_argument("--buffer-m", type=float, default=250.0, help="Buffer radius used to build the unburned ring around each fire perimeter.")
    parser.add_argument("--erode-m", type=float, default=20.0, help="Inward erosion of the perimeter before sampling burned pixels (avoids mixed edge pixels).")
    return parser.parse_args()


# Must match 01_fetch_sentinel2.py so image filenames can be traced back to their fire.
ID_CANDIDATES = ("fire_id", "id", "fireid", "nfireid", "fire_number", "objectid")


def _find_matching_column(columns, *candidates: str) -> str | None:
    columns = list(columns)
    lowered = [str(c).lower() for c in columns]
    for candidate in candidates:
        if candidate.lower() in lowered:
            return columns[lowered.index(candidate.lower())]
    return None


def _safe_name(value) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(value).strip())


def _read_raster(path: Path):
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    return data, profile, transform, crs


def _ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red
    return np.divide(nir - red, denom, out=np.zeros_like(red, dtype=np.float32), where=denom != 0)


def _dfi(red: np.ndarray, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    denominator = np.clip(swir1, 1e-6, None)
    return 100.0 * (1.0 - np.divide(swir2, denominator, out=np.zeros_like(swir1, dtype=np.float32), where=denominator != 0)) * (np.divide(red, nir, out=np.zeros_like(red, dtype=np.float32), where=nir != 0))


def _valid_mask(scl: np.ndarray, data_mask: np.ndarray) -> np.ndarray:
    valid = (data_mask == 1) & (np.isin(scl, [4, 5]))
    return valid


def _build_mask_from_geoms(geometries, transform, shape_hw, fill_value: int = 1):
    if not geometries:
        return np.zeros(shape_hw, dtype=np.uint8)
    mask = rasterize(
        [(geom, fill_value) for geom in geometries],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def _worldcover_mask(worldcover_path: str | None, transform, crs, out_shape):
    if worldcover_path is None:
        return np.ones(out_shape, dtype=bool)
    # Resample WorldCover onto the Sentinel-2 grid so any extent/projection works (e.g. a regional mosaic).
    worldcover = np.zeros(out_shape, dtype=np.uint8)
    with rasterio.open(worldcover_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=worldcover,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return np.isin(worldcover, [10, 20, 30])


def _load_baseline_ndvi(baseline_dir: str | None, transform, out_shape):
    if not baseline_dir:
        return np.zeros(out_shape, dtype=np.float32)
    baseline_dir = Path(baseline_dir)
    ndvi_layers = []
    for tif in sorted(baseline_dir.glob("**/*.tif")):
        try:
            data, _, _, _ = _read_raster(tif)
            if data.shape[1:] != out_shape:
                continue
            red = data[1].astype(np.float32)
            nir = data[2].astype(np.float32)
            ndvi = _ndvi(red, nir)
            ndvi_layers.append(ndvi)
        except Exception as exc:  # pragma: no cover - best-effort baseline loader
            LOGGER.warning("Skipping baseline raster %s: %s", tif, exc)
    if not ndvi_layers:
        LOGGER.warning("No baseline NDVI rasters found; anomaly will be zero.")
        return np.zeros(out_shape, dtype=np.float32)
    stack = np.stack(ndvi_layers, axis=0)
    return stack.mean(axis=0)


def _sample_pixels(mask: np.ndarray, values: np.ndarray, n_samples: int, label: int):
    valid_positions = np.argwhere(mask)
    if len(valid_positions) == 0:
        return []
    idx = np.random.default_rng(42).choice(len(valid_positions), size=min(len(valid_positions), n_samples), replace=False)
    rows, cols = valid_positions[idx].T
    return [(int(r), int(c), float(values[r, c]), label) for r, c in zip(rows, cols)]


def _rasterize_fire_geom(geom, transform, shape_hw):
    mask = rasterize(
        [(geom, 1)],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def _record(fire_index: int, fire_label: str, burned: int, r, c, ndvi, anomaly, dfi, image_name: str) -> dict:
    return {
        "fire_id": fire_index,
        "fire_label": fire_label,
        "burned": burned,
        "ndvi": float(ndvi[r, c]),
        "anomaly": float(anomaly[r, c]),
        "dfi": float(dfi[r, c]),
        "source_image": image_name,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    perimeters = gpd.read_file(args.effis_shp)
    if perimeters.empty:
        raise ValueError("EFFIS shapefile is empty.")
    if perimeters.crs is None:
        raise ValueError("EFFIS shapefile has no CRS; cannot align perimeters with the imagery.")
    id_col = _find_matching_column(perimeters.columns, *ID_CANDIDATES)
    if id_col is None:
        raise ValueError(f"Could not locate a fire-id column (tried {ID_CANDIDATES}) in {args.effis_shp}.")
    perimeters = perimeters[perimeters.geometry.notna() & ~perimeters.geometry.is_empty].copy()
    perimeters["_key"] = perimeters[id_col].map(_safe_name)

    all_records = []
    tif_paths = sorted(image_dir.glob("**/*.tif"))
    for fire_index, tif_path in enumerate(tif_paths):
        try:
            data, profile, transform, crs = _read_raster(tif_path)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable raster %s: %s", tif_path, exc)
            continue
        if data.shape[0] < 7:
            LOGGER.warning("Raster %s does not contain B03, B04, B08, B11, B12, SCL, dataMask; skipping.", tif_path)
            continue

        # Filenames are "<fire id>__<start>_<end>.tif" (see 01_fetch_sentinel2.py).
        fire_key = tif_path.stem.split("__")[0]
        if not (perimeters["_key"] == fire_key).any():
            LOGGER.warning("No perimeter with id %r for %s; skipping.", fire_key, tif_path.name)
            continue

        b03, b04, b08, b11, b12, scl, data_mask = [data[i] for i in range(7)]
        shape_hw = b04.shape
        valid = _valid_mask(scl, data_mask)
        ndvi = _ndvi(b04.astype(np.float32), b08.astype(np.float32))
        dfi = _dfi(b04.astype(np.float32), b08.astype(np.float32), b11.astype(np.float32), b12.astype(np.float32))
        valid &= _worldcover_mask(args.worldcover_raster, transform, crs, shape_hw)

        baseline_mean = _load_baseline_ndvi(args.baseline_dir, transform, shape_hw)
        anomaly = ndvi - baseline_mean

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

        try:
            fire_mask = _rasterize_fire_geom(fire_geom, transform, shape_hw)
            core = fire_geom.buffer(-args.erode_m)
            burned_mask = _rasterize_fire_geom(core, transform, shape_hw) if not core.is_empty else fire_mask
            ring_mask = _rasterize_fire_geom(fire_geom.buffer(args.buffer_m), transform, shape_hw) & ~fire_mask
            if not other_fires.empty:
                ring_mask &= ~_rasterize_fire_geom(other_fires.union_all(), transform, shape_hw)
        except Exception as exc:  # pragma: no cover - robustness guard
            LOGGER.warning("Fire processing failed for %s: %s", tif_path.name, exc)
            continue

        burned_pixels = np.where(valid & burned_mask)
        unburned_pixels = np.where(valid & ring_mask)
        if len(burned_pixels[0]) == 0 or len(unburned_pixels[0]) == 0:
            LOGGER.warning("No cloud-free burned/unburned pixels for %s; skipping.", tif_path.name)
            continue

        burn_samples = np.random.default_rng(42 + fire_index).choice(len(burned_pixels[0]), size=min(len(burned_pixels[0]), args.max_samples_per_class), replace=False)
        unburn_samples = np.random.default_rng(43 + fire_index).choice(len(unburned_pixels[0]), size=min(len(unburned_pixels[0]), args.max_samples_per_class), replace=False)

        for j in burn_samples:
            all_records.append(_record(fire_index, fire_key, 1, burned_pixels[0][j], burned_pixels[1][j], ndvi, anomaly, dfi, tif_path.name))
        for j in unburn_samples:
            all_records.append(_record(fire_index, fire_key, 0, unburned_pixels[0][j], unburned_pixels[1][j], ndvi, anomaly, dfi, tif_path.name))
        LOGGER.info("%s: %d burned, %d unburned samples", tif_path.name, len(burn_samples), len(unburn_samples))

    if not all_records:
        raise ValueError("No valid burned/unburned pixels were produced. Check the imagery, masks, and fire polygons.")

    df = pd.DataFrame(all_records)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Dataset written to %s with %d rows.", output_path, len(df))
    print(f"OK: {output_path}")


if __name__ == "__main__":
    main()
