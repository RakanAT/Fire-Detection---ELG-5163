"""Build burned vs unburned pixel datasets from the downloaded scenes.

Per scene (load_scene): compute the pre-fire indices (NDVI, Temporal NDVI from the prior-year scenes, DFI, NDMI,
NBR, dead-fuel neighbourhood shares, dry clumps), the landscape factors (slope, valley/ridge position, distance to
firebreaks), dNDVI and dNBR from the post-fire scene; mask clouds and non-vegetation (ESA WorldCover); rasterize
the fire perimeter in the scene's UTM grid and measure every pixel's distance from it. Then
    burned pixels   - inside the perimeter eroded by erode_m (avoids mixed edge pixels)
    unburned pixels - outside the perimeter and any concurrent fire, sampled per distance zone (e.g. 0-250 m)
build_dataset samples both into a CSV and, for the main dataset, writes one summary row per fire, including a
check for fires whose "pre-fire" scene already shows the burn scar (fire date later than the real fire).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import array_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

from . import indices, landscape, terrain, worldcover
from .config import (BANDS, CONCURRENT_FIRE_DAYS, DEAD_FUEL_NDMI, ID_CANDIDATES, LAND_COVER_NAMES, NEIGHBOURHOOD_RADII_M,
                     POST_FIRE_SCL_CLASSES, PRE_BURN_NDVI_DROP, PRE_BURN_SWIR_EXCESS, SceneSet)
from .raster import clear_sky_mask, rasterize_geom, read_raster, vegetation_mask
from .utils import find_column, load_excluded, safe_name

LOGGER = logging.getLogger(__name__)

# Per-pixel layers written to the dataset CSV (each is a Scene attribute).
PIXEL_LAYERS = ["ndvi", "temporal_ndvi", "dfi", "ndmi", "nbr", "dead_fuel_100m", "dead_fuel_300m", "dry_patch_ha",
                "dist_large_dry_patch_m", "slope", "tpi_300m", "dist_nonfuel_m", "dndvi", "dnbr"]


@dataclass
class Scene:
    """One fire's pre-fire scene with its index, landscape and mask layers, all on the scene's grid."""

    path: Path
    fire_key: str
    transform: object
    crs: object
    bands: np.ndarray  # float32, in config.BANDS order
    ndvi: np.ndarray
    temporal_ndvi: np.ndarray
    dfi: np.ndarray
    ndmi: np.ndarray
    nbr: np.ndarray
    dead_fuel_100m: np.ndarray  # share of dry vegetation (NDMI < 0) around each pixel
    dead_fuel_300m: np.ndarray
    dry_patch_ha: np.ndarray  # see landscape.py
    dist_large_dry_patch_m: np.ndarray
    slope: np.ndarray  # NaN without a DEM
    tpi_300m: np.ndarray  # NaN without a DEM
    dist_nonfuel_m: np.ndarray
    dndvi: np.ndarray  # pre-fire minus post-fire NDVI; NaN without a post-fire scene or where it is cloudy
    dnbr: np.ndarray  # pre-fire minus post-fire NBR; NaN like dndvi
    baseline_ndvi: np.ndarray
    postfire_bands: np.ndarray | None
    land_cover: np.ndarray  # WorldCover classes
    elevation: np.ndarray | None  # metres, None without a DEM
    clear: np.ndarray  # cloud-free vegetation / bare-soil pixels (SCL)
    valid: np.ndarray  # clear, vegetation land cover and a clear prior year
    fire_mask: np.ndarray  # the full perimeter
    burned_mask: np.ndarray  # eroded perimeter, valid pixels only
    unburned_mask: np.ndarray  # outside the perimeter and concurrent fires, valid pixels only
    distance_m: np.ndarray  # distance from the perimeter: positive outside, negative inside

    @property
    def pixel_size_m(self) -> float:
        return abs(self.transform.a)

    def band(self, name: str) -> np.ndarray:
        return self.bands[BANDS.index(name)]

    def zone_mask(self, min_m: float, max_m: float) -> np.ndarray:
        """Unburned pixels between min_m (exclusive) and max_m (inclusive) from the perimeter."""
        return self.unburned_mask & (self.distance_m > min_m) & (self.distance_m <= max_m)


def load_perimeters(fires_file) -> gpd.GeoDataFrame:
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
    return perimeters


def _same_grid_scenes(directory: Path, fire_key: str, transform, shape_hw):
    """Bands of the fire's scenes in directory that are on the pre-fire scene's grid."""
    for tif in sorted(directory.glob(f"{fire_key}__*.tif")):
        data, scene_transform, _ = read_raster(tif)
        if data.shape[1:] != shape_hw or scene_transform != transform:
            LOGGER.warning("%s is not on the same grid as the pre-fire scene; ignoring it.", tif)
            continue
        yield data[:len(BANDS)].astype(np.float32)


def _baseline_ndvi(baseline_dir: Path, fire_key: str, transform, shape_hw) -> np.ndarray | None:
    """Mean prior-year NDVI on the scene's grid (NaN where no prior year is clear), or None if there are no baselines."""
    ndvi_layers, clear_layers = [], []
    for data in _same_grid_scenes(baseline_dir, fire_key, transform, shape_hw):
        ndvi_layers.append(indices.ndvi(data[1], data[2]))
        clear_layers.append(clear_sky_mask(data[5], data[6]))
    return indices.mean_ndvi(ndvi_layers, clear_layers) if ndvi_layers else None


def _distance_from_perimeter(fire_mask: np.ndarray, pixel_size_m: float) -> np.ndarray:
    outside = distance_transform_edt(~fire_mask)
    inside = distance_transform_edt(fire_mask)
    return (np.where(fire_mask, -inside, outside) * pixel_size_m).astype(np.float32)


def load_scene(tif_path: Path, perimeters: gpd.GeoDataFrame, scene_set: SceneSet, erode_m: float = 20.0) -> Scene | None:
    """Compute indices, landscape factors and masks for one pre-fire scene; None (with a warning) if it can't be used.

    Needs the fire's baseline scenes and WorldCover patch (downloaded on demand if missing); post-fire scenes and
    DEM patches are used when present.
    """
    try:
        data, transform, crs = read_raster(tif_path)
    except Exception as exc:
        LOGGER.warning("Skipping unreadable raster %s: %s", tif_path, exc)
        return None
    if data.shape[0] < len(BANDS):
        LOGGER.warning("Raster %s does not contain bands %s; skipping.", tif_path, ", ".join(BANDS))
        return None

    # Filenames are "<fire id>__<start>_<end>.tif" (see sentinel2.scene_path).
    fire_key = tif_path.stem.split("__")[0]
    if not (perimeters["_key"] == fire_key).any():
        LOGGER.warning("No perimeter with id %r for %s; skipping.", fire_key, tif_path.name)
        return None
    baseline = _baseline_ndvi(scene_set.baseline_dir, fire_key, transform, data.shape[1:])
    if baseline is None:
        LOGGER.warning("No baseline scenes for %s in %s; skipping (run the fetch stage).", tif_path.name, scene_set.baseline_dir)
        return None

    bands = data[:len(BANDS)].astype(np.float32)
    b03, b04, b08, b11, b12, scl, data_mask = bands
    shape_hw = b04.shape
    pixel_size_m = abs(transform.a)
    nan_layer = np.full(shape_hw, np.nan, dtype=np.float32)
    ndvi = indices.ndvi(b04, b08)
    ndmi = indices.ndmi(b08, b11)
    nbr = indices.nbr(b08, b12)
    clear = clear_sky_mask(scl, data_mask)
    land_cover = worldcover.land_cover(transform, crs, shape_hw, scene_set.worldcover_dir / tif_path.name)
    vegetation = clear & vegetation_mask(land_cover)
    # Every index is sampled on the same pixels, so only pixels with a clear prior year are kept.
    valid = vegetation & np.isfinite(baseline)

    # Fuel continuity and clumps of dry fuel, counted over cloud-free vegetation pixels.
    dry = ndmi < DEAD_FUEL_NDMI
    dead_fuel = {name: indices.neighbourhood_fraction(dry, vegetation, radius_m / pixel_size_m)
                 for name, radius_m in NEIGHBOURHOOD_RADII_M.items()}
    dry_patch_ha, dist_large_dry_patch_m = landscape.dry_patches(dry, vegetation, pixel_size_m)

    postfire = next(_same_grid_scenes(scene_set.postfire_dir, fire_key, transform, shape_hw), None)
    dndvi = dnbr = nan_layer
    if postfire is not None:
        post_clear = clear_sky_mask(postfire[5], postfire[6], POST_FIRE_SCL_CLASSES)
        dndvi = np.where(post_clear, ndvi - indices.ndvi(postfire[1], postfire[2]), np.nan).astype(np.float32)
        dnbr = np.where(post_clear, nbr - indices.nbr(postfire[2], postfire[4]), np.nan).astype(np.float32)

    elevation, slope, tpi = None, nan_layer, nan_layer
    dem_path = scene_set.dem_dir / tif_path.name
    if dem_path.exists():
        elevation = terrain.elevation(transform, crs, shape_hw, dem_path)
        slope = terrain.slope_aspect(elevation, pixel_size_m)[0]
        tpi = landscape.topographic_position(elevation, pixel_size_m)

    # Only perimeters overlapping this image matter; reproject them to the image's UTM grid
    # so buffer/erosion distances are in metres.
    footprint = gpd.GeoSeries([box(*array_bounds(*shape_hw, transform))], crs=crs).to_crs(perimeters.crs).iloc[0]
    nearby = perimeters[perimeters.intersects(footprint)].to_crs(crs)
    is_target = nearby["_key"] == fire_key
    if not is_target.any():
        LOGGER.warning("Perimeter %r does not overlap %s; skipping.", fire_key, tif_path.name)
        return None
    fire_geom = nearby[is_target].union_all()
    # Fires around the same date (or undated) may be part of the same event: keep them out of the unburned zones.
    dates = pd.to_datetime(nearby["fire_date"], errors="coerce")
    days_apart = (dates[~is_target] - dates[is_target].iloc[0]).dt.days.abs()
    concurrent = nearby[~is_target][days_apart.isna() | (days_apart <= CONCURRENT_FIRE_DAYS)]

    fire_mask = rasterize_geom(fire_geom, transform, shape_hw)
    core = fire_geom.buffer(-erode_m)
    burned_mask = rasterize_geom(core, transform, shape_hw) if not core.is_empty else fire_mask
    unburned_mask = ~fire_mask & valid
    if not concurrent.empty:
        unburned_mask &= ~rasterize_geom(concurrent.union_all(), transform, shape_hw)

    return Scene(
        path=tif_path, fire_key=fire_key, transform=transform, crs=crs, bands=bands,
        ndvi=ndvi, temporal_ndvi=indices.temporal_ndvi(ndvi, baseline), dfi=indices.dfi(b04, b08, b11, b12),
        ndmi=ndmi, nbr=nbr, dead_fuel_100m=dead_fuel["dead_fuel_100m"], dead_fuel_300m=dead_fuel["dead_fuel_300m"],
        dry_patch_ha=dry_patch_ha, dist_large_dry_patch_m=dist_large_dry_patch_m, slope=slope, tpi_300m=tpi,
        dist_nonfuel_m=landscape.distance_to_nonfuel(land_cover, pixel_size_m), dndvi=dndvi, dnbr=dnbr,
        baseline_ndvi=baseline, postfire_bands=postfire, land_cover=land_cover, elevation=elevation,
        clear=clear, valid=valid, fire_mask=fire_mask, burned_mask=burned_mask & valid, unburned_mask=unburned_mask,
        distance_m=_distance_from_perimeter(fire_mask, pixel_size_m),
    )


def pre_burn_check(scene: Scene, ring_m: float = 250.0) -> dict:
    """Signs that the burn scar is already in the "pre-fire" scene, comparing the perimeter with its ring.

    ndvi_drop_vs_ring: how much more NDVI fell below the prior-year mean inside the perimeter than in the ring.
    swir_excess:       B12/B11 inside minus in the ring; charred ground pushes B12/B11 above ~1.
    Both must exceed their thresholds.
    """
    b11, b12 = scene.band("B11"), scene.band("B12")
    swir_ratio = np.divide(b12, b11, out=np.full(b11.shape, np.nan, dtype=np.float32), where=b11 > 0)
    inside = scene.fire_mask & scene.clear
    ring = scene.zone_mask(0, ring_m)
    swir_excess = float(np.nanmean(swir_ratio[inside]) - np.nanmean(swir_ratio[ring]))
    drop = scene.baseline_ndvi - scene.ndvi
    ndvi_drop = float(np.nanmean(drop[inside]) - np.nanmean(drop[ring]))
    suspect = swir_excess > PRE_BURN_SWIR_EXCESS and ndvi_drop > PRE_BURN_NDVI_DROP
    return {"ndvi_drop_vs_ring": round(ndvi_drop, 3), "swir_excess": round(swir_excess, 3), "suspect_pre_burned": bool(suspect)}


def _land_cover_shares(scene: Scene) -> dict:
    classes = scene.land_cover[scene.burned_mask]
    return {f"burned_{name}_pct": round(100 * float(np.mean(classes == code)), 1) if classes.size else float("nan")
            for code, name in LAND_COVER_NAMES.items()}


def _sample(mask: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    picks = np.random.default_rng(seed).choice(len(rows), size=min(len(rows), n), replace=False)
    return rows[picks], cols[picks]


def _pixel_table(scene: Scene, rows, cols, fire_index: int, zone: str) -> pd.DataFrame:
    return pd.DataFrame({
        "fire_id": fire_index,
        "fire_label": scene.fire_key,
        "zone": zone,
        "burned": int(zone == "burned"),
        **{name: getattr(scene, name)[rows, cols].astype(np.float64) for name in PIXEL_LAYERS},
        "land_cover": scene.land_cover[rows, cols],
        "source_image": scene.path.name,
    })


def build_dataset(scene_set: SceneSet, fires_file: str, selected_csv: str, output_csv: str, zones: list[tuple[str, float, float]],
                  summary_csv: str | None = None, excluded_file: str | None = None, erode_m: float = 20.0,
                  max_samples_per_class: int = 5000) -> Path:
    """Sample burned pixels and each unburned zone of the selected fires' scenes into output_csv.

    zones are (name, min distance, max distance) in metres from the perimeter. With summary_csv, one summary row
    per fire (land cover, pre-burn check) is written too.
    """
    if not scene_set.prefire_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {scene_set.prefire_dir}")
    perimeters = load_perimeters(fires_file)
    selected = set(pd.read_csv(selected_csv, dtype={"fire_id": str})["fire_id"].map(safe_name))
    excluded = load_excluded(excluded_file)
    fire_info = perimeters.drop_duplicates("_key").set_index("_key")

    tables, summaries = [], []
    for fire_index, tif_path in enumerate(sorted(scene_set.prefire_dir.glob("*.tif"))):
        fire_key = tif_path.stem.split("__")[0]
        if fire_key not in selected:
            continue
        if fire_key in excluded:
            LOGGER.info("Fire %s is on the exclude list (%s); skipping.", fire_key, excluded[fire_key])
            continue
        scene = load_scene(tif_path, perimeters, scene_set, erode_m)
        if scene is None:
            continue
        zone_masks = [(name, scene.zone_mask(lo, hi)) for name, lo, hi in zones]
        if not scene.burned_mask.any() or not any(mask.any() for _, mask in zone_masks):
            LOGGER.warning("No cloud-free burned/unburned pixels for %s; skipping.", tif_path.name)
            continue

        counts = {}
        for k, (zone, mask) in enumerate([("burned", scene.burned_mask), *zone_masks]):
            rows, cols = _sample(mask, max_samples_per_class, 42 + k + fire_index)
            counts[zone] = len(rows)
            tables.append(_pixel_table(scene, rows, cols, fire_index, zone))
        LOGGER.info("%s: %s samples", tif_path.name, ", ".join(f"{n} {zone}" for zone, n in counts.items()))

        if summary_csv:
            info = fire_info.loc[fire_key]
            check = pre_burn_check(scene, zones[0][2])
            if check["suspect_pre_burned"]:
                LOGGER.warning("Fire %s: the pre-fire scene already looks burned inside the perimeter (NDVI drop vs ring %.2f, "
                               "SWIR excess %.2f). Check its date; add it to the exclude list if confirmed.",
                               fire_key, check["ndvi_drop_vs_ring"], check["swir_excess"])
            summaries.append({
                "fire_label": fire_key,
                "fire_date": info.get("fire_date"),
                "area_ha": round(float(info["area_ha"]), 1) if "area_ha" in info else float("nan"),
                "source_image": tif_path.name,
                "burned_pixels": int(scene.burned_mask.sum()),
                **{f"{zone}_pixels": int(mask.sum()) for zone, mask in zone_masks},
                "has_postfire": scene.postfire_bands is not None,
                "has_dem": scene.elevation is not None,
                **_land_cover_shares(scene),
                **check,
            })

    if not tables:
        raise ValueError("No valid burned/unburned pixels were produced. Check the imagery, masks, and fire polygons.")

    df = pd.concat(tables, ignore_index=True)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Dataset written to %s with %d rows from %d fires.", output_path, len(df), df["fire_label"].nunique())
    if summary_csv:
        pd.DataFrame(summaries).to_csv(summary_csv, index=False)
        suspects = [s["fire_label"] for s in summaries if s["suspect_pre_burned"]]
        if suspects:
            LOGGER.warning("%d fire(s) look already burned in their pre-fire scene: %s (see suspect_pre_burned in the per-fire table).",
                           len(suspects), ", ".join(suspects))
    return output_path
