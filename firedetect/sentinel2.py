"""Fetch Sentinel-2 L2A scenes for the selected fires through the Sentinel Hub Process API.

Each scene covers the fire's bounding box plus a margin, in the fire's UTM zone, at the scene set's resolution
(coarsened only when a side would exceed 2500 px). Sentinel Hub builds a least-cloudy mosaic over the date window.

Per fire and scene set (config.SceneSet), all on the same grid so they line up pixel for pixel:
    pre-fire scene     60 to 10 days before the fire
    baseline scenes    the same window 1..N years earlier, for Temporal NDVI
    post-fire scene    5 to 45 days after the fire, for the dNDVI control (main set only)
    WorldCover and DEM patches (free, see worldcover.py and terrain.py)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import terrain, worldcover
from .cdse import CdseSession
from .config import BANDS, ID_CANDIDATES, MAX_CLOUD_COVER, MAX_SIDE_PX, POST_FIRE_DAYS, ROOT, SceneSet
from .utils import find_column, load_excluded, parse_date, safe_name

LOGGER = logging.getLogger(__name__)

EVALSCRIPT = f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{bands: {list(BANDS)}, units: "DN"}}],
    output: {{id: "default", bands: {len(BANDS)}, sampleType: "INT16"}}
  }};
}}
function evaluatePixel(s) {{
  return [{", ".join(f"s.{b}" for b in BANDS)}];
}}
"""


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def date_window(fire_date: datetime, days_before: int, days_after: int) -> tuple[str, str]:
    """Pre-fire window: from days_before to days_after days before the fire, as YYYY-MM-DD strings."""
    start = (fire_date - timedelta(days=days_before)).strftime("%Y-%m-%d")
    end = (fire_date - timedelta(days=days_after)).strftime("%Y-%m-%d")
    if end < start:
        raise ValueError("The requested pre-fire window is invalid; check the window size parameters.")
    return start, end


def post_fire_window(fire_date: datetime) -> tuple[str, str]:
    first, last = POST_FIRE_DAYS
    return (fire_date + timedelta(days=first)).strftime("%Y-%m-%d"), (fire_date + timedelta(days=last)).strftime("%Y-%m-%d")


def scene_path(out_dir: Path, fire_id, start: str, end: str) -> Path:
    # Double underscore separates the ID from the dates, so IDs may contain single underscores/dashes.
    return out_dir / f"{safe_name(fire_id)}__{start}_{end}.tif"


def scene_grid(fire: gpd.GeoDataFrame, scene_set: SceneSet) -> tuple[int, tuple[float, float, float, float], float]:
    """(UTM EPSG code, bbox in metres, pixel size in metres) of a fire's scenes in a scene set."""
    centroid = fire.to_crs(4326).geometry.iloc[0].centroid
    epsg = utm_epsg(centroid.x, centroid.y)
    bbox = fire.to_crs(epsg=epsg).geometry.iloc[0].buffer(scene_set.margin_m).bounds
    resolution_m = max(scene_set.min_resolution_m, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / MAX_SIDE_PX)
    return epsg, bbox, resolution_m


def processing_units(fire: gpd.GeoDataFrame, scene_set: SceneSet) -> float:
    """CDSE cost of one scene: (pixels / 512^2) x (6 bands / 3); dataMask is free, minimum 0.01 PU per request."""
    _, (xmin, ymin, xmax, ymax), res = scene_grid(fire, scene_set)
    return max((xmax - xmin) / res * (ymax - ymin) / res / 512 ** 2, 0.01) * (len(BANDS) - 1) / 3


def fetch_scene(session: CdseSession, fire: gpd.GeoDataFrame, scene_set: SceneSet, start: str, end: str, out_path: Path) -> Path:
    """Download one scene around a fire perimeter (a one-row GeoDataFrame); skip it if it already exists."""
    if out_path.exists():
        LOGGER.info("Already downloaded, skipping: %s", out_path)
        return out_path

    epsg, (xmin, ymin, xmax, ymax), resolution_m = scene_grid(fire, scene_set)
    # The output grid follows the bbox CRS (UTM), so resx/resy are in metres.
    request = {
        "input": {
            "bounds": {"bbox": [xmin, ymin, xmax, ymax], "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    # The API needs full ISO-8601 timestamps; the end day is included.
                    "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                    "mosaickingOrder": "leastCC",
                    "maxCloudCoverage": MAX_CLOUD_COVER,
                },
                "processing": {"upsampling": "BILINEAR", "downsampling": "BILINEAR"},
            }],
        },
        "output": {
            "resx": resolution_m,
            "resy": resolution_m,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }
    content = session.process(request)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so an interrupted run never leaves a partial file that a re-run would skip.
    partial = out_path.with_suffix(".part")
    partial.write_bytes(content)
    partial.replace(out_path)
    LOGGER.info("Saved Sentinel-2 GeoTIFF to %s (%.1f m/px, %.1f MB)", out_path, resolution_m, len(content) / 1e6)
    return out_path


def scene_requests(fire_id, fire_date: datetime, scene_set: SceneSet, baseline_years: int, post_fire: bool,
                   days_before: int, days_after: int) -> list[Path]:
    """Output paths of every Sentinel-2 scene of one fire in a scene set; the pre-fire scene comes first."""
    paths = [scene_path(scene_set.prefire_dir, fire_id, *date_window(fire_date, days_before, days_after))]
    for years_back in range(1, baseline_years + 1):
        start, end = date_window(fire_date - pd.DateOffset(years=years_back), days_before, days_after)
        paths.append(scene_path(scene_set.baseline_dir, fire_id, start, end))
    if post_fire:
        paths.append(scene_path(scene_set.postfire_dir, fire_id, *post_fire_window(fire_date)))
    return paths


def _load_selection(fires_file: str, selected_csv: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    selected_path = ROOT / selected_csv
    if not selected_path.exists():
        raise SystemExit(f"{selected_csv} not found; run the 'select' stage first.")
    selected = pd.read_csv(selected_path, dtype={"fire_id": str, "fire_date": str})
    perimeters = gpd.read_file(ROOT / fires_file)
    if perimeters.crs is None:
        raise SystemExit(f"{fires_file} has no CRS; cannot locate the fires on the map.")
    id_col = find_column(perimeters.columns, ID_CANDIDATES)
    perimeters["_key"] = perimeters[id_col].astype(str).str.strip()
    return selected, perimeters


def fetch_selected(session: CdseSession | None, fires_file: str, selected_csv: str, scene_sets: list[tuple[SceneSet, bool]],
                   baseline_years: int, days_before: int, days_after: int, continue_on_error: bool,
                   estimate_only: bool = False, excluded_file: str | None = None) -> None:
    """Fetch every scene and patch of every selected, non-excluded fire, for each (scene set, with post-fire scene) pair.

    With estimate_only, nothing is downloaded: the processing units still needed are printed instead.
    """
    selected, perimeters = _load_selection(fires_file, selected_csv)
    selected = selected[~selected["fire_id"].map(safe_name).isin(load_excluded(excluded_file))]
    failed = []
    dem_jobs = {}
    cost = {str(scene_set.root.relative_to(ROOT)): 0.0 for scene_set, _ in scene_sets}
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        if not estimate_only:
            print(f"\n[{i}/{len(selected)}] fire {row.fire_id}", flush=True)
        try:
            fire = perimeters[perimeters["_key"] == str(row.fire_id).strip()].iloc[[0]]
            if not (isinstance(row.fire_date, str) and row.fire_date):
                raise ValueError(f"Fire {row.fire_id} has no date; pass --fire-date.")
            fire_date = parse_date(row.fire_date)
            for scene_set, post_fire in scene_sets:
                paths = scene_requests(row.fire_id, fire_date, scene_set, baseline_years, post_fire, days_before, days_after)
                if estimate_only:
                    missing = sum(not p.exists() for p in paths)
                    cost[str(scene_set.root.relative_to(ROOT))] += missing * processing_units(fire, scene_set)
                    continue
                for path in paths:
                    print(f"OK: {fetch_scene(session, fire, scene_set, *path.stem.split('__')[1].split('_'), path)}")
                prefire = paths[0]
                print(f"OK: {worldcover.fetch_for_scene(prefire, scene_set.worldcover_dir)}")
                dem_jobs.setdefault(safe_name(row.fire_id), []).append((prefire, scene_set.dem_dir))
        except Exception as exc:
            LOGGER.error("Fetch failed for fire %s: %s", row.fire_id, exc)
            if not continue_on_error:
                raise
            failed.append(row.fire_id)
    if dem_jobs:
        print(f"\nDEM patches for {len(dem_jobs)} fires ({terrain.PARALLEL_DOWNLOADS} in parallel) ...", flush=True)
        dem_failed = terrain.fetch_patches(dem_jobs, ROOT / "data" / "dem_raw")
        if dem_failed and not continue_on_error:
            raise RuntimeError(f"DEM download failed for fires {', '.join(dem_failed)}")
        failed += dem_failed
    if estimate_only:
        for name, pu in cost.items():
            print(f"Estimated processing units still needed for {name}: {pu:.0f}")
        print(f"Total: {sum(cost.values()):.0f} PU (free tier: 10,000 per month). Nothing was downloaded.")
    if failed:
        print(f"WARNING: fetch failed for fire IDs: {', '.join(failed)}")
